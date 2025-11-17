"""src/model.py
Model, tokenizer and custom optimiser factory (RoMoS-LR & EBS-LR).
"""
from __future__ import annotations

import collections
import math
from typing import Any

import torch
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from peft import LoraConfig, get_peft_model
except ImportError:
    LoraConfig = None
    get_peft_model = None

_CACHE_DIR = ".cache/"

# ---------------------------------------------------------------------------
# tokenizer / model ----------------------------------------------------------
# ---------------------------------------------------------------------------

def build_tokenizer(cfg):
    name = cfg.model.get("tokenizer_name", cfg.model.name)
    tok = AutoTokenizer.from_pretrained(name, cache_dir=_CACHE_DIR, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return tok


def _apply_lora(cfg, model):
    if not cfg.model.get("lora"):
        return model
    if LoraConfig is None:
        raise ImportError("peft required for LoRA")
    lo = cfg.model.lora
    lcfg = LoraConfig(r=lo.rank, lora_alpha=lo.alpha, lora_dropout=lo.dropout,
                      target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                      task_type="CAUSAL_LM", bias="none")
    return get_peft_model(model, lcfg)


def build_model(cfg, tokenizer):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(str(cfg.model.dtype).lower(), torch.bfloat16)
    quant_cfg = BitsAndBytesConfig(load_in_8bit=True) if cfg.model.get("quant") == "8bit" else None
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, torch_dtype=dtype, cache_dir=_CACHE_DIR, device_map="auto", quantization_config=quant_cfg)
    model.resize_token_embeddings(len(tokenizer))
    model = _apply_lora(cfg, model)
    return model

# ---------------------------------------------------------------------------
# robust LR controllers ------------------------------------------------------
# ---------------------------------------------------------------------------

class _RingBuf:
    def __init__(self, M: int):
        self.buf = collections.deque(maxlen=M)

    def append(self, x: float):
        self.buf.append(x)

    def blocks(self, K: int):
        n = len(self.buf)
        if n < K:
            return []
        blk = n // K
        return [self.buf[i * blk:(i + 1) * blk] for i in range(K)]

    def mean_var(self):
        if not self.buf:
            return 0.0, 0.0
        t = torch.tensor(list(self.buf))
        return t.mean().item(), t.var(unbiased=False).item()


class RoMoSAdamW(AdamW):
    """AdamW with Robust Median-of-Means Secant LR controller."""

    def __init__(self, params, *, lr: float, cfg):
        super().__init__(params, lr=lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps, weight_decay=cfg.weight_decay)
        r = cfg.romos_lr
        self.lr_base = lr
        self.delta0 = r.delta0
        self.kappa = r.kappa
        self.M = r.window_M
        self.K = r.num_blocks_K
        self.clip_low, self.clip_high = r.clip_low, r.clip_high
        self.buf = _RingBuf(self.M)
        self.prev_gdot = self.prev_dir2 = self.prev_lr = None

    def _robust_lower(self, delta):
        blocks = self.buf.blocks(self.K)
        if not blocks:
            return -1.0
        b_means = torch.tensor([sum(b) / len(b) for b in blocks])
        med = b_means.median().item()
        mad = (b_means - med).abs().median().item() * 1.4826 + 1e-12
        radius = 2 * mad * math.sqrt(2 * math.log(2 / delta) / (self.M / self.K))
        return med - radius

    @torch.no_grad()
    def _safe_lr(self, loss: float, gdotd: float, dir2: float):
        if self.prev_gdot is not None:
            h_hat = (gdotd - self.prev_gdot) / ((self.prev_lr or 1e-12) * (self.prev_dir2 or 1e-12))
            self.buf.append(float(h_hat))
        delta_t = self.delta0 * len(self.buf) / self.M if self.M else self.delta0
        h_low = self._robust_lower(max(delta_t, 1e-6))
        if h_low > 0:
            lr_star = -gdotd / (h_low * dir2 + 1e-12)
        else:
            lr_star = self.kappa * loss / (abs(gdotd) + 1e-12)
        return max(self.clip_low * self.lr_base, min(self.clip_high * self.lr_base, lr_star))

    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("RoMoSAdamW requires closure returning loss")
        loss = closure()
        gdotd = dir2 = 0.0
        lr_now = self.param_groups[0]["lr"]
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                d = -lr_now * g
                gdotd += (g * d).sum().item()
                dir2 += (d * d).sum().item()
        lr_used = self._safe_lr(loss.item(), gdotd, dir2)
        for group in self.param_groups:
            group["lr"] = lr_used
        super().step(lambda: loss)
        self.prev_gdot, self.prev_dir2, self.prev_lr = gdotd, dir2, lr_used
        return loss


class EBSAdamW(AdamW):
    """AdamW with Empirical-Bernstein LR controller (naïve)."""

    def __init__(self, params, *, lr: float, cfg):
        super().__init__(params, lr=lr, betas=(cfg.beta1, cfg.beta2), eps=cfg.eps, weight_decay=cfg.weight_decay)
        e = cfg.ebs_lr
        self.lr_base = lr
        self.delta0 = e.delta0
        self.M = e.window_M
        self.clip_low, self.clip_high = e.clip_low, e.clip_high
        self.smooth = e.var_smoothing
        self.buf = _RingBuf(self.M)
        self.prev_gdot = self.prev_dir2 = self.prev_lr = None

    def _bernstein_lower(self, delta):
        mu, var = self.buf.mean_var()
        n = len(self.buf.buf)
        if n < 2:
            return -1.0
        rad = math.sqrt(2 * var * math.log(2 / delta) / n + 3 * math.log(2 / delta) / (n - 1))
        return mu - rad

    @torch.no_grad()
    def _safe_lr(self, loss, gdotd, dir2):
        if self.prev_gdot is not None:
            h_hat = (gdotd - self.prev_gdot) / ((self.prev_lr or 1e-12) * (self.prev_dir2 or 1e-12))
            self.buf.append(float(h_hat))
        delta_t = self.delta0 * len(self.buf.buf) / self.M if self.M else self.delta0
        h_low = self._bernstein_lower(max(delta_t, 1e-6))
        if h_low > 0:
            lr_star = -gdotd / (h_low * dir2 + self.smooth)
        else:
            lr_star = 0.02 * loss / (abs(gdotd) + 1e-12)
        return max(self.clip_low * self.lr_base, min(self.clip_high * self.lr_base, lr_star))

    def step(self, closure=None):
        if closure is None:
            raise RuntimeError("EBSAdamW requires closure returning loss")
        loss = closure()
        gdotd = dir2 = 0.0
        lr_now = self.param_groups[0]["lr"]
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad.detach()
                d = -lr_now * g
                gdotd += (g * d).sum().item()
                dir2 += (d * d).sum().item()
        lr_used = self._safe_lr(loss.item(), gdotd, dir2)
        for group in self.param_groups:
            group["lr"] = lr_used
        super().step(lambda: loss)
        self.prev_gdot, self.prev_dir2, self.prev_lr = gdotd, dir2, lr_used
        return loss


# ---------------------------------------------------------------------------
# factory -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def build_optimizer(cfg, model):
    base_lr = cfg.training.learning_rate_base
    scheduler = str(cfg.training.lr_scheduler).lower()
    if scheduler == "romos-lr":
        return RoMoSAdamW(model.parameters(), lr=base_lr, cfg=cfg.training)
    if scheduler == "ebs-lr":
        return EBSAdamW(model.parameters(), lr=base_lr, cfg=cfg.training)
    return AdamW(model.parameters(), lr=base_lr, betas=(cfg.training.beta1, cfg.training.beta2),
                 eps=cfg.training.eps, weight_decay=cfg.training.weight_decay)
