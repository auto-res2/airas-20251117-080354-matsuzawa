"""src/train.py
Single-run executor invoked by src.main.  Implements full training,
Optuna HPO, and exhaustive WandB logging.
"""
from __future__ import annotations

import math
import warnings
from pathlib import Path

import hydra
import optuna
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src import model as mdl
from src import preprocess as prep
from src.main import _prepare_cfg  # reuse identical helper

warnings.filterwarnings("ignore", message=".*torch.distributed.*")
_CACHE_DIR = ".cache/"

# ---------------------------------------------------------------------------
# helper utils ---------------------------------------------------------------
# ---------------------------------------------------------------------------

def _is_min_metric(name: str) -> bool:
    name = name.lower()
    return any(t in name for t in ("loss", "error", "perplexity"))


def _token_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    mask = labels.ne(-100)
    correct = (preds.eq(labels) & mask).sum().item()
    total = mask.sum().item()
    return correct / max(total, 1)


@torch.no_grad()
def _eval_epoch(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    tot_loss, tot_corr, tot_tok = 0.0, 0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = model(**batch)
            logits = out.logits
            loss = out.loss
        tok_acc = _token_accuracy(logits, batch["labels"])
        tokens = batch["labels"].ne(-100).sum().item()
        tot_loss += loss.item() * tokens
        tot_corr += tok_acc * tokens
        tot_tok += tokens
    return {"val_loss": tot_loss / tot_tok, "val_token_acc": tot_corr / tot_tok}


def _train_epoch(model: torch.nn.Module, loader: DataLoader, optimiser: torch.optim.Optimizer,
                 device: torch.device, epoch_idx: int, cfg: DictConfig, log: bool):
    model.train()
    for step, batch in enumerate(loader):
        if cfg.mode == "trial" and step >= 2:
            break  # tiny budget in trial-mode
        batch = {k: v.to(device) for k, v in batch.items()}

        def closure():
            optimiser.zero_grad(set_to_none=True)
            out = model(**batch)
            out.loss.backward()
            return out.loss

        loss_tensor = optimiser.step(closure) if hasattr(optimiser, "step") else closure()

        # per-batch logging ------------------------------------------------
        if log and wandb.run and wandb.run.mode != "disabled":
            logits = model(**batch).logits.detach()
            tok_acc = _token_accuracy(logits, batch["labels"])
            wandb.log({
                "train_step_loss": loss_tensor.item(),
                "train_step_token_acc": tok_acc,
                "epoch": epoch_idx,
            })


# ---------------------------------------------------------------------------
# Optuna hyper-parameter search ---------------------------------------------
# ---------------------------------------------------------------------------

def _sample(trial: optuna.Trial, space: dict) -> dict:
    sampled: dict = {}
    for hp, spec in space.items():
        tp = spec.get("type", "uniform")
        if tp == "loguniform":
            sampled[hp] = trial.suggest_float(hp, spec["low"], spec["high"], log=True)
        elif tp == "uniform":
            sampled[hp] = trial.suggest_float(hp, spec["low"], spec["high"])
        elif tp == "categorical":
            sampled[hp] = trial.suggest_categorical(hp, spec["choices"])
        else:
            raise ValueError(f"Unsupported Optuna type: {tp}")
    return sampled


def _objective(trial: optuna.Trial, base_cfg: DictConfig, device: torch.device):
    cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    OmegaConf.set_struct(cfg, False)
    for k, v in _sample(trial, cfg.optuna.search_space).items():
        # hierarchical update (training.*, romos_lr.*, ebs_lr.*)
        if k in cfg.training:
            cfg.training[k] = v
        elif "romos_lr" in cfg.training and k in cfg.training.romos_lr:
            cfg.training.romos_lr[k] = v
        elif "ebs_lr" in cfg.training and k in cfg.training.ebs_lr:
            cfg.training.ebs_lr[k] = v
    OmegaConf.set_struct(cfg, True)

    tokenizer = mdl.build_tokenizer(cfg)
    train_dl, val_dl = prep.build_dataloaders(cfg, tokenizer)
    model = mdl.build_model(cfg, tokenizer).to(device)
    optimiser = mdl.build_optimizer(cfg, model)

    # run one mini-epoch for quick evaluation ---------------------------
    _train_epoch(model, train_dl, optimiser, device, 0, cfg, log=False)
    metrics = _eval_epoch(model, val_dl, device)

    target = cfg.optuna.metric
    val = metrics[target]
    return val if not _is_min_metric(target) else -val


# ---------------------------------------------------------------------------
# top-level training routine -------------------------------------------------
# ---------------------------------------------------------------------------

def _run(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.training.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.training.seed)

    # -------------------- Optuna --------------------------------------
    if cfg.optuna.n_trials > 0:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: _objective(t, cfg, device), n_trials=cfg.optuna.n_trials)
        print("[Optuna] best", study.best_params)
        OmegaConf.set_struct(cfg, False)
        for k, v in study.best_params.items():
            if k in cfg.training:
                cfg.training[k] = v
            elif "romos_lr" in cfg.training and k in cfg.training.romos_lr:
                cfg.training.romos_lr[k] = v
            elif "ebs_lr" in cfg.training and k in cfg.training.ebs_lr:
                cfg.training.ebs_lr[k] = v
        OmegaConf.set_struct(cfg, True)

    # -------------------- WandB ---------------------------------------
    wandb_mode = cfg.wandb.mode if cfg.wandb.mode else "online"
    wandb.init(entity=cfg.wandb.entity,
               project=cfg.wandb.project,
               id=cfg.run_id,
               resume="allow",
               config=OmegaConf.to_container(cfg, resolve=True),
               mode=wandb_mode)
    print("[wandb]", wandb.run.get_url())

    # -------------------- data / model --------------------------------
    tokenizer = mdl.build_tokenizer(cfg)
    train_dl, val_dl = prep.build_dataloaders(cfg, tokenizer)
    model = mdl.build_model(cfg, tokenizer).to(device)
    optimiser = mdl.build_optimizer(cfg, model)

    best_val = -math.inf
    for epoch in range(cfg.training.max_epochs):
        _train_epoch(model, train_dl, optimiser, device, epoch, cfg, log=True)
        val_stats = _eval_epoch(model, val_dl, device)
        wandb.log({**val_stats, "epoch": epoch})
        if val_stats["val_token_acc"] > best_val:
            best_val = val_stats["val_token_acc"]
            ckpt_dir = Path(cfg.results_dir) / cfg.run_id
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / "best.pt"
            torch.save({
                "model": model.state_dict(),
                "tokenizer": tokenizer.name_or_path,
                "cfg": OmegaConf.to_container(cfg, resolve=True),
            }, ckpt_path)
            wandb.summary["best_val_token_acc"] = best_val
            wandb.summary["checkpoint"] = str(ckpt_path)
    wandb.finish()


# ---------------------------------------------------------------------------
# hydra entry ---------------------------------------------------------------
# ---------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def _hydra_entry(cfg: DictConfig):
    cfg = _prepare_cfg(cfg)
    _run(cfg)


if __name__ == "__main__":
    _hydra_entry()
