"""src/preprocess.py
End-to-end dataset pipeline for GSM8K fine-tuning.
Returns PyTorch DataLoader objects.
"""
from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

_CACHE_DIR = ".cache/"


def _tokenise(example: Dict[str, Any], tokenizer: PreTrainedTokenizerBase, cfg) -> Dict[str, Any]:
    prompt = f"Question: {example['question']}\nAnswer:"
    answer = example["answer"].split("####")[-1].strip()
    text = prompt + " " + answer + tokenizer.eos_token

    enc = tokenizer(
        text,
        max_length=cfg.dataset.preprocessing.max_seq_length,
        truncation=True,
        padding="max_length" if cfg.dataset.preprocessing.get("pad_to_max_length", False) else False,
        return_attention_mask=True,
    )
    labels = enc["input_ids"].copy()
    prompt_len = len(tokenizer(prompt)["input_ids"])
    labels[:prompt_len] = [-100] * prompt_len  # ignore prompt
    enc["labels"] = labels
    return enc


class _Collator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokeniser = tokenizer

    def __call__(self, batch: List[Dict[str, Any]]):
        ids = [b["input_ids"] for b in batch]
        att = [b["attention_mask"] for b in batch]
        labels = [b["labels"] for b in batch]
        batch_enc = self.tokeniser.pad({"input_ids": ids, "attention_mask": att}, return_tensors="pt")
        max_len = batch_enc["input_ids"].size(1)
        padded_lbl = [l + [-100] * (max_len - len(l)) for l in labels]
        batch_enc["labels"] = torch.tensor(padded_lbl, dtype=torch.long)
        return batch_enc


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------

def build_dataloaders(cfg, tokenizer: PreTrainedTokenizerBase) -> Tuple[DataLoader, DataLoader]:
    raw = load_dataset("openai/gsm8k", cfg.dataset.split, cache_dir=_CACHE_DIR)
    val_frac = 0.1
    idx = list(range(len(raw["train"])))
    random.seed(cfg.dataset.preprocessing.seed)
    random.shuffle(idx)
    val_sz = int(len(idx) * val_frac)
    val_idx, train_idx = idx[:val_sz], idx[val_sz:]

    train_ds = raw["train"].select(train_idx)
    val_ds = raw["train"].select(val_idx)

    remove = [c for c in train_ds.column_names if c not in {"question", "answer"}]
    train_ds = train_ds.map(lambda ex: _tokenise(ex, tokenizer, cfg), remove_columns=remove)
    val_ds = val_ds.map(lambda ex: _tokenise(ex, tokenizer, cfg), remove_columns=remove)

    coll = _Collator(tokenizer)
    train_loader = DataLoader(train_ds, shuffle=cfg.dataset.preprocessing.shuffle, batch_size=cfg.training.micro_batch_size, collate_fn=coll)
    val_loader = DataLoader(val_ds, shuffle=False, batch_size=cfg.training.micro_batch_size, collate_fn=coll)
    return train_loader, val_loader
