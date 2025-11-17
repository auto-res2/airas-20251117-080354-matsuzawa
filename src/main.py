"""src/main.py
Orchestrator – launches *one* training run in a fresh subprocess with the
appropriate Hydra overrides.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig


def _prepare_cfg(cfg: DictConfig) -> DictConfig:
    """Shared helper used by src.main and src.train for mode-based overrides."""
    # absolute results_dir ---------------------------------------------------
    root = Path(get_original_cwd())
    if not Path(cfg.results_dir).is_absolute():
        cfg.results_dir = str((root / cfg.results_dir).resolve())

    # mode-specific tweaks ----------------------------------------------------
    if cfg.mode == "trial":
        cfg.wandb.mode = "disabled"
        cfg.optuna.n_trials = 0
        cfg.training.max_epochs = 1
        cfg.training.micro_batch_size = min(cfg.training.micro_batch_size, 2)
    elif cfg.mode != "full":
        raise ValueError("mode must be 'full' or 'trial'")

    return cfg


@hydra.main(config_path="../config", config_name="config", version_base="1.3")
def _hydra_main(cfg: DictConfig):
    cfg = _prepare_cfg(cfg)

    # determine selected runs-config -----------------------------------
    run_cfg_name = cfg.hydra.runtime.choices.get("runs")
    if not run_cfg_name:
        raise ValueError("No runs config selected. Use +runs=<id> or runs=<id> on CLI.")

    cmd = [sys.executable, "-u", "-m", "src.train",
           f"runs={run_cfg_name}", f"results_dir={cfg.results_dir}", f"mode={cfg.mode}"]
    print("[main]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    _hydra_main()
