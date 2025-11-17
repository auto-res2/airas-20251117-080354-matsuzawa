"""src/evaluate.py
Independent evaluation & visualisation script executed AFTER all training
jobs have finished.

CLI
----
uv run python -m src.evaluate results_dir=<path> \
                      run_ids='["run-1", "run-2"]'
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wandb
from omegaconf import OmegaConf
from scipy import stats as st

sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# helper functions -----------------------------------------------------------
# ---------------------------------------------------------------------------

def _load_global_cfg() -> Dict:
    cfg_path = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
    return OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)


def _export_metrics(out_dir: Path, history: pd.DataFrame, summary: Dict, cfg: Dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "history": history.to_dict(orient="list"),
            "summary": summary,
            "config": cfg,
        }, f, indent=2)


def _learning_curve(history: pd.DataFrame, out_dir: Path, run_id: str) -> Path:
    plt.figure(figsize=(7, 4))
    if "train_loss" in history.columns:
        sns.lineplot(history, x="_step", y="train_loss", label="train")
    if "val_loss" in history.columns:
        sns.lineplot(history, x="_step", y="val_loss", label="val")
    plt.title(f"Learning curve – {run_id}")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.tight_layout()
    fp = out_dir / f"{run_id}_learning_curve.pdf"
    plt.savefig(fp)
    plt.close()
    return fp


def _confusion_matrix(history: pd.DataFrame, out_dir: Path, run_id: str) -> Path:
    """Synthetic confusion matrix: train-vs-val per-step correctness (>0.5 acc)."""
    train_acc = history.loc[history.train_step_token_acc.notna(), "train_step_token_acc"].values
    val_acc = history.loc[history.val_step_token_acc.notna(), "val_step_token_acc"].values
    n = min(len(train_acc), len(val_acc))
    if n == 0:
        return Path("")  # nothing to plot
    y_pred = train_acc[:n] > 0.5
    y_true = val_acc[:n] > 0.5
    cm = np.zeros((2, 2), int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    labels = ["incorrect", "correct"]
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("train prediction")
    plt.ylabel("val ground truth")
    plt.title(f"Confusion – {run_id}")
    plt.tight_layout()
    fp = out_dir / f"{run_id}_confusion_matrix.pdf"
    plt.savefig(fp)
    plt.close()
    return fp


def _box_plot(primary: Dict[str, float], comparison_dir: Path) -> Path:
    data = pd.DataFrame({"run_id": list(primary.keys()), "value": list(primary.values())})
    data["group"] = data.run_id.apply(lambda r: "proposed" if "proposed" in r else ("baseline" if ("baseline" in r or "comparative" in r) else "other"))
    plt.figure(figsize=(5, 4))
    sns.boxplot(data=data, x="group", y="value")
    sns.stripplot(data=data, x="group", y="value", color="black", jitter=True, size=4)
    plt.title("Primary metric distribution")
    plt.tight_layout()
    fp = comparison_dir / "comparison_primary_metric_boxplot.pdf"
    plt.savefig(fp)
    plt.close()
    return fp


# ---------------------------------------------------------------------------
# main evaluation ------------------------------------------------------------
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("run_ids", help="JSON list of WandB run IDs")
    args = ap.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    run_ids: List[str] = json.loads(args.run_ids)

    wandb_cfg = _load_global_cfg()["wandb"]
    entity, project = wandb_cfg["entity"], wandb_cfg["project"]
    api = wandb.Api()

    aggregated: Dict[str, Dict[str, float]] = {}
    primary_metric_name = "best_val_token_acc"
    primary_vals: Dict[str, float] = {}

    generated: List[str] = []

    # ------------------------------------------------------------------
    # per-run processing ------------------------------------------------
    # ------------------------------------------------------------------
    for rid in run_ids:
        run = api.run(f"{entity}/{project}/{rid}")
        hist = run.history()  # DataFrame
        summ = run.summary._json_dict
        cfg = dict(run.config)

        out_dir = results_dir / rid
        _export_metrics(out_dir, hist, summ, cfg)
        generated.append(str(out_dir / "metrics.json"))

        # learning curve
        lc = _learning_curve(hist, out_dir, rid)
        generated.append(str(lc))
        # confusion matrix
        cm = _confusion_matrix(hist, out_dir, rid)
        if cm.exists():
            generated.append(str(cm))

        # collect metrics ----------------------------------------------
        for k, v in summ.items():
            aggregated.setdefault(k, {})[rid] = v
        if primary_metric_name in summ:
            primary_vals[rid] = summ[primary_metric_name]

    # ------------------------------------------------------------------
    # aggregated analysis ----------------------------------------------
    # ------------------------------------------------------------------
    comp_dir = results_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # bar chart ---------------------------------------------------------
    plt.figure(figsize=(6, 4))
    sns.barplot(x=list(primary_vals.keys()), y=list(primary_vals.values()))
    plt.xticks(rotation=45, ha="right")
    plt.title(primary_metric_name)
    for i, v in enumerate(primary_vals.values()):
        plt.text(i, v, f"{v:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    bar_fp = comp_dir / "comparison_primary_metric_bar_chart.pdf"
    plt.savefig(bar_fp)
    plt.close()
    generated.append(str(bar_fp))

    # box plot ----------------------------------------------------------
    bp_fp = _box_plot(primary_vals, comp_dir)
    generated.append(str(bp_fp))

    # significance test -------------------------------------------------
    proposed = [v for k, v in primary_vals.items() if "proposed" in k]
    baseline = [v for k, v in primary_vals.items() if ("baseline" in k or "comparative" in k)]
    p_val = None
    if proposed and baseline:
        t_stat, p_val = st.ttest_ind(proposed, baseline, equal_var=False)

    # build aggregated JSON --------------------------------------------
    best_prop = max(((v, k) for k, v in primary_vals.items() if "proposed" in k), default=(None, None))
    best_base = max(((v, k) for k, v in primary_vals.items() if ("baseline" in k or "comparative" in k)), default=(None, None))
    gap = None
    if best_prop[1] and best_base[1]:
        gap = (best_prop[0] - best_base[0]) / best_base[0] * 100
    agg_json = {
        "primary_metric": "1. Validation accuracy after 3 epochs. 2. GPU-hours to reach 60 % accuracy. 3. Empirical violation rate (# steps with ΔL>0) versus theoretical δ_t. 4. Outlier resilience: slowdown (extra steps to 60 %) on the stress-tail benchmark.",
        "metrics": aggregated,
        "best_proposed": {"run_id": best_prop[1], "value": best_prop[0]},
        "best_baseline": {"run_id": best_base[1], "value": best_base[0]},
        "gap": gap,
        "statistical_significance": {
            "test": "Welch t-test",
            "p_value": p_val,
        },
    }
    with open(comp_dir / "aggregated_metrics.json", "w") as f:
        json.dump(agg_json, f, indent=2)
    generated.append(str(comp_dir / "aggregated_metrics.json"))

    # performance table -------------------------------------------------
    df_tbl = pd.DataFrame({"run_id": list(primary_vals.keys()), primary_metric_name: list(primary_vals.values())})
    tbl_fp = comp_dir / "comparison_primary_metric_table.csv"
    df_tbl.to_csv(tbl_fp, index=False)
    generated.append(str(tbl_fp))

    # stdout paths ------------------------------------------------------
    for fp in generated:
        print(fp)


if __name__ == "__main__":
    main()
