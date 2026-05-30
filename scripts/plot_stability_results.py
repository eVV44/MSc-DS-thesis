# -- IMPORTS --
from __future__ import annotations
import argparse
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(root / "thesis_runs/shared/mplconfig"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

EXPERIMENTS = {
    "cross_encoder": {
        "label": "Cross-encoder",
        "summary_file": root / "thesis_runs/cross_encoder/stability/stability_summary.csv",
        "out_file": root / "thesis_runs/cross_encoder/stability/stability_plots.png"},

    "duot5": {
        "label": "DuoT5",
        "summary_file": root / "thesis_runs/duot5/stability/stability_summary.csv",
        "out_file": root / "thesis_runs/duot5/stability/stability_plots.png"}}

METHOD_LABELS = {
    "pairwise_ig": "Pairwise IG",
    "pointwise_ig": "Pointwise IG",
    "loo_pairwise": "LOO pairwise",
    "loo_pointwise": "LOO pointwise"}

METHOD_ORDER = ["pairwise_ig", "pointwise_ig", "loo_pairwise", "loo_pointwise"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["cross_encoder", "duot5", "all"], default="all")
    return parser.parse_args()


def plot_one(experiment_key: str):
    cfg = EXPERIMENTS[experiment_key]
    summary_file = cfg["summary_file"]
    if not summary_file.exists():
        print(f"Skipping {cfg['label']}: missing {summary_file}")
        return

    df = pd.read_csv(summary_file)
    if df.empty:
        print(f"Skipping {cfg['label']}: empty summary file")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 5.5))
    fig.suptitle(f"Stability Evaluation: {cfg['label']}", fontsize=18)

    for method in METHOD_ORDER:
        subset = df[df["method"] == method].sort_values("k_masked_words")
        if subset.empty:
            continue
        ax.plot(
            subset["k_masked_words"],
            subset["mean_spearman_all"],
            marker="o",
            label=METHOD_LABELS.get(method, method))

    ax.set_title("Mean Spearman Stability")
    ax.set_xlabel("Masked low-information words")
    ax.set_ylabel("Mean Spearman")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.85, 1.00)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True)

    plt.tight_layout(rect=[0, 0.10, 1, 0.92])
    cfg["out_file"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg["out_file"], dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved stability plot to {cfg['out_file']}")


def main():
    args = parse_args()
    if args.experiment == "all":
        for experiment_key in ["cross_encoder", "duot5"]:
            plot_one(experiment_key)
    else:
        plot_one(args.experiment)


if __name__ == "__main__":
    main()