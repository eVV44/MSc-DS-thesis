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
import numpy as np
import pandas as pd


EXPERIMENTS = {
    "cross_encoder": {
        "label": "Cross-encoder",
        "summary_file": root / "thesis_runs/cross_encoder/stability/stability_summary.csv",
        "bar_out": root / "thesis_runs/cross_encoder/stability/stability_bar_chart.png",
        "heatmap_out": root / "thesis_runs/cross_encoder/stability/stability_heatmap.png",
    },
    "duot5": {
        "label": "DuoT5",
        "summary_file": root / "thesis_runs/duot5/stability/stability_summary.csv",
        "bar_out": root / "thesis_runs/duot5/stability/stability_bar_chart.png",
        "heatmap_out": root / "thesis_runs/duot5/stability/stability_heatmap.png",
    },
}

METHOD_ORDER = ["pairwise_ig", "pointwise_ig", "loo_pairwise", "loo_pointwise"]
METHOD_LABELS = {
    "pairwise_ig": "Pairwise IG",
    "pointwise_ig": "Pointwise IG",
    "loo_pairwise": "Pairwise LOO",
    "loo_pointwise": "Pointwise LOO",
}
K_ORDER = [1, 3, 5]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["cross_encoder", "duot5", "all"], default="all")
    return parser.parse_args()


def load_filtered_df(summary_file: Path) -> pd.DataFrame:
    df = pd.read_csv(summary_file)
    if df.empty:
        return df
    df = df[df["k_masked_words"].isin(K_ORDER)].copy()
    df["method"] = pd.Categorical(df["method"], categories=METHOD_ORDER, ordered=True)
    return df.sort_values(["method", "k_masked_words"]).reset_index(drop=True)


def plot_bar_chart(df: pd.DataFrame, title: str, out_file: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    methods = [m for m in METHOD_ORDER if m in set(df["method"].astype(str))]
    x = np.arange(len(methods))
    width = 0.22
    offsets = np.linspace(-width, width, len(K_ORDER))
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    for offset, k, color in zip(offsets, K_ORDER, colors):
        vals = []
        for method in methods:
            subset = df[(df["method"].astype(str) == method) & (df["k_masked_words"] == k)]
            vals.append(float(subset["mean_spearman_all"].iloc[0]) if not subset.empty else np.nan)
        ax.bar(x + offset, vals, width=width, label=f"k={k}", color=color)

    ax.set_title(f"Stability by Method: {title}")
    ax.set_xlabel("Explanation method")
    ax.set_ylabel("Mean Spearman ($\\rho_{all}$)")
    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in methods], rotation=15, ha="right")
    ax.set_ylim(0.85, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=True)
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(df: pd.DataFrame, title: str, out_file: Path) -> None:
    methods = [m for m in METHOD_ORDER if m in set(df["method"].astype(str))]
    matrix = []
    for method in methods:
        row = []
        for k in K_ORDER:
            subset = df[(df["method"].astype(str) == method) & (df["k_masked_words"] == k)]
            row.append(float(subset["mean_spearman_all"].iloc[0]) if not subset.empty else np.nan)
        matrix.append(row)
    data = np.array(matrix, dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    im = ax.imshow(data, cmap="YlGnBu", vmin=0.85, vmax=1.0, aspect="auto")

    ax.set_title(f"Stability Heatmap: {title}")
    ax.set_xlabel("Masked low-information words")
    ax.set_ylabel("Explanation method")
    ax.set_xticks(np.arange(len(K_ORDER)))
    ax.set_xticklabels(K_ORDER)
    ax.set_yticks(np.arange(len(methods)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in methods])

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            label = "NA" if np.isnan(value) else f"{value:.3f}"
            ax.text(j, i, label, ha="center", va="center", color="black", fontsize=10)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Spearman ($\\rho_{all}$)")
    plt.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_one(experiment_key: str) -> None:
    cfg = EXPERIMENTS[experiment_key]
    df = load_filtered_df(cfg["summary_file"])
    if df.empty:
        print(f"Skipping {cfg['label']}: empty or missing summary file")
        return
    plot_bar_chart(df, cfg["label"], cfg["bar_out"])
    plot_heatmap(df, cfg["label"], cfg["heatmap_out"])
    print(f"Saved bar chart to {cfg['bar_out']}")
    print(f"Saved heatmap to {cfg['heatmap_out']}")


def main():
    args = parse_args()
    if args.experiment == "all":
        for experiment_key in ["cross_encoder", "duot5"]:
            plot_one(experiment_key)
    else:
        plot_one(args.experiment)


if __name__ == "__main__":
    main()