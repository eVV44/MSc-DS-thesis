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


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = {
    "cross_encoder": {
        "label": "Cross-encoder",
        "summary_file": ROOT / "thesis_runs/cross_encoder/faithfulness/retrieval_faithfulness_summary.csv",
        "out_file": ROOT / "thesis_runs/cross_encoder/faithfulness/retrieval_faithfulness_plots.png",
    },
    "duot5": {
        "label": "DuoT5",
        "summary_file": ROOT / "thesis_runs/duot5/faithfulness/retrieval_faithfulness_summary.csv",
        "out_file": ROOT / "thesis_runs/duot5/faithfulness/retrieval_faithfulness_plots.png",
    },
}

METHOD_LABELS = {
    "pairwise_ig": "Pairwise IG",
    "pointwise_ig": "Pointwise IG",
    "loo_pairwise": "LOO pairwise",
    "loo_pointwise": "LOO pointwise",
}

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

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Retrieval-based Faithfulness: {cfg['label']}", fontsize=20)
    axes = axes.ravel()

    panels = [
        ("mean_delta_g", "Replacement: Mean Δg", "Mean Δg"),
        ("preference_preserved_rate", "Replacement: Preference Preserved", "Rate"),
        ("mean_replacement_similarity", "Replacement: Mean Retrieval Similarity", "Mean similarity"),
        ("mean_replacement_bm25_rank", "Replacement: Mean Candidate Rank", "Mean candidate rank"),
    ]

    for ax, (metric, title, ylabel) in zip(axes, panels):
        for method in METHOD_ORDER:
            for condition, linestyle in [("explanation", "-"), ("random", "--")]:
                subset = df[(df["method"] == method) & (df["condition"] == condition)].sort_values("k")
                if subset.empty:
                    continue
                ax.plot(
                    subset["k"],
                    subset[metric],
                    marker="o",
                    linestyle=linestyle,
                    label=f"{METHOD_LABELS.get(method, method)} ({condition})",
                )

        ax.set_title(title)
        ax.set_xlabel("Top-k tokens used for retrieval")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if metric == "preference_preserved_rate":
            ax.set_ylim(0.0, 1.0)
        elif metric == "mean_replacement_similarity":
            ax.set_ylim(0.0, 1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    cfg["out_file"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg["out_file"], dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved retrieval-faithfulness plot to {cfg['out_file']}")


def main():
    args = parse_args()
    if args.experiment == "all":
        for experiment_key in ["cross_encoder", "duot5"]:
            plot_one(experiment_key)
    else:
        plot_one(args.experiment)


if __name__ == "__main__":
    main()