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


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = {
    "cross_encoder": {
        "label": "Cross-encoder",
        "summary_file": ROOT / "thesis_runs/cross_encoder/faithfulness/sentence_retrieval_faithfulness_summary_sent1.csv",
        "out_file": ROOT / "thesis_runs/cross_encoder/faithfulness/sentence_retrieval_faithfulness_plots_sent1.png",
    },
    "duot5": {
        "label": "DuoT5",
        "summary_file": ROOT / "thesis_runs/duot5/faithfulness/sentence_retrieval_faithfulness_summary_sent1.csv",
        "out_file": ROOT / "thesis_runs/duot5/faithfulness/sentence_retrieval_faithfulness_plots_sent1.png",
    },
}

METHOD_LABELS = {
    "pairwise_ig": "Pairwise IG",
    "pointwise_ig": "Pointwise IG",
    "loo_pairwise": "LOO pairwise",
    "loo_pointwise": "LOO pointwise",
}

METHOD_ORDER = ["pairwise_ig", "pointwise_ig", "loo_pairwise", "loo_pointwise"]
COLORS = {
    "explanation": "#1f77b4",
    "random": "#ff7f0e",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["cross_encoder", "duot5", "all"], default="all")
    return parser.parse_args()


def plot_one(experiment_key: str):
    cfg = EXPERIMENTS[experiment_key]
    if not cfg["summary_file"].exists():
        print(f"Missing summary: {cfg['summary_file']}")
        return

    df = pd.read_csv(cfg["summary_file"])
    if df.empty:
        print(f"Empty summary: {cfg['summary_file']}")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Sentence-Based Evidence Transfer: {cfg['label']}", fontsize=18)

    panels = [
        ("preference_preserved_rate", "Preference preserved", "Rate", True),
        ("mean_abs_score_gap", "Mean |g - g'|", "Mean absolute score gap", False),
        ("mean_replacement_similarity", "Replacement similarity", "Mean TF-IDF cosine similarity", True),
    ]

    x = np.arange(len(METHOD_ORDER))
    width = 0.35

    for ax, (metric, title, ylabel, zero_to_one) in zip(axes, panels):
        exp_vals = []
        rnd_vals = []
        for method in METHOD_ORDER:
            exp_row = df[(df["method"] == method) & (df["condition"] == "explanation")]
            rnd_row = df[(df["method"] == method) & (df["condition"] == "random")]
            exp_vals.append(float(exp_row.iloc[0][metric]) if not exp_row.empty else np.nan)
            rnd_vals.append(float(rnd_row.iloc[0][metric]) if not rnd_row.empty else np.nan)

        ax.bar(x - width / 2, exp_vals, width, label="Explanation-guided", color=COLORS["explanation"])
        ax.bar(x + width / 2, rnd_vals, width, label="Random baseline", color=COLORS["random"])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=20, ha="right")
        ax.grid(axis="y", alpha=0.25)
        if zero_to_one:
            ax.set_ylim(0.0, 1.0)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True)
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    cfg["out_file"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(cfg["out_file"], dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sentence-retrieval plot to {cfg['out_file']}")


def main():
    args = parse_args()
    targets = ["cross_encoder", "duot5"] if args.experiment == "all" else [args.experiment]
    for target in targets:
        plot_one(target)


if __name__ == "__main__":
    main()