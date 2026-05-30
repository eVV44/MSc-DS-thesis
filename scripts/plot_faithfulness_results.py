# -- IMPORTS --
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


root = Path(__file__).resolve().parents[1]

EXPERIMENTS = {
    "cross_encoder": {
        "label": "Cross-encoder",
        "summary_file": root / "thesis_runs/cross_encoder/faithfulness/faithfulness_summary.csv",
        "out_png": root / "thesis_runs/cross_encoder/faithfulness/faithfulness_plots.png"},

    "duot5": {
        "label": "DuoT5",
        "summary_file": root / "thesis_runs/duot5/faithfulness/faithfulness_summary.csv",
        "out_png": root / "thesis_runs/duot5/faithfulness/faithfulness_plots.png"}}

method_labels = {
    "pairwise_ig": "Pairwise IG",
    "pointwise_ig": "Pointwise IG",
    "loo_pairwise": "LOO pairwise",
    "loo_pointwise": "LOO pointwise"}

line_styles = {"explanation": "-", "random": "--"}


def add_anchor_points(df: pd.DataFrame, check: str) -> pd.DataFrame:
    if df.empty:
        return df

    anchors = []
    for _, row in df.sort_values("k").groupby(["method", "condition"], as_index=False).first().iterrows():
        base = row.to_dict()
        base["k"] = 0
        if check == "deletion":
            base["mean_delta_g"] = 0.0
            base["flip_rate"] = 0.0
        anchors.append(base)

    return pd.concat([pd.DataFrame(anchors), df], ignore_index=True).sort_values("k").reset_index(drop=True)


def plot_experiment(summary_df: pd.DataFrame, title: str, out_png: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for method in summary_df["method"].unique():
        label = method_labels.get(method, method)
        for condition in ["explanation", "random"]:
            subset = summary_df[(summary_df["method"] == method) & (summary_df["condition"] == condition)]
            if subset.empty:
                continue
            deletion = add_anchor_points(subset[subset["check"] == "deletion"].sort_values("k"), "deletion")
            preservation = subset[subset["check"] == "preservation"].sort_values("k")

            axes[0].plot(
                deletion["k"],
                deletion["mean_delta_g"],
                marker="o",
                linestyle=line_styles[condition],
                label=f"{label} ({condition})")
            
            axes[1].plot(
                deletion["k"],
                deletion["flip_rate"],
                marker="o",
                linestyle=line_styles[condition],
                label=f"{label} ({condition})")
            
            axes[2].plot(
                preservation["k"],
                preservation["mean_abs_g_gap"],
                marker="o",
                linestyle=line_styles[condition],
                label=f"{label} ({condition})")
            
            axes[3].plot(
                preservation["k"],
                preservation["sign_preservation_rate"],
                marker="o",
                linestyle=line_styles[condition],
                label=f"{label} ({condition})")

    axes[0].set_title("Deletion: Mean Δg")
    axes[0].set_ylabel("Mean Δg")
    axes[1].set_title("Deletion: Flip Rate")
    axes[1].set_ylabel("Flip rate")
    axes[2].set_title("Preservation: Mean |g - g'|")
    axes[2].set_ylabel("Mean |g - g'|")
    axes[3].set_title("Preservation: Sign Preservation")
    axes[3].set_ylabel("Sign preservation rate")

    for ax in axes:
        ax.set_xlabel("Budget k")
        ax.grid(alpha=0.3)

    axes[0].set_xticks([0, 1, 3, 5, 10, 20])
    axes[1].set_xticks([0, 1, 3, 5, 10, 20])
    axes[2].set_xticks([1, 3, 5, 10, 20])
    axes[3].set_xticks([1, 3, 5, 10, 20])

    fig.suptitle(f"Faithfulness Evaluation: {title}", y=1.03)
    axes[-1].legend(loc="best")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    for name, cfg in EXPERIMENTS.items():
        summary_df = pd.read_csv(cfg["summary_file"])
        plot_experiment(summary_df, cfg["label"], cfg["out_png"])
        print(f"Saved {name} faithfulness plot to {cfg['out_png']}")


if __name__ == "__main__":
    main()