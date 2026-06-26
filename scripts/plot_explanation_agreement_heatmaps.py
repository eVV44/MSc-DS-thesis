# -- IMPORTS --
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = {
    "cross_encoder": {
        "label": "Cross-encoder",
        "input_file": ROOT / "thesis_runs/cross_encoder/faithfulness/explanation_agreement_summary.csv",
        "detail_file": ROOT / "thesis_runs/cross_encoder/faithfulness/explanation_agreement_results.csv",
        "out_file": ROOT / "thesis_runs/cross_encoder/faithfulness/explanation_agreement_top5_heatmap.png",
        "methods": ["pairwise_ig", "pointwise_ig", "loo_pairwise", "loo_pointwise"],
        "display_names": {
            "pairwise_ig": "Pairwise IG",
            "pointwise_ig": "Pointwise IG",
            "loo_pairwise": "Pairwise LOO",
            "loo_pointwise": "Pointwise LOO",
            "random": "Random",
        },
    },
    "duot5": {
        "label": "DuoT5",
        "input_file": ROOT / "thesis_runs/duot5/faithfulness/explanation_agreement_summary.csv",
        "detail_file": ROOT / "thesis_runs/duot5/faithfulness/explanation_agreement_results.csv",
        "out_file": ROOT / "thesis_runs/duot5/faithfulness/explanation_agreement_top5_heatmap.png",
        "methods": ["pairwise_ig", "pointwise_ig", "loo_pairwise", "loo_pointwise"],
        "display_names": {
            "pairwise_ig": "Pairwise IG",
            "pointwise_ig": "Pointwise IG proxy",
            "loo_pairwise": "Pairwise LOO",
            "loo_pointwise": "Pointwise LOO proxy",
            "random": "Random",
        },
    },
}


def build_topk_matrix(
    df: pd.DataFrame,
    detail_df: pd.DataFrame,
    methods: list[str],
    topk_col: str = "mean_top5_overlap",
    random_label: str = "random",
) -> pd.DataFrame:
    labels = methods + [random_label]
    matrix = pd.DataFrame(np.eye(len(labels)), index=labels, columns=labels, dtype=float)

    for _, row in df.iterrows():
        method_a = row["method_a"]
        method_b = row["method_b"]
        value = float(row[topk_col])
        if method_a in matrix.index and method_b in matrix.columns:
            matrix.loc[method_a, method_b] = value
            matrix.loc[method_b, method_a] = value

    # For a uniformly random top-k set drawn from N aligned token positions,
    # the expected overlap with another top-k set is k / N.
    k = int(topk_col.removeprefix("mean_top").split("_")[0])
    aligned_sizes = detail_df.groupby(["qid", "pid_i", "pid_j"])["n_aligned"].first()
    random_overlap = float((k / aligned_sizes).mean())
    for method in methods:
        matrix.loc[method, random_label] = random_overlap
        matrix.loc[random_label, method] = random_overlap
    matrix.loc[random_label, random_label] = random_overlap

    return matrix


def plot_heatmap(
    matrix: pd.DataFrame,
    display_names: dict[str, str],
    title: str,
    out_file: Path,
) -> None:
    labels = [display_names[m] for m in matrix.index]
    values = matrix.to_numpy()

    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    im = ax.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title, pad=14)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            text_color = "white" if values[i, j] >= 0.7 else "black"
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", color=text_color, fontsize=10)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Top-5 overlap")

    ax.set_xlabel("Explanation method")
    ax.set_ylabel("Explanation method")
    fig.subplots_adjust(left=0.22, right=0.92, bottom=0.28, top=0.90)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=180)
    plt.close(fig)


def main() -> None:
    for name, cfg in EXPERIMENTS.items():
        df = pd.read_csv(cfg["input_file"])
        detail_df = pd.read_csv(cfg["detail_file"])
        matrix = build_topk_matrix(df, detail_df, cfg["methods"], topk_col="mean_top5_overlap")
        plot_heatmap(
            matrix,
            cfg["display_names"],
            f"{cfg['label']}: explanation agreement at top-5",
            cfg["out_file"],
        )
        print(f"Saved {name} heatmap to {cfg['out_file']}")


if __name__ == "__main__":
    main()