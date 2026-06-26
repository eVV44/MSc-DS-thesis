from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "thesis_runs" / "shared" / "qualitative_examples"


def load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def normalize_token(token: str) -> str:
    return (
        token.replace("▁", "")
        .replace("##", "")
        .replace("[CLS]", "")
        .replace("[SEP]", "")
        .strip()
    )


def clean_word_pairs(tokens, scores, max_tokens: int = 45):
    pairs = []
    for token, score in zip(tokens, scores):
        tok = normalize_token(str(token))
        if not tok:
            continue
        pairs.append((tok, float(score)))
    return pairs[:max_tokens]


def wrap_pairs(pairs, max_chars: int = 58):
    lines: list[list[tuple[str, float]]] = []
    current: list[tuple[str, float]] = []
    current_len = 0
    for token, score in pairs:
        tok_len = len(token) + 1
        if current and current_len + tok_len > max_chars:
            lines.append(current)
            current = []
            current_len = 0
        current.append((token, score))
        current_len += tok_len
    if current:
        lines.append(current)
    return lines


def score_norm(pairs):
    max_abs = max(abs(score) for _, score in pairs) if pairs else 1.0
    max_abs = max(max_abs, 1e-6)
    return colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)


def draw_token_heatmap(ax, pairs, title: str, subtitle: str | None = None, norm=None, cmap=None):
    pairs = clean_word_pairs([t for t, _ in pairs], [s for _, s in pairs], max_tokens=len(pairs))
    if not pairs:
        ax.axis("off")
        ax.set_title(title, fontsize=11, loc="left")
        return

    norm = norm or score_norm(pairs)
    cmap = cmap or plt.get_cmap("coolwarm")
    lines = wrap_pairs(pairs)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, max(1, len(lines)))
    ax.axis("off")
    ax.set_title(title, fontsize=11, loc="left", pad=10)
    if subtitle:
        ax.text(0.0, len(lines) + 0.08, subtitle, fontsize=8, color="dimgray", va="bottom")

    y = len(lines) - 0.5
    for line in lines:
        x = 0.0
        for token, score in line:
            width = 0.014 * len(token) + 0.022
            face = cmap(norm(score))
            rgba = list(face)
            rgba[3] = 0.9
            ax.text(
                x,
                y,
                token,
                fontsize=9,
                va="center",
                ha="left",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": tuple(rgba),
                    "edgecolor": "white",
                    "linewidth": 0.6,
                },
            )
            x += width
        y -= 1.0


def global_score_norm(*pair_groups):
    all_scores = [float(score) for pairs in pair_groups for _, score in pairs]
    max_abs = max(abs(score) for score in all_scores) if all_scores else 1.0
    max_abs = max(max_abs, 1e-6)
    return colors.TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)


def extract_cross_encoder_record(qid: str, pid_i: str, pid_j: str):
    data = load_pickle(ROOT / "thesis_runs" / "cross_encoder" / "explanations" / "attributions_ig.pkl")
    return next(r for r in data if r["qid"] == qid and r["pid_i"] == pid_i and r["pid_j"] == pid_j)


def extract_duot5_pairwise_record(qid: str, pid_i: str, pid_j: str):
    data = load_pickle(ROOT / "thesis_runs" / "duot5" / "explanations" / "attributions_pairwise_ig.pkl")
    return next(r for r in data if r["qid"] == qid and r["pid_i"] == pid_i and r["pid_j"] == pid_j)


def extract_duot5_pointwise_record(qid: str, pid_i: str, pid_j: str):
    data = load_pickle(ROOT / "thesis_runs" / "duot5" / "explanations" / "attributions_pointwise_ig.pkl")
    return next(r for r in data if r["qid"] == qid and r["pid_i"] == pid_i and r["pid_j"] == pid_j)


def extract_loo_rows(path: Path, qid: int, pid_i: int, pid_j: int, side: str):
    df = pd.read_csv(path)
    sub = (
        df[(df["qid"] == qid) & (df["pid_i"] == pid_i) & (df["pid_j"] == pid_j) & (df["side"] == side)]
        .sort_values("position")
        .copy()
    )
    return list(zip(sub["token"].astype(str), sub["support_score"].astype(float)))


def make_cross_encoder_equivalence():
    qid, pid_i, pid_j = "747937", "7504079", "7504078"
    rec = extract_cross_encoder_record(qid, pid_i, pid_j)

    pair_doc_i = list(zip(rec["pairwise_ig"]["doc_i"]["word_tokens"], rec["pairwise_ig"]["doc_i"]["word_scores"]))
    point_doc_i = list(zip(rec["pointwise_ig_i"]["doc_word_tokens"], rec["pointwise_ig_i"]["doc_word_scores"]))
    pair_doc_j = list(zip(rec["pairwise_ig"]["doc_j"]["word_tokens"], rec["pairwise_ig"]["doc_j"]["word_scores"]))
    point_doc_j = list(zip(rec["pointwise_ig_j"]["doc_word_tokens"], rec["pointwise_ig_j"]["doc_word_scores"]))

    fig, axes = plt.subplots(2, 2, figsize=(14, 6))
    fig.suptitle("Qualitative Example 1: Cross-encoder Pairwise and Pointwise IG Collapse", fontsize=14, y=1.02)

    draw_token_heatmap(axes[0, 0], pair_doc_i[:36], "Pairwise IG: preferred passage")
    draw_token_heatmap(axes[0, 1], point_doc_i[:36], "Pointwise IG: preferred passage")
    draw_token_heatmap(axes[1, 0], pair_doc_j[:28], "Pairwise IG: comparison passage")
    draw_token_heatmap(axes[1, 1], point_doc_j[:28], "Pointwise IG: comparison passage")

    fig.text(0.01, -0.02, "Query: what is floof", fontsize=10)
    out = OUT_DIR / "cross_encoder_equivalence_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def make_duot5_pairwise_advantage():
    qid, pid_i, pid_j = "525868", "7633123", "4241268"
    pair_rec = extract_duot5_pairwise_record(qid, pid_i, pid_j)
    point_rec = extract_duot5_pointwise_record(qid, pid_i, pid_j)

    pair_doc0 = list(zip(pair_rec["pairwise_ig"]["doc0"]["word_tokens"], pair_rec["pairwise_ig"]["doc0"]["word_scores"]))
    pair_doc1 = list(zip(pair_rec["pairwise_ig"]["doc1"]["word_tokens"], pair_rec["pairwise_ig"]["doc1"]["word_scores"]))
    point_doc0 = list(zip(point_rec["pointwise_ig_i"]["doc0"]["word_tokens"], point_rec["pointwise_ig_i"]["doc0"]["word_scores"]))
    point_doc1 = list(zip(point_rec["pointwise_ig_i"]["doc1"]["word_tokens"], point_rec["pointwise_ig_i"]["doc1"]["word_scores"]))

    fig, axes = plt.subplots(2, 2, figsize=(15, 7))
    fig.suptitle("Qualitative Example 2: DuoT5 Pairwise IG vs Pointwise Proxy IG", fontsize=14, y=1.02)

    draw_token_heatmap(axes[0, 0], pair_doc0[:38], "Pairwise IG: preferred passage")
    draw_token_heatmap(axes[0, 1], pair_doc1[:34], "Pairwise IG: comparison passage")
    draw_token_heatmap(axes[1, 0], point_doc0[:38], "Pointwise proxy IG: target passage")
    draw_token_heatmap(axes[1, 1], point_doc1[:34], "Pointwise proxy IG: reference passage")

    fig.text(0.01, -0.02, "Query: two rivers supervisory union", fontsize=10)
    out = OUT_DIR / "duot5_pairwise_advantage_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def make_duot5_pairwise_advantage_shared_scale():
    qid, pid_i, pid_j = "525868", "7633123", "4241268"
    pair_rec = extract_duot5_pairwise_record(qid, pid_i, pid_j)
    point_rec = extract_duot5_pointwise_record(qid, pid_i, pid_j)

    pair_doc0 = clean_word_pairs(
        pair_rec["pairwise_ig"]["doc0"]["word_tokens"],
        pair_rec["pairwise_ig"]["doc0"]["word_scores"],
        max_tokens=38,
    )
    pair_doc1 = clean_word_pairs(
        pair_rec["pairwise_ig"]["doc1"]["word_tokens"],
        pair_rec["pairwise_ig"]["doc1"]["word_scores"],
        max_tokens=34,
    )
    point_doc0 = clean_word_pairs(
        point_rec["pointwise_ig_i"]["doc0"]["word_tokens"],
        point_rec["pointwise_ig_i"]["doc0"]["word_scores"],
        max_tokens=38,
    )
    point_doc1 = clean_word_pairs(
        point_rec["pointwise_ig_i"]["doc1"]["word_tokens"],
        point_rec["pointwise_ig_i"]["doc1"]["word_scores"],
        max_tokens=34,
    )

    cmap = plt.get_cmap("coolwarm")
    norm = global_score_norm(pair_doc0, pair_doc1, point_doc0, point_doc1)

    fig = plt.figure(figsize=(15.2, 8.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.15, hspace=0.30)
    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
    ]
    cax = fig.add_subplot(gs[:, 2])

    fig.suptitle("Qualitative DuoT5 Example: Pairwise Versus Pointwise Proxy IG", fontsize=16, y=0.985)
    fig.text(0.5, 0.948, "Query: two rivers supervisory union", ha="center", fontsize=12)

    draw_token_heatmap(
        axes[0],
        pair_doc0,
        "PAIRWISE IG: PREFERRED PASSAGE",
        subtitle="pid 7633123 — Vermont school district (Two Rivers Supervisory Union)",
        norm=norm,
        cmap=cmap,
    )
    draw_token_heatmap(
        axes[1],
        pair_doc1,
        "PAIRWISE IG: COMPARISON PASSAGE",
        subtitle="pid 4241268 — Tigris–Euphrates / Mesopotamia",
        norm=norm,
        cmap=cmap,
    )
    draw_token_heatmap(
        axes[2],
        point_doc0,
        "POINTWISE PROXY IG: TARGET PASSAGE",
        subtitle="Document0 = Vermont passage, Document1 = Three Rivers MI reference",
        norm=norm,
        cmap=cmap,
    )
    draw_token_heatmap(
        axes[3],
        point_doc1,
        "POINTWISE PROXY IG: REFERENCE PASSAGE",
        subtitle="ref pid 8147418 — Chicago → Three Rivers, MI driving time",
        norm=norm,
        cmap=cmap,
    )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("Integrated Gradients attribution", fontsize=11)

    out = OUT_DIR / "duot5_pairwise_advantage_heatmap_shared_scale.png"
    fig.subplots_adjust(left=0.04, right=0.96, bottom=0.05, top=0.90)
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def make_ig_loo_disagreement():
    qid, pid_i, pid_j = "1001108", "7870036", "302808"
    rec = extract_cross_encoder_record(qid, pid_i, pid_j)
    ig_doc_i = list(zip(rec["pairwise_ig"]["doc_i"]["word_tokens"], rec["pairwise_ig"]["doc_i"]["word_scores"]))
    loo_doc_i = extract_loo_rows(
        ROOT / "thesis_runs" / "cross_encoder" / "explanations" / "attributions_loo_pairwise_rows.csv",
        int(qid),
        int(pid_i),
        int(pid_j),
        "doc_i",
    )
    loo_doc_i = sorted(loo_doc_i, key=lambda x: x[1], reverse=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    fig.suptitle("Qualitative Example 3: Cross-encoder Pairwise IG vs Pairwise LOO", fontsize=14, y=1.02)

    draw_token_heatmap(
        axes[0],
        ig_doc_i[:36],
        "Pairwise IG: preferred passage",
        subtitle="Low agreement with LOO for this query ($\\rho=0.299$, top-5 overlap = 0.0).",
    )
    draw_token_heatmap(axes[1], loo_doc_i[:36], "Pairwise LOO: preferred passage")

    fig.text(0.01, -0.02, "Query: where the chromosomes are moving towards the poles of the cell", fontsize=10)
    out = OUT_DIR / "cross_encoder_ig_loo_disagreement_heatmap.png"
    fig.tight_layout()
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        make_cross_encoder_equivalence(),
        make_duot5_pairwise_advantage(),
        make_duot5_pairwise_advantage_shared_scale(),
        make_ig_loo_disagreement(),
    ]
    for path in outputs:
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
