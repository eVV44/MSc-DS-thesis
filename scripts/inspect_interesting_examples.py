# -- IMPORTS --
from __future__ import annotations
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd


root = Path(__file__).resolve().parents[1]


EXPERIMENTS = {
    "cross_encoder": {
        "label": "Cross-encoder",
        "faithfulness_file": root / "thesis_runs/cross_encoder/faithfulness/faithfulness_results.csv",
        "pair_file": root / "thesis_runs/cross_encoder/explanations/explanation_pairs_1_per_query_on_duot5_queries_seed42.csv",
        "ig_file": root / "thesis_runs/cross_encoder/explanations/attributions_ig.pkl",
        "loo_file": root / "thesis_runs/cross_encoder/explanations/attributions_loo.pkl",
        "out_dir": root / "thesis_runs/cross_encoder/faithfulness"},

    "duot5": {
        "label": "DuoT5",
        "faithfulness_file": root / "thesis_runs/duot5/faithfulness/faithfulness_results.csv",
        "pair_file": root / "thesis_runs/duot5/explanations/explanation_pairs_1_per_query_seed42.csv",
        "pairwise_ig_file": root / "thesis_runs/duot5/explanations/attributions_pairwise_ig.pkl",
        "pointwise_ig_file": root / "thesis_runs/duot5/explanations/attributions_pointwise_ig.pkl",
        "loo_file": root / "thesis_runs/duot5/explanations/attributions_loo.pkl",
        "out_dir": root / "thesis_runs/duot5/faithfulness"}}


def parse_args():
    parser = argparse.ArgumentParser(description="Find and inspect interesting explanation examples.")
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--k", type=int, default=5, help="Deletion budget k used to surface examples.")
    parser.add_argument("--top-words", type=int, default=8, help="How many top words to show per side.")
    parser.add_argument("--negative-limit", type=int, default=5, help="How many negative delta examples to summarize.")
    return parser.parse_args()


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def pair_key(qid, pid_i, pid_j):
    return str(qid), str(pid_i), str(pid_j)


def top_word_pairs(tokens, scores, limit):
    pairs = sorted(zip(tokens, scores), key=lambda x: x[1], reverse=True)[:limit]
    return [(word, round(float(score), 4)) for word, score in pairs]


def top_loo_words(rows, limit):
    pairs = sorted(rows, key=lambda x: x["support_score"], reverse=True)[:limit]
    return [(row["word"], round(float(row["support_score"]), 4)) for row in pairs]


def collect_examples(faithfulness_df: pd.DataFrame, k: int):
    examples = {}

    def pick(method: str, ascending: bool, name: str):
        subset = faithfulness_df[
            (faithfulness_df["method"] == method)
            & (faithfulness_df["condition"] == "explanation")
            & (faithfulness_df["check"] == "deletion")
            & (faithfulness_df["k"] == k)
        ].sort_values("delta_g", ascending=ascending)
        if not subset.empty:
            examples[name] = subset.iloc[0]

    pick("pairwise_ig", False, "strong_pairwise_ig")
    pick("pointwise_ig", False, "strong_pointwise_ig")
    pick("loo_pairwise", False, "strong_loo_pairwise")
    pick("loo_pointwise", False, "strong_loo_pointwise")
    pick("pointwise_ig", True, "weird_pointwise_ig")
    return examples


def negative_cases(faithfulness_df: pd.DataFrame, method: str, limit: int):
    subset = faithfulness_df[
        (faithfulness_df["method"] == method)
        & (faithfulness_df["condition"] == "explanation")
        & (faithfulness_df["check"] == "deletion")
        & (faithfulness_df["delta_g"] < 0)
    ].sort_values("delta_g")
    return subset.head(limit)


def dataframe_to_plain_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "No rows."
    headers = list(df.columns)
    rows = [[str(v) for v in row] for row in df.to_numpy()]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def fmt_row(values):
        return " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(values))

    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt_row(headers), sep]
    lines.extend(fmt_row(row) for row in rows)
    return "\n".join(lines)


def render_cross_example(record_ig, record_loo, example_row, example_name, top_words):
    lines = []
    lines.append(f"## {example_name}")
    lines.append(f"- `qid`: {example_row['qid']}")
    lines.append(f"- `pid_i`: {example_row['pid_i']}")
    lines.append(f"- `pid_j`: {example_row['pid_j']}")
    lines.append(f"- `g`: {example_row['g']:.4f}")
    lines.append(f"- `g_perturbed`: {example_row['g_perturbed']:.4f}")
    lines.append(f"- `delta_g`: {example_row['delta_g']:.4f}")
    lines.append(f"- `preference_flipped`: {int(example_row['preference_flipped'])}")
    lines.append("")
    lines.append(f"Query: {record_ig['query']}")
    lines.append("")

    if "pairwise" in example_name or "pointwise_ig" in example_name:
        if "pairwise" in example_name:
            doc_i = top_word_pairs(
                record_ig["pairwise_ig"]["doc_i"]["doc_word_tokens"],
                record_ig["pairwise_ig"]["doc_i"]["doc_word_scores"],
                top_words)
            
            doc_j = top_word_pairs(
                record_ig["pairwise_ig"]["doc_j"]["doc_word_tokens"],
                record_ig["pairwise_ig"]["doc_j"]["doc_word_scores"],
                top_words)
            
        else:
            doc_i = top_word_pairs(
                record_ig["pointwise_ig_i"]["doc_word_tokens"],
                record_ig["pointwise_ig_i"]["doc_word_scores"],
                top_words)
            
            doc_j = top_word_pairs(
                record_ig["pointwise_ig_j"]["doc_word_tokens"],
                record_ig["pointwise_ig_j"]["doc_word_scores"],
                top_words)
            
        lines.append(f"Top doc_i words: {doc_i}")
        lines.append(f"Top doc_j words: {doc_j}")
    else:
        if "pairwise" in example_name:
            doc_i = top_loo_words(record_loo["loo_pairwise"]["doc_i"], top_words)
            doc_j = top_loo_words(record_loo["loo_pairwise"]["doc_j"], top_words)
        else:
            doc_i = top_loo_words(record_loo["loo_pointwise_i"]["doc"], top_words)
            doc_j = top_loo_words(record_loo["loo_pointwise_j"]["doc"], top_words)
        lines.append(f"Top doc_i words: {doc_i}")
        lines.append(f"Top doc_j words: {doc_j}")
    lines.append("")
    return "\n".join(lines)


def render_duot5_example(record_pw, record_pt, record_loo, example_row, example_name, top_words):
    lines = []
    lines.append(f"## {example_name}")
    lines.append(f"- `qid`: {example_row['qid']}")
    lines.append(f"- `pid_i`: {example_row['pid_i']}")
    lines.append(f"- `pid_j`: {example_row['pid_j']}")
    lines.append(f"- `g`: {example_row['g']:.4f}")
    lines.append(f"- `g_perturbed`: {example_row['g_perturbed']:.4f}")
    lines.append(f"- `delta_g`: {example_row['delta_g']:.4f}")
    lines.append(f"- `preference_flipped`: {int(example_row['preference_flipped'])}")
    lines.append("")
    lines.append(f"Query: {record_pw['query'] if record_pw else record_pt['query']}")
    lines.append("")

    if example_name == "strong_pairwise_ig":
        doc_i = top_word_pairs(record_pw["pairwise_ig"]["doc0"]["word_tokens"], record_pw["pairwise_ig"]["doc0"]["word_scores"], top_words)
        doc_j = top_word_pairs(record_pw["pairwise_ig"]["doc1"]["word_tokens"], record_pw["pairwise_ig"]["doc1"]["word_scores"], top_words)
        lines.append(f"Top doc_i words: {doc_i}")
        lines.append(f"Top doc_j words: {doc_j}")
    elif example_name == "strong_pointwise_ig" or example_name == "weird_pointwise_ig":
        doc_i = top_word_pairs(record_pt["pointwise_ig_i"]["doc0"]["word_tokens"], record_pt["pointwise_ig_i"]["doc0"]["word_scores"], top_words)
        doc_j = top_word_pairs(record_pt["pointwise_ig_j"]["doc0"]["word_tokens"], record_pt["pointwise_ig_j"]["doc0"]["word_scores"], top_words)
        lines.append(f"Reference pid i: {record_pt['ref_pid_i']}")
        lines.append(f"Reference pid j: {record_pt['ref_pid_j']}")
        lines.append(f"Pointwise true-logit gap: {record_pt['pointwise_true_logit_g']:.4f}")
        lines.append(f"Top doc_i words: {doc_i}")
        lines.append(f"Top doc_j words: {doc_j}")
    elif example_name == "strong_loo_pairwise":
        doc_i = top_loo_words(record_loo["loo_pairwise"]["doc_i"], top_words)
        doc_j = top_loo_words(record_loo["loo_pairwise"]["doc_j"], top_words)
        lines.append(f"Top doc_i words: {doc_i}")
        lines.append(f"Top doc_j words: {doc_j}")
    elif example_name == "strong_loo_pointwise":
        doc_i = top_loo_words(record_loo["loo_pointwise_i"]["doc"], top_words)
        doc_j = top_loo_words(record_loo["loo_pointwise_j"]["doc"], top_words)
        lines.append(f"Reference pid i: {record_loo['ref_pid_i']}")
        lines.append(f"Reference pid j: {record_loo['ref_pid_j']}")
        lines.append(f"Pointwise score gap: {record_loo['pointwise_score_gap']:.4f}")
        lines.append(f"Top doc_i words: {doc_i}")
        lines.append(f"Top doc_j words: {doc_j}")
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    cfg = EXPERIMENTS[args.experiment]

    faithfulness_df = pd.read_csv(cfg["faithfulness_file"])
    pair_df = pd.read_csv(cfg["pair_file"], dtype={"qid": str, "pid_i": str, "pid_j": str})

    examples = collect_examples(faithfulness_df, args.k)
    interesting_csv = cfg["out_dir"] / f"interesting_examples_k{args.k}.csv"
    interesting_md = cfg["out_dir"] / f"interesting_examples_k{args.k}.md"

    lines = []
    lines.append(f"# Interesting Examples: {cfg['label']}")
    lines.append("")
    lines.append(f"- deletion budget: `k={args.k}`")
    lines.append(f"- top words shown per side: `{args.top_words}`")
    lines.append("")
    lines.append("Purpose:")
    lines.append("- strong examples: check whether highlighted tokens look semantically sensible")
    lines.append("- weird pointwise examples: check whether unstable behavior looks like a methodological limitation rather than a coding bug")
    lines.append("- negative delta cases: check whether masking supposedly supportive tokens can sometimes help the model")
    lines.append("")

    selected_rows = []

    if args.experiment == "cross_encoder":
        ig_records = load_pickle(cfg["ig_file"])
        loo_records = load_pickle(cfg["loo_file"])
        ig_lookup = {pair_key(r["qid"], r["pid_i"], r["pid_j"]): r for r in ig_records}
        loo_lookup = {pair_key(r["qid"], r["pid_i"], r["pid_j"]): r for r in loo_records}

        for name, row in examples.items():
            key = pair_key(row["qid"], row["pid_i"], row["pid_j"])
            selected_rows.append(
                {
                    "example_name": name,
                    "qid": row["qid"],
                    "pid_i": row["pid_i"],
                    "pid_j": row["pid_j"],
                    "method": row["method"],
                    "k": row["k"],
                    "g": row["g"],
                    "g_perturbed": row["g_perturbed"],
                    "delta_g": row["delta_g"],
                    "preference_flipped": row["preference_flipped"]})
            lines.append(render_cross_example(ig_lookup[key], loo_lookup[key], row, name, args.top_words))

        neg = negative_cases(faithfulness_df, "pairwise_ig", args.negative_limit)
        lines.append("## Negative delta cases for pairwise_ig")
        lines.append("")
        if neg.empty:
            lines.append("No negative deletion cases found.")
        else:
            lines.append(
                dataframe_to_plain_table(
                    neg[["qid", "pid_i", "pid_j", "k", "g", "g_perturbed", "delta_g", "preference_flipped"]]))

    else:
        pairwise_ig_records = load_pickle(cfg["pairwise_ig_file"])
        pointwise_ig_records = load_pickle(cfg["pointwise_ig_file"])
        loo_records = load_pickle(cfg["loo_file"])
        pw_lookup = {pair_key(r["qid"], r["pid_i"], r["pid_j"]): r for r in pairwise_ig_records}
        pt_lookup = {pair_key(r["qid"], r["pid_i"], r["pid_j"]): r for r in pointwise_ig_records}
        loo_lookup = {pair_key(r["qid"], r["pid_i"], r["pid_j"]): r for r in loo_records}

        for name, row in examples.items():
            key = pair_key(row["qid"], row["pid_i"], row["pid_j"])
            selected_rows.append(
                {
                    "example_name": name,
                    "qid": row["qid"],
                    "pid_i": row["pid_i"],
                    "pid_j": row["pid_j"],
                    "method": row["method"],
                    "k": row["k"],
                    "g": row["g"],
                    "g_perturbed": row["g_perturbed"],
                    "delta_g": row["delta_g"],
                    "preference_flipped": row["preference_flipped"]})
            lines.append(render_duot5_example(pw_lookup.get(key), pt_lookup.get(key), loo_lookup.get(key), row, name, args.top_words))

        neg = negative_cases(faithfulness_df, "pointwise_ig", args.negative_limit)
        lines.append("## Negative delta cases for pointwise_ig")
        lines.append("")
        if neg.empty:
            lines.append("No negative deletion cases found.")
        else:
            lines.append(
                dataframe_to_plain_table(
                    neg[["qid", "pid_i", "pid_j", "k", "g", "g_perturbed", "delta_g", "preference_flipped"]]))

    pd.DataFrame(selected_rows).to_csv(interesting_csv, index=False)
    interesting_md.write_text("\n".join(lines) + "\n")

    print(f"Saved selected example metadata to {interesting_csv}")
    print(f"Saved example inspection notes to {interesting_md}")


if __name__ == "__main__":
    main()