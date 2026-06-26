# -- IMPORTS --
from __future__ import annotations
import argparse
import gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from run_faithfulness_final import build_support_table
from run_retrieval_faithfulness_final import (
    EXPERIMENTS,
    POOL_TOP_N,
    load_records,
    load_top100_pools,
    original_model_score,
    replacement_model_score,
    setup_cross_runtime,
    setup_duot5_runtime)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--window-radius", type=int, default=3)
    return parser.parse_args()


def build_query_pools_tfidf(pair_df: pd.DataFrame) -> dict[str, dict]:
    qids = set(pair_df["qid"].astype(str))
    pool_df = load_top100_pools(qids)
    pools = {}
    for qid, group in pool_df.groupby("qid", sort=False):
        rows = group.reset_index(drop=True).copy()
        passages = rows["passage"].astype(str).tolist()
        vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
        matrix = vectorizer.fit_transform(passages)
        pools[str(qid)] = {
            "rows": rows,
            "vectorizer": vectorizer,
            "matrix": matrix,
        }
    return pools


def retrieve_replacement(pool: dict, query_text: str, exclude_pids: set[str]):
    if not query_text.strip():
        return None
    query_vec = pool["vectorizer"].transform([query_text])
    if query_vec.nnz == 0:
        return None
    sims = linear_kernel(query_vec, pool["matrix"]).ravel()
    order = np.argsort(-sims)
    for idx in order:
        row = pool["rows"].iloc[int(idx)]
        pid = str(row["pid"])
        if pid in exclude_pids:
            continue
        return {
            "pid": pid,
            "passage": str(row["passage"]),
            "bm25_rank": int(row["bm25_rank"]),
            "similarity": float(sims[int(idx)]),
        }
    return None


def doc_i_rows(record, method: str, family: str) -> pd.DataFrame:
    table = build_support_table(record, method, family)
    if table.empty:
        return table
    subset = table[table["side"] == "doc_i"].copy()
    if subset.empty:
        return subset
    subset["position"] = subset["position"].astype(int)
    return subset.sort_values("position").reset_index(drop=True)


def positive_rows(record, method: str, family: str) -> pd.DataFrame:
    rows = doc_i_rows(record, method, family)
    if rows.empty:
        return rows
    return rows[rows["support_score"] > 0].sort_values("support_score", ascending=False).reset_index(drop=True)


def local_window_query(rows: pd.DataFrame, anchor_position: int, window_radius: int) -> tuple[str, int]:
    subset = rows[(rows["position"] >= anchor_position - window_radius) & (rows["position"] <= anchor_position + window_radius)].copy()
    subset = subset.sort_values("position")
    tokens = [str(tok).strip() for tok in subset["token"].tolist() if str(tok).strip()]
    return " ".join(tokens).strip(), len(tokens)


def explanation_window_query(record, method: str, family: str, window_radius: int):
    all_rows = doc_i_rows(record, method, family)
    pos_rows = positive_rows(record, method, family)
    if all_rows.empty or pos_rows.empty:
        return "", 0, np.nan
    anchor = pos_rows.iloc[0]
    query_text, window_len = local_window_query(all_rows, int(anchor["position"]), window_radius)
    return query_text, window_len, float(anchor["support_score"])


def random_window_query(record, method: str, family: str, window_radius: int, rng):
    all_rows = doc_i_rows(record, method, family)
    pos_rows = positive_rows(record, method, family)
    if all_rows.empty or pos_rows.empty:
        return "", 0
    sampled = pos_rows.sample(n=1, replace=False, random_state=int(rng.integers(0, 1_000_000_000))).iloc[0]
    query_text, window_len = local_window_query(all_rows, int(sampled["position"]), window_radius)
    return query_text, window_len


def summarize(results_df: pd.DataFrame) -> pd.DataFrame:
    results_df = results_df.copy()
    results_df["abs_score_gap"] = (results_df["original_g"] - results_df["replacement_g"]).abs()
    grouped = (
        results_df.groupby(["experiment", "model_label", "method", "condition", "window_radius"], as_index=False)
        .agg(
            n_queries=("qid", "nunique"),
            mean_original_g=("original_g", "mean"),
            mean_replacement_g=("replacement_g", "mean"),
            mean_delta_g=("delta_g", "mean"),
            mean_abs_score_gap=("abs_score_gap", "mean"),
            preference_preserved_rate=("preference_preserved", "mean"),
            mean_replacement_similarity=("replacement_similarity", "mean"),
            mean_replacement_bm25_rank=("replacement_bm25_rank", "mean"),
            mean_query_token_count=("query_token_count", "mean"),
        )
    )
    return grouped.sort_values(["condition", "method", "window_radius"]).reset_index(drop=True)


def run_window_retrieval_faithfulness(records, pair_df, cfg, window_radius: int):
    family = cfg["family"]
    rng = np.random.default_rng(42)
    scoring_runtime = setup_cross_runtime(cfg) if family == "cross_encoder" else setup_duot5_runtime(cfg)
    query_pools = build_query_pools_tfidf(pair_df)
    methods = ["pairwise_ig", "pointwise_ig", "loo_pairwise", "loo_pointwise"]
    rows = []

    for record in records:
        qid = str(record["qid"])
        pool = query_pools[qid]
        excluded = {str(record["pid_i"]), str(record["pid_j"])}
        original_g = original_model_score(record, family, scoring_runtime)

        for method in methods:
            query_text, query_len, anchor_score = explanation_window_query(record, method, family, window_radius)
            if query_text:
                replacement = retrieve_replacement(pool, query_text, excluded)
                if replacement is not None:
                    replacement_g = replacement_model_score(record, family, scoring_runtime, replacement["passage"])
                    rows.append(
                        {
                            "experiment": family,
                            "model_label": cfg["label"],
                            "method": method,
                            "condition": "explanation",
                            "qid": qid,
                            "pid_i": record["pid_i"],
                            "pid_j": record["pid_j"],
                            "window_radius": window_radius,
                            "original_g": original_g,
                            "replacement_g": replacement_g,
                            "delta_g": original_g - replacement_g,
                            "preference_preserved": int(np.sign(original_g) == np.sign(replacement_g)),
                            "replacement_pid": replacement["pid"],
                            "replacement_bm25_rank": replacement["bm25_rank"],
                            "replacement_similarity": replacement["similarity"],
                            "query_token_count": query_len,
                            "anchor_support_score": anchor_score,
                            "retrieval_query": query_text,
                        }
                    )

            random_query, random_len = random_window_query(record, method, family, window_radius, rng)
            if random_query:
                replacement = retrieve_replacement(pool, random_query, excluded)
                if replacement is not None:
                    replacement_g = replacement_model_score(record, family, scoring_runtime, replacement["passage"])
                    rows.append(
                        {
                            "experiment": family,
                            "model_label": cfg["label"],
                            "method": method,
                            "condition": "random",
                            "qid": qid,
                            "pid_i": record["pid_i"],
                            "pid_j": record["pid_j"],
                            "window_radius": window_radius,
                            "original_g": original_g,
                            "replacement_g": replacement_g,
                            "delta_g": original_g - replacement_g,
                            "preference_preserved": int(np.sign(original_g) == np.sign(replacement_g)),
                            "replacement_pid": replacement["pid"],
                            "replacement_bm25_rank": replacement["bm25_rank"],
                            "replacement_similarity": replacement["similarity"],
                            "query_token_count": random_len,
                            "anchor_support_score": np.nan,
                            "retrieval_query": random_query,
                        }
                    )

        gc.collect()
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    cfg = EXPERIMENTS[args.experiment]
    pair_df, records = load_records(cfg)
    if args.limit is not None:
        records = records[: args.limit]
        keep = {(str(r["qid"]), str(r["pid_i"]), str(r["pid_j"])) for r in records}
        pair_df = pair_df[
            [(str(r["qid"]), str(r["pid_i"]), str(r["pid_j"])) in keep for r in pair_df.to_dict("records")]
        ].reset_index(drop=True)

    out_dir = cfg["faithfulness_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {cfg['label']} ({cfg['family']})")
    print(f"Loaded records: {len(records)}")
    print(f"Replacement pool: BM25 top-{POOL_TOP_N} documents per query")
    print("Retriever: query-specific TF-IDF over candidate passages")
    print(f"Window retrieval radius: +/-{args.window_radius} token positions")

    results_df = run_window_retrieval_faithfulness(records, pair_df, cfg, window_radius=args.window_radius)
    summary_df = summarize(results_df)

    suffix_parts = [f"win{args.window_radius}"]
    if args.limit is not None:
        suffix_parts.append(f"limit{args.limit}")
    suffix = "_" + "_".join(suffix_parts)
    results_out = out_dir / f"window_retrieval_faithfulness_results{suffix}.csv"
    summary_out = out_dir / f"window_retrieval_faithfulness_summary{suffix}.csv"
    results_df.to_csv(results_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    print(f"Saved detailed window-retrieval faithfulness results to {results_out}")
    print(f"Saved summary window-retrieval faithfulness results to {summary_out}")


if __name__ == "__main__":
    main()