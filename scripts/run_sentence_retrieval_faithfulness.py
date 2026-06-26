from __future__ import annotations

import argparse
import gc
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from run_faithfulness_final import build_support_table, tokenize_cross, tokenize_duo_pair
from run_retrieval_faithfulness_final import (
    EXPERIMENTS,
    POOL_TOP_N,
    ROOT,
    SEED,
    load_records,
    load_top100_pools,
    original_model_score,
    replacement_model_score,
    setup_cross_runtime,
    setup_duot5_runtime,
)


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-sentences", type=int, default=1)
    parser.add_argument("--score-agg", choices=["sum", "mean", "max"], default="sum")
    return parser.parse_args()


def split_sentences(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", str(text)).strip()
    if not normalized:
        return []
    pieces = [piece.strip() for piece in SENTENCE_SPLIT_RE.split(normalized) if piece.strip()]
    return pieces if pieces else [normalized]


def sentence_token_lengths(tokenizer, sentences: list[str], space_sensitive: bool) -> list[int]:
    lengths = []
    for idx, sentence in enumerate(sentences):
        text = sentence if idx == 0 or not space_sensitive else f" {sentence}"
        token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        lengths.append(len(token_ids))
    return lengths


def assign_sentence_positions(tokenizer, sentences: list[str], doc_positions: list[int], space_sensitive: bool) -> list[dict]:
    if not sentences or not doc_positions:
        return []

    lengths = sentence_token_lengths(tokenizer, sentences, space_sensitive=space_sensitive)
    spans = []
    cursor = 0
    for sentence, length in zip(sentences, lengths):
        if cursor >= len(doc_positions):
            break
        if length <= 0:
            continue
        next_cursor = min(cursor + length, len(doc_positions))
        assigned = list(doc_positions[cursor:next_cursor])
        if assigned:
            spans.append(
                {
                    "sentence": sentence,
                    "positions": assigned,
                    "sentence_index": len(spans),
                }
            )
        cursor = next_cursor

    if not spans:
        return [{"sentence": " ".join(sentences).strip(), "positions": list(doc_positions), "sentence_index": 0}]

    if cursor < len(doc_positions):
        spans[-1]["positions"].extend(doc_positions[cursor:])

    return spans


def doc_i_sentence_spans(record, family: str, runtime) -> list[dict]:
    tokenizer = runtime[0]
    sentences = split_sentences(record["passage_i"])
    if family == "cross_encoder":
        _, _, device = runtime
        tok = tokenize_cross(tokenizer, device, record["query"], record["passage_i"])
        return assign_sentence_positions(tokenizer, sentences, tok["doc_positions"], space_sensitive=False)

    _, _, device, *_ = runtime
    tok = tokenize_duo_pair(tokenizer, device, record["query"], record["passage_i"], record["passage_j"])
    return assign_sentence_positions(tokenizer, sentences, tok["doc0_positions"], space_sensitive=True)


def row_positions(row) -> list[int]:
    positions = row.get("positions")
    if isinstance(positions, list):
        return [int(p) for p in positions]
    return [int(row["position"])]


def aggregate_scores(scores: list[float], mode: str) -> float:
    if not scores:
        return 0.0
    if mode == "sum":
        return float(np.sum(scores))
    if mode == "mean":
        return float(np.mean(scores))
    if mode == "max":
        return float(np.max(scores))
    raise ValueError(f"Unsupported score aggregation mode: {mode}")


def score_sentences(record, method: str, family: str, spans: list[dict], score_agg: str) -> list[dict]:
    table = build_support_table(record, method, family)
    if table.empty:
        return []

    subset = table[(table["side"] == "doc_i") & (table["support_score"] > 0)].copy()
    if subset.empty:
        return []

    scored = []
    for span in spans:
        pos_set = set(int(p) for p in span["positions"])
        support_values = []
        matched_tokens = []
        for _, row in subset.iterrows():
            positions = row_positions(row)
            if any(int(p) in pos_set for p in positions):
                support_values.append(float(row["support_score"]))
                matched_tokens.append(str(row["token"]))
        scored.append(
            {
                **span,
                "support_score": aggregate_scores(support_values, score_agg),
                "matched_support_count": len(support_values),
                "sentence_token_count": len(span["positions"]),
                "sentence_char_count": len(span["sentence"]),
                "matched_tokens": matched_tokens,
            }
        )
    return scored


def select_top_sentences(scored_spans: list[dict], n_sentences: int) -> list[dict]:
    if not scored_spans:
        return []
    ordered = sorted(
        scored_spans,
        key=lambda row: (-float(row["support_score"]), int(row["sentence_index"])),
    )
    return ordered[: min(n_sentences, len(ordered))]


def select_random_sentences(spans: list[dict], n_sentences: int, rng) -> list[dict]:
    if not spans:
        return []
    n = min(n_sentences, len(spans))
    indices = rng.choice(len(spans), size=n, replace=False)
    selected = [spans[int(idx)] for idx in sorted(indices)]
    return selected


def retrieval_query_from_sentences(selected_spans: list[dict]) -> str:
    text = " ".join(span["sentence"].strip() for span in selected_spans if span["sentence"].strip()).strip()
    return text


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

    vectorizer = pool["vectorizer"]
    matrix = pool["matrix"]
    rows = pool["rows"]
    query_vec = vectorizer.transform([query_text])
    if query_vec.nnz == 0:
        return None

    sims = linear_kernel(query_vec, matrix).ravel()
    order = np.argsort(-sims)
    for idx in order:
        row = rows.iloc[int(idx)]
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


def summarize(results_df: pd.DataFrame) -> pd.DataFrame:
    results_df = results_df.copy()
    results_df["abs_score_gap"] = (results_df["original_g"] - results_df["replacement_g"]).abs()
    grouped = (
        results_df.groupby(["experiment", "model_label", "method", "condition", "n_sentences", "score_agg"], as_index=False)
        .agg(
            n_queries=("qid", "nunique"),
            mean_original_g=("original_g", "mean"),
            mean_replacement_g=("replacement_g", "mean"),
            mean_delta_g=("delta_g", "mean"),
            mean_abs_score_gap=("abs_score_gap", "mean"),
            preference_preserved_rate=("preference_preserved", "mean"),
            mean_replacement_similarity=("replacement_similarity", "mean"),
            mean_replacement_bm25_rank=("replacement_bm25_rank", "mean"),
            mean_selected_sentence_token_count=("selected_sentence_token_count", "mean"),
            mean_selected_sentence_char_count=("selected_sentence_char_count", "mean"),
            mean_doc_sentence_count=("doc_sentence_count", "mean"),
            mean_doc_token_count=("doc_token_count", "mean"),
        )
    )
    return grouped.sort_values(["score_agg", "condition", "method", "n_sentences"]).reset_index(drop=True)


def run_sentence_retrieval_faithfulness(records, pair_df, cfg, n_sentences: int, score_agg: str):
    family = cfg["family"]
    rng = np.random.default_rng(SEED)
    scoring_runtime = setup_cross_runtime(cfg) if family == "cross_encoder" else setup_duot5_runtime(cfg)
    query_pools = build_query_pools_tfidf(pair_df)
    methods = ["pairwise_ig", "pointwise_ig", "loo_pairwise", "loo_pointwise"]
    rows = []

    for record in records:
        qid = str(record["qid"])
        pool = query_pools[qid]
        excluded = {str(record["pid_i"]), str(record["pid_j"])}
        original_g = original_model_score(record, family, scoring_runtime)
        spans = doc_i_sentence_spans(record, family, scoring_runtime)
        doc_sentence_count = len(spans)
        doc_token_count = sum(len(span["positions"]) for span in spans)

        for method in methods:
            scored_spans = score_sentences(record, method, family, spans, score_agg=score_agg)
            explanation_spans = select_top_sentences(scored_spans, n_sentences=n_sentences)
            if explanation_spans:
                query_text = retrieval_query_from_sentences(explanation_spans)
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
                            "n_sentences": n_sentences,
                            "score_agg": score_agg,
                            "original_g": original_g,
                            "replacement_g": replacement_g,
                            "delta_g": original_g - replacement_g,
                            "preference_preserved": int(np.sign(original_g) == np.sign(replacement_g)),
                            "replacement_pid": replacement["pid"],
                            "replacement_bm25_rank": replacement["bm25_rank"],
                            "replacement_similarity": replacement["similarity"],
                            "selected_sentence_token_count": float(np.mean([len(span["positions"]) for span in explanation_spans])),
                            "selected_sentence_char_count": float(np.mean([len(span["sentence"]) for span in explanation_spans])),
                            "doc_sentence_count": doc_sentence_count,
                            "doc_token_count": doc_token_count,
                            "retrieval_query": query_text,
                            "selected_sentences": " || ".join(span["sentence"] for span in explanation_spans),
                        }
                    )

            random_spans = select_random_sentences(spans, n_sentences=n_sentences, rng=rng)
            if random_spans:
                random_query = retrieval_query_from_sentences(random_spans)
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
                            "n_sentences": n_sentences,
                            "score_agg": score_agg,
                            "original_g": original_g,
                            "replacement_g": replacement_g,
                            "delta_g": original_g - replacement_g,
                            "preference_preserved": int(np.sign(original_g) == np.sign(replacement_g)),
                            "replacement_pid": replacement["pid"],
                            "replacement_bm25_rank": replacement["bm25_rank"],
                            "replacement_similarity": replacement["similarity"],
                            "selected_sentence_token_count": float(np.mean([len(span["positions"]) for span in random_spans])),
                            "selected_sentence_char_count": float(np.mean([len(span["sentence"]) for span in random_spans])),
                            "doc_sentence_count": doc_sentence_count,
                            "doc_token_count": doc_token_count,
                            "retrieval_query": random_query,
                            "selected_sentences": " || ".join(span["sentence"] for span in random_spans),
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
    print(f"Selected evidence spans: top-{args.n_sentences} sentence(s)")
    print(f"Sentence scoring: {args.score_agg}")

    results_df = run_sentence_retrieval_faithfulness(records, pair_df, cfg, n_sentences=args.n_sentences, score_agg=args.score_agg)
    summary_df = summarize(results_df)

    suffix_parts = [f"sent{args.n_sentences}", args.score_agg]
    if args.limit is not None:
        suffix_parts.append(f"limit{args.limit}")
    suffix = "_" + "_".join(suffix_parts)

    results_out = out_dir / f"sentence_retrieval_faithfulness_results{suffix}.csv"
    summary_out = out_dir / f"sentence_retrieval_faithfulness_summary{suffix}.csv"
    results_df.to_csv(results_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    print(f"Saved detailed sentence-retrieval faithfulness results to {results_out}")
    print(f"Saved summary sentence-retrieval faithfulness results to {summary_out}")


if __name__ == "__main__":
    main()
