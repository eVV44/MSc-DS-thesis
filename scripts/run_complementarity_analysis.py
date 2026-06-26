# -- IMPORTS --
from __future__ import annotations
import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from run_faithfulness_final import (
    BUDGETS,
    EXPERIMENTS,
    load_cross_records,
    load_duot5_records,
    pair_key,
    perturb_cross,
    perturb_duot5_pairwise,
    perturb_duot5_pointwise,
    record_target_score,
    selection_to_payload,
)


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
MAX_LENGTH = 512


def setup_cross_runtime(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], local_files_only=True)
    ce = CrossEncoder(cfg["model_name"], max_length=MAX_LENGTH, local_files_only=True)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = ce.model.to(device)
    model.eval()
    return tokenizer, model, device


def setup_duot5_runtime(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"], local_files_only=True)
    device = torch.device("cpu")
    model = model.to(device)
    model.eval()
    decoder_start_token_id = model.config.decoder_start_token_id
    if decoder_start_token_id is None:
        decoder_start_token_id = tokenizer.pad_token_id
    true_ids = tokenizer.encode("true", add_special_tokens=False)
    false_ids = tokenizer.encode("false", add_special_tokens=False)
    if len(true_ids) != 1 or len(false_ids) != 1:
        raise ValueError("Expected 'true' and 'false' to map to single tokens.")
    return tokenizer, model, device, int(decoder_start_token_id), int(true_ids[0]), int(false_ids[0])

COMPARISONS = [
    ("pairwise_ig", "loo_pairwise", "pairwise_ig_vs_loo_pairwise"),
    ("pointwise_ig", "loo_pointwise", "pointwise_ig_vs_loo_pointwise"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["cross_encoder", "duot5", "all"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def selection_id(row: pd.Series):
    return (str(row["side"]), int(row["word_index"]))


def merge_wordpiece_spans(tokens, positions, scores):
    spans = []
    current_tokens, current_positions, current_scores = [], [], []
    for token, pos, score in zip(tokens, positions, scores):
        if token.startswith("##") and current_tokens:
            current_tokens.append(token)
            current_positions.append(int(pos))
            current_scores.append(float(score))
        else:
            if current_tokens:
                spans.append((current_tokens, current_positions, current_scores))
            current_tokens = [token]
            current_positions = [int(pos)]
            current_scores = [float(score)]
    if current_tokens:
        spans.append((current_tokens, current_positions, current_scores))

    rows = []
    for word_index, (toks, poss, scs) in enumerate(spans):
        word = toks[0].replace("##", "")
        for tok in toks[1:]:
            word += tok.replace("##", "")
        rows.append({"word_index": word_index, "token": word, "positions": poss, "support_score": float(sum(scs))})
    return rows


def merge_t5_spans(tokens, positions, scores):
    spans = []
    current_tokens, current_positions, current_scores = [], [], []
    for token, pos, score in zip(tokens, positions, scores):
        starts_new = token.startswith("▁") or not current_tokens
        if starts_new and current_tokens:
            spans.append((current_tokens, current_positions, current_scores))
            current_tokens, current_positions, current_scores = [], [], []
        current_tokens.append(token)
        current_positions.append(int(pos))
        current_scores.append(float(score))
    if current_tokens:
        spans.append((current_tokens, current_positions, current_scores))

    rows = []
    for word_index, (toks, poss, scs) in enumerate(spans):
        pieces = [tok.replace("▁", "") for tok in toks]
        word = "".join(pieces).strip()
        rows.append({"word_index": word_index, "token": word, "positions": poss, "support_score": float(sum(scs))})
    return rows


def build_word_support_table(record, method: str, family: str) -> pd.DataFrame:
    rows = []
    if family == "cross_encoder":
        if method == "pairwise_ig":
            for side, attr, sign in [("doc_i", record["pairwise_ig"]["doc_i"], 1.0), ("doc_j", record["pairwise_ig"]["doc_j"], 1.0)]:
                doc_positions = list(range(int(attr["sep_idx"]) + 1, int(attr["sep_idx"]) + 1 + len(attr["doc_tokens"])))
                for row in merge_wordpiece_spans(attr["doc_tokens"], doc_positions, np.asarray(attr["doc_token_scores"]) * sign):
                    rows.append({"side": side, **row})
        elif method == "pointwise_ig":
            for side, attr, sign in [("doc_i", record["pointwise_ig_i"], 1.0), ("doc_j", record["pointwise_ig_j"], -1.0)]:
                doc_positions = list(range(int(attr["sep_idx"]) + 1, int(attr["sep_idx"]) + 1 + len(attr["doc_tokens"])))
                for row in merge_wordpiece_spans(attr["doc_tokens"], doc_positions, np.asarray(attr["doc_token_scores"]) * sign):
                    rows.append({"side": side, **row})
        elif method == "loo_pairwise":
            for side, entries, sign in [("doc_i", record["loo_pairwise"]["doc_i"], 1.0), ("doc_j", record["loo_pairwise"]["doc_j"], 1.0)]:
                for word_index, entry in enumerate(entries):
                    rows.append({"side": side, "word_index": word_index, "token": entry["word"], "positions": [int(p) for p in entry["positions"]], "support_score": float(sign * entry["support_score"])})
        elif method == "loo_pointwise":
            for side, entries, sign in [("doc_i", record["loo_pointwise_i"]["doc"], 1.0), ("doc_j", record["loo_pointwise_j"]["doc"], -1.0)]:
                for word_index, entry in enumerate(entries):
                    rows.append({"side": side, "word_index": word_index, "token": entry["word"], "positions": [int(p) for p in entry["positions"]], "support_score": float(sign * entry["support_score"])})
    elif family == "duot5":
        if method == "pairwise_ig":
            for side, attr, sign in [("doc_i", record["pairwise_ig"]["doc0"], 1.0), ("doc_j", record["pairwise_ig"]["doc1"], 1.0)]:
                for row in merge_t5_spans(attr["tokens"], attr["positions"], np.asarray(attr["token_scores"]) * sign):
                    rows.append({"side": side, **row})
        elif method == "pointwise_ig":
            for side, attr, sign in [("doc_i", record["pointwise_ig_i"]["doc0"], 1.0), ("doc_j", record["pointwise_ig_j"]["doc0"], -1.0)]:
                for row in merge_t5_spans(attr["tokens"], attr["positions"], np.asarray(attr["token_scores"]) * sign):
                    rows.append({"side": side, **row})
        elif method == "loo_pairwise":
            for side, entries, sign in [("doc_i", record["loo_pairwise"]["doc_i"], 1.0), ("doc_j", record["loo_pairwise"]["doc_j"], 1.0)]:
                for word_index, entry in enumerate(entries):
                    rows.append({"side": side, "word_index": word_index, "token": entry["word"], "positions": [int(p) for p in entry["positions"]], "support_score": float(sign * entry["support_score"])})
        elif method == "loo_pointwise":
            for side, entries, sign in [("doc_i", record["loo_pointwise_i"]["doc"], 1.0), ("doc_j", record["loo_pointwise_j"]["doc"], -1.0)]:
                for word_index, entry in enumerate(entries):
                    rows.append({"side": side, "word_index": word_index, "token": entry["word"], "positions": [int(p) for p in entry["positions"]], "support_score": float(sign * entry["support_score"])})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    return table.sort_values("support_score", ascending=False).reset_index(drop=True)


def select_word_tokens(record, method: str, family: str, k: int) -> pd.DataFrame:
    table = build_word_support_table(record, method, family)
    if table.empty:
        return table
    table = table[table["support_score"] > 0].copy().reset_index(drop=True)
    return table.head(k).reset_index(drop=True)


def overlap_stats(sel_a: pd.DataFrame, sel_b: pd.DataFrame):
    ids_a = {selection_id(row) for _, row in sel_a.iterrows()}
    ids_b = {selection_id(row) for _, row in sel_b.iterrows()}
    overlap = ids_a & ids_b
    unique_a = ids_a - ids_b
    unique_b = ids_b - ids_a
    denom = min(len(ids_a), len(ids_b))
    overlap_rate = float(len(overlap) / denom) if denom else np.nan
    return {
        "overlap_count": int(len(overlap)),
        "overlap_rate": overlap_rate,
        "unique_a_count": int(len(unique_a)),
        "unique_b_count": int(len(unique_b)),
    }


def build_priority_merge(record, method_a: str, method_b: str, family: str, k: int) -> pd.DataFrame:
    pool_a = build_word_support_table(record, method_a, family)
    pool_b = build_word_support_table(record, method_b, family)
    pool_a = pool_a[pool_a["support_score"] > 0].reset_index(drop=True)
    pool_b = pool_b[pool_b["support_score"] > 0].reset_index(drop=True)
    if pool_a.empty and pool_b.empty:
        return pd.DataFrame()

    chosen_rows = []
    seen = set()
    idx_a = 0
    idx_b = 0
    turn = 0

    while len(chosen_rows) < k and (idx_a < len(pool_a) or idx_b < len(pool_b)):
        use_a = (turn % 2 == 0)
        if use_a and idx_a < len(pool_a):
            row = pool_a.iloc[idx_a]
            idx_a += 1
        elif (not use_a) and idx_b < len(pool_b):
            row = pool_b.iloc[idx_b]
            idx_b += 1
        elif idx_a < len(pool_a):
            row = pool_a.iloc[idx_a]
            idx_a += 1
        elif idx_b < len(pool_b):
            row = pool_b.iloc[idx_b]
            idx_b += 1
        else:
            break

        row_id = selection_id(row)
        if row_id in seen:
            turn += 1
            continue
        seen.add(row_id)
        chosen_rows.append(row.to_dict())
        turn += 1

    if not chosen_rows:
        return pd.DataFrame()
    return pd.DataFrame(chosen_rows).reset_index(drop=True)


def evaluate_selection(record, family: str, runtime, method_family: str, payload, check: str) -> float:
    query = record["query"]
    passage_i = record["passage_i"]
    passage_j = record["passage_j"]

    if family == "cross_encoder":
        return perturb_cross(runtime, query, passage_i, passage_j, payload, check)
    if method_family == "pairwise":
        return perturb_duot5_pairwise(runtime, query, passage_i, passage_j, payload, check)
    return perturb_duot5_pointwise(runtime, record, payload, check)


def run_one(experiment_key: str, limit: int | None = None):
    cfg = EXPERIMENTS[experiment_key]
    family = cfg["family"]
    records = load_cross_records(cfg) if family == "cross_encoder" else load_duot5_records(cfg)
    if limit is not None:
        records = records[:limit]
    runtime = setup_cross_runtime(cfg) if family == "cross_encoder" else setup_duot5_runtime(cfg)
    rows = []

    for record in records:
        key = pair_key(record["qid"], record["pid_i"], record["pid_j"])
        for method_a, method_b, comparison_label in COMPARISONS:
            method_family = "pairwise" if method_a.startswith("pairwise") or method_a == "loo_pairwise" else "pointwise"
            original_score = record_target_score(record, method_a, family)

            for k in BUDGETS:
                sel_a = select_word_tokens(record, method_a, family, k)
                sel_b = select_word_tokens(record, method_b, family, k)
                merged = build_priority_merge(record, method_a, method_b, family, k)
                if sel_a.empty or sel_b.empty or merged.empty:
                    continue

                stats = overlap_stats(sel_a, sel_b)
                payload_a = selection_to_payload(sel_a)
                payload_b = selection_to_payload(sel_b)
                payload_m = selection_to_payload(merged)

                for check in ["deletion", "preservation"]:
                    score_a = evaluate_selection(record, family, runtime, method_family, payload_a, check)
                    score_b = evaluate_selection(record, family, runtime, method_family, payload_b, check)
                    score_m = evaluate_selection(record, family, runtime, method_family, payload_m, check)

                    rows.append(
                        {
                            "experiment": family,
                            "model_label": cfg["label"],
                            "qid": key[0],
                            "pid_i": key[1],
                            "pid_j": key[2],
                            "comparison": comparison_label,
                            "method_a": method_a,
                            "method_b": method_b,
                            "k": k,
                            "check": check,
                            "overlap_count": stats["overlap_count"],
                            "overlap_rate": stats["overlap_rate"],
                            "unique_a_count": stats["unique_a_count"],
                            "unique_b_count": stats["unique_b_count"],
                            "g": float(original_score),
                            "g_a": float(score_a),
                            "g_b": float(score_b),
                            "g_merged": float(score_m),
                            "delta_a": float(original_score - score_a),
                            "delta_b": float(original_score - score_b),
                            "delta_merged": float(original_score - score_m),
                            "flip_a": int(np.sign(original_score) != np.sign(score_a)),
                            "flip_b": int(np.sign(original_score) != np.sign(score_b)),
                            "flip_merged": int(np.sign(original_score) != np.sign(score_m)),
                            "abs_gap_a": float(abs(original_score - score_a)),
                            "abs_gap_b": float(abs(original_score - score_b)),
                            "abs_gap_merged": float(abs(original_score - score_m)),
                            "sign_pres_a": int(np.sign(original_score) == np.sign(score_a)),
                            "sign_pres_b": int(np.sign(original_score) == np.sign(score_b)),
                            "sign_pres_merged": int(np.sign(original_score) == np.sign(score_m)),
                            "merged_minus_mean_delta": float((original_score - score_m) - np.mean([original_score - score_a, original_score - score_b])),
                            "merged_minus_best_delta": float((original_score - score_m) - max(original_score - score_a, original_score - score_b)),
                        }
                    )

    detail_df = pd.DataFrame(rows)
    summary_rows = []
    for (comparison, check, k), group in detail_df.groupby(["comparison", "check", "k"], dropna=False):
        summary_rows.append(
            {
                "experiment": family,
                "model_label": cfg["label"],
                "comparison": comparison,
                "check": check,
                "k": int(k),
                "n_pairs": int(len(group)),
                "mean_overlap_count": float(group["overlap_count"].mean()),
                "mean_overlap_rate": float(group["overlap_rate"].mean()),
                "mean_unique_a_count": float(group["unique_a_count"].mean()),
                "mean_unique_b_count": float(group["unique_b_count"].mean()),
                "mean_delta_a": float(group["delta_a"].mean()),
                "mean_delta_b": float(group["delta_b"].mean()),
                "mean_delta_merged": float(group["delta_merged"].mean()),
                "flip_rate_a": float(group["flip_a"].mean()),
                "flip_rate_b": float(group["flip_b"].mean()),
                "flip_rate_merged": float(group["flip_merged"].mean()),
                "mean_abs_gap_a": float(group["abs_gap_a"].mean()),
                "mean_abs_gap_b": float(group["abs_gap_b"].mean()),
                "mean_abs_gap_merged": float(group["abs_gap_merged"].mean()),
                "sign_pres_rate_a": float(group["sign_pres_a"].mean()),
                "sign_pres_rate_b": float(group["sign_pres_b"].mean()),
                "sign_pres_rate_merged": float(group["sign_pres_merged"].mean()),
                "mean_merged_minus_mean_delta": float(group["merged_minus_mean_delta"].mean()),
                "mean_merged_minus_best_delta": float(group["merged_minus_best_delta"].mean()),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values(["comparison", "check", "k"]).reset_index(drop=True)

    out_dir = cfg["faithfulness_dir"]
    detail_out = out_dir / "complementarity_results.csv"
    summary_out = out_dir / "complementarity_summary.csv"
    detail_df.to_csv(detail_out, index=False)
    summary_df.to_csv(summary_out, index=False)
    print(f"Saved complementarity results to {detail_out}")
    print(f"Saved complementarity summary to {summary_out}")


def main():
    args = parse_args()
    if args.experiment == "all":
        for experiment_key in ["cross_encoder", "duot5"]:
            run_one(experiment_key, limit=args.limit)
    else:
        run_one(args.experiment, limit=args.limit)


if __name__ == "__main__":
    main()