# -- IMPORTS --
from __future__ import annotations
import argparse
import gc
import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(root / "thesis_runs/shared/mplconfig"))

import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from run_faithfulness_final import (
    build_support_table,
    load_cross_records,
    load_duot5_records,
    pair_key,
)


ROOT = Path(__file__).resolve().parents[1]

SEED = 42
MAX_LENGTH = 256
TOP_KS = [1, 3, 5, 10, 20]
POOL_TOP_N = 100
EMBED_BATCH_SIZE = 16
FORCE_CPU = True
RETRIEVER_MODEL = "castorini/monot5-base-msmarco"

EXPERIMENTS = {
    "cross_encoder": {
        "label": "Cross-encoder",
        "family": "cross_encoder",
        "pair_file": ROOT / "thesis_runs/cross_encoder/explanations/explanation_pairs_1_per_query_on_duot5_queries_seed42.csv",
        "faithfulness_dir": ROOT / "thesis_runs/cross_encoder/faithfulness",
        "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    },
    "duot5": {
        "label": "DuoT5",
        "family": "duot5",
        "pair_file": ROOT / "thesis_runs/duot5/explanations/explanation_pairs_1_per_query_seed42.csv",
        "faithfulness_dir": ROOT / "thesis_runs/duot5/faithfulness",
        "model_name": "castorini/duot5-base-msmarco",
    },
}

TOP1000_FILE = ROOT / "data/msmarco_passage_dev/raw/top1000.dev"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_top100_pools(qids: set[str]):
    df = pd.read_csv(
        TOP1000_FILE,
        sep="\t",
        header=None,
        names=["qid", "pid", "query", "passage"],
        dtype={"qid": str, "pid": str, "query": str, "passage": str},
    )
    df = df[df["qid"].isin(qids)].copy()
    df["bm25_rank"] = df.groupby("qid").cumcount() + 1
    return df[df["bm25_rank"] <= POOL_TOP_N].reset_index(drop=True)


def setup_retriever():
    tokenizer = AutoTokenizer.from_pretrained(RETRIEVER_MODEL, use_fast=False, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(RETRIEVER_MODEL, local_files_only=True)
    device = torch.device("cpu") if FORCE_CPU else (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    model = model.to(device)
    model.eval()
    return tokenizer, model, device


def encode_texts(tokenizer, model, device, texts: list[str], batch_size: int = EMBED_BATCH_SIZE):
    vectors = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            encoder_outputs = model.get_encoder()(input_ids=input_ids, attention_mask=attention_mask)
            hidden = encoder_outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1)
            summed = (hidden * mask).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1)
            pooled = summed / lengths
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        vectors.append(pooled.cpu())
    return torch.cat(vectors, dim=0).numpy()


def build_query_pools(retriever_runtime, pair_df: pd.DataFrame):
    qids = set(pair_df["qid"].astype(str))
    pool_df = load_top100_pools(qids)
    tokenizer, model, device = retriever_runtime
    pools = {}
    for qid, group in pool_df.groupby("qid", sort=False):
        passages = group["passage"].astype(str).tolist()
        embeddings = encode_texts(tokenizer, model, device, passages)
        pools[str(qid)] = {
            "rows": group.reset_index(drop=True),
            "embeddings": embeddings,
        }
    return pools


def setup_cross_runtime(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], local_files_only=True)
    ce = CrossEncoder(cfg["model_name"], max_length=512, local_files_only=True)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = ce.model.to(device)
    model.eval()
    return tokenizer, model, device


def score_cross(runtime, query, passage_i, passage_j):
    tokenizer, model, device = runtime
    encoded_i = tokenizer(query, passage_i, max_length=512, truncation=True, padding=False, return_tensors="pt")
    encoded_j = tokenizer(query, passage_j, max_length=512, truncation=True, padding=False, return_tensors="pt")

    def run_one(encoded):
        kwargs = {
            "input_ids": encoded["input_ids"].to(device),
            "attention_mask": encoded["attention_mask"].to(device),
        }
        if "token_type_ids" in encoded:
            kwargs["token_type_ids"] = encoded["token_type_ids"].to(device)
        with torch.no_grad():
            out = model(**kwargs)
        return float(out.logits.squeeze(-1).detach().cpu().item())

    return run_one(encoded_i) - run_one(encoded_j)


def setup_duot5_runtime(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False, local_files_only=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg["model_name"], local_files_only=True)
    device = torch.device("cpu") if FORCE_CPU else (torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu"))
    model = model.to(device)
    model.eval()
    decoder_start_token_id = model.config.decoder_start_token_id
    if decoder_start_token_id is None:
        decoder_start_token_id = tokenizer.pad_token_id
    true_ids = tokenizer.encode("true", add_special_tokens=False)
    false_ids = tokenizer.encode("false", add_special_tokens=False)
    if len(true_ids) != 1 or len(false_ids) != 1:
        raise ValueError("Expected true/false to map to single tokens.")
    return tokenizer, model, device, int(decoder_start_token_id), int(true_ids[0]), int(false_ids[0])


def duo_input(query, doc0, doc1):
    return f"Query: {query} Document0: {doc0} Document1: {doc1} Relevant:"


def score_duot5_pairwise(runtime, query, passage_i, passage_j):
    tokenizer, model, device, decoder_start_token_id, true_token_id, false_token_id = runtime
    enc = tokenizer(
        duo_input(query, passage_i, passage_j),
        max_length=512,
        truncation=True,
        padding=False,
        return_tensors="pt",
    )
    decoder_input_ids = torch.full((1, 1), decoder_start_token_id, dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
            decoder_input_ids=decoder_input_ids,
        )
    logits = outputs.logits[:, 0, :]
    return float((logits[:, true_token_id] - logits[:, false_token_id]).detach().cpu().item())


def select_doc_i_tokens(record, method, family, k):
    table = build_support_table(record, method, family)
    if table.empty:
        return []
    subset = table[(table["side"] == "doc_i") & (table["support_score"] > 0)].copy()
    subset = subset.sort_values("support_score", ascending=False).head(k)
    return subset["token"].astype(str).tolist()


def select_random_doc_i_tokens(record, method, family, k, rng):
    table = build_support_table(record, method, family)
    if table.empty:
        return []
    subset = table[table["side"] == "doc_i"].copy()
    if subset.empty:
        return []
    n = min(k, len(subset))
    picked = subset.sample(n=n, replace=False, random_state=int(rng.integers(0, 1_000_000_000)))
    return picked["token"].astype(str).tolist()


def retrieval_query_from_tokens(query: str, tokens: list[str]) -> str:
    unique_tokens = []
    seen = set()
    for token in tokens:
        norm = token.strip()
        if not norm:
            continue
        if norm not in seen:
            unique_tokens.append(norm)
            seen.add(norm)
    # The replacement pool is already query-specific, so re-appending the full
    # query can drown out the contribution of the explanation-selected tokens.
    # We therefore retrieve from the candidate pool using the selected passage
    # evidence itself, falling back to the query only if token selection fails.
    if unique_tokens:
        return " ".join(unique_tokens)
    return query.strip()


def retrieve_replacement(pool, retriever_runtime, query_text: str, exclude_pids: set[str]):
    tokenizer, model, device = retriever_runtime
    query_vec = encode_texts(tokenizer, model, device, [query_text])[0]
    embeddings = pool["embeddings"]
    sims = embeddings @ query_vec
    rows = pool["rows"]
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


def load_records(cfg):
    pair_df = pd.read_csv(cfg["pair_file"], dtype={"qid": str, "pid_i": str, "pid_j": str})
    if cfg["family"] == "cross_encoder":
        records = load_cross_records({
            "pair_file": cfg["pair_file"],
            "explanation_dir": ROOT / "thesis_runs/cross_encoder/explanations",
        })
    else:
        records = load_duot5_records({
            "pair_file": cfg["pair_file"],
            "explanation_dir": ROOT / "thesis_runs/duot5/explanations",
            "ranked_results_file": ROOT / "thesis_runs/duot5/reranking/duot5_final/ranked_results.csv",
        })
    record_lookup = {pair_key(r["qid"], r["pid_i"], r["pid_j"]): r for r in records}
    ordered_records = []
    for row in pair_df.to_dict("records"):
        key = pair_key(row["qid"], row["pid_i"], row["pid_j"])
        if key in record_lookup:
            ordered_records.append(record_lookup[key])
    return pair_df, ordered_records


def original_model_score(record, family, runtime):
    if family == "cross_encoder":
        return float(score_cross(runtime, record["query"], record["passage_i"], record["passage_j"]))
    return float(score_duot5_pairwise(runtime, record["query"], record["passage_i"], record["passage_j"]))


def replacement_model_score(record, family, runtime, replacement_passage):
    if family == "cross_encoder":
        return float(score_cross(runtime, record["query"], replacement_passage, record["passage_j"]))
    return float(score_duot5_pairwise(runtime, record["query"], replacement_passage, record["passage_j"]))


def run_retrieval_faithfulness(records, pair_df, cfg):
    family = cfg["family"]
    rng = np.random.default_rng(SEED)
    retriever_runtime = setup_retriever()
    scoring_runtime = setup_cross_runtime(cfg) if family == "cross_encoder" else setup_duot5_runtime(cfg)
    query_pools = build_query_pools(retriever_runtime, pair_df)
    methods = ["pairwise_ig", "pointwise_ig", "loo_pairwise", "loo_pointwise"]
    rows = []

    for record in records:
        qid = str(record["qid"])
        pool = query_pools[qid]
        excluded = {str(record["pid_i"]), str(record["pid_j"])}
        original_g = original_model_score(record, family, scoring_runtime)

        for method in methods:
            for k in TOP_KS:
                explanation_tokens = select_doc_i_tokens(record, method, family, k)
                if explanation_tokens:
                    query_text = retrieval_query_from_tokens(record["query"], explanation_tokens)
                    replacement = retrieve_replacement(pool, retriever_runtime, query_text, excluded)
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
                                "k": k,
                                "original_g": original_g,
                                "replacement_g": replacement_g,
                                "delta_g": original_g - replacement_g,
                                "preference_preserved": int(np.sign(original_g) == np.sign(replacement_g)),
                                "replacement_pid": replacement["pid"],
                                "replacement_bm25_rank": replacement["bm25_rank"],
                                "replacement_similarity": replacement["similarity"],
                                "retrieval_query": query_text,
                                "selected_tokens": " | ".join(explanation_tokens),
                            }
                        )

                random_tokens = select_random_doc_i_tokens(record, method, family, k, rng)
                if random_tokens:
                    random_query_text = retrieval_query_from_tokens(record["query"], random_tokens)
                    random_replacement = retrieve_replacement(pool, retriever_runtime, random_query_text, excluded)
                    if random_replacement is not None:
                        replacement_g = replacement_model_score(record, family, scoring_runtime, random_replacement["passage"])
                        rows.append(
                            {
                                "experiment": family,
                                "model_label": cfg["label"],
                                "method": method,
                                "condition": "random",
                                "qid": qid,
                                "pid_i": record["pid_i"],
                                "pid_j": record["pid_j"],
                                "k": k,
                                "original_g": original_g,
                                "replacement_g": replacement_g,
                                "delta_g": original_g - replacement_g,
                                "preference_preserved": int(np.sign(original_g) == np.sign(replacement_g)),
                                "replacement_pid": random_replacement["pid"],
                                "replacement_bm25_rank": random_replacement["bm25_rank"],
                                "replacement_similarity": random_replacement["similarity"],
                                "retrieval_query": random_query_text,
                                "selected_tokens": " | ".join(random_tokens),
                            }
                        )

        gc.collect()
        if torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    return pd.DataFrame(rows)


def summarize(results_df):
    grouped = (
        results_df.groupby(["experiment", "model_label", "method", "condition", "k"], as_index=False)
        .agg(
            n_queries=("qid", "nunique"),
            mean_original_g=("original_g", "mean"),
            mean_replacement_g=("replacement_g", "mean"),
            mean_delta_g=("delta_g", "mean"),
            preference_preserved_rate=("preference_preserved", "mean"),
            mean_replacement_similarity=("replacement_similarity", "mean"),
            mean_replacement_bm25_rank=("replacement_bm25_rank", "mean"),
        )
    )
    return grouped.sort_values(["condition", "method", "k"]).reset_index(drop=True)


def main():
    args = parse_args()
    cfg = EXPERIMENTS[args.experiment]
    pair_df, records = load_records(cfg)
    if args.limit is not None:
        records = records[: args.limit]
        keep = {pair_key(r["qid"], r["pid_i"], r["pid_j"]) for r in records}
        pair_df = pair_df[[pair_key(r["qid"], r["pid_i"], r["pid_j"]) in keep for r in pair_df.to_dict("records")]].reset_index(drop=True)

    out_dir = cfg["faithfulness_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment: {cfg['label']} ({cfg['family']})")
    print(f"Loaded records: {len(records)}")
    print(f"Replacement pool: BM25 top-{POOL_TOP_N} documents per query")
    print(f"Retriever: {RETRIEVER_MODEL} mean-pooled encoder embeddings")
    print(f"Top-k token budgets: {TOP_KS}")

    results_df = run_retrieval_faithfulness(records, pair_df, cfg)
    summary_df = summarize(results_df)

    suffix = f"_limit{args.limit}" if args.limit is not None else ""
    results_out = out_dir / f"retrieval_faithfulness_results{suffix}.csv"
    summary_out = out_dir / f"retrieval_faithfulness_summary{suffix}.csv"
    results_df.to_csv(results_out, index=False)
    summary_df.to_csv(summary_out, index=False)

    print(f"Saved detailed retrieval-faithfulness results to {results_out}")
    print(f"Saved summary retrieval-faithfulness results to {summary_out}")


if __name__ == "__main__":
    main()
