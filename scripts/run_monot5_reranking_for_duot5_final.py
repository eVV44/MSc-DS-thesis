# -- IMPORTS --
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]

QUERIES_SCOPE_FILE = ROOT / "thesis_runs/shared/pair_definitions/duot5_queries_150_seed42_top100.csv"
CANDIDATES_FILE = ROOT / "data/msmarco_passage_dev/raw/top1000.dev"
QRELS_FILE = ROOT / "data/msmarco_passage_dev/raw/qrels.dev.tsv"
OUT_DIR = ROOT / "thesis_runs/duot5/reranking/monot5_first_stage"

MODEL_NAME = "castorini/monot5-base-msmarco"
TOP_K = 100
BATCH_SIZE = 8
MAX_LENGTH = 512
MAX_PAIRS_PER_QUERY = 5
SEED = 42


def load_query_scope() -> dict[str, str]:
    queries = pd.read_csv(QUERIES_SCOPE_FILE, dtype={"qid": str, "query": str})
    return dict(zip(queries["qid"], queries["query"]))


def load_candidates(query_map: dict[str, str]) -> pd.DataFrame:
    candidates_df = pd.read_csv(
        CANDIDATES_FILE,
        sep="\t",
        header=None,
        names=["qid", "pid", "query", "passage"],
        dtype={0: str, 1: str},
    )
    candidates_df["qid"] = candidates_df["qid"].astype(str)
    candidates_df["pid"] = candidates_df["pid"].astype(str)
    candidates_df = candidates_df[candidates_df["qid"].isin(query_map)]
    candidates_df = candidates_df.groupby("qid", sort=False).head(TOP_K).reset_index(drop=True)
    return candidates_df


def load_qrels() -> dict[tuple[str, str], int]:
    qrels_df = pd.read_csv(
        QRELS_FILE,
        sep="\t",
        header=None,
        names=["qid", "_", "pid", "relevance"],
        dtype={0: str, 2: str},
    )
    qrels_df["qid"] = qrels_df["qid"].astype(str)
    qrels_df["pid"] = qrels_df["pid"].astype(str)
    return {(row.qid, row.pid): int(row.relevance) for row in qrels_df.itertuples(index=False)}


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model = model.to(device)
    model.eval()

    true_ids = tokenizer.encode("true", add_special_tokens=False)
    false_ids = tokenizer.encode("false", add_special_tokens=False)
    if len(true_ids) != 1 or len(false_ids) != 1:
        raise ValueError("Expected 'true' and 'false' to map to single tokens for monoT5 scoring.")
    decoder_start_token_id = model.config.decoder_start_token_id
    if decoder_start_token_id is None:
        decoder_start_token_id = tokenizer.pad_token_id

    return tokenizer, model, device, int(true_ids[0]), int(false_ids[0]), int(decoder_start_token_id)


def mono_input(query: str, passage: str) -> str:
    return f"Query: {query} Document: {passage} Relevant:"


def monot5_true_probs(texts, tokenizer, model, device, true_token_id, false_token_id, decoder_start_token_id):
    probs = []
    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start:start + BATCH_SIZE]
        batch = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        batch = {k: v.to(device) for k, v in batch.items()}
        decoder_input_ids = torch.full(
            (len(batch_texts), 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=device,
        )

        with torch.no_grad():
            outputs = model(**batch, decoder_input_ids=decoder_input_ids)

        logits = outputs.logits[:, 0, :]
        tf_logits = logits[:, [false_token_id, true_token_id]]
        batch_probs = torch.softmax(tf_logits, dim=-1)[:, 1].detach().cpu().numpy()
        probs.extend(batch_probs.tolist())

    return np.array(probs, dtype=float)


def rerank(query_map, candidates_df, qrel_map):
    tokenizer, model, device, true_token_id, false_token_id, decoder_start_token_id = load_model()
    ranked_results = {}

    for qid, group in candidates_df.groupby("qid"):
        query_text = query_map[qid]
        passages = group["passage"].tolist()
        pids = group["pid"].tolist()

        inputs = [mono_input(query_text, passage) for passage in passages]
        scores = monot5_true_probs(inputs, tokenizer, model, device, true_token_id, false_token_id, decoder_start_token_id)
        order = np.argsort(scores)[::-1]

        ranked_results[qid] = [
            {
                "pid": pids[i],
                "passage": passages[i],
                "score": float(scores[i]),
                "rank": rank + 1,
                "relevant": int(qrel_map.get((qid, pids[i]), 0) > 0),
            }
            for rank, i in enumerate(order)
        ]

    return ranked_results


def build_pairwise_table(ranked_results: dict[str, list[dict]], query_map: dict[str, str]) -> pd.DataFrame:
    pairwise_records = []
    rng = np.random.default_rng(SEED)

    for qid, results in ranked_results.items():
        query_text = query_map[qid]
        rel_entries = [entry for entry in results if entry["relevant"]]
        nonrel_entries = [entry for entry in results if not entry["relevant"]]

        if not rel_entries or not nonrel_entries:
            continue

        pairs = [(di, dj) for di in rel_entries for dj in nonrel_entries]
        if MAX_PAIRS_PER_QUERY and len(pairs) > MAX_PAIRS_PER_QUERY:
            idx = rng.choice(len(pairs), size=MAX_PAIRS_PER_QUERY, replace=False)
            pairs = [pairs[i] for i in idx]

        for di, dj in pairs:
            g = di["score"] - dj["score"]
            pairwise_records.append(
                {
                    "qid": qid,
                    "query": query_text,
                    "pid_i": di["pid"],
                    "passage_i": di["passage"],
                    "score_i": di["score"],
                    "pid_j": dj["pid"],
                    "passage_j": dj["passage"],
                    "score_j": dj["score"],
                    "g_score": g,
                    "correct_pref": int(g > 0),
                }
            )

    return pd.DataFrame(pairwise_records)


def save_outputs(ranked_results: dict[str, list[dict]], pairs_df: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ranked_pkl = OUT_DIR / "ranked_results.pkl"
    ranked_csv = OUT_DIR / "ranked_results.csv"
    pairs_pkl = OUT_DIR / "pairwise_scores.pkl"
    pairs_csv = OUT_DIR / "pairwise_scores.csv"
    pairs_parquet = OUT_DIR / "pairwise_scores.parquet"
    mono_run_out = OUT_DIR / "mono_run.txt"

    with ranked_pkl.open("wb") as f:
        pickle.dump(ranked_results, f)

    ranked_rows = [{"qid": qid, **entry} for qid, results in ranked_results.items() for entry in results]
    pd.DataFrame(ranked_rows).to_csv(ranked_csv, index=False)

    run_rows = []
    for qid, results in ranked_results.items():
        for entry in results:
            run_rows.append([qid, "Q0", entry["pid"], entry["rank"], entry["score"], "monoT5"])
    pd.DataFrame(run_rows).to_csv(mono_run_out, sep=" ", header=False, index=False)

    pairs_df.to_pickle(pairs_pkl)
    pairs_df.to_csv(pairs_csv, index=False)
    try:
        pairs_df.to_parquet(pairs_parquet, index=False)
    except Exception as exc:
        print(f"Skipping parquet save: {exc}")

    print(f"Saved monoT5 first-stage run to {mono_run_out}")


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    query_map = load_query_scope()
    qrel_map = load_qrels()
    candidates_df = load_candidates(query_map)

    print(f"DuoT5 query subset size: {len(query_map):,}")
    print(f"monoT5 candidate rows after top-{TOP_K}: {len(candidates_df):,}")

    ranked_results = rerank(query_map, candidates_df, qrel_map)
    pairs_df = build_pairwise_table(ranked_results, query_map)
    save_outputs(ranked_results, pairs_df)

    print(f"Reranked queries: {len(ranked_results):,}")
    print(f"Pairwise rows: {len(pairs_df):,}")
    print(f"Queries covered in pair table: {pairs_df['qid'].nunique():,}")


if __name__ == "__main__":
    main()