# -- IMPORTS --
from __future__ import annotations
import pickle
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]

QUERIES_SCOPE_FILE = ROOT / "thesis_runs/shared/pair_definitions/duot5_queries_150_seed42_top100.csv"
CANDIDATES_FILE = ROOT / "data/msmarco_passage_dev/raw/top1000.dev"
QRELS_FILE = ROOT / "data/msmarco_passage_dev/raw/qrels.dev.tsv"
MONO_RUN_FILE = ROOT / "thesis_runs/duot5/reranking/monot5_first_stage/mono_run.txt"
OUT_DIR = ROOT / "thesis_runs/duot5/reranking/duot5_final"

MODEL_NAME = "castorini/duot5-base-msmarco"
MONO_CANDIDATE_DEPTH = 100
DUOT5_RERANK_DEPTH = 50
BATCH_SIZE = 8
MAX_LENGTH = 512
MAX_PAIRS_PER_QUERY = 5
SEED = 42


def load_query_scope() -> dict[str, str]:
    queries = pd.read_csv(QUERIES_SCOPE_FILE, dtype={"qid": str, "query": str})
    return dict(zip(queries["qid"], queries["query"]))


def load_candidates_base() -> pd.DataFrame:
    candidates_df = pd.read_csv(
        CANDIDATES_FILE,
        sep="\t",
        header=None,
        names=["qid", "pid", "query", "passage"],
        dtype={0: str, 1: str},
    )
    candidates_df["qid"] = candidates_df["qid"].astype(str)
    candidates_df["pid"] = candidates_df["pid"].astype(str)
    return candidates_df


def load_query_candidates(query_map: dict[str, str], candidates_base_df: pd.DataFrame) -> pd.DataFrame:
    if not MONO_RUN_FILE.exists():
        raise FileNotFoundError(
            f"Missing monoT5 run file: {MONO_RUN_FILE}. Run scripts/run_monot5_reranking_for_duot5_final.py first."
        )

    mono_run_df = pd.read_csv(
        MONO_RUN_FILE,
        sep=r"\s+",
        header=None,
        names=["qid", "Q0", "pid", "rank", "score", "tag"],
        dtype={0: str, 2: str},
    )
    mono_run_df["qid"] = mono_run_df["qid"].astype(str)
    mono_run_df["pid"] = mono_run_df["pid"].astype(str)
    mono_run_df = mono_run_df[mono_run_df["qid"].isin(query_map)]
    mono_run_df = (
        mono_run_df.sort_values(["qid", "rank"])
        .groupby("qid", sort=False)
        .head(DUOT5_RERANK_DEPTH)
    )

    candidates_df = mono_run_df.merge(
        candidates_base_df[["qid", "pid", "passage"]].drop_duplicates(["qid", "pid"]),
        on=["qid", "pid"],
        how="left",
    )
    candidates_df = candidates_df.dropna(subset=["passage"]).copy()
    candidates_df["query"] = candidates_df["qid"].map(query_map)
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
        raise ValueError("Expected 'true' and 'false' to map to single tokens for DuoT5 scoring.")

    decoder_start_token_id = model.config.decoder_start_token_id
    if decoder_start_token_id is None:
        decoder_start_token_id = tokenizer.pad_token_id

    return tokenizer, model, device, int(true_ids[0]), int(false_ids[0]), int(decoder_start_token_id)


def duo_input(query: str, doc0: str, doc1: str) -> str:
    return f"Query: {query} Document0: {doc0} Document1: {doc1} Relevant:"


def duo_true_probs(texts, tokenizer, model, device, true_token_id, false_token_id, decoder_start_token_id):
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


def duo_aggregate_scores_sym(query, passages, tokenizer, model, device, true_token_id, false_token_id, decoder_start_token_id):
    n = len(passages)
    if n == 1:
        return np.array([1.0], dtype=float)

    pair_indices = list(combinations(range(n), 2))
    inputs = []
    pair_meta = []

    for i, j in pair_indices:
        inputs.append(duo_input(query, passages[i], passages[j]))
        pair_meta.append((i, j, "ij"))
        inputs.append(duo_input(query, passages[j], passages[i]))
        pair_meta.append((i, j, "ji"))

    probs = duo_true_probs(inputs, tokenizer, model, device, true_token_id, false_token_id, decoder_start_token_id)
    scores = np.zeros(n, dtype=float)

    for prob, (i, j, direction) in zip(probs, pair_meta):
        if direction == "ij":
            scores[i] += prob
            scores[j] += 1.0 - prob
        else:
            scores[j] += prob
            scores[i] += 1.0 - prob

    scores /= (2 * (n - 1))
    return scores


def rerank(query_map, candidates_df, qrel_map):
    tokenizer, model, device, true_token_id, false_token_id, decoder_start_token_id = load_model()
    ranked_results = {}

    for qid, group in candidates_df.groupby("qid"):
        query_text = query_map[qid]
        passages = group["passage"].tolist()
        pids = group["pid"].tolist()

        scores = duo_aggregate_scores_sym(
            query_text,
            passages,
            tokenizer,
            model,
            device,
            true_token_id,
            false_token_id,
            decoder_start_token_id,
        )
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

    with ranked_pkl.open("wb") as f:
        pickle.dump(ranked_results, f)

    ranked_rows = [{"qid": qid, **entry} for qid, results in ranked_results.items() for entry in results]
    pd.DataFrame(ranked_rows).to_csv(ranked_csv, index=False)

    pairs_df.to_pickle(pairs_pkl)
    pairs_df.to_csv(pairs_csv, index=False)
    try:
        pairs_df.to_parquet(pairs_parquet, index=False)
    except Exception as exc:
        print(f"Skipping parquet save: {exc}")

    print(f"Saved DuoT5 ranked results to {ranked_pkl} and {ranked_csv}")
    print(f"Saved pairwise scores to {pairs_pkl}, {pairs_csv}")


def main() -> None:
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    query_map = load_query_scope()
    qrel_map = load_qrels()
    candidates_base_df = load_candidates_base()
    candidates_df = load_query_candidates(query_map, candidates_base_df)

    print(f"DuoT5 query subset size: {len(query_map):,}")
    print(
        "DuoT5 candidate rows after monoT5 "
        f"top-{DUOT5_RERANK_DEPTH} selection (from BM25/monoT5 top-{MONO_CANDIDATE_DEPTH} pool): "
        f"{len(candidates_df):,}"
    )

    ranked_results = rerank(query_map, candidates_df, qrel_map)
    pairs_df = build_pairwise_table(ranked_results, query_map)
    save_outputs(ranked_results, pairs_df)

    print(f"Reranked queries: {len(ranked_results):,}")
    print(f"Pairwise rows: {len(pairs_df):,}")
    print(f"Queries covered in pair table: {pairs_df['qid'].nunique():,}")


if __name__ == "__main__":
    main()