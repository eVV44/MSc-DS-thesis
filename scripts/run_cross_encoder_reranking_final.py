# -- IMPORTS --
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sentence_transformers import CrossEncoder


root = Path(__file__).resolve().parents[1]

queries_scope_file = root / "thesis_runs/shared/pair_definitions/cross_encoder_queries_full_eligible_top100.csv"
candidates_file = root / "data/msmarco_passage_dev/raw/top1000.dev"
qrels_file = root / "data/msmarco_passage_dev/raw/qrels.dev.tsv"
out_dir = root / "thesis_runs/cross_encoder/reranking"

model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
top_k = 100
batch_size = 64
max_pairs_per_query = 5
seed = 42


def load_query_scope() -> dict[str, str]:
    queries = pd.read_csv(queries_scope_file, dtype={"qid": str, "query": str})
    return dict(zip(queries["qid"], queries["query"]))


def load_candidates(query_map: dict[str, str]) -> pd.DataFrame:
    candidates_df = pd.read_csv(
        candidates_file,
        sep="\t",
        header=None,
        names=["qid", "pid", "query", "passage"],
        dtype={0: str, 1: str})
    
    candidates_df["qid"] = candidates_df["qid"].astype(str)
    candidates_df["pid"] = candidates_df["pid"].astype(str)
    candidates_df = candidates_df[candidates_df["qid"].isin(query_map)]
    candidates_df = candidates_df.groupby("qid", sort=False).head(top_k).reset_index(drop=True)
    return candidates_df


def load_qrels() -> dict[tuple[str, str], int]:
    qrels_df = pd.read_csv(
        qrels_file,
        sep="\t",
        header=None,
        names=["qid", "_", "pid", "relevance"],
        dtype={0: str, 2: str})
    
    qrels_df["qid"] = qrels_df["qid"].astype(str)
    qrels_df["pid"] = qrels_df["pid"].astype(str)
    return {(row.qid, row.pid): int(row.relevance) for row in qrels_df.itertuples(index=False)}


def rerank(query_map: dict[str, str], candidates_df: pd.DataFrame, qrel_map: dict[tuple[str, str], int]) -> dict[str, list[dict]]:
    model = CrossEncoder(model_name, max_length=512)
    if torch.backends.mps.is_available():
        model.model = model.model.to("mps")

    ranked_results: dict[str, list[dict]] = {}
    for qid, group in candidates_df.groupby("qid"):
        query_text = query_map[qid]
        passages = group["passage"].tolist()
        pids = group["pid"].tolist()

        pairs = [[query_text, passage] for passage in passages]
        scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        order = np.argsort(scores)[::-1]

        ranked_results[qid] = [
            {
                "pid": pids[i],
                "passage": passages[i],
                "score": float(scores[i]),
                "rank": rank + 1,
                "relevant": int(qrel_map.get((qid, pids[i]), 0) > 0)}
            for rank, i in enumerate(order)]

    return ranked_results


def build_pairwise_table(ranked_results: dict[str, list[dict]], query_map: dict[str, str]) -> pd.DataFrame:
    pairwise_records = []
    rng = np.random.default_rng(seed)

    for qid, results in ranked_results.items():
        query_text = query_map[qid]
        rel_entries = [entry for entry in results if entry["relevant"]]
        nonrel_entries = [entry for entry in results if not entry["relevant"]]

        if not rel_entries or not nonrel_entries:
            continue

        pairs = [(di, dj) for di in rel_entries for dj in nonrel_entries]
        if max_pairs_per_query and len(pairs) > max_pairs_per_query:
            idx = rng.choice(len(pairs), size=max_pairs_per_query, replace=False)
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
                    "correct_pref": int(g > 0)})

    return pd.DataFrame(pairwise_records)


def save_outputs(ranked_results: dict[str, list[dict]], pairs_df: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    ranked_pkl = out_dir / "ranked_results.pkl"
    ranked_csv = out_dir / "ranked_results.csv"
    pairs_pkl = out_dir / "pairwise_scores.pkl"
    pairs_csv = out_dir / "pairwise_scores.csv"
    pairs_parquet = out_dir / "pairwise_scores.parquet"

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

    print(f"Saved ranked results to {ranked_pkl} and {ranked_csv}")
    print(f"Saved pairwise scores to {pairs_pkl}, {pairs_csv}")


def main() -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)

    query_map = load_query_scope()
    qrel_map = load_qrels()
    candidates_df = load_candidates(query_map)

    print(f"Cross-encoder queries: {len(query_map):,}")
    print(f"Candidate rows after top-{top_k}: {len(candidates_df):,}")

    ranked_results = rerank(query_map, candidates_df, qrel_map)
    pairs_df = build_pairwise_table(ranked_results, query_map)
    save_outputs(ranked_results, pairs_df)

    print(f"Reranked queries: {len(ranked_results):,}")
    print(f"Pairwise rows: {len(pairs_df):,}")
    print(f"Queries covered in pair table: {pairs_df['qid'].nunique():,}")


if __name__ == "__main__":
    main()