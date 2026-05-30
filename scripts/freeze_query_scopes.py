# -- IMPORTS --
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


root = Path(__file__).resolve().parents[1]

queries_file = root / "data/msmarco_passage_dev/raw/queries.dev.tsv"
candidates_file = root / "data/msmarco_passage_dev/raw/top1000.dev"
qrels_file = root / "data/msmarco_passage_dev/raw/qrels.dev.tsv"

out_dir = root / "thesis_runs/shared/pair_definitions"

cross_encoder_out = out_dir / "cross_encoder_queries_full_eligible_top100.csv"
duot5_out = out_dir / "duot5_queries_150_seed42_top100.csv"
info_out = out_dir / "query_scope_info.json"

duot5_n_queries = 150
duot5_random_seed = 42
candidate_depth = 100


def load_queries() -> pd.DataFrame:
    loaded_queries = pd.read_csv(
        queries_file,
        sep="\t",
        header=None,
        names=["qid", "query"],
        dtype={0: str, 1: str})
    
    return loaded_queries


def load_candidate_query_ids() -> set[str]:
    loaded_candidates = set(
        pd.read_csv(
            candidates_file,
            sep="\t",
            header=None,
            usecols=[0],
            names=["qid"],
            dtype={0: str})["qid"])
    
    return loaded_candidates


def load_qrel_query_ids() -> set[str]:
    qrels = pd.read_csv(
        qrels_file,
        sep="\t",
        header=None,
        names=["qid", "_", "pid", "relevance"],
        dtype={0: str, 2: str})
    
    qrels_set = set(qrels["qid"])
    return qrels_set


def load_qrels_df() -> pd.DataFrame:
    loaded_qrels = pd.read_csv(
        qrels_file,
        sep="\t",
        header=None,
        names=["qid", "_", "pid", "relevance"],
        dtype={0: str, 2: str})
    
    return loaded_qrels


def build_eligible_query_pool() -> pd.DataFrame:
    queries = load_queries()
    candidate_qids = load_candidate_query_ids()
    qrel_qids = load_qrel_query_ids()

    eligible_qids = candidate_qids & qrel_qids
    eligible_queries = queries[queries["qid"].isin(eligible_qids)].copy().reset_index(drop=True)
    return eligible_queries


def build_top100_relevant_query_pool() -> pd.DataFrame:
    queries = load_queries()
    qrels = load_qrels_df()
    qrels = qrels[qrels["relevance"] > 0][["qid", "pid"]].drop_duplicates()

    candidates = pd.read_csv(
        candidates_file,
        sep="\t",
        header=None,
        names=["qid", "pid", "query", "passage"],
        dtype={0: str, 1: str})
    
    candidates["qid"] = candidates["qid"].astype(str)
    candidates["pid"] = candidates["pid"].astype(str)
    candidates = candidates.groupby("qid", sort=False).head(candidate_depth)[["qid", "pid"]]

    qids_with_relevant_in_top100 = set(candidates.merge(qrels, on=["qid", "pid"], how="inner")["qid"])
    final_top100 = queries[queries["qid"].isin(qids_with_relevant_in_top100)].copy().reset_index(drop=True)
    return final_top100


def save_query_scopes() -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    eligible_queries = build_eligible_query_pool()
    top100_relevant_queries = build_top100_relevant_query_pool()
    eligible_queries.to_csv(cross_encoder_out, index=False)

    duot5_queries = (top100_relevant_queries.sample(n=duot5_n_queries, random_state=duot5_random_seed)
                     .sort_values("qid")
                     .reset_index(drop=True))
    
    duot5_queries.to_csv(duot5_out, index=False)

    info = {
        "cross_encoder": {
            "query_scope": "full eligible query set",
            "n_queries": int(len(eligible_queries)),
            "candidate_depth": candidate_depth,
            "output_file": cross_encoder_out.name,
            "eligibility_rule": "query appears in candidate set and in qrels"},
        
        "duot5": {
            "query_scope": "fixed random subset of queries with at least one judged relevant passage in the BM25 top-100",
            "n_queries": duot5_n_queries,
            "candidate_depth": candidate_depth,
            "random_seed": duot5_random_seed,
            "output_file": duot5_out.name,
            "eligibility_rule": "sampled from queries whose BM25 top-100 candidate set contains at least one qrel-positive passage"},
        
        "shared_pool": {
            "n_eligible_queries": int(len(eligible_queries)),
            "n_queries_with_relevant_in_top100": int(len(top100_relevant_queries)),
            "queries_file": str(queries_file.relative_to(root)),
            "candidates_file": str(candidates_file.relative_to(root)),
            "qrels_file": str(qrels_file.relative_to(root))}}

    info_out.write_text(json.dumps(info, indent=2) + "\n")

    print(f"Saved cross-encoder query list to {cross_encoder_out}")
    print(f"Saved DuoT5 query list to {duot5_out}")
    print(f"Saved query scope info to {info_out}")
    print(f"Eligible query pool size: {len(eligible_queries)}")
    print(f"Queries with a judged relevant in BM25 top-{candidate_depth}: {len(top100_relevant_queries)}")


if __name__ == "__main__":
    save_query_scopes()