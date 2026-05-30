# -- IMPORTS --
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


root = Path(__file__).resolve().parents[1]

seed = 42
n_queries = 50

cross_source = root / "thesis_runs/cross_encoder/explanations/explanation_pairs_1_per_query_on_duot5_queries_seed42.csv"
duo_source = root / "thesis_runs/duot5/explanations/explanation_pairs_1_per_query_seed42.csv"

shared_dir = root / "thesis_runs/shared/pair_definitions"
cross_out_dir = root / "thesis_runs/cross_encoder/stability"
duo_out_dir = root / "thesis_runs/duot5/stability"


def main():
    rng = np.random.default_rng(seed)

    cross_df = pd.read_csv(cross_source, dtype={"qid": str, "pid_i": str, "pid_j": str})
    duo_df = pd.read_csv(duo_source, dtype={"qid": str, "pid_i": str, "pid_j": str})

    common_qids = sorted(set(cross_df["qid"]) & set(duo_df["qid"]))
    if len(common_qids) < n_queries:
        raise ValueError(f"Only found {len(common_qids)} shared queries, expected at least {n_queries}.")

    selected_qids = sorted(rng.choice(common_qids, size=n_queries, replace=False).tolist())

    shared_queries = (
        duo_df.loc[duo_df["qid"].isin(selected_qids), ["qid", "query"]]
        .drop_duplicates()
        .sort_values("qid")
        .reset_index(drop=True))
    
    cross_subset = cross_df[cross_df["qid"].isin(selected_qids)].copy().sort_values("qid").reset_index(drop=True)
    duo_subset = duo_df[duo_df["qid"].isin(selected_qids)].copy().sort_values("qid").reset_index(drop=True)

    shared_dir.mkdir(parents=True, exist_ok=True)
    cross_out_dir.mkdir(parents=True, exist_ok=True)
    duo_out_dir.mkdir(parents=True, exist_ok=True)

    shared_queries.to_csv(shared_dir / "stability_queries_50_seed42.csv", index=False)
    cross_subset.to_csv(cross_out_dir / "stability_pairs_50_queries_seed42.csv", index=False)
    duo_subset.to_csv(duo_out_dir / "stability_pairs_50_queries_seed42.csv", index=False)

    info = pd.DataFrame(
        [{
                "seed": seed,
                "n_queries": n_queries,
                "selection_rule": "shared random sample from aligned 150-query explanation subset",
                "cross_pairs": len(cross_subset),
                "duot5_pairs": len(duo_subset)}])
    info.to_csv(shared_dir / "stability_subset_info.csv", index=False)

    print(f"Saved shared query list to {shared_dir / 'stability_queries_50_seed42.csv'}")
    print(f"Saved cross-encoder stability pairs to {cross_out_dir / 'stability_pairs_50_queries_seed42.csv'}")
    print(f"Saved DuoT5 stability pairs to {duo_out_dir / 'stability_pairs_50_queries_seed42.csv'}")


if __name__ == "__main__":
    main()