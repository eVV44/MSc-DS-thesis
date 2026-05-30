# -- IMPORTS --
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd


root = Path(__file__).resolve().parents[1]

cross_input = root / "thesis_runs/cross_encoder/reranking/pairwise_scores.csv"
duot5_input = root / "thesis_runs/duot5/reranking/duot5_final/pairwise_scores.csv"
duot5_query_scope = root / "thesis_runs/shared/pair_definitions/duot5_queries_150_seed42_top100.csv"

cross_out_dir = root / "thesis_runs/cross_encoder/explanations"
duot5_out_dir = root / "thesis_runs/duot5/explanations"
shared_out_dir = root / "thesis_runs/shared/pair_definitions"

cross_out = cross_out_dir / "explanation_pairs_1_per_query_on_duot5_queries_seed42.csv"
duot5_out = duot5_out_dir / "explanation_pairs_1_per_query_seed42.csv"
info_out = shared_out_dir / "explanation_subset_info.json"

seed = 42


def load_pairs(path: Path) -> pd.DataFrame:
    loaded_pairs = pd.read_csv(path, dtype={"qid": str, "pid_i": str, "pid_j": str})
    return loaded_pairs


def load_duot5_query_scope() -> set[str]:
    df = pd.read_csv(duot5_query_scope, dtype={"qid": str})
    loaded_scope = set(df["qid"])
    return loaded_scope


def save_outputs(cross_df: pd.DataFrame, duot5_df: pd.DataFrame) -> None:
    cross_out_dir.mkdir(parents=True, exist_ok=True)
    duot5_out_dir.mkdir(parents=True, exist_ok=True)
    shared_out_dir.mkdir(parents=True, exist_ok=True)

    cross_df.to_csv(cross_out, index=False)
    duot5_df.to_csv(duot5_out, index=False)

    info = {
        "cross_encoder": {
            "input_file": str(cross_input.relative_to(root)),
            "output_file": str(cross_out.relative_to(root)),
            "selection_rule": "one randomly sampled pair per query, restricted to the frozen DuoT5 150-query scope",
            "random_seed": seed,
            "n_rows": int(len(cross_df)),
            "n_queries": int(cross_df["qid"].nunique()),},

        "duot5": {
            "input_file": str(duot5_input.relative_to(root)),
            "output_file": str(duot5_out.relative_to(root)),
            "selection_rule": "one randomly sampled pair per query from the frozen reranking pair table",
            "random_seed": seed,
            "n_rows": int(len(duot5_df)),
            "n_queries": int(duot5_df["qid"].nunique()),},}
    info_out.write_text(json.dumps(info, indent=2) + "\n")

    print(f"Saved cross-encoder explanation pairs to {cross_out}")
    print(f"Saved DuoT5 explanation pairs to {duot5_out}")
    print(f"Saved explanation subset info to {info_out}")
    print(f"Cross-encoder rows: {len(cross_df):,} across {cross_df['qid'].nunique():,} queries")
    print(f"DuoT5 rows: {len(duot5_df):,} across {duot5_df['qid'].nunique():,} queries")


def main() -> None:
    cross_df = load_pairs(cross_input)
    duot5_df = load_pairs(duot5_input)
    duot5_query_scope_set = load_duot5_query_scope()

    cross_df = cross_df[cross_df["qid"].isin(duot5_query_scope_set)].copy()

    cross_subset = (
        cross_df.groupby("qid", group_keys=False)
        .sample(n=1, random_state=seed)
        .sort_values(["qid", "pid_i", "pid_j"])
        .reset_index(drop=True))

    duot5_subset = (
        duot5_df.groupby("qid", group_keys=False)
        .sample(n=1, random_state=seed)
        .sort_values(["qid", "pid_i", "pid_j"])
        .reset_index(drop=True))

    save_outputs(cross_subset, duot5_subset)


if __name__ == "__main__":
    main()