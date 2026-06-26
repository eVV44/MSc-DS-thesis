# -- IMPORTS --
from __future__ import annotations
import argparse
import itertools
import os
import pickle
import random
from pathlib import Path

root = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(root / "thesis_runs/shared/mplconfig"))

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TOP_KS = [1, 3, 5, 10, 20]
RANDOM_SEED = 42
RANDOM_SAMPLES = 200

EXPERIMENTS = {
    "cross_encoder": {
        "label": "Cross-encoder",
        "ig_file": ROOT / "thesis_runs/cross_encoder/explanations/attributions_ig.pkl",
        "loo_file": ROOT / "thesis_runs/cross_encoder/explanations/attributions_loo.pkl",
        "out_dir": ROOT / "thesis_runs/cross_encoder/faithfulness",
        "methods": ["pairwise_ig", "pointwise_ig", "loo_pairwise", "loo_pointwise"],
    },
    "duot5": {
        "label": "DuoT5",
        "pairwise_ig_file": ROOT / "thesis_runs/duot5/explanations/attributions_pairwise_ig.pkl",
        "pointwise_ig_file": ROOT / "thesis_runs/duot5/explanations/attributions_pointwise_ig.pkl",
        "loo_file": ROOT / "thesis_runs/duot5/explanations/attributions_loo.pkl",
        "out_dir": ROOT / "thesis_runs/duot5/faithfulness",
        "methods": ["pairwise_ig", "pointwise_ig", "loo_pairwise", "loo_pointwise"],
    },
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["cross_encoder", "duot5", "all"], default="all")
    return parser.parse_args()


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def pair_key(qid, pid_i, pid_j):
    return str(qid), str(pid_i), str(pid_j)


def spearman_similarity(x, y):
    x = pd.Series(x, dtype=float)
    y = pd.Series(y, dtype=float)
    valid = ~(x.isna() | y.isna())
    x = x[valid]
    y = y[valid]
    if len(x) < 2:
        return np.nan
    xr = x.rank(method="average")
    yr = y.rank(method="average")
    if xr.nunique() < 2 or yr.nunique() < 2:
        return np.nan
    return float(xr.corr(yr, method="pearson"))


def build_cross_lookup(cfg):
    ig_records = load_pickle(cfg["ig_file"])
    loo_records = load_pickle(cfg["loo_file"])
    ig_lookup = {pair_key(r["qid"], r["pid_i"], r["pid_j"]): r for r in ig_records}
    loo_lookup = {pair_key(r["qid"], r["pid_i"], r["pid_j"]): r for r in loo_records}
    keys = sorted(set(ig_lookup) & set(loo_lookup))
    return keys, ig_lookup, loo_lookup


def build_duot5_lookup(cfg):
    pw_records = load_pickle(cfg["pairwise_ig_file"])
    pt_records = load_pickle(cfg["pointwise_ig_file"])
    loo_records = load_pickle(cfg["loo_file"])
    pw_lookup = {pair_key(r["qid"], r["pid_i"], r["pid_j"]): r for r in pw_records}
    pt_lookup = {pair_key(r["qid"], r["pid_i"], r["pid_j"]): r for r in pt_records}
    loo_lookup = {pair_key(r["qid"], r["pid_i"], r["pid_j"]): r for r in loo_records}
    keys = sorted(set(pw_lookup) & set(pt_lookup) & set(loo_lookup))
    return keys, pw_lookup, pt_lookup, loo_lookup


def rows_from_cross(key, method, ig_lookup, loo_lookup):
    rows = []
    if method == "pairwise_ig":
        record = ig_lookup[key]
        for side, attr in [("doc_i", record["pairwise_ig"]["doc_i"]), ("doc_j", record["pairwise_ig"]["doc_j"])]:
            for idx, (tok, score) in enumerate(zip(attr["doc_word_tokens"], attr["doc_word_scores"])):
                rows.append({"side": side, "position": idx, "token": tok, "support_score": float(score)})
    elif method == "pointwise_ig":
        record = ig_lookup[key]
        for side, attr, sign in [
            ("doc_i", record["pointwise_ig_i"], 1.0),
            ("doc_j", record["pointwise_ig_j"], -1.0),
        ]:
            for idx, (tok, score) in enumerate(zip(attr["doc_word_tokens"], attr["doc_word_scores"])):
                rows.append({"side": side, "position": idx, "token": tok, "support_score": float(sign * score)})
    elif method == "loo_pairwise":
        record = loo_lookup[key]
        for side, entries in [("doc_i", record["loo_pairwise"]["doc_i"]), ("doc_j", record["loo_pairwise"]["doc_j"])]:
            for entry in entries:
                rows.append({"side": side, "position": int(entry["position"]), "token": entry["word"], "support_score": float(entry["support_score"])})
    elif method == "loo_pointwise":
        record = loo_lookup[key]
        for side, attr_key, sign in [("doc_i", "loo_pointwise_i", 1.0), ("doc_j", "loo_pointwise_j", -1.0)]:
            for entry in record[attr_key]["doc"]:
                rows.append({"side": side, "position": int(entry["position"]), "token": entry["word"], "support_score": float(sign * entry["support_score"])})
    else:
        raise ValueError(method)
    return pd.DataFrame(rows)


def rows_from_duot5(key, method, pw_lookup, pt_lookup, loo_lookup):
    rows = []
    if method == "pairwise_ig":
        record = pw_lookup[key]
        for side, attr in [("doc_i", record["pairwise_ig"]["doc0"]), ("doc_j", record["pairwise_ig"]["doc1"])]:
            for idx, (tok, score) in enumerate(zip(attr["word_tokens"], attr["word_scores"])):
                rows.append({"side": side, "position": idx, "token": tok, "support_score": float(score)})
    elif method == "pointwise_ig":
        record = pt_lookup[key]
        for side, attr, sign in [("doc_i", record["pointwise_ig_i"]["doc0"], 1.0), ("doc_j", record["pointwise_ig_j"]["doc0"], -1.0)]:
            for idx, (tok, score) in enumerate(zip(attr["word_tokens"], attr["word_scores"])):
                rows.append({"side": side, "position": idx, "token": tok, "support_score": float(sign * score)})
    elif method == "loo_pairwise":
        record = loo_lookup[key]
        for side, entries in [("doc_i", record["loo_pairwise"]["doc_i"]), ("doc_j", record["loo_pairwise"]["doc_j"])]:
            for entry in entries:
                rows.append({"side": side, "position": int(entry["position"]), "token": entry["word"], "support_score": float(entry["support_score"])})
    elif method == "loo_pointwise":
        record = loo_lookup[key]
        for side, attr_key, sign in [("doc_i", "loo_pointwise_i", 1.0), ("doc_j", "loo_pointwise_j", -1.0)]:
            for entry in record[attr_key]["doc"]:
                rows.append({"side": side, "position": int(entry["position"]), "token": entry["word"], "support_score": float(sign * entry["support_score"])})
    else:
        raise ValueError(method)
    return pd.DataFrame(rows)


def top_k_tokens(table: pd.DataFrame, k: int):
    if table.empty:
        return set()
    subset = table.sort_values("support_score", ascending=False).head(k)
    return {(str(r["side"]), int(r["position"])) for _, r in subset.iterrows()}


def overlap_at_k(table_a: pd.DataFrame, table_b: pd.DataFrame, k: int):
    top_a = top_k_tokens(table_a, k)
    top_b = top_k_tokens(table_b, k)
    if not top_a or not top_b:
        return np.nan
    denom = min(k, len(top_a), len(top_b))
    if denom == 0:
        return np.nan
    return float(len(top_a & top_b) / denom)


def sampled_random_overlap_at_k(
    table: pd.DataFrame,
    k: int,
    rng: random.Random,
    n_samples: int = RANDOM_SAMPLES,
):
    if table.empty:
        return np.nan

    coords = [
        (str(r["side"]), int(r["position"]))
        for _, r in table[["side", "position"]].drop_duplicates().iterrows()
    ]
    if not coords:
        return np.nan

    kk = min(k, len(coords))
    top = top_k_tokens(table, kk)
    if not top:
        return np.nan

    overlaps = []
    for _ in range(n_samples):
        sampled = set(rng.sample(coords, kk))
        overlaps.append(len(set(top) & sampled) / kk)
    return float(np.mean(overlaps))


def full_spearman(table_a: pd.DataFrame, table_b: pd.DataFrame):
    if table_a.empty or table_b.empty:
        return np.nan, 0
    a = table_a[["side", "position", "support_score"]].rename(columns={"support_score": "a"})
    b = table_b[["side", "position", "support_score"]].rename(columns={"support_score": "b"})
    merged = a.merge(b, on=["side", "position"], how="inner")
    if merged.empty:
        return np.nan, 0
    return spearman_similarity(merged["a"].to_numpy(), merged["b"].to_numpy()), int(len(merged))


def run_one(experiment_key: str):
    cfg = EXPERIMENTS[experiment_key]
    if experiment_key == "cross_encoder":
        keys, ig_lookup, loo_lookup = build_cross_lookup(cfg)
        getter = lambda key, method: rows_from_cross(key, method, ig_lookup, loo_lookup)
    else:
        keys, pw_lookup, pt_lookup, loo_lookup = build_duot5_lookup(cfg)
        getter = lambda key, method: rows_from_duot5(key, method, pw_lookup, pt_lookup, loo_lookup)

    detail_rows = []
    pair_comparisons = list(itertools.combinations(cfg["methods"], 2))
    for key in keys:
        tables = {method: getter(key, method) for method in cfg["methods"]}
        for method_a, method_b in pair_comparisons:
            table_a = tables[method_a]
            table_b = tables[method_b]
            rho, n_aligned = full_spearman(table_a, table_b)
            row = {
                "experiment": experiment_key,
                "model_label": cfg["label"],
                "qid": key[0],
                "pid_i": key[1],
                "pid_j": key[2],
                "method_a": method_a,
                "method_b": method_b,
                "spearman_full": rho,
                "n_aligned": n_aligned,
            }
            for k in TOP_KS:
                row[f"top{k}_overlap"] = overlap_at_k(table_a, table_b, k)
            detail_rows.append(row)

        for method in cfg["methods"]:
            table = tables[method]
            random_rng = random.Random(
                f"{RANDOM_SEED}:{experiment_key}:{key[0]}:{key[1]}:{key[2]}:{method}"
            )
            row = {
                "experiment": experiment_key,
                "model_label": cfg["label"],
                "qid": key[0],
                "pid_i": key[1],
                "pid_j": key[2],
                "method_a": method,
                "method_b": "random",
                "spearman_full": np.nan,
                "n_aligned": int(len(table[["side", "position"]].drop_duplicates())),
            }
            for k in TOP_KS:
                row[f"top{k}_overlap"] = sampled_random_overlap_at_k(table, k, random_rng)
            detail_rows.append(row)

    detail_df = pd.DataFrame(detail_rows)
    summary_rows = []
    for (experiment, model_label, method_a, method_b), group in detail_df.groupby(
        ["experiment", "model_label", "method_a", "method_b"], dropna=False
    ):
        summary = {
            "experiment": experiment,
            "model_label": model_label,
            "method_a": method_a,
            "method_b": method_b,
            "n_pairs": int(len(group)),
            "mean_spearman_full": float(group["spearman_full"].dropna().mean()) if group["spearman_full"].notna().any() else np.nan,
            "mean_aligned": float(group["n_aligned"].mean()),
        }
        for k in TOP_KS:
            col = f"top{k}_overlap"
            summary[f"mean_{col}"] = float(group[col].dropna().mean()) if group[col].notna().any() else np.nan
        summary_rows.append(summary)

    summary_df = pd.DataFrame(summary_rows).sort_values(["method_a", "method_b"]).reset_index(drop=True)

    out_dir = cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_out = out_dir / "explanation_agreement_results.csv"
    summary_out = out_dir / "explanation_agreement_summary.csv"
    detail_df.to_csv(detail_out, index=False)
    summary_df.to_csv(summary_out, index=False)
    print(f"Saved explanation agreement results to {detail_out}")
    print(f"Saved explanation agreement summary to {summary_out}")


def main():
    args = parse_args()
    if args.experiment == "all":
        for experiment_key in ["cross_encoder", "duot5"]:
            run_one(experiment_key)
    else:
        run_one(args.experiment)


if __name__ == "__main__":
    main()
