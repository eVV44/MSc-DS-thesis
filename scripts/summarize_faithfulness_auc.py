# -- IMPORTS --
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = {
    "cross_encoder": {
        "label": "Cross-encoder",
        "summary_file": ROOT / "thesis_runs/cross_encoder/faithfulness/faithfulness_summary.csv",
        "out_file": ROOT / "thesis_runs/cross_encoder/faithfulness/faithfulness_auc_summary.csv",
    },
    "duot5": {
        "label": "DuoT5",
        "summary_file": ROOT / "thesis_runs/duot5/faithfulness/faithfulness_summary.csv",
        "out_file": ROOT / "thesis_runs/duot5/faithfulness/faithfulness_auc_summary.csv",
    },
}

CURVES = [
    {
        "curve_name": "deletion_delta_g",
        "check": "deletion",
        "metric": "mean_delta_g",
        "higher_is_better": True,
    },
    {
        "curve_name": "deletion_flip_rate",
        "check": "deletion",
        "metric": "flip_rate",
        "higher_is_better": True,
    },
    {
        "curve_name": "preservation_sign_preservation",
        "check": "preservation",
        "metric": "sign_preservation_rate",
        "higher_is_better": True,
    },
    {
        "curve_name": "preservation_abs_gap",
        "check": "preservation",
        "metric": "mean_abs_g_gap",
        "higher_is_better": False,
    },
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["cross_encoder", "duot5", "all"], default="all")
    return parser.parse_args()


def trapezoid_auc(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2:
        return np.nan, np.nan
    auc = float(np.trapz(y, x))
    width = float(x.max() - x.min())
    normalized_auc = float(auc / width) if width > 0 else np.nan
    return auc, normalized_auc


def summarize_one(experiment_key: str):
    cfg = EXPERIMENTS[experiment_key]
    df = pd.read_csv(cfg["summary_file"])

    rows = []
    for (experiment, model_label, method, condition), group in df.groupby(
        ["experiment", "model_label", "method", "condition"], dropna=False
    ):
        for curve in CURVES:
            subset = group[group["check"] == curve["check"]].sort_values("k")
            if subset.empty:
                continue
            auc, normalized_auc = trapezoid_auc(subset["k"], subset[curve["metric"]])
            rows.append(
                {
                    "experiment": experiment,
                    "model_label": model_label,
                    "method": method,
                    "condition": condition,
                    "curve_name": curve["curve_name"],
                    "check": curve["check"],
                    "metric": curve["metric"],
                    "higher_is_better": curve["higher_is_better"],
                    "min_k": int(subset["k"].min()),
                    "max_k": int(subset["k"].max()),
                    "n_points": int(len(subset)),
                    "auc": auc,
                    "normalized_auc": normalized_auc,
                }
            )

    out_df = pd.DataFrame(rows).sort_values(
        ["condition", "curve_name", "method"]
    ).reset_index(drop=True)
    cfg["out_file"].parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(cfg["out_file"], index=False)
    print(f"Saved AUC summary to {cfg['out_file']}")
    return out_df


def main():
    args = parse_args()
    if args.experiment == "all":
        for experiment_key in ["cross_encoder", "duot5"]:
            summarize_one(experiment_key)
    else:
        summarize_one(args.experiment)


if __name__ == "__main__":
    main()