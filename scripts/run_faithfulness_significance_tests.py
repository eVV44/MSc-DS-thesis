from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


ROOT = Path(__file__).resolve().parents[1]
K_BUDGETS = np.array([1, 3, 5, 10, 20], dtype=float)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Apply Holm correction to a list of p-values."""
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)
    running_max = 0.0

    for rank, idx in enumerate(order):
        value = (n - rank) * p_values[idx]
        running_max = max(running_max, value)
        adjusted[idx] = min(running_max, 1.0)

    return adjusted.tolist()


def per_query_auc(
    df: pd.DataFrame,
    *,
    method: str,
    check: str,
    value_col: str,
) -> pd.Series:
    subset = df[
        (df["condition"] == "explanation")
        & (df["trial"] == 0)
        & (df["method"] == method)
        & (df["check"] == check)
    ]
    pivot = (
        subset.pivot_table(index="qid", columns="k", values=value_col, aggfunc="first")
        .reindex(columns=K_BUDGETS)
        .sort_index()
    )
    return pd.Series(
        np.trapezoid(pivot.to_numpy(), x=K_BUDGETS, axis=1),
        index=pivot.index,
        name=method,
    )


def main() -> None:
    comparisons = [
        ("cross_encoder", "pairwise_ig", "loo_pairwise", "CE LOO vs IG (pairwise)"),
        ("duot5", "pairwise_ig", "loo_pairwise", "DuoT5 LOO vs IG (pairwise)"),
        ("duot5", "loo_pointwise", "loo_pairwise", "DuoT5 pairwise vs pointwise LOO"),
        ("duot5", "pointwise_ig", "pairwise_ig", "DuoT5 pairwise vs pointwise IG"),
    ]
    metrics = [
        ("deletion", "delta_g", r"Deletion $\Delta g$", True),
        ("preservation", "abs_g_gap", r"Preservation $|g-g'|$", False),
    ]

    rows: list[dict[str, object]] = []
    p_values: list[float] = []

    for experiment, method_a, method_b, label in comparisons:
        path = ROOT / "thesis_runs" / experiment / "faithfulness" / "faithfulness_results.csv"
        df = pd.read_csv(path)

        for check, value_col, metric_label, higher_is_better in metrics:
            auc_a = per_query_auc(df, method=method_a, check=check, value_col=value_col)
            auc_b = per_query_auc(df, method=method_b, check=check, value_col=value_col)

            common_qids = auc_a.index.intersection(auc_b.index)
            auc_a = auc_a.loc[common_qids]
            auc_b = auc_b.loc[common_qids]
            diff = auc_b - auc_a

            stat, p_value = wilcoxon(
                auc_a,
                auc_b,
                alternative="two-sided",
                zero_method="wilcox",
                method="approx",
            )

            better_method = method_b
            if higher_is_better:
                if auc_a.mean() > auc_b.mean():
                    better_method = method_a
            else:
                if auc_a.mean() < auc_b.mean():
                    better_method = method_a

            rows.append(
                {
                    "experiment": experiment,
                    "comparison": label,
                    "metric": metric_label,
                    "method_a": method_a,
                    "method_b": method_b,
                    "n_queries": len(common_qids),
                    "auc_mean_a": auc_a.mean(),
                    "auc_mean_b": auc_b.mean(),
                    "median_diff_b_minus_a": float(np.median(diff)),
                    "wilcoxon_stat": float(stat),
                    "p_value": float(p_value),
                    "higher_better": higher_is_better,
                    "better_method": better_method,
                }
            )
            p_values.append(float(p_value))

    adjusted = holm_adjust(p_values)
    for row, p_holm in zip(rows, adjusted):
        row["p_holm"] = p_holm

    out_df = pd.DataFrame(rows)
    out_path = ROOT / "thesis_runs" / "shared" / "faithfulness_significance_summary.csv"
    out_df.to_csv(out_path, index=False)

    print(out_df.to_string(index=False))
    print(f"\nSaved significance summary to {out_path}")


if __name__ == "__main__":
    main()
