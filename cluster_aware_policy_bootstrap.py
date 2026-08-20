#!/usr/bin/env python3
"""
cluster_aware_policy_bootstrap.py
=================================

Cluster-aware robustness analysis for the leakage-free selective adaptation policy.

Why
---
The 20 transfer tasks are not fully independent because multiple antibiotics
can share the same species and source->target cohort.

Therefore ordinary transfer-level bootstrap may be optimistic.

Cluster definition
------------------
    source -> target | species

Examples:
    DRIAMS-A->DRIAMS-C | Escherichia coli
    DRIAMS-A->DRIAMS-B | Staphylococcus epidermidis

All antibiotic tasks inside the same cluster are resampled together.

Primary comparisons
-------------------
For budgets 20, 30, 50:
    selective_policy_v2 - always_random
    selective_policy_v2 - always_hybrid

Also reports:
    - mean PR-AUC difference
    - cluster-bootstrap 95% CI
    - two-sided bootstrap p-style tail probability
    - number of unique clusters
    - number of transfers

Expected inputs
---------------
outputs/selective_adaptation_policy_v2_b20/policy_predictions.csv
outputs/selective_adaptation_policy_v2/policy_predictions.csv
outputs/selective_adaptation_policy_v2_b50/policy_predictions.csv

Outputs
-------
outputs/cluster_aware_policy_bootstrap/
    cluster_aware_results.csv
    cluster_inventory.csv
    run_config.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def make_cluster_id(df):
    return (
        df["source"].astype(str)
        + "->"
        + df["target"].astype(str)
        + "|"
        + df["species"].astype(str)
    )


def cluster_bootstrap_mean_diff(
    df,
    diff_col,
    cluster_col="cluster_id",
    reps=20000,
    seed=0,
):
    """
    Resample clusters with replacement.
    If a cluster is drawn multiple times, all rows from that cluster are included
    multiple times in the bootstrap replicate.
    """
    clusters = np.array(sorted(df[cluster_col].dropna().unique()))
    if len(clusters) < 2:
        return dict(
            estimate=np.nan,
            ci_low=np.nan,
            ci_high=np.nan,
            p_two_sided=np.nan,
            n_clusters=len(clusters),
        )

    estimate = float(df[diff_col].mean())
    rng = np.random.default_rng(seed)
    vals = []

    grouped = {c: df[df[cluster_col] == c][diff_col].to_numpy(float) for c in clusters}

    for _ in range(reps):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)

        replicate_parts = []
        for c in sampled:
            replicate_parts.append(grouped[c])

        x = np.concatenate(replicate_parts)
        vals.append(float(np.mean(x)))

    vals = np.asarray(vals, dtype=float)
    lo, hi = np.percentile(vals, [2.5, 97.5])

    # two-sided bootstrap tail probability around zero
    p_left = np.mean(vals <= 0)
    p_right = np.mean(vals >= 0)
    p_two = min(1.0, 2 * min(p_left, p_right))

    return dict(
        estimate=estimate,
        ci_low=float(lo),
        ci_high=float(hi),
        p_two_sided=float(p_two),
        n_clusters=int(len(clusters)),
    )


def summarize_budget(path, budget, reps, seed):
    df = pd.read_csv(path)
    df["cluster_id"] = make_cluster_id(df)

    df["diff_policy_random"] = df["prauc_policy"] - df["prauc_random"]
    df["diff_policy_hybrid"] = df["prauc_policy"] - df["prauc_hybrid"]

    rows = []

    for comparison, col in [
        ("selective_policy_v2 - always_random", "diff_policy_random"),
        ("selective_policy_v2 - always_hybrid", "diff_policy_hybrid"),
    ]:
        res = cluster_bootstrap_mean_diff(
            df,
            diff_col=col,
            cluster_col="cluster_id",
            reps=reps,
            seed=seed,
        )

        rows.append(dict(
            budget_n=budget,
            comparison=comparison,
            n_transfers=len(df),
            mean_delta=float(df[col].mean()),
            median_delta=float(df[col].median()),
            fraction_positive=float(np.mean(df[col] > 0)),
            fraction_zero=float(np.mean(df[col] == 0)),
            **res,
        ))

    return df, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ap.add_argument(
        "--b20",
        default="outputs/selective_adaptation_policy_v2_b20/policy_predictions.csv",
    )
    ap.add_argument(
        "--b30",
        default="outputs/selective_adaptation_policy_v2/policy_predictions.csv",
    )
    ap.add_argument(
        "--b50",
        default="outputs/selective_adaptation_policy_v2_b50/policy_predictions.csv",
    )
    ap.add_argument(
        "--out",
        default="outputs/cluster_aware_policy_bootstrap",
    )
    ap.add_argument(
        "--bootstrap-reps",
        type=int,
        default=20000,
    )

    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_rows = []
    inventory_rows = []

    for budget, path, seed in [
        (20, args.b20, 20),
        (30, args.b30, 30),
        (50, args.b50, 50),
    ]:
        df, res = summarize_budget(
            path=path,
            budget=budget,
            reps=args.bootstrap_reps,
            seed=seed,
        )

        all_rows.append(res)

        inv = (
            df.groupby("cluster_id")
            .agg(
                n_transfers=("transfer_id", "size"),
                source=("source", "first"),
                target=("target", "first"),
                species=("species", "first"),
            )
            .reset_index()
        )
        inv["budget_n"] = budget
        inventory_rows.append(inv)

    results = pd.concat(all_rows, ignore_index=True)
    inventory = pd.concat(inventory_rows, ignore_index=True)

    results.to_csv(
        out / "cluster_aware_results.csv",
        index=False,
    )

    inventory.to_csv(
        out / "cluster_inventory.csv",
        index=False,
    )

    config = vars(args).copy()
    config["cluster_definition"] = "source->target|species"

    (out / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=== CLUSTER-AWARE POLICY BOOTSTRAP ===")
    print("\nCluster definition: source->target | species")

    print("\n=== CLUSTER INVENTORY ===")
    inv_show = (
        inventory[inventory.budget_n == 30]
        .sort_values("cluster_id")
    )
    print(
        inv_show[
            [
                "cluster_id",
                "n_transfers",
            ]
        ].to_string(index=False)
    )

    print("\n=== CLUSTER-AWARE RESULTS ===")
    print(
        results[
            [
                "budget_n",
                "comparison",
                "n_transfers",
                "n_clusters",
                "mean_delta",
                "median_delta",
                "ci_low",
                "ci_high",
                "p_two_sided",
                "fraction_positive",
                "fraction_zero",
            ]
        ].to_string(index=False)
    )

    print(f"\nOutputs: {out}")


if __name__ == "__main__":
    main()
