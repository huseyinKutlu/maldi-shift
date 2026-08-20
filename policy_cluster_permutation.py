#!/usr/bin/env python3
"""
policy_cluster_permutation.py
=============================

Cluster-level permutation sensitivity test for the leakage-free selective policy.

Why
---
The 20 transfer tasks are not fully independent because multiple antibiotics may
share the same species and source->target cohort. This script uses
source->target|species as the permutation unit.

For each budget (20, 30, 50):
  - build the same leakage-free transfer-level policy table
  - compute observed nested LOTO policy statistics
  - permute Hybrid-Random delta PR-AUC at the CLUSTER level
  - preserve all tasks within a cluster together
  - rerun the full nested policy pipeline
  - derive empirical p-values for:
      Policy - Random
      Policy - Hybrid
      sign accuracy
      mean regret

Important
---------
With only 7 clusters, the null space is limited. This analysis is therefore a
conservative sensitivity analysis, not the primary inferential test.

Expected inputs
---------------
outputs/multitask_active_validation_final/
    raw_results.csv
    paired_hybrid_vs_random.csv

outputs/shift_predictors/
    transfer_predictors.csv

Outputs
-------
outputs/policy_cluster_permutation/
    observed_summary.csv
    cluster_inventory.csv
    permutation_raw.csv
    permutation_summary.csv
    run_config.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PREDICTORS = [
    "domain_auc_svd",
    "centroid_distance_svd",
    "covariance_distance_svd",
    "source_to_target_nn_distance",
    "source_internal_diversity",
]


def build_transfer_table(raw, paired, predictors, budget):
    keys = ["source", "target", "species", "drug"]

    z = raw[
        raw.strategy.isin(["random", "hybrid"])
        & (raw.budget_n == budget)
    ].copy()

    perf = (
        z.groupby(keys + ["strategy"])
        .agg(
            auroc=("auroc", "mean"),
            prauc=("prauc", "mean"),
            brier=("brier", "mean"),
        )
        .reset_index()
    )

    wide = perf.pivot_table(
        index=keys,
        columns="strategy",
        values=["auroc", "prauc", "brier"],
    )
    wide.columns = [f"{m}_{s}" for m, s in wide.columns]
    wide = wide.reset_index()

    hp = paired[
        (paired.metric == "prauc")
        & (paired.budget_n == budget)
    ][keys + ["delta_mean"]].rename(
        columns={"delta_mean": "delta_prauc_hybrid_random"}
    )

    out = wide.merge(hp, on=keys, how="inner")
    out = out.merge(predictors, on=keys, how="left")

    out["transfer_id"] = (
        out.source.astype(str)
        + "->"
        + out.target.astype(str)
        + "|"
        + out.species.astype(str)
        + "|"
        + out.drug.astype(str)
    )

    out["cluster_id"] = (
        out.source.astype(str)
        + "->"
        + out.target.astype(str)
        + "|"
        + out.species.astype(str)
    )

    return out.reset_index(drop=True)


def prepare_X(df):
    avail = [p for p in PREDICTORS if p in df.columns]
    X = df[avail].copy()

    for p in avail:
        X[p] = X[p].fillna(X[p].median())

    return X.to_numpy(dtype=float)


def choose_alpha(X, y, alphas):
    n = len(y)
    best_alpha = None
    best_rmse = np.inf

    for alpha in alphas:
        pred = np.zeros(n, dtype=float)

        for i in range(n):
            tr = np.arange(n) != i

            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ])

            pipe.fit(X[tr], y[tr])
            pred[i] = pipe.predict(X[[i]])[0]

        rmse = math.sqrt(mean_squared_error(y, pred))

        if rmse < best_rmse:
            best_rmse = rmse
            best_alpha = float(alpha)

    return best_alpha


def outer_logo_policy(df, y_learning, alphas):
    X = prepare_X(df)
    true_delta = df["delta_prauc_hybrid_random"].to_numpy(dtype=float)

    n = len(df)
    pred_delta = np.zeros(n, dtype=float)

    for i in range(n):
        tr = np.arange(n) != i

        alpha = choose_alpha(
            X[tr],
            y_learning[tr],
            alphas,
        )

        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ])

        pipe.fit(X[tr], y_learning[tr])
        pred_delta[i] = pipe.predict(X[[i]])[0]

    choose_hybrid = pred_delta > 0

    pr_random = df["prauc_random"].to_numpy(dtype=float)
    pr_hybrid = df["prauc_hybrid"].to_numpy(dtype=float)

    pr_policy = np.where(
        choose_hybrid,
        pr_hybrid,
        pr_random,
    )

    pr_oracle = np.maximum(
        pr_random,
        pr_hybrid,
    )

    regret = pr_oracle - pr_policy

    return dict(
        mean_policy_prauc=float(pr_policy.mean()),
        mean_gain_vs_random=float((pr_policy - pr_random).mean()),
        mean_gain_vs_hybrid=float((pr_policy - pr_hybrid).mean()),
        sign_accuracy=float(
            np.mean(
                choose_hybrid
                == (true_delta > 0)
            )
        ),
        mean_regret=float(regret.mean()),
        zero_regret_fraction=float(np.mean(regret == 0)),
    )


def cluster_permute_target(df, rng):
    """
    Permute target vectors at cluster level.

    Each cluster contributes its entire vector of task-level deltas.
    Since clusters differ in size, we map donor clusters to recipient clusters
    and resize donor vectors by cyclic repetition/truncation when necessary.
    This preserves within-cluster dependence while breaking the original
    predictor-outcome association.
    """
    clusters = list(df["cluster_id"].drop_duplicates())
    donor_order = list(rng.permutation(clusters))

    y_perm = np.empty(len(df), dtype=float)

    for recipient, donor in zip(clusters, donor_order):
        rec_idx = np.where(df["cluster_id"].to_numpy() == recipient)[0]
        donor_vals = df.loc[
            df["cluster_id"] == donor,
            "delta_prauc_hybrid_random"
        ].to_numpy(dtype=float)

        if len(donor_vals) == 0:
            raise RuntimeError("Empty donor cluster encountered.")

        vals = np.resize(donor_vals, len(rec_idx))
        y_perm[rec_idx] = vals

    return y_perm


def empirical_p_greater(null, obs):
    null = np.asarray(null, dtype=float)
    return float(
        (1 + np.sum(null >= obs))
        / (1 + len(null))
    )


def empirical_p_less(null, obs):
    null = np.asarray(null, dtype=float)
    return float(
        (1 + np.sum(null <= obs))
        / (1 + len(null))
    )


def summarize_null(values):
    values = np.asarray(values, dtype=float)

    return dict(
        null_mean=float(values.mean()),
        null_sd=float(values.std(ddof=1)),
        null_ci_low=float(np.percentile(values, 2.5)),
        null_ci_high=float(np.percentile(values, 97.5)),
    )


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ap.add_argument(
        "--multitask-dir",
        default="outputs/multitask_active_validation_final",
    )
    ap.add_argument(
        "--shift-dir",
        default="outputs/shift_predictors",
    )
    ap.add_argument(
        "--out",
        default="outputs/policy_cluster_permutation",
    )
    ap.add_argument(
        "--budgets",
        default="20,30,50",
    )
    ap.add_argument(
        "--permutations",
        type=int,
        default=5000,
    )
    ap.add_argument(
        "--alphas",
        default="0.01,0.1,1,10,100",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = ap.parse_args()

    md = Path(args.multitask_dir)
    sd = Path(args.shift_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(md / "raw_results.csv")
    paired = pd.read_csv(md / "paired_hybrid_vs_random.csv")
    predictors = pd.read_csv(sd / "transfer_predictors.csv")

    budgets = [
        int(x)
        for x in args.budgets.split(",")
    ]

    alphas = [
        float(x)
        for x in args.alphas.split(",")
    ]

    rng = np.random.default_rng(args.seed)

    observed_rows = []
    raw_rows = []
    summary_rows = []
    inventory_rows = []

    print("=== CLUSTER-LEVEL POLICY PERMUTATION ===")
    print(f"Budgets: {budgets}")
    print(f"Permutations: {args.permutations}")
    print("Cluster unit: source->target|species")

    for budget in budgets:
        print(f"\n--- Budget {budget} ---")

        df = build_transfer_table(
            raw,
            paired,
            predictors,
            budget,
        )

        inv = (
            df.groupby("cluster_id")
            .size()
            .reset_index(name="n_transfers")
        )
        inv["budget_n"] = budget
        inventory_rows.append(inv)

        true_y = df[
            "delta_prauc_hybrid_random"
        ].to_numpy(dtype=float)

        obs = outer_logo_policy(
            df,
            true_y,
            alphas,
        )

        observed_rows.append(
            dict(
                budget_n=budget,
                n_transfers=len(df),
                n_clusters=df.cluster_id.nunique(),
                **obs,
            )
        )

        print(
            f"Observed: "
            f"Policy-Random={obs['mean_gain_vs_random']:+.6f}, "
            f"sign_acc={obs['sign_accuracy']:.3f}, "
            f"regret={obs['mean_regret']:.6f}"
        )

        budget_rows = []

        for b in range(args.permutations):
            yp = cluster_permute_target(
                df,
                rng,
            )

            res = outer_logo_policy(
                df,
                yp,
                alphas,
            )

            row = dict(
                budget_n=budget,
                permutation=b,
                **res,
            )

            budget_rows.append(row)

            if (b + 1) % 250 == 0:
                print(
                    f"  permutation {b+1}/{args.permutations}",
                    flush=True,
                )

        bp = pd.DataFrame(budget_rows)
        raw_rows.extend(budget_rows)

        tests = [
            ("mean_gain_vs_random", "greater"),
            ("mean_gain_vs_hybrid", "greater"),
            ("sign_accuracy", "greater"),
            ("mean_regret", "less"),
            ("zero_regret_fraction", "greater"),
        ]

        for metric, direction in tests:
            null = bp[metric].to_numpy(dtype=float)
            observed = obs[metric]

            if direction == "greater":
                p = empirical_p_greater(
                    null,
                    observed,
                )
            else:
                p = empirical_p_less(
                    null,
                    observed,
                )

            summary_rows.append(
                dict(
                    budget_n=budget,
                    metric=metric,
                    observed=observed,
                    empirical_p=p,
                    direction=direction,
                    **summarize_null(null),
                )
            )

    observed_df = pd.DataFrame(observed_rows)
    raw_df = pd.DataFrame(raw_rows)
    summary_df = pd.DataFrame(summary_rows)
    inventory_df = pd.concat(
        inventory_rows,
        ignore_index=True,
    )

    observed_df.to_csv(
        out / "observed_summary.csv",
        index=False,
    )

    raw_df.to_csv(
        out / "permutation_raw.csv",
        index=False,
    )

    summary_df.to_csv(
        out / "permutation_summary.csv",
        index=False,
    )

    inventory_df.to_csv(
        out / "cluster_inventory.csv",
        index=False,
    )

    (out / "run_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "predictors": PREDICTORS,
                "cluster_unit": "source->target|species",
                "note": (
                    "Cluster-level permutation sensitivity; "
                    "limited to seven species-site clusters."
                ),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("\n=== OBSERVED SUMMARY ===")
    print(observed_df.to_string(index=False))

    print("\n=== CLUSTER PERMUTATION SUMMARY ===")
    print(
        summary_df[
            [
                "budget_n",
                "metric",
                "observed",
                "null_mean",
                "null_sd",
                "null_ci_low",
                "null_ci_high",
                "empirical_p",
            ]
        ].to_string(index=False)
    )

    print(f"\nOutputs: {out}")


if __name__ == "__main__":
    main()
