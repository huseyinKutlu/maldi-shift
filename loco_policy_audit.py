#!/usr/bin/env python3
"""
loco_policy_audit.py
====================

Diagnostic audit for leave-one-cluster-out (LOCO) selective policy results.

Purpose
-------
This script investigates WHY the LOCO policy succeeds or fails for each held-out
source->target|species cluster.

It does NOT retrain MALDI classifiers or modify the policy. It audits the already
generated LOCO predictions and asks whether poor policy performance is associated
with descriptor-space extrapolation.

For each held-out cluster it reports:
    - observed mean ΔPR-AUC (Hybrid - Random)
    - predicted mean ΔPR-AUC
    - policy choices and direction accuracy
    - mean regret
    - held-out cluster mean for each shift descriptor
    - training-cluster min / max
    - held-out percentile relative to training-cluster means
    - standardized z-score relative to training clusters
    - number of descriptors outside the training-cluster range
    - maximum absolute z-score
    - Euclidean distance in standardized descriptor space to nearest training cluster
    - mean distance to all training clusters

Expected inputs
---------------
outputs/leave_one_cluster_out_policy/
    policy_predictions.csv
    alpha_by_outer_cluster.csv

outputs/shift_predictors/
    transfer_predictors.csv

Example
-------
python loco_policy_audit.py \
  --loco-dir outputs/leave_one_cluster_out_policy \
  --shift-dir outputs/shift_predictors \
  --out outputs/loco_policy_audit

Outputs
-------
cluster_audit.csv
descriptor_audit_long.csv
task_audit.csv
audit_summary.txt
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


PREDICTORS = [
    "domain_auc_svd",
    "centroid_distance_svd",
    "covariance_distance_svd",
    "source_to_target_nn_distance",
    "source_internal_diversity",
]


def make_cluster_id(df):
    return (
        df["source"].astype(str)
        + "->"
        + df["target"].astype(str)
        + "|"
        + df["species"].astype(str)
    )


def percentile_against_training(value, train_values):
    train_values = np.asarray(train_values, dtype=float)
    train_values = train_values[np.isfinite(train_values)]
    if len(train_values) == 0 or not np.isfinite(value):
        return np.nan
    # mid-rank empirical percentile
    less = np.sum(train_values < value)
    equal = np.sum(train_values == value)
    return 100.0 * (less + 0.5 * equal) / len(train_values)


def safe_z(value, train_values):
    train_values = np.asarray(train_values, dtype=float)
    train_values = train_values[np.isfinite(train_values)]
    if len(train_values) < 2 or not np.isfinite(value):
        return np.nan
    mu = np.mean(train_values)
    sd = np.std(train_values, ddof=1)
    if sd <= 1e-12:
        return 0.0 if abs(value - mu) <= 1e-12 else np.sign(value - mu) * np.inf
    return (value - mu) / sd


def standardized_distances(held_vec, train_matrix):
    """
    Standardize using training-cluster mean and SD, then calculate Euclidean distance
    between held-out cluster and every training cluster.
    """
    X = np.asarray(train_matrix, dtype=float)
    h = np.asarray(held_vec, dtype=float)

    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=1)
    sd[~np.isfinite(sd) | (sd <= 1e-12)] = 1.0

    # training-cluster matrix is expected complete; guard anyway
    X_imp = np.where(np.isfinite(X), X, mu)
    h_imp = np.where(np.isfinite(h), h, mu)

    Xz = (X_imp - mu) / sd
    hz = (h_imp - mu) / sd
    d = np.sqrt(np.sum((Xz - hz) ** 2, axis=1))
    return d


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument(
        "--loco-dir",
        default="outputs/leave_one_cluster_out_policy",
    )
    ap.add_argument(
        "--shift-dir",
        default="outputs/shift_predictors",
    )
    ap.add_argument(
        "--out",
        default="outputs/loco_policy_audit",
    )
    args = ap.parse_args()

    loco_dir = Path(args.loco_dir)
    shift_dir = Path(args.shift_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pred_path = loco_dir / "policy_predictions.csv"
    alpha_path = loco_dir / "alpha_by_outer_cluster.csv"
    shift_path = shift_dir / "transfer_predictors.csv"

    for p in [pred_path, shift_path]:
        if not p.exists():
            raise SystemExit(f"Missing required input: {p}")

    pred = pd.read_csv(pred_path)
    shift = pd.read_csv(shift_path)

    required_pred = [
        "budget_n", "cluster_id", "transfer_id",
        "predicted_delta_prauc_hybrid_random",
        "observed_delta_prauc_hybrid_random",
        "policy_choice", "oracle_choice",
        "prauc_random", "prauc_hybrid", "prauc_policy",
        "prauc_oracle", "policy_correct_direction", "regret_prauc"
    ]
    missing = [c for c in required_pred if c not in pred.columns]
    if missing:
        raise SystemExit(f"policy_predictions.csv missing columns: {missing}")

    if "cluster_id" not in shift.columns:
        shift["cluster_id"] = make_cluster_id(shift)

    # keep only predictors actually available
    predictors = [p for p in PREDICTORS if p in shift.columns]
    if len(predictors) != len(PREDICTORS):
        missing_p = sorted(set(PREDICTORS) - set(predictors))
        print("WARNING: missing predictor columns:", missing_p)

    keys = ["source", "target", "species", "drug"]
    keep = keys + ["cluster_id"] + predictors
    shift_small = shift[keep].drop_duplicates(keys).copy()

    # attach descriptors to task-level LOCO predictions
    task = pred.merge(
        shift_small,
        on=["source", "target", "species", "drug", "cluster_id"],
        how="left",
        validate="many_to_one",
    )
    task.to_csv(out / "task_audit.csv", index=False)

    # cluster-level descriptor means using the full predictor table.
    # This is label-free and independent of budget.
    cluster_desc = (
        shift_small.groupby("cluster_id", as_index=False)[predictors]
        .mean()
    )

    cluster_rows = []
    long_rows = []

    for budget in sorted(task["budget_n"].unique()):
        pb = task[task["budget_n"] == budget].copy()

        for held_cluster in sorted(pb["cluster_id"].unique()):
            hg = pb[pb["cluster_id"] == held_cluster].copy()

            # held-out cluster descriptor vector
            held_row = cluster_desc[cluster_desc.cluster_id == held_cluster]
            if held_row.empty:
                print(f"WARNING: no descriptors for {held_cluster}")
                continue

            train_desc = cluster_desc[cluster_desc.cluster_id != held_cluster].copy()

            held_vec = held_row[predictors].iloc[0].to_numpy(float)
            train_matrix = train_desc[predictors].to_numpy(float)
            dists = standardized_distances(held_vec, train_matrix)

            descriptor_outside_count = 0
            abs_zs = []
            percentiles = []

            descriptor_stats = {}

            for p in predictors:
                value = float(held_row[p].iloc[0])
                vals = train_desc[p].to_numpy(float)

                tmin = float(np.nanmin(vals))
                tmax = float(np.nanmax(vals))
                tmean = float(np.nanmean(vals))
                tsd = float(np.nanstd(vals, ddof=1))
                pct = percentile_against_training(value, vals)
                z = safe_z(value, vals)
                outside = bool(value < tmin or value > tmax)

                descriptor_outside_count += int(outside)
                if np.isfinite(z):
                    abs_zs.append(abs(z))
                if np.isfinite(pct):
                    percentiles.append(pct)

                descriptor_stats[p] = value

                long_rows.append({
                    "budget_n": int(budget),
                    "held_cluster": held_cluster,
                    "predictor": p,
                    "held_value": value,
                    "train_cluster_min": tmin,
                    "train_cluster_max": tmax,
                    "train_cluster_mean": tmean,
                    "train_cluster_sd": tsd,
                    "held_percentile": pct,
                    "z_vs_train_clusters": z,
                    "outside_training_range": outside,
                })

            observed = hg["observed_delta_prauc_hybrid_random"].to_numpy(float)
            predicted = hg["predicted_delta_prauc_hybrid_random"].to_numpy(float)

            choices = hg["policy_choice"].value_counts().to_dict()
            oracle_choices = hg["oracle_choice"].value_counts().to_dict()

            row = {
                "budget_n": int(budget),
                "held_cluster": held_cluster,
                "n_tasks": len(hg),
                "observed_mean_delta_prauc": float(np.mean(observed)),
                "predicted_mean_delta_prauc": float(np.mean(predicted)),
                "prediction_error_mean": float(np.mean(predicted - observed)),
                "direction_accuracy": float(hg["policy_correct_direction"].mean()),
                "mean_gain_vs_random": float((hg["prauc_policy"] - hg["prauc_random"]).mean()),
                "mean_gain_vs_hybrid": float((hg["prauc_policy"] - hg["prauc_hybrid"]).mean()),
                "mean_regret": float(hg["regret_prauc"].mean()),
                "policy_hybrid_fraction": float((hg["policy_choice"] == "hybrid").mean()),
                "oracle_hybrid_fraction": float((hg["oracle_choice"] == "hybrid").mean()),
                "n_descriptors_outside_training_range": int(descriptor_outside_count),
                "max_abs_z_vs_train_clusters": float(max(abs_zs)) if abs_zs else np.nan,
                "min_descriptor_percentile": float(min(percentiles)) if percentiles else np.nan,
                "max_descriptor_percentile": float(max(percentiles)) if percentiles else np.nan,
                "nearest_training_cluster_distance_z": float(np.min(dists)),
                "mean_training_cluster_distance_z": float(np.mean(dists)),
                "nearest_training_cluster": str(
                    train_desc.iloc[int(np.argmin(dists))]["cluster_id"]
                ),
            }
            row.update(descriptor_stats)
            cluster_rows.append(row)

    cluster_audit = pd.DataFrame(cluster_rows)
    long_audit = pd.DataFrame(long_rows)

    if alpha_path.exists():
        alpha_tab = pd.read_csv(alpha_path).rename(
            columns={"outer_held_cluster": "held_cluster"}
        )
        cluster_audit = cluster_audit.merge(
            alpha_tab[
                ["budget_n", "held_cluster", "selected_alpha", "n_train_clusters"]
            ],
            on=["budget_n", "held_cluster"],
            how="left",
        )

    # useful rankings
    cluster_audit["failure_rank_by_regret"] = (
        cluster_audit.groupby("budget_n")["mean_regret"]
        .rank(method="min", ascending=False)
    )
    cluster_audit["ood_rank_by_nearest_distance"] = (
        cluster_audit.groupby("budget_n")["nearest_training_cluster_distance_z"]
        .rank(method="min", ascending=False)
    )

    cluster_audit = cluster_audit.sort_values(
        ["budget_n", "mean_regret"],
        ascending=[True, False]
    )
    cluster_audit.to_csv(out / "cluster_audit.csv", index=False)
    long_audit.to_csv(out / "descriptor_audit_long.csv", index=False)

    # Summaries to console/text
    lines = []
    lines.append("LOCO POLICY AUDIT")
    lines.append("=" * 72)
    lines.append(
        "Goal: determine whether poor held-out-cluster policy performance "
        "coincides with descriptor-space extrapolation."
    )
    lines.append("")

    for budget in sorted(cluster_audit["budget_n"].unique()):
        g = cluster_audit[cluster_audit.budget_n == budget].copy()

        worst = g.sort_values("mean_regret", ascending=False).iloc[0]
        most_ood = g.sort_values(
            "nearest_training_cluster_distance_z", ascending=False
        ).iloc[0]

        lines.append(f"BUDGET {int(budget)}")
        lines.append("-" * 72)
        lines.append(
            f"Worst-regret cluster: {worst.held_cluster} | "
            f"regret={worst.mean_regret:.6f} | "
            f"gain_vs_random={worst.mean_gain_vs_random:+.6f} | "
            f"direction_acc={worst.direction_accuracy:.3f}"
        )
        lines.append(
            f"Most descriptor-OOD cluster: {most_ood.held_cluster} | "
            f"nearest standardized cluster distance="
            f"{most_ood.nearest_training_cluster_distance_z:.3f} | "
            f"outside-range descriptors="
            f"{int(most_ood.n_descriptors_outside_training_range)}"
        )

        # rank association: does OOD distance track regret?
        if len(g) >= 4:
            rho = g[
                ["nearest_training_cluster_distance_z", "mean_regret"]
            ].corr(method="spearman").iloc[0, 1]
            lines.append(
                f"Spearman(OOD nearest-distance, regret) across clusters = {rho:+.3f}"
            )

        lines.append("")

    # Specific focus: Staphylococcus epidermidis A->B if present
    focus_mask = cluster_audit["held_cluster"].str.contains(
        r"DRIAMS-A->DRIAMS-B\|Staphylococcus epidermidis",
        regex=True,
        na=False,
    )
    focus = cluster_audit[focus_mask]
    if not focus.empty:
        lines.append("FOCUS: DRIAMS-A->DRIAMS-B|Staphylococcus epidermidis")
        lines.append("-" * 72)
        for _, r in focus.sort_values("budget_n").iterrows():
            lines.append(
                f"Budget {int(r.budget_n)} | "
                f"obsΔ={r.observed_mean_delta_prauc:+.6f} | "
                f"predΔ={r.predicted_mean_delta_prauc:+.6f} | "
                f"acc={r.direction_accuracy:.3f} | "
                f"regret={r.mean_regret:.6f} | "
                f"nearest_dist={r.nearest_training_cluster_distance_z:.3f} | "
                f"outside={int(r.n_descriptors_outside_training_range)} | "
                f"nearest={r.nearest_training_cluster}"
            )
        lines.append("")

    lines.append("INTERPRETATION GUIDE")
    lines.append("-" * 72)
    lines.append(
        "If the worst-policy clusters also have high standardized nearest-cluster "
        "distance and/or descriptors outside the training range, failure is compatible "
        "with descriptor-space extrapolation."
    )
    lines.append(
        "If poor performance occurs despite low OOD distance and all descriptors lying "
        "inside the training range, the current five descriptors are insufficient to "
        "distinguish acquisition regimes, rather than the failure being explained simply "
        "by extrapolation."
    )

    summary_text = "\n".join(lines)
    (out / "audit_summary.txt").write_text(summary_text, encoding="utf-8")

    print("\n=== LOCO POLICY AUDIT: CLUSTER SUMMARY ===")
    display_cols = [
        "budget_n",
        "held_cluster",
        "n_tasks",
        "observed_mean_delta_prauc",
        "predicted_mean_delta_prauc",
        "direction_accuracy",
        "mean_gain_vs_random",
        "mean_regret",
        "policy_hybrid_fraction",
        "oracle_hybrid_fraction",
        "n_descriptors_outside_training_range",
        "max_abs_z_vs_train_clusters",
        "nearest_training_cluster_distance_z",
        "nearest_training_cluster",
    ]
    print(cluster_audit[display_cols].to_string(index=False))

    print("\n=== AUDIT SUMMARY ===")
    print(summary_text)
    print(f"\nOutputs: {out}")


if __name__ == "__main__":
    main()
