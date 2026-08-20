#!/usr/bin/env python3
"""
leave_one_cluster_out_policy.py
===============================

Cluster-held-out validation of the leakage-free selective adaptation policy.

Why this analysis?
------------------
The 20 antibiotic-specific transfer tasks are not fully independent because several
tasks share the same source->target site pair and species. The manuscript identifies
7 such species-site clusters.

This script therefore evaluates the policy under OUTER leave-one-cluster-out (LOCO):
    - all antibiotic tasks belonging to one source->target|species cluster are held out
    - the policy is trained only on the remaining clusters
    - ridge alpha is selected by INNER leave-one-cluster-out on the training clusters
    - the held-out cluster contributes no task to model fitting or hyperparameter tuning

To avoid clusters with many antibiotics dominating training, Ridge receives
inverse-cluster-size sample weights. Inner model selection averages validation MSE
equally across held-out clusters.

Predictors are strictly target-label-free:
    domain_auc_svd
    centroid_distance_svd
    covariance_distance_svd
    source_to_target_nn_distance
    source_internal_diversity

Expected inputs
---------------
outputs/multitask_active_validation_final/
    raw_results.csv
    paired_hybrid_vs_random.csv

outputs/shift_predictors/
    transfer_predictors.csv

Example
-------
python leave_one_cluster_out_policy.py \
  --budgets 20,30,50 \
  --out outputs/leave_one_cluster_out_policy

Outputs
-------
policy_predictions.csv
budget_summary.csv
paired_comparisons.csv
cluster_summary.csv
alpha_by_outer_cluster.csv
ridge_coefficients.csv
run_config.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


PREDICTORS = [
    "domain_auc_svd",
    "centroid_distance_svd",
    "covariance_distance_svd",
    "source_to_target_nn_distance",
    "source_internal_diversity",
]


def cluster_id(df):
    return (
        df["source"].astype(str)
        + "->"
        + df["target"].astype(str)
        + "|"
        + df["species"].astype(str)
    )


def bootstrap_ci(x, reps=10000, seed=123):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=float)
    n = len(x)
    for i in range(reps):
        vals[i] = np.mean(rng.choice(x, size=n, replace=True))
    return tuple(np.percentile(vals, [2.5, 97.5]).astype(float))


def cluster_bootstrap_ci(df, value_col, reps=10000, seed=123):
    """Resample clusters with replacement; preserve all tasks in selected clusters."""
    clusters = df["cluster_id"].drop_duplicates().tolist()
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=float)

    by_cluster = {c: df.loc[df.cluster_id == c, value_col].to_numpy(float) for c in clusters}

    for i in range(reps):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        pieces = [by_cluster[c] for c in sampled]
        vals[i] = np.mean(np.concatenate(pieces))

    return tuple(np.percentile(vals, [2.5, 97.5]).astype(float))


def build_transfer_table(raw, paired, predictors, budget):
    keys = ["source", "target", "species", "drug"]

    z = raw[
        raw["strategy"].isin(["random", "hybrid"])
        & (raw["budget_n"] == budget)
    ].copy()

    perf = (
        z.groupby(keys + ["strategy"], as_index=False)
        .agg(
            auroc=("auroc", "mean"),
            prauc=("prauc", "mean"),
            brier=("brier", "mean"),
        )
    )

    wide = perf.pivot_table(
        index=keys,
        columns="strategy",
        values=["auroc", "prauc", "brier"],
    )
    wide.columns = [f"{m}_{s}" for m, s in wide.columns]
    wide = wide.reset_index()

    hp = paired[
        (paired["metric"] == "prauc")
        & (paired["budget_n"] == budget)
    ][keys + ["delta_mean"]].rename(
        columns={"delta_mean": "delta_prauc_hybrid_random"}
    )

    out = wide.merge(hp, on=keys, how="inner")
    out = out.merge(predictors, on=keys, how="left")

    out["transfer_id"] = (
        out["source"].astype(str)
        + "->"
        + out["target"].astype(str)
        + "|"
        + out["species"].astype(str)
        + "|"
        + out["drug"].astype(str)
    )
    out["cluster_id"] = cluster_id(out)
    return out


def impute_from_train(train, test, predictors):
    available = [p for p in predictors if p in train.columns]
    Xtr = train[available].copy()
    Xte = test[available].copy()

    medians = {}
    for p in available:
        med = float(Xtr[p].median())
        medians[p] = med
        Xtr[p] = Xtr[p].fillna(med)
        Xte[p] = Xte[p].fillna(med)

    return available, Xtr.to_numpy(float), Xte.to_numpy(float), medians


def cluster_balanced_weights(train):
    counts = train["cluster_id"].value_counts()
    w = train["cluster_id"].map(lambda c: 1.0 / counts[c]).to_numpy(float)
    # normalize mean weight to 1 (does not change relative weighting)
    w = w / np.mean(w)
    return w


def fit_weighted_ridge(train, test, predictors, alpha):
    available, Xtr, Xte, _ = impute_from_train(train, test, predictors)
    ytr = train["delta_prauc_hybrid_random"].to_numpy(float)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    weights = cluster_balanced_weights(train)

    model = Ridge(alpha=float(alpha))
    model.fit(Xtr_s, ytr, sample_weight=weights)
    pred = model.predict(Xte_s)

    return pred, available, model.coef_


def choose_alpha_inner_loco(train, predictors, alphas):
    """Inner LOCO; equal weight is given to each validation cluster."""
    clusters = sorted(train["cluster_id"].unique())
    if len(clusters) < 3:
        # With too few clusters, use a conservative default.
        return 10.0, pd.DataFrame()

    rows = []
    for alpha in alphas:
        fold_mse = []

        for held_cluster in clusters:
            inner_test = train[train.cluster_id == held_cluster].copy()
            inner_train = train[train.cluster_id != held_cluster].copy()

            if inner_train["cluster_id"].nunique() < 2:
                continue

            pred, _, _ = fit_weighted_ridge(
                inner_train, inner_test, predictors, alpha
            )
            obs = inner_test["delta_prauc_hybrid_random"].to_numpy(float)
            fold_mse.append(float(np.mean((pred - obs) ** 2)))

        if fold_mse:
            rows.append({
                "alpha": float(alpha),
                "mean_cluster_mse": float(np.mean(fold_mse)),
                "rmse": float(math.sqrt(np.mean(fold_mse))),
                "n_inner_clusters": len(fold_mse),
            })

    if not rows:
        return 10.0, pd.DataFrame()

    tab = pd.DataFrame(rows).sort_values(
        ["mean_cluster_mse", "alpha"], ascending=[True, True]
    )
    return float(tab.iloc[0]["alpha"]), tab


def strategy_from_delta(x):
    return "hybrid" if x > 0 else "random"


def oracle_from_row(row):
    return "hybrid" if row["prauc_hybrid"] >= row["prauc_random"] else "random"


def evaluate_budget(table, budget, alphas):
    clusters = sorted(table["cluster_id"].unique())
    rows = []
    alpha_rows = []
    coef_rows = []

    print(f"\n=== BUDGET {budget} ===")
    print(f"Transfers: {len(table)} | clusters: {len(clusters)}")

    for j, held_cluster in enumerate(clusters, 1):
        test = table[table.cluster_id == held_cluster].copy()
        train = table[table.cluster_id != held_cluster].copy()

        alpha, inner_tab = choose_alpha_inner_loco(train, PREDICTORS, alphas)
        pred, available, coefs = fit_weighted_ridge(
            train, test, PREDICTORS, alpha
        )

        alpha_rows.append({
            "budget_n": budget,
            "outer_held_cluster": held_cluster,
            "selected_alpha": alpha,
            "n_train_clusters": train.cluster_id.nunique(),
            "n_test_tasks": len(test),
        })

        for p, c in zip(available, coefs):
            coef_rows.append({
                "budget_n": budget,
                "outer_held_cluster": held_cluster,
                "predictor": p,
                "coefficient": float(c),
                "selected_alpha": alpha,
            })

        print(
            f"[{j}/{len(clusters)}] hold out {held_cluster} | "
            f"test tasks={len(test)} | alpha={alpha:g}"
        )

        for k, (_, r) in enumerate(test.iterrows()):
            pred_delta = float(pred[k])
            choice = strategy_from_delta(pred_delta)
            oracle = oracle_from_row(r)

            pr_policy = float(r[f"prauc_{choice}"])
            pr_oracle = float(r[f"prauc_{oracle}"])

            rows.append({
                "budget_n": budget,
                "cluster_id": held_cluster,
                "transfer_id": r["transfer_id"],
                "source": r["source"],
                "target": r["target"],
                "species": r["species"],
                "drug": r["drug"],
                "predicted_delta_prauc_hybrid_random": pred_delta,
                "observed_delta_prauc_hybrid_random": float(r["delta_prauc_hybrid_random"]),
                "selected_alpha": alpha,
                "policy_choice": choice,
                "oracle_choice": oracle,
                "prauc_random": float(r["prauc_random"]),
                "prauc_hybrid": float(r["prauc_hybrid"]),
                "prauc_policy": pr_policy,
                "prauc_oracle": pr_oracle,
                "policy_correct_direction": bool(choice == oracle),
                "regret_prauc": float(pr_oracle - pr_policy),
            })

    return pd.DataFrame(rows), pd.DataFrame(alpha_rows), pd.DataFrame(coef_rows)


def safe_corr(x, y, method="pearson"):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return np.nan
    try:
        if method == "pearson":
            return float(pearsonr(x[mask], y[mask])[0])
        return float(spearmanr(x[mask], y[mask])[0])
    except Exception:
        return np.nan


def summarize_predictions(pred, bootstrap_reps):
    budget_rows = []
    comp_rows = []
    cluster_rows = []

    for budget, g in pred.groupby("budget_n", sort=True):
        d_random = (g["prauc_policy"] - g["prauc_random"]).to_numpy(float)
        d_hybrid = (g["prauc_policy"] - g["prauc_hybrid"]).to_numpy(float)

        lo_r, hi_r = cluster_bootstrap_ci(
            g.assign(delta=d_random), "delta", reps=bootstrap_reps, seed=100 + int(budget)
        )
        lo_h, hi_h = cluster_bootstrap_ci(
            g.assign(delta=d_hybrid), "delta", reps=bootstrap_reps, seed=200 + int(budget)
        )

        observed = g["observed_delta_prauc_hybrid_random"].to_numpy(float)
        predicted = g["predicted_delta_prauc_hybrid_random"].to_numpy(float)

        budget_rows.append({
            "budget_n": int(budget),
            "n_transfers": len(g),
            "n_clusters": g.cluster_id.nunique(),
            "mean_prauc_random": float(g.prauc_random.mean()),
            "mean_prauc_hybrid": float(g.prauc_hybrid.mean()),
            "mean_prauc_policy": float(g.prauc_policy.mean()),
            "mean_prauc_oracle": float(g.prauc_oracle.mean()),
            "mean_gain_vs_random": float(d_random.mean()),
            "gain_vs_random_cluster_ci_low": lo_r,
            "gain_vs_random_cluster_ci_high": hi_r,
            "mean_gain_vs_hybrid": float(d_hybrid.mean()),
            "gain_vs_hybrid_cluster_ci_low": lo_h,
            "gain_vs_hybrid_cluster_ci_high": hi_h,
            "direction_accuracy": float(g.policy_correct_direction.mean()),
            "mean_regret": float(g.regret_prauc.mean()),
            "zero_regret_fraction": float((g.regret_prauc == 0).mean()),
            "pearson_pred_obs": safe_corr(predicted, observed, "pearson"),
            "spearman_pred_obs": safe_corr(predicted, observed, "spearman"),
        })

        comp_rows.extend([
            {
                "budget_n": int(budget),
                "comparison": "LOCO policy - Always-Random",
                "mean_delta": float(d_random.mean()),
                "cluster_ci_low": lo_r,
                "cluster_ci_high": hi_r,
                "fraction_positive": float(np.mean(d_random > 0)),
                "fraction_equal": float(np.mean(d_random == 0)),
            },
            {
                "budget_n": int(budget),
                "comparison": "LOCO policy - Always-Hybrid",
                "mean_delta": float(d_hybrid.mean()),
                "cluster_ci_low": lo_h,
                "cluster_ci_high": hi_h,
                "fraction_positive": float(np.mean(d_hybrid > 0)),
                "fraction_equal": float(np.mean(d_hybrid == 0)),
            },
        ])

        for cid, cg in g.groupby("cluster_id"):
            dr = (cg.prauc_policy - cg.prauc_random).to_numpy(float)
            cluster_rows.append({
                "budget_n": int(budget),
                "cluster_id": cid,
                "n_tasks": len(cg),
                "mean_gain_vs_random": float(dr.mean()),
                "direction_accuracy": float(cg.policy_correct_direction.mean()),
                "mean_regret": float(cg.regret_prauc.mean()),
            })

    return (
        pd.DataFrame(budget_rows),
        pd.DataFrame(comp_rows),
        pd.DataFrame(cluster_rows),
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
        default="outputs/leave_one_cluster_out_policy",
    )
    ap.add_argument(
        "--budgets",
        default="20,30,50",
    )
    ap.add_argument(
        "--alphas",
        default="0.01,0.1,1,10,100",
    )
    ap.add_argument(
        "--bootstrap-reps",
        type=int,
        default=10000,
    )
    args = ap.parse_args()

    budgets = [int(x.strip()) for x in args.budgets.split(",") if x.strip()]
    alphas = [float(x.strip()) for x in args.alphas.split(",") if x.strip()]

    multitask_dir = Path(args.multitask_dir)
    shift_dir = Path(args.shift_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw_path = multitask_dir / "raw_results.csv"
    paired_path = multitask_dir / "paired_hybrid_vs_random.csv"
    pred_path = shift_dir / "transfer_predictors.csv"

    for p in [raw_path, paired_path, pred_path]:
        if not p.exists():
            raise SystemExit(f"Missing required input: {p}")

    raw = pd.read_csv(raw_path)
    paired = pd.read_csv(paired_path)
    predictors = pd.read_csv(pred_path)

    all_pred = []
    all_alpha = []
    all_coef = []

    print("=== LEAVE-ONE-CLUSTER-OUT SELECTIVE POLICY ===")
    print("Outer unit: source->target|species")
    print("Predictors:", ", ".join(PREDICTORS))
    print("Budgets:", budgets)

    for budget in budgets:
        table = build_transfer_table(raw, paired, predictors, budget)

        if table["cluster_id"].nunique() < 4:
            raise SystemExit(
                f"Budget {budget}: only {table['cluster_id'].nunique()} clusters available."
            )

        pred, alpha_tab, coef_tab = evaluate_budget(
            table, budget, alphas
        )
        all_pred.append(pred)
        all_alpha.append(alpha_tab)
        all_coef.append(coef_tab)

    pred = pd.concat(all_pred, ignore_index=True)
    alpha_tab = pd.concat(all_alpha, ignore_index=True)
    coef_tab = pd.concat(all_coef, ignore_index=True)

    summary, comparisons, cluster_summary = summarize_predictions(
        pred, args.bootstrap_reps
    )

    pred.to_csv(out / "policy_predictions.csv", index=False)
    summary.to_csv(out / "budget_summary.csv", index=False)
    comparisons.to_csv(out / "paired_comparisons.csv", index=False)
    cluster_summary.to_csv(out / "cluster_summary.csv", index=False)
    alpha_tab.to_csv(out / "alpha_by_outer_cluster.csv", index=False)
    coef_tab.to_csv(out / "ridge_coefficients.csv", index=False)

    config = {
        "outer_validation": "leave-one-source-target-species-cluster-out",
        "inner_validation": "leave-one-cluster-out",
        "cluster_balanced_training": True,
        "predictors": PREDICTORS,
        "budgets": budgets,
        "alphas": alphas,
        "bootstrap_reps": args.bootstrap_reps,
        "multitask_dir": str(multitask_dir),
        "shift_dir": str(shift_dir),
    }
    (out / "run_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    print("\n=== LOCO BUDGET SUMMARY ===")
    print(summary.to_string(index=False))

    print("\n=== LOCO PAIRED COMPARISONS ===")
    print(comparisons.to_string(index=False))

    print("\n=== CLUSTER-LEVEL SUMMARY ===")
    print(cluster_summary.to_string(index=False))

    print(f"\nOutputs: {out}")


if __name__ == "__main__":
    main()
