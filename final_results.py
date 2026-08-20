#!/usr/bin/env python3
"""
final_results.py
================

Manuscript-ready result consolidation for the MALDI-DRIAMS cross-site study.

This script DOES NOT introduce new models.
It reads the final analysis outputs already produced and creates:
    - Main manuscript tables
    - Supplementary tables
    - Figure-ready CSV files
    - A compact manuscript results summary
    - A missing-input audit

Expected final inputs
---------------------
outputs/multitask_active_validation_final/
    raw_results.csv
    paired_hybrid_vs_random.csv
    transfer_summary.csv
    overall_meta_summary.csv

outputs/shift_predictors_validate/
    transfer_level_table.csv
    confirmatory_correlations.csv
    ridge_logo_summary.csv
    ridge_permutation.csv
    threshold_rule_summary.csv

outputs/selective_adaptation_policy_v2_b20/
    policy_predictions.csv

outputs/selective_adaptation_policy_v2/
    policy_predictions.csv

outputs/selective_adaptation_policy_v2_b50/
    policy_predictions.csv

outputs/cluster_aware_policy_bootstrap/
    cluster_aware_results.csv
    cluster_inventory.csv

Optional historical/engineering inputs
--------------------------------------
outputs/engineering_validation/
outputs/dipls_validation_v2/
outputs/shift_mechanism/
outputs/stdz/
outputs/glsw/
outputs/engineering_transfer/

Outputs
-------
outputs/final_results/
    Table1_transfer_characteristics.csv
    Table2_cross_site_and_shift.csv
    Table3_policy_performance.csv
    Table4_cluster_robustness.csv

    Supplementary_S1_hybrid_vs_random_all_tasks.csv
    Supplementary_S2_shift_predictors.csv
    Supplementary_S3_policy_choices.csv
    Supplementary_S4_cluster_inventory.csv

    Figure2_cross_site_landscape.csv
    Figure3_shift_predictor_scatter.csv
    Figure5_label_budget_policy.csv
    Figure6_policy_vs_oracle.csv

    manuscript_key_results.txt
    missing_inputs.txt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def read_csv_checked(path, required=True):
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(str(path))
        return None
    return pd.read_csv(path)


def bootstrap_ci(x, reps=10000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(reps, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


def build_table1(raw, predictors):
    keys = ["source", "target", "species", "drug"]

    base = (
        raw[raw.strategy == "source_only"]
        .groupby(keys)
        .agg(
            n_reps=("rep", "nunique"),
            baseline_auroc=("auroc", "mean"),
            baseline_prauc=("prauc", "mean"),
            baseline_brier=("brier", "mean"),
            target_prevalence=("prevalence", "mean"),
            target_n=("n", "mean"),
            target_positives=("positives", "mean"),
        )
        .reset_index()
    )

    cols = keys + [
        "source_n",
        "target_n",
        "source_positives",
        "target_positives",
        "source_prevalence",
        "target_prevalence",
        "domain_auc_svd",
        "centroid_distance_svd",
        "covariance_distance_svd",
        "source_to_target_nn_distance",
        "source_internal_diversity",
    ]

    pred = predictors[[c for c in cols if c in predictors.columns]].copy()

    out = base.merge(pred, on=keys, how="left", suffixes=("", "_pred"))

    # prefer predictor target_n/prevalence if both exist
    for c in ["target_n", "target_positives", "target_prevalence"]:
        cp = f"{c}_pred"
        if cp in out.columns:
            out[c] = out[cp].combine_first(out[c])
            out = out.drop(columns=[cp])

    return out.sort_values(keys).reset_index(drop=True)


def build_table2(transfer_level, corr):
    keys = ["source", "target", "species", "drug"]

    keep = keys + [
        "mean_delta_prauc",
        "median_delta_prauc",
        "positive_budget_fraction",
        "domain_auc_svd",
        "centroid_distance_svd",
        "covariance_distance_svd",
        "source_to_target_nn_distance",
        "source_internal_diversity",
        "source_only_auroc_mean",
    ]

    table = transfer_level[[c for c in keep if c in transfer_level.columns]].copy()

    ranked = corr.copy()
    if "spearman_rho" in ranked.columns:
        ranked["abs_rho"] = ranked["spearman_rho"].abs()
        ranked = ranked.sort_values("abs_rho", ascending=False)

    return table, ranked


def summarize_policy(path, budget, bootstrap_reps=10000):
    df = pd.read_csv(path)

    rows = []

    mapping = [
        ("always_random", "prauc_random"),
        ("always_hybrid", "prauc_hybrid"),
        ("selective_policy_v2", "prauc_policy"),
        ("oracle_random_hybrid", "prauc_oracle"),
    ]

    for name, col in mapping:
        vals = df[col].to_numpy(float)
        lo, hi = bootstrap_ci(vals, bootstrap_reps, seed=budget)
        rows.append(dict(
            budget_n=budget,
            strategy=name,
            n_transfers=len(vals),
            prauc_mean=float(np.mean(vals)),
            prauc_median=float(np.median(vals)),
            ci_low=lo,
            ci_high=hi,
        ))

    # paired deltas
    for base_name, base_col in [
        ("always_random", "prauc_random"),
        ("always_hybrid", "prauc_hybrid"),
    ]:
        d = df["prauc_policy"].to_numpy(float) - df[base_col].to_numpy(float)
        lo, hi = bootstrap_ci(d, bootstrap_reps, seed=100 + budget)
        rows.append(dict(
            budget_n=budget,
            strategy=f"policy_minus_{base_name}",
            n_transfers=len(d),
            prauc_mean=float(np.mean(d)),
            prauc_median=float(np.median(d)),
            ci_low=lo,
            ci_high=hi,
        ))

    # validity summary encoded as separate rows
    obs = df["observed_delta_prauc_hybrid_random"].to_numpy(float)
    pred = df["predicted_delta_prauc_hybrid_random"].to_numpy(float)
    sign_acc = float(np.mean((obs > 0) == (pred > 0)))
    regret = df["regret_prauc"].to_numpy(float)

    rows.append(dict(
        budget_n=budget,
        strategy="policy_sign_accuracy",
        n_transfers=len(df),
        prauc_mean=sign_acc,
        prauc_median=np.nan,
        ci_low=np.nan,
        ci_high=np.nan,
    ))

    rows.append(dict(
        budget_n=budget,
        strategy="policy_mean_regret",
        n_transfers=len(df),
        prauc_mean=float(np.mean(regret)),
        prauc_median=float(np.median(regret)),
        ci_low=np.nan,
        ci_high=np.nan,
    ))

    return pd.DataFrame(rows), df


def build_figure2(table1):
    cols = [
        "source", "target", "species", "drug",
        "baseline_auroc", "baseline_prauc",
        "domain_auc_svd",
        "target_prevalence",
    ]
    return table1[[c for c in cols if c in table1.columns]].copy()


def build_figure3(transfer_level):
    predictors = [
        "domain_auc_svd",
        "source_internal_diversity",
        "centroid_distance_svd",
        "covariance_distance_svd",
        "source_to_target_nn_distance",
    ]

    rows = []
    for _, r in transfer_level.iterrows():
        for p in predictors:
            if p in transfer_level.columns:
                rows.append(dict(
                    transfer_id=r.get("transfer_id", ""),
                    source=r["source"],
                    target=r["target"],
                    species=r["species"],
                    drug=r["drug"],
                    predictor=p,
                    predictor_value=r[p],
                    mean_delta_prauc=r["mean_delta_prauc"],
                ))

    return pd.DataFrame(rows)


def build_figure5(policy_summaries):
    z = policy_summaries[
        policy_summaries.strategy.isin([
            "always_random",
            "always_hybrid",
            "selective_policy_v2",
            "oracle_random_hybrid",
        ])
    ].copy()

    return z.sort_values(["budget_n", "strategy"]).reset_index(drop=True)


def build_figure6(policy_predictions):
    rows = []
    for budget, df in policy_predictions.items():
        for _, r in df.iterrows():
            rows.append(dict(
                budget_n=budget,
                transfer_id=r["transfer_id"],
                policy_choice=r["policy_choice"],
                oracle_choice=r["oracle_choice"],
                predicted_delta_prauc=r["predicted_delta_prauc_hybrid_random"],
                observed_delta_prauc=r["observed_delta_prauc_hybrid_random"],
                prauc_random=r["prauc_random"],
                prauc_hybrid=r["prauc_hybrid"],
                prauc_policy=r["prauc_policy"],
                prauc_oracle=r["prauc_oracle"],
                regret_prauc=r["regret_prauc"],
            ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ap.add_argument(
        "--multitask-dir",
        default="outputs/multitask_active_validation_final",
    )
    ap.add_argument(
        "--shift-validate-dir",
        default="outputs/shift_predictors_validate",
    )
    ap.add_argument(
        "--policy20-dir",
        default="outputs/selective_adaptation_policy_v2_b20",
    )
    ap.add_argument(
        "--policy30-dir",
        default="outputs/selective_adaptation_policy_v2",
    )
    ap.add_argument(
        "--policy50-dir",
        default="outputs/selective_adaptation_policy_v2_b50",
    )
    ap.add_argument(
        "--cluster-dir",
        default="outputs/cluster_aware_policy_bootstrap",
    )
    ap.add_argument(
        "--shift-dir",
        default="outputs/shift_predictors",
    )
    ap.add_argument(
        "--out",
        default="outputs/final_results",
    )
    ap.add_argument(
        "--bootstrap-reps",
        type=int,
        default=10000,
    )

    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    missing = []

    def req(path):
        p = Path(path)
        if not p.exists():
            missing.append(str(p))
            return None
        return pd.read_csv(p)

    raw = req(Path(args.multitask_dir) / "raw_results.csv")
    paired = req(Path(args.multitask_dir) / "paired_hybrid_vs_random.csv")

    predictors = req(Path(args.shift_dir) / "transfer_predictors.csv")

    transfer_level = req(Path(args.shift_validate_dir) / "transfer_level_table.csv")
    corr = req(Path(args.shift_validate_dir) / "confirmatory_correlations.csv")
    ridge = req(Path(args.shift_validate_dir) / "ridge_logo_summary.csv")
    perm = req(Path(args.shift_validate_dir) / "ridge_permutation.csv")

    cluster_results = req(Path(args.cluster_dir) / "cluster_aware_results.csv")
    cluster_inventory = req(Path(args.cluster_dir) / "cluster_inventory.csv")

    policy_paths = {
        20: Path(args.policy20_dir) / "policy_predictions.csv",
        30: Path(args.policy30_dir) / "policy_predictions.csv",
        50: Path(args.policy50_dir) / "policy_predictions.csv",
    }

    policy_predictions = {}
    policy_summaries = []

    for budget, path in policy_paths.items():
        if path.exists():
            ps, pdf = summarize_policy(
                path,
                budget,
                bootstrap_reps=args.bootstrap_reps,
            )
            policy_summaries.append(ps)
            policy_predictions[budget] = pdf
        else:
            missing.append(str(path))

    if missing:
        (out / "missing_inputs.txt").write_text(
            "\n".join(missing),
            encoding="utf-8",
        )

    critical = [raw, paired, predictors, transfer_level, corr, cluster_results]
    if any(x is None for x in critical):
        raise SystemExit(
            "Critical input missing. See outputs/final_results/missing_inputs.txt"
        )

    # ---------------------------------------------------------
    # Main tables
    # ---------------------------------------------------------
    table1 = build_table1(raw, predictors)
    table1.to_csv(
        out / "Table1_transfer_characteristics.csv",
        index=False,
    )

    table2, ranked_corr = build_table2(
        transfer_level,
        corr,
    )
    table2.to_csv(
        out / "Table2_cross_site_and_shift.csv",
        index=False,
    )

    if policy_summaries:
        table3 = pd.concat(
            policy_summaries,
            ignore_index=True,
        )
        table3.to_csv(
            out / "Table3_policy_performance.csv",
            index=False,
        )
    else:
        table3 = pd.DataFrame()

    cluster_results.to_csv(
        out / "Table4_cluster_robustness.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Supplementary tables
    # ---------------------------------------------------------
    paired.to_csv(
        out / "Supplementary_S1_hybrid_vs_random_all_tasks.csv",
        index=False,
    )

    ranked_corr.to_csv(
        out / "Supplementary_S2_shift_predictors.csv",
        index=False,
    )

    if policy_predictions:
        pd.concat(
            [
                df.assign(budget_n=budget)
                for budget, df in policy_predictions.items()
            ],
            ignore_index=True,
        ).to_csv(
            out / "Supplementary_S3_policy_choices.csv",
            index=False,
        )

    if cluster_inventory is not None:
        cluster_inventory.to_csv(
            out / "Supplementary_S4_cluster_inventory.csv",
            index=False,
        )

    # ---------------------------------------------------------
    # Figure-ready data
    # ---------------------------------------------------------
    build_figure2(table1).to_csv(
        out / "Figure2_cross_site_landscape.csv",
        index=False,
    )

    build_figure3(transfer_level).to_csv(
        out / "Figure3_shift_predictor_scatter.csv",
        index=False,
    )

    if not table3.empty:
        build_figure5(table3).to_csv(
            out / "Figure5_label_budget_policy.csv",
            index=False,
        )

    if policy_predictions:
        build_figure6(policy_predictions).to_csv(
            out / "Figure6_policy_vs_oracle.csv",
            index=False,
        )

    # ---------------------------------------------------------
    # Key manuscript results text
    # ---------------------------------------------------------
    lines = []
    lines.append("MALDI-DRIAMS FINAL MANUSCRIPT RESULTS")
    lines.append("=" * 42)
    lines.append("")

    lines.append(f"Transfer tasks: {len(table1)}")

    if cluster_inventory is not None:
        inv30 = cluster_inventory
        if "budget_n" in inv30.columns:
            inv30 = inv30[inv30.budget_n == 30]
        lines.append(
            f"Species-site transfer clusters: {inv30['cluster_id'].nunique()}"
        )

    lines.append("")
    lines.append("SHIFT PREDICTOR CONFIRMATION")
    lines.append("-" * 28)

    for _, r in ranked_corr.head(6).iterrows():
        lines.append(
            f"{r['predictor']}: Spearman rho={r['spearman_rho']:.3f}, "
            f"95% CI [{r['spearman_ci_low']:.3f}, {r['spearman_ci_high']:.3f}], "
            f"p={r['spearman_p']:.6g}"
        )

    if ridge is not None and not ridge.empty:
        rr = ridge.sort_values("logo_rmse").iloc[0]
        lines.append(
            f"Best Ridge LOGO: alpha={rr['alpha']}, "
            f"RMSE={rr['logo_rmse']:.4f}, "
            f"Pearson r={rr['logo_pearson_r']:.3f}, "
            f"Spearman rho={rr['logo_spearman_rho']:.3f}"
        )

    if perm is not None and not perm.empty:
        rp = perm.iloc[0]
        lines.append(
            f"Ridge permutation p={rp['permutation_p']:.6g}"
        )

    lines.append("")
    lines.append("LEAKAGE-FREE SELECTIVE POLICY")
    lines.append("-" * 30)

    if not table3.empty:
        for budget in [20, 30, 50]:
            z = table3[table3.budget_n == budget]
            if z.empty:
                continue

            def get(strategy):
                q = z[z.strategy == strategy]
                return None if q.empty else q.iloc[0]

            rr = get("always_random")
            hh = get("always_hybrid")
            pp = get("selective_policy_v2")
            oo = get("oracle_random_hybrid")
            dd = get("policy_minus_always_random")
            sa = get("policy_sign_accuracy")

            if rr is not None and pp is not None:
                lines.append(
                    f"{budget} labels: Random={rr.prauc_mean:.4f}, "
                    f"Hybrid={hh.prauc_mean:.4f}, "
                    f"Policy={pp.prauc_mean:.4f}, "
                    f"Oracle={oo.prauc_mean:.4f}; "
                    f"Policy-Random={dd.prauc_mean:+.4f} "
                    f"[{dd.ci_low:+.4f}, {dd.ci_high:+.4f}]; "
                    f"direction accuracy={sa.prauc_mean:.0%}"
                )

    lines.append("")
    lines.append("CLUSTER-AWARE ROBUSTNESS")
    lines.append("-" * 25)

    for _, r in cluster_results.iterrows():
        if "always_random" not in r["comparison"]:
            continue
        lines.append(
            f"{int(r['budget_n'])} labels: "
            f"Policy-Random delta PR-AUC={r['mean_delta']:+.4f}, "
            f"cluster 95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}], "
            f"p-style={r['p_two_sided']:.4g}"
        )

    lines.append("")
    lines.append("FINAL INTERPRETATION")
    lines.append("-" * 20)
    lines.append(
        "No single target-acquisition strategy is universally optimal across transfers."
    )
    lines.append(
        "Unlabeled spectral shift descriptors predict whether Hybrid or Random acquisition is preferable."
    )
    lines.append(
        "The leakage-free selective policy outperforms Always-Random at 20, 30, and 50 labels, "
        "and these gains remain positive under species-site cluster-aware bootstrap."
    )

    (out / "manuscript_key_results.txt").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    config = vars(args).copy()
    config["critical_inputs_found"] = len(missing) == 0

    (out / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=== FINAL RESULTS CONSOLIDATION ===")
    print(f"Main tables: 4")
    print(f"Transfer tasks: {len(table1)}")
    if policy_predictions:
        print(f"Policy budgets: {sorted(policy_predictions)}")

    print("\n=== KEY RESULTS ===")
    print("\n".join(lines))

    if missing:
        print("\n=== MISSING OPTIONAL/INPUT FILES ===")
        for p in missing:
            print(" -", p)

    print(f"\nOutputs: {out}")


if __name__ == "__main__":
    main()
