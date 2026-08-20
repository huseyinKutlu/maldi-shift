#!/usr/bin/env python3
"""
selective_adaptation_policy_v2.py
================================

Leakage-free deployment policy for active target-site adaptation.

Goal
----
For a NEW source->target transfer, choose between:

    RANDOM target-label acquisition
    HYBRID target-label acquisition

using ONLY unlabeled / deployment-available shift descriptors.

No target-label-derived predictor is used for the policy.

Baselines
---------
1) always_random
2) always_hybrid
3) selective_policy_v2
4) oracle_random_hybrid

Default budget
--------------
30 target labels.

Leakage-free predictors
-----------------------
1) domain_auc_svd
2) centroid_distance_svd
3) covariance_distance_svd
4) source_to_target_nn_distance
5) source_internal_diversity

Explicitly excluded
-------------------
- source_only_auroc_mean
- source_only_prauc_mean
- target prevalence
- target positives
- any held-out target performance
- any target-label-derived calibration statistic

Protocol
--------
Outer leave-one-transfer-out (LOTO):
  - held-out transfer is never used to fit the policy
  - alpha is selected using inner LOTO on training transfers only
  - policy predicts:
        delta_prauc = PR-AUC(Hybrid) - PR-AUC(Random)
  - if predicted delta > 0 => Hybrid
    else => Random

Evaluation
----------
For the held-out transfer, compare the selected strategy's observed performance
against:
    always_random
    always_hybrid
    oracle_random_hybrid

Also report:
    - benefit prediction Pearson/Spearman
    - sign accuracy
    - policy regret vs oracle
    - bootstrap CI for paired PR-AUC differences

Expected inputs
---------------
outputs/multitask_active_validation_final/
    raw_results.csv
    paired_hybrid_vs_random.csv

outputs/shift_predictors/
    transfer_predictors.csv

Outputs
-------
outputs/selective_adaptation_policy_v2/
    policy_predictions.csv
    policy_summary.csv
    metric_summary.csv
    paired_policy_comparisons.csv
    benefit_prediction_validity.csv
    regret_summary.csv
    ridge_coefficients_by_fold.csv
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
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


LEAKAGE_FREE_PREDICTORS = [
    "domain_auc_svd",
    "centroid_distance_svd",
    "covariance_distance_svd",
    "source_to_target_nn_distance",
    "source_internal_diversity",
]


def bootstrap_ci(x, reps=10000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    bs = rng.choice(
        x,
        size=(reps, len(x)),
        replace=True,
    ).mean(axis=1)

    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


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

    wide.columns = [
        f"{metric}_{strategy}"
        for metric, strategy in wide.columns
    ]
    wide = wide.reset_index()

    hp = paired[
        (paired.metric == "prauc")
        & (paired.budget_n == budget)
    ][
        keys + [
            "delta_mean",
            "delta_ci_low",
            "delta_ci_high",
            "significant",
        ]
    ].rename(
        columns={
            "delta_mean": "delta_prauc_hybrid_random",
            "delta_ci_low": "delta_prauc_ci_low",
            "delta_ci_high": "delta_prauc_ci_high",
            "significant": "delta_prauc_significant",
        }
    )

    out = wide.merge(hp, on=keys, how="inner")
    out = out.merge(
        predictors,
        on=keys,
        how="left",
    )

    out["transfer_id"] = (
        out.source.astype(str)
        + "->"
        + out.target.astype(str)
        + "|"
        + out.species.astype(str)
        + "|"
        + out.drug.astype(str)
    )

    return out


def prepare_xy(train, test, predictors):
    available = [
        p for p in predictors
        if p in train.columns
    ]

    Xtr = train[available].copy()
    Xte = test[available].copy()

    for p in available:
        med = Xtr[p].median()
        Xtr[p] = Xtr[p].fillna(med)
        Xte[p] = Xte[p].fillna(med)

    return (
        available,
        Xtr.to_numpy(dtype=float),
        Xte.to_numpy(dtype=float),
    )


def choose_alpha_inner_logo(train, predictors, alphas):
    available = [
        p for p in predictors
        if p in train.columns
    ]

    X = train[available].copy()

    for p in available:
        X[p] = X[p].fillna(X[p].median())

    X = X.to_numpy(dtype=float)
    y = train["delta_prauc_hybrid_random"].to_numpy(dtype=float)

    rows = []

    for alpha in alphas:
        pred = np.zeros(len(train), dtype=float)

        for i in range(len(train)):
            tr = np.arange(len(train)) != i

            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ])

            pipe.fit(X[tr], y[tr])
            pred[i] = pipe.predict(X[[i]])[0]

        rmse = math.sqrt(mean_squared_error(y, pred))
        rows.append((float(alpha), float(rmse)))

    rows.sort(key=lambda x: x[1])
    return rows[0][0]


def predict_delta_outer_fold(train, test, predictors, alphas):
    alpha = choose_alpha_inner_logo(
        train,
        predictors,
        alphas,
    )

    available, Xtr, Xte = prepare_xy(
        train,
        test,
        predictors,
    )

    ytr = train[
        "delta_prauc_hybrid_random"
    ].to_numpy(dtype=float)

    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ])

    pipe.fit(Xtr, ytr)
    pred = float(pipe.predict(Xte)[0])

    coef_rows = [
        dict(
            predictor=p,
            coefficient=float(c),
            alpha=float(alpha),
        )
        for p, c in zip(
            available,
            pipe.named_steps["ridge"].coef_,
        )
    ]

    return pred, alpha, coef_rows


def policy_choice(predicted_delta):
    return "hybrid" if predicted_delta > 0 else "random"


def oracle_choice(row):
    if row.prauc_hybrid >= row.prauc_random:
        return "hybrid"
    return "random"


def metric_value(row, metric, strategy):
    return float(row[f"{metric}_{strategy}"])


def summarize_vector(name, vals, bootstrap_reps, seed=0):
    vals = np.asarray(vals, dtype=float)
    lo, hi = bootstrap_ci(
        vals,
        reps=bootstrap_reps,
        seed=seed,
    )

    return dict(
        strategy=name,
        n_transfers=int(np.isfinite(vals).sum()),
        mean=float(np.nanmean(vals)),
        median=float(np.nanmedian(vals)),
        ci_low=lo,
        ci_high=hi,
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
        default="outputs/selective_adaptation_policy_v2",
    )

    ap.add_argument(
        "--budget",
        type=int,
        default=30,
    )

    ap.add_argument(
        "--bootstrap-reps",
        type=int,
        default=10000,
    )

    ap.add_argument(
        "--alphas",
        default="0.01,0.1,1,10,100",
    )

    args = ap.parse_args()

    multitask_dir = Path(args.multitask_dir)
    shift_dir = Path(args.shift_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(
        multitask_dir / "raw_results.csv"
    )

    paired = pd.read_csv(
        multitask_dir / "paired_hybrid_vs_random.csv"
    )

    predictors = pd.read_csv(
        shift_dir / "transfer_predictors.csv"
    )

    table = build_transfer_table(
        raw,
        paired,
        predictors,
        budget=args.budget,
    )

    if len(table) < 8:
        raise SystemExit(
            f"Only {len(table)} transfers available."
        )

    alphas = [
        float(x)
        for x in args.alphas.split(",")
    ]

    print("=== SELECTIVE ADAPTATION POLICY V2 ===")
    print(f"Transfers: {len(table)}")
    print(f"Budget: {args.budget}")
    print(
        "Leakage-free predictors: "
        + ", ".join(LEAKAGE_FREE_PREDICTORS)
    )

    rows = []
    coef_rows = []

    for i in range(len(table)):
        test = table.iloc[[i]].copy()
        train = table.drop(
            index=table.index[i]
        ).copy()

        pred_delta, alpha, coefs = predict_delta_outer_fold(
            train,
            test,
            LEAKAGE_FREE_PREDICTORS,
            alphas,
        )

        row = test.iloc[0]
        choice = policy_choice(pred_delta)
        oracle = oracle_choice(row)

        rows.append(dict(
            transfer_id=row.transfer_id,
            source=row.source,
            target=row.target,
            species=row.species,
            drug=row.drug,

            predicted_delta_prauc_hybrid_random=pred_delta,
            observed_delta_prauc_hybrid_random=row.delta_prauc_hybrid_random,

            selected_alpha=float(alpha),
            policy_choice=choice,
            oracle_choice=oracle,

            prauc_random=float(row.prauc_random),
            prauc_hybrid=float(row.prauc_hybrid),
            prauc_policy=metric_value(row, "prauc", choice),
            prauc_oracle=metric_value(row, "prauc", oracle),

            auroc_random=float(row.auroc_random),
            auroc_hybrid=float(row.auroc_hybrid),
            auroc_policy=metric_value(row, "auroc", choice),
            auroc_oracle=metric_value(row, "auroc", oracle),

            brier_random=float(row.brier_random),
            brier_hybrid=float(row.brier_hybrid),
            brier_policy=metric_value(row, "brier", choice),
            brier_oracle=metric_value(row, "brier", oracle),

            policy_correct_direction=bool(
                choice == oracle
            ),
            regret_prauc=float(
                metric_value(row, "prauc", oracle)
                - metric_value(row, "prauc", choice)
            ),
        ))

        for c in coefs:
            coef_rows.append(dict(
                transfer_id=row.transfer_id,
                **c
            ))

        print(
            f"[{i+1:02d}/{len(table)}] "
            f"{row.species}/{row.drug} "
            f"{row.source}->{row.target} | "
            f"predΔPR={pred_delta:+.4f} | "
            f"obsΔPR={row.delta_prauc_hybrid_random:+.4f} | "
            f"choice={choice} | oracle={oracle}",
            flush=True,
        )

    pred = pd.DataFrame(rows)

    pred.to_csv(
        out / "policy_predictions.csv",
        index=False,
    )

    pd.DataFrame(coef_rows).to_csv(
        out / "ridge_coefficients_by_fold.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # PR-AUC summary
    # ---------------------------------------------------------
    summary_rows = []

    for name, col in [
        ("always_random", "prauc_random"),
        ("always_hybrid", "prauc_hybrid"),
        ("selective_policy_v2", "prauc_policy"),
        ("oracle_random_hybrid", "prauc_oracle"),
    ]:
        summary_rows.append(
            summarize_vector(
                name,
                pred[col].to_numpy(dtype=float),
                args.bootstrap_reps,
                seed=0,
            )
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        out / "policy_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Metric summary
    # ---------------------------------------------------------
    metric_rows = []

    strategy_suffixes = [
        ("always_random", "random"),
        ("always_hybrid", "hybrid"),
        ("selective_policy_v2", "policy"),
        ("oracle_random_hybrid", "oracle"),
    ]

    for strategy, suffix in strategy_suffixes:
        for metric in ["prauc", "auroc", "brier"]:
            vals = pred[
                f"{metric}_{suffix}"
            ].to_numpy(dtype=float)

            lo, hi = bootstrap_ci(
                vals,
                reps=args.bootstrap_reps,
                seed=10,
            )

            metric_rows.append(dict(
                strategy=strategy,
                metric=metric,
                mean=float(np.nanmean(vals)),
                median=float(np.nanmedian(vals)),
                ci_low=lo,
                ci_high=hi,
            ))

    metric_summary = pd.DataFrame(metric_rows)
    metric_summary.to_csv(
        out / "metric_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Paired comparisons
    # ---------------------------------------------------------
    comp_rows = []

    for base_name, base_col in [
        ("always_random", "prauc_random"),
        ("always_hybrid", "prauc_hybrid"),
    ]:
        d = (
            pred.prauc_policy.to_numpy(dtype=float)
            - pred[base_col].to_numpy(dtype=float)
        )

        lo, hi = bootstrap_ci(
            d,
            reps=args.bootstrap_reps,
            seed=20,
        )

        comp_rows.append(dict(
            comparison=(
                f"selective_policy_v2 - {base_name}"
            ),
            mean_delta=float(np.mean(d)),
            median_delta=float(np.median(d)),
            ci_low=lo,
            ci_high=hi,
            fraction_policy_better=float(np.mean(d > 0)),
            fraction_policy_equal=float(np.mean(d == 0)),
        ))

    comparisons = pd.DataFrame(comp_rows)

    comparisons.to_csv(
        out / "paired_policy_comparisons.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Predictive validity
    # ---------------------------------------------------------
    observed = pred[
        "observed_delta_prauc_hybrid_random"
    ].to_numpy(dtype=float)

    predicted = pred[
        "predicted_delta_prauc_hybrid_random"
    ].to_numpy(dtype=float)

    try:
        pr = float(pearsonr(predicted, observed)[0])
    except Exception:
        pr = np.nan

    try:
        sr = float(spearmanr(predicted, observed)[0])
    except Exception:
        sr = np.nan

    sign_acc = float(
        np.mean(
            (predicted > 0)
            == (observed > 0)
        )
    )

    validity = pd.DataFrame([dict(
        pearson_r=pr,
        spearman_rho=sr,
        sign_accuracy=sign_acc,
        n_transfers=len(pred),
    )])

    validity.to_csv(
        out / "benefit_prediction_validity.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Regret
    # ---------------------------------------------------------
    regret = pred.regret_prauc.to_numpy(dtype=float)

    lo, hi = bootstrap_ci(
        regret,
        reps=args.bootstrap_reps,
        seed=30,
    )

    regret_summary = pd.DataFrame([dict(
        mean_regret=float(np.mean(regret)),
        median_regret=float(np.median(regret)),
        ci_low=lo,
        ci_high=hi,
        zero_regret_fraction=float(np.mean(regret == 0)),
        correct_direction_fraction=float(
            pred.policy_correct_direction.mean()
        ),
    )])

    regret_summary.to_csv(
        out / "regret_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Console
    # ---------------------------------------------------------
    print("\n=== POLICY PR-AUC SUMMARY ===")
    print(summary.to_string(index=False))

    print("\n=== PAIRED POLICY COMPARISONS ===")
    print(comparisons.to_string(index=False))

    print("\n=== BENEFIT PREDICTION VALIDITY ===")
    print(validity.to_string(index=False))

    print("\n=== REGRET SUMMARY ===")
    print(regret_summary.to_string(index=False))

    print("\n=== POLICY CHOICES ===")
    print(
        pred[
            [
                "transfer_id",
                "predicted_delta_prauc_hybrid_random",
                "observed_delta_prauc_hybrid_random",
                "policy_choice",
                "oracle_choice",
                "prauc_random",
                "prauc_hybrid",
                "prauc_policy",
                "prauc_oracle",
                "regret_prauc",
            ]
        ].to_string(index=False)
    )

    config = vars(args).copy()
    config["leakage_free_predictors"] = LEAKAGE_FREE_PREDICTORS
    config["policy"] = (
        "Outer LOTO; inner LOTO alpha selection; "
        "Hybrid if predicted delta PR-AUC > 0, else Random."
    )

    (out / "run_config.json").write_text(
        json.dumps(
            config,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print(f"\nOutputs: {out}")


if __name__ == "__main__":
    main()
