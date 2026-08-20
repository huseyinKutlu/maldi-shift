#!/usr/bin/env python3
"""
multitask_active_validation.py
==============================

Locked multi-task validation of label-efficient active target-site adaptation.

Purpose
-------
Validate whether the HYBRID selection strategy generalizes beyond one
species-drug-transfer pair.

Locked strategies
-----------------
1) source_only
2) random
3) hybrid = uncertainty + spectral diversity

Important:
- The HYBRID scoring rule is fixed and is NOT tuned per task.
- Target held-out test labels are never used for selection.
- The uncertainty baseline bug from the exploratory script is irrelevant here:
  this script evaluates only RANDOM and the already-correct HYBRID strategy.

Candidate task discovery
------------------------
Automatically finds species-drug pairs with:
- enough tested spectra in source and target
- both classes present
- minimum resistant counts in source and target

Default transfer set:
    A -> B
    A -> C
    B -> C
when data are available.

Budgets:
    20, 30, 50 labeled target samples

Outputs:
- task_inventory.csv
- raw_results.csv
- aggregate_results.csv
- paired_hybrid_vs_random.csv
- transfer_summary.csv
- overall_meta_summary.csv

Place next to nested_cv.py.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from nested_cv import load_spectra, gather_rows, patient_map, group_key

warnings.filterwarnings("ignore")

CFG = dict(
    objective="binary",
    num_leaves=31,
    n_estimators=300,
    learning_rate=0.05,
    colsample_bytree=0.3,
    subsample=0.8,
    subsample_freq=1,
    verbose=-1,
    n_jobs=12,
)

CACHE = {}


def spectra(mdir, site):
    if site not in CACHE:
        CACHE[site] = load_spectra(mdir, site)
    return CACHE[site]


def cal_slope(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    lo = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        m = LogisticRegression(
            penalty=None, solver="lbfgs", max_iter=1000
        ).fit(lo, y)
        return float(m.coef_[0, 0])
    except Exception:
        return np.nan


def metrics(y, p):
    return dict(
        auroc=float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        prauc=float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        brier=float(brier_score_loss(y, p)),
        slope=cal_slope(y, p),
        n=int(len(y)),
        positives=int(np.sum(y)),
        prevalence=float(np.mean(y)),
    )


def load_task(lab, mdir, root, species, drug, site):
    sel = lab[
        (lab.species == species)
        & (lab.drug == drug)
        & lab.tested
        & lab.has_spectrum
        & (lab.site == site)
    ].copy()

    if sel.empty:
        return None

    try:
        xs, _, idx = spectra(mdir, site)
    except FileNotFoundError:
        return None

    sel = sel[sel.code.isin(idx.keys())].copy()
    if sel.empty:
        return None

    X = gather_rows(xs, [idx[c] for c in sel.code]).astype(np.float32)
    y = sel.label_RI.to_numpy(dtype=int)

    try:
        pmap = patient_map(root, site)
        g = group_key(sel.code.to_numpy(), pmap, "patient")
    except Exception:
        g = sel.code.astype(str).to_numpy()

    return dict(
        df=sel.reset_index(drop=True),
        X=X,
        y=y,
        g=np.asarray(g),
    )


def fit_predict(Xtr, ytr, Xte, seed=0):
    mdl = lgb.LGBMClassifier(**CFG, random_state=seed)
    mdl.fit(Xtr, ytr)
    return mdl, mdl.predict_proba(Xte)[:, 1]


def fit_source_svd(Xs, ncomp=100, seed=0):
    mu = Xs.mean(axis=0, keepdims=True).astype(np.float32)
    nc = max(2, min(ncomp, min(Xs.shape) - 1))

    _, _, Vt = randomized_svd(
        Xs.astype(np.float64) - mu,
        n_components=nc,
        random_state=seed,
    )

    return mu, Vt.T.astype(np.float32)


def transform_svd(X, mu, P):
    return ((X - mu) @ P).astype(np.float32)


def normalize01(x):
    x = np.asarray(x, dtype=float)
    lo, hi = np.min(x), np.max(x)

    if hi - lo < 1e-12:
        return np.zeros_like(x)

    return (x - lo) / (hi - lo)


def select_hybrid(Z, p, budget, alpha=0.5):
    """
    Fixed hybrid criterion:
        alpha * uncertainty + (1-alpha) * diversity

    uncertainty = 1 - 2*|p-0.5|
    diversity is greedy minimum distance to already selected observations.
    """
    n = len(Z)
    budget = min(budget, n)

    if budget >= n:
        return np.arange(n)

    uncertainty = normalize01(1.0 - 2.0 * np.abs(p - 0.5))

    first = int(np.argmax(uncertainty))
    selected = [first]

    chosen = np.zeros(n, dtype=bool)
    chosen[first] = True

    min_dist = np.linalg.norm(Z - Z[first], axis=1)

    while len(selected) < budget:
        diversity = normalize01(min_dist)

        score = alpha * uncertainty + (1 - alpha) * diversity
        score[chosen] = -np.inf

        j = int(np.argmax(score))
        selected.append(j)
        chosen[j] = True

        d = np.linalg.norm(Z - Z[j], axis=1)
        min_dist = np.minimum(min_dist, d)

    return np.asarray(selected, dtype=int)


def choose_target_split(y, g, seed=0, test_fraction=0.5):
    n_splits = max(2, min(5, int(round(1.0 / test_fraction))))

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    best = None
    best_err = np.inf

    for tr, te in sgkf.split(np.zeros(len(y)), y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        if len(np.unique(y[te])) < 2:
            continue

        err = abs(len(te) / len(y) - test_fraction)

        if err < best_err:
            best = (tr, te)
            best_err = err

    if best is None:
        raise ValueError("Could not create valid patient/group-aware target split.")

    return best


def bootstrap_ci(x, reps=5000, seed=0):
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


def task_inventory(
    lab,
    source_sites,
    target_sites,
    min_source_n=300,
    min_target_n=150,
    min_source_pos=20,
    min_target_pos=10,
):
    z = lab[
        lab.tested & lab.has_spectrum
    ][["site", "species", "drug", "code", "label_RI"]].copy()

    counts = (
        z.groupby(["site", "species", "drug"])
        .agg(
            n=("code", "size"),
            positives=("label_RI", "sum"),
            prevalence=("label_RI", "mean"),
        )
        .reset_index()
    )

    rows = []

    for ss, ts in zip(source_sites, target_sites):
        a = counts[counts.site == ss].copy()
        b = counts[counts.site == ts].copy()

        m = a.merge(
            b,
            on=["species", "drug"],
            suffixes=("_source", "_target"),
        )

        m["source"] = ss
        m["target"] = ts

        m["eligible"] = (
            (m.n_source >= min_source_n)
            & (m.n_target >= min_target_n)
            & (m.positives_source >= min_source_pos)
            & (m.positives_target >= min_target_pos)
            & ((m.n_source - m.positives_source) >= min_source_pos)
            & ((m.n_target - m.positives_target) >= min_target_pos)
        )

        rows.append(m)

    if not rows:
        return pd.DataFrame()

    inv = pd.concat(rows, ignore_index=True)

    cols = [
        "source", "target",
        "species", "drug",
        "n_source", "positives_source", "prevalence_source",
        "n_target", "positives_target", "prevalence_target",
        "eligible",
    ]

    return inv[cols]


def aggregate_raw(raw, bootstrap_reps=5000):
    rows = []

    for keys, g in raw.groupby(
        ["source", "target", "species", "drug", "strategy", "budget_n"]
    ):
        source, target, species, drug, strategy, budget = keys

        row = dict(
            source=source,
            target=target,
            species=species,
            drug=drug,
            strategy=strategy,
            budget_n=int(budget),
            n_reps=len(g),
            selected_pos_mean=float(g.selected_pos.mean()),
            selected_pos_rate_mean=float(g.selected_pos_rate.mean()),
        )

        for met in ["auroc", "prauc", "brier", "slope"]:
            vals = g[met].to_numpy(dtype=float)

            row[f"{met}_mean"] = np.nanmean(vals)
            row[f"{met}_std"] = (
                np.nanstd(vals, ddof=1)
                if np.isfinite(vals).sum() > 1
                else np.nan
            )

            lo, hi = bootstrap_ci(
                vals,
                reps=bootstrap_reps,
                seed=0,
            )

            row[f"{met}_ci_low"] = lo
            row[f"{met}_ci_high"] = hi

        rows.append(row)

    return pd.DataFrame(rows)


def paired_hybrid_vs_random(raw, bootstrap_reps=5000):
    rows = []

    task_cols = ["source", "target", "species", "drug", "budget_n"]

    hybrid = raw[raw.strategy == "hybrid"].copy()
    random = raw[raw.strategy == "random"].copy()

    for keys, gh in hybrid.groupby(task_cols):
        source, target, species, drug, budget = keys

        gr = random[
            (random.source == source)
            & (random.target == target)
            & (random.species == species)
            & (random.drug == drug)
            & (random.budget_n == budget)
        ]

        m = gh.merge(
            gr,
            on="rep",
            suffixes=("_hybrid", "_random"),
        )

        if len(m) < 2:
            continue

        for met in ["auroc", "prauc", "brier", "selected_pos"]:
            d = (
                m[f"{met}_hybrid"].to_numpy(dtype=float)
                - m[f"{met}_random"].to_numpy(dtype=float)
            )

            lo, hi = bootstrap_ci(
                d,
                reps=bootstrap_reps,
                seed=1,
            )

            rows.append(dict(
                source=source,
                target=target,
                species=species,
                drug=drug,
                budget_n=int(budget),
                metric=met,
                n_pairs=len(m),
                hybrid_mean=float(m[f"{met}_hybrid"].mean()),
                random_mean=float(m[f"{met}_random"].mean()),
                delta_mean=float(np.nanmean(d)),
                delta_ci_low=lo,
                delta_ci_high=hi,
                significant=bool(
                    np.isfinite(lo)
                    and np.isfinite(hi)
                    and (lo > 0 or hi < 0)
                ),
            ))

    return pd.DataFrame(rows)


def transfer_summary(paired):
    p = paired[paired.metric == "prauc"].copy()

    rows = []

    for keys, g in p.groupby(["source", "target", "species", "drug"]):
        source, target, species, drug = keys

        rows.append(dict(
            source=source,
            target=target,
            species=species,
            drug=drug,
            n_budgets=len(g),
            hybrid_better_budgets=int((g.delta_mean > 0).sum()),
            hybrid_significant_budgets=int(
                ((g.delta_mean > 0) & g.significant).sum()
            ),
            mean_delta_prauc=float(g.delta_mean.mean()),
            median_delta_prauc=float(g.delta_mean.median()),
            min_delta_prauc=float(g.delta_mean.min()),
            max_delta_prauc=float(g.delta_mean.max()),
        ))

    return pd.DataFrame(rows)


def overall_meta(paired):
    p = paired[paired.metric == "prauc"].copy()

    rows = []

    for budget, g in p.groupby("budget_n"):
        vals = g.delta_mean.to_numpy(dtype=float)

        lo, hi = bootstrap_ci(
            vals,
            reps=10000,
            seed=int(budget),
        )

        rows.append(dict(
            budget_n=int(budget),
            n_transfers=len(g),
            mean_delta_prauc=float(np.nanmean(vals)),
            median_delta_prauc=float(np.nanmedian(vals)),
            ci_low=lo,
            ci_high=hi,
            fraction_hybrid_better=float(np.mean(vals > 0)),
            fraction_hybrid_significant=float(
                np.mean((g.delta_mean > 0) & g.significant)
            ),
        ))

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ap.add_argument(
        "--labels",
        default="outputs/driams_long.parquet",
    )
    ap.add_argument(
        "--matrices",
        default="matrices",
    )
    ap.add_argument(
        "--root",
        default="~/data/DRIAMS",
    )
    ap.add_argument(
        "--out",
        default="outputs/multitask_active_validation",
    )

    ap.add_argument(
        "--transfers",
        default="DRIAMS-A>DRIAMS-B,DRIAMS-A>DRIAMS-C,DRIAMS-B>DRIAMS-C",
    )

    ap.add_argument(
        "--budgets",
        default="20,30,50",
    )

    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--test-fraction", type=float, default=0.5)
    ap.add_argument("--ncomp", type=int, default=100)

    # Locked hybrid hyperparameter
    ap.add_argument("--hybrid-alpha", type=float, default=0.5)

    # task eligibility
    ap.add_argument("--min-source-n", type=int, default=300)
    ap.add_argument("--min-target-n", type=int, default=150)
    ap.add_argument("--min-source-pos", type=int, default=20)
    ap.add_argument("--min-target-pos", type=int, default=10)

    ap.add_argument("--max-tasks", type=int, default=20)
    ap.add_argument("--bootstrap-reps", type=int, default=5000)

    args = ap.parse_args()

    mdir = Path(args.matrices)
    root = Path(args.root).expanduser()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lab = pd.read_parquet(args.labels)

    transfer_pairs = []

    for item in args.transfers.split(","):
        s, t = item.strip().split(">")
        transfer_pairs.append((s, t))

    source_sites = [x[0] for x in transfer_pairs]
    target_sites = [x[1] for x in transfer_pairs]

    inv = task_inventory(
        lab,
        source_sites,
        target_sites,
        min_source_n=args.min_source_n,
        min_target_n=args.min_target_n,
        min_source_pos=args.min_source_pos,
        min_target_pos=args.min_target_pos,
    )

    inv.to_csv(out / "task_inventory.csv", index=False)

    eligible = inv[inv.eligible].copy()

    # Prioritize tasks with larger target positive count and adequate target size
    eligible = eligible.sort_values(
        ["positives_target", "n_target"],
        ascending=[False, False],
    )

    if args.max_tasks > 0:
        eligible = eligible.head(args.max_tasks)

    if eligible.empty:
        raise SystemExit("No eligible multi-task transfers found.")

    print("=== MULTITASK ACTIVE VALIDATION ===")
    print(f"Eligible tasks: {len(eligible)}")
    print(
        eligible[
            [
                "source", "target", "species", "drug",
                "n_source", "positives_source",
                "n_target", "positives_target",
            ]
        ].to_string(index=False)
    )

    budgets = [int(x) for x in args.budgets.split(",")]

    rows = []

    for task_id, r in eligible.reset_index(drop=True).iterrows():
        source = r.source
        target = r.target
        species = r.species
        drug = r.drug

        print(
            f"\n[{task_id+1}/{len(eligible)}] "
            f"{species} / {drug} | {source}->{target}",
            flush=True,
        )

        src = load_task(
            lab, mdir, root,
            species, drug, source,
        )
        tgt = load_task(
            lab, mdir, root,
            species, drug, target,
        )

        if src is None or tgt is None:
            print("  skipped: task data unavailable", flush=True)
            continue

        for rep in range(args.reps):
            try:
                adapt_ix, test_ix = choose_target_split(
                    tgt["y"],
                    tgt["g"],
                    seed=rep,
                    test_fraction=args.test_fraction,
                )
            except Exception as e:
                print(f"  rep {rep}: split failed: {e}", flush=True)
                continue

            Xa = tgt["X"][adapt_ix]
            ya = tgt["y"][adapt_ix]

            Xte = tgt["X"][test_ix]
            yte = tgt["y"][test_ix]

            source_model, p_source = fit_predict(
                src["X"], src["y"],
                Xte,
                seed=rep,
            )

            base_met = metrics(yte, p_source)

            rows.append(dict(
                task_id=task_id,
                source=source,
                target=target,
                species=species,
                drug=drug,
                rep=rep,
                strategy="source_only",
                budget_n=0,
                selected_pos=0,
                selected_pos_rate=0.0,
                **base_met,
            ))

            # Candidate uncertainty
            p_adapt = source_model.predict_proba(Xa)[:, 1]

            # Source-fitted geometry
            mu, P = fit_source_svd(
                src["X"],
                ncomp=args.ncomp,
                seed=rep,
            )

            Zs = transform_svd(src["X"], mu, P)
            Za = transform_svd(Xa, mu, P)

            scaler = StandardScaler().fit(Zs)

            Zs = scaler.transform(Zs).astype(np.float32)
            Za = scaler.transform(Za).astype(np.float32)

            rng = np.random.default_rng(
                task_id * 100000 + rep
            )

            for budget in budgets:
                if budget > len(Xa):
                    continue

                ix_random = rng.choice(
                    len(Xa),
                    size=budget,
                    replace=False,
                )

                ix_hybrid = select_hybrid(
                    Za,
                    p_adapt,
                    budget=budget,
                    alpha=args.hybrid_alpha,
                )

                for strategy, bix in [
                    ("random", ix_random),
                    ("hybrid", ix_hybrid),
                ]:
                    Xb = Xa[bix]
                    yb = ya[bix]

                    Xtr = np.vstack([
                        src["X"],
                        Xb,
                    ])
                    ytr = np.r_[
                        src["y"],
                        yb,
                    ]

                    _, p = fit_predict(
                        Xtr,
                        ytr,
                        Xte,
                        seed=rep,
                    )

                    met = metrics(yte, p)

                    rows.append(dict(
                        task_id=task_id,
                        source=source,
                        target=target,
                        species=species,
                        drug=drug,
                        rep=rep,
                        strategy=strategy,
                        budget_n=budget,
                        selected_pos=int(yb.sum()),
                        selected_pos_rate=float(yb.mean()),
                        **met,
                    ))

            print(
                f"  rep {rep+1}/{args.reps} complete",
                flush=True,
            )

    raw = pd.DataFrame(rows)
    raw.to_csv(out / "raw_results.csv", index=False)

    agg = aggregate_raw(
        raw,
        bootstrap_reps=args.bootstrap_reps,
    )
    agg.to_csv(
        out / "aggregate_results.csv",
        index=False,
    )

    paired = paired_hybrid_vs_random(
        raw,
        bootstrap_reps=args.bootstrap_reps,
    )
    paired.to_csv(
        out / "paired_hybrid_vs_random.csv",
        index=False,
    )

    transfer = transfer_summary(paired)
    transfer.to_csv(
        out / "transfer_summary.csv",
        index=False,
    )

    meta = overall_meta(paired)
    meta.to_csv(
        out / "overall_meta_summary.csv",
        index=False,
    )

    config = vars(args).copy()
    config["locked_hybrid_definition"] = (
        "0.5*uncertainty + 0.5*greedy spectral diversity"
    )

    (out / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== TRANSFER SUMMARY ===")
    print(
        transfer.sort_values(
            "mean_delta_prauc",
            ascending=False,
        ).to_string(index=False)
    )

    print("\n=== OVERALL META SUMMARY ===")
    print(meta.to_string(index=False))

    print("\n=== PAIRED PR-AUC: FIRST 30 ROWS ===")
    print(
        paired[
            paired.metric == "prauc"
        ].head(30).to_string(index=False)
    )

    print(f"\nOutputs: {out}")


if __name__ == "__main__":
    main()
