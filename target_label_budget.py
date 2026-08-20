#!/usr/bin/env python3
"""
target_label_budget.py
======================

Few-shot target-label budget analysis for cross-site MALDI-TOF AMR.

Question
--------
How much labeled target-site data are required to recover useful performance?

Scenarios
---------
1) source_only
   Train on all DRIAMS-A, test on held-out target.

2) target_only
   Train only on the labeled target adaptation subset, test on held-out target.

3) source_plus_target
   Train on all DRIAMS-A plus the labeled target adaptation subset,
   test on held-out target.

Protocol
--------
For each repetition:
  - split target patients/groups into adaptation pool and held-out test
  - within adaptation pool, sample label budgets:
      1%, 2.5%, 5%, 10%, 20%, 50%, 100%
  - preserve patient/group independence
  - repeat across seeds
  - save per-repetition metrics and aggregate mean/std/95% bootstrap CI

Target labels are never used in the held-out test split.

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

    xs, _, idx = spectra(mdir, site)
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


def fit_predict(Xtr, ytr, Xte, seed=0):
    m = lgb.LGBMClassifier(**CFG, random_state=seed)
    m.fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1]


def choose_test_split(y, g, seed=0, test_fraction=0.5):
    """
    Group-aware split. Uses StratifiedGroupKFold and chooses the fold whose
    size is closest to requested test_fraction.
    """
    n_splits = max(2, int(round(1 / test_fraction)))
    n_splits = min(n_splits, 5)

    sgkf = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    best = None
    best_err = np.inf

    for tr, te in sgkf.split(np.zeros(len(y)), y, g):
        frac = len(te) / len(y)
        err = abs(frac - test_fraction)
        if len(np.unique(y[te])) < 2:
            continue
        if len(np.unique(y[tr])) < 2:
            continue
        if err < best_err:
            best = (tr, te)
            best_err = err

    if best is None:
        raise ValueError("Could not create valid group-aware target split.")

    return best


def sample_budget_indices(y, g, fraction, seed=0, min_pos=2):
    """
    Sample whole groups from the adaptation pool until approximately the
    requested sample fraction is reached. Repeats attempts to retain both classes.
    """
    rng = np.random.default_rng(seed)

    groups = np.unique(g)
    target_n = max(10, int(round(fraction * len(y))))
    target_n = min(target_n, len(y))

    group_to_idx = {u: np.where(g == u)[0] for u in groups}

    best = None
    best_gap = np.inf

    for _ in range(200):
        order = rng.permutation(groups)
        chosen = []
        n = 0

        for u in order:
            chosen.append(u)
            n += len(group_to_idx[u])
            if n >= target_n:
                break

        idx = np.concatenate([group_to_idx[u] for u in chosen])
        yy = y[idx]

        if len(np.unique(yy)) < 2:
            continue
        if int((yy == 1).sum()) < min_pos:
            continue

        gap = abs(len(idx) - target_n)
        if gap < best_gap:
            best = idx
            best_gap = gap

        if gap == 0:
            break

    return best


def bootstrap_ci(x, reps=5000, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        return np.nan, np.nan

    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(reps, len(x)), replace=True).mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])

    return float(lo), float(hi)


def aggregate_results(df, bootstrap_reps=5000):
    rows = []

    group_cols = ["scenario", "budget_fraction"]

    for keys, g in df.groupby(group_cols, dropna=False):
        scenario, budget = keys

        row = dict(
            scenario=scenario,
            budget_fraction=budget,
            n_reps=len(g),
            mean_adapt_n=float(g.adapt_n.mean()) if "adapt_n" in g else np.nan,
            mean_adapt_pos=float(g.adapt_pos.mean()) if "adapt_pos" in g else np.nan,
        )

        for met in ["auroc", "prauc", "brier", "slope"]:
            vals = g[met].to_numpy(dtype=float)
            row[f"{met}_mean"] = np.nanmean(vals)
            row[f"{met}_std"] = np.nanstd(vals, ddof=1) if np.isfinite(vals).sum() > 1 else np.nan
            lo, hi = bootstrap_ci(vals, reps=bootstrap_reps, seed=0)
            row[f"{met}_ci_low"] = lo
            row[f"{met}_ci_high"] = hi

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ap.add_argument("--species", required=True)
    ap.add_argument("--drug", required=True)

    ap.add_argument("--source", default="DRIAMS-A")
    ap.add_argument("--target", default="DRIAMS-C")

    ap.add_argument("--labels", default="outputs/driams_long.parquet")
    ap.add_argument("--matrices", default="matrices")
    ap.add_argument("--root", default="~/data/DRIAMS")
    ap.add_argument("--out", default="outputs/target_label_budget")

    ap.add_argument(
        "--budgets",
        default="0.01,0.025,0.05,0.10,0.20,0.50,1.00"
    )

    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--test-fraction", type=float, default=0.50)
    ap.add_argument("--bootstrap-reps", type=int, default=5000)
    ap.add_argument("--min-pos", type=int, default=2)

    args = ap.parse_args()

    mdir = Path(args.matrices)
    root = Path(args.root).expanduser()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lab = pd.read_parquet(args.labels)

    src = load_task(
        lab, mdir, root,
        args.species, args.drug, args.source
    )

    tgt = load_task(
        lab, mdir, root,
        args.species, args.drug, args.target
    )

    if src is None or tgt is None:
        raise SystemExit("Source or target task unavailable.")

    if len(np.unique(tgt["y"])) < 2:
        raise SystemExit("Target has only one class.")

    budgets = [float(x) for x in args.budgets.split(",")]

    print("=== TARGET LABEL BUDGET ===")
    print(
        f"{args.species}/{args.drug}\n"
        f"source {args.source}: n={len(src['y'])}, R={src['y'].mean():.3f}\n"
        f"target {args.target}: n={len(tgt['y'])}, R={tgt['y'].mean():.3f}\n"
        f"reps={args.reps}, held-out target fraction={args.test_fraction}"
    )

    rows = []

    for rep in range(args.reps):
        adapt_ix, test_ix = choose_test_split(
            tgt["y"], tgt["g"],
            seed=rep,
            test_fraction=args.test_fraction,
        )

        Xa = tgt["X"][adapt_ix]
        ya = tgt["y"][adapt_ix]
        ga = tgt["g"][adapt_ix]

        Xte = tgt["X"][test_ix]
        yte = tgt["y"][test_ix]

        # source-only baseline evaluated on exactly the same held-out target
        p = fit_predict(
            src["X"], src["y"],
            Xte,
            seed=rep,
        )
        met = metrics(yte, p)
        rows.append(dict(
            rep=rep,
            scenario="source_only",
            budget_fraction=0.0,
            adapt_n=0,
            adapt_pos=0,
            test_n=len(yte),
            test_pos=int(yte.sum()),
            **met,
        ))

        for frac in budgets:
            if frac >= 1.0:
                bix = np.arange(len(ya))
            else:
                bix = sample_budget_indices(
                    ya, ga,
                    fraction=frac,
                    seed=10000 * rep + int(round(frac * 10000)),
                    min_pos=args.min_pos,
                )

            if bix is None or len(bix) < 10:
                rows.append(dict(
                    rep=rep,
                    scenario="budget_unavailable",
                    budget_fraction=frac,
                    adapt_n=0,
                    adapt_pos=0,
                    test_n=len(yte),
                    test_pos=int(yte.sum()),
                    auroc=np.nan,
                    prauc=np.nan,
                    brier=np.nan,
                    slope=np.nan,
                    n=len(yte),
                    positives=int(yte.sum()),
                    prevalence=float(yte.mean()),
                ))
                continue

            Xb = Xa[bix]
            yb = ya[bix]

            if len(np.unique(yb)) < 2:
                continue

            # target-only
            p_t = fit_predict(
                Xb, yb,
                Xte,
                seed=rep,
            )
            met_t = metrics(yte, p_t)
            rows.append(dict(
                rep=rep,
                scenario="target_only",
                budget_fraction=frac,
                adapt_n=len(yb),
                adapt_pos=int(yb.sum()),
                test_n=len(yte),
                test_pos=int(yte.sum()),
                **met_t,
            ))

            # source + target
            Xst = np.vstack([src["X"], Xb])
            yst = np.r_[src["y"], yb]

            p_st = fit_predict(
                Xst, yst,
                Xte,
                seed=rep,
            )
            met_st = metrics(yte, p_st)
            rows.append(dict(
                rep=rep,
                scenario="source_plus_target",
                budget_fraction=frac,
                adapt_n=len(yb),
                adapt_pos=int(yb.sum()),
                test_n=len(yte),
                test_pos=int(yte.sum()),
                **met_st,
            ))

        print(f"rep {rep+1}/{args.reps} complete", flush=True)

    raw = pd.DataFrame(rows)

    tag = (
        f"{args.species.replace(' ', '_')}__"
        f"{args.drug}__{args.source}_to_{args.target}"
    )

    raw_path = out / f"{tag}__raw.csv"
    raw.to_csv(raw_path, index=False)

    agg = aggregate_results(
        raw[raw.scenario != "budget_unavailable"].copy(),
        bootstrap_reps=args.bootstrap_reps,
    )

    agg_path = out / f"{tag}__summary.csv"
    agg.to_csv(agg_path, index=False)

    # delta vs source-only at repetition level
    source_ref = (
        raw[raw.scenario == "source_only"][
            ["rep", "auroc", "prauc", "brier"]
        ]
        .rename(columns={
            "auroc": "source_auroc",
            "prauc": "source_prauc",
            "brier": "source_brier",
        })
    )

    comp = raw[
        raw.scenario.isin(["target_only", "source_plus_target"])
    ].merge(source_ref, on="rep", how="left")

    comp["delta_auroc_vs_source"] = comp.auroc - comp.source_auroc
    comp["delta_prauc_vs_source"] = comp.prauc - comp.source_prauc
    comp["delta_brier_vs_source"] = comp.brier - comp.source_brier

    delta_rows = []

    for (scenario, budget), g in comp.groupby(
        ["scenario", "budget_fraction"]
    ):
        row = dict(
            scenario=scenario,
            budget_fraction=budget,
            n_reps=len(g),
        )

        for col in [
            "delta_auroc_vs_source",
            "delta_prauc_vs_source",
            "delta_brier_vs_source",
        ]:
            vals = g[col].to_numpy(dtype=float)
            row[f"{col}_mean"] = np.nanmean(vals)
            lo, hi = bootstrap_ci(
                vals,
                reps=args.bootstrap_reps,
                seed=1,
            )
            row[f"{col}_ci_low"] = lo
            row[f"{col}_ci_high"] = hi

        delta_rows.append(row)

    delta = pd.DataFrame(delta_rows)
    delta_path = out / f"{tag}__delta_vs_source.csv"
    delta.to_csv(delta_path, index=False)

    config = vars(args).copy()
    (out / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== SUMMARY ===")
    show_cols = [
        "scenario",
        "budget_fraction",
        "n_reps",
        "mean_adapt_n",
        "mean_adapt_pos",
        "auroc_mean",
        "prauc_mean",
        "brier_mean",
        "slope_mean",
    ]
    print(
        agg.sort_values(
            ["scenario", "budget_fraction"]
        )[show_cols].to_string(index=False)
    )

    print("\n=== DELTA VS SOURCE-ONLY ===")
    print(
        delta.sort_values(
            ["scenario", "budget_fraction"]
        ).to_string(index=False)
    )

    print(f"\nSaved raw: {raw_path}")
    print(f"Saved summary: {agg_path}")
    print(f"Saved deltas: {delta_path}")


if __name__ == "__main__":
    main()
