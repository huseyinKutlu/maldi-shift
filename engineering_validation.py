#!/usr/bin/env python3
"""
engineering_validation.py
=========================

Two-stage validation for engineering-inspired MALDI transfer methods.

Stage 1: Development / hyperparameter selection
    Source: DRIAMS-A
    Validation target: DRIAMS-B

Stage 2: Locked external test
    Source: DRIAMS-A
    External target: DRIAMS-C

Methods
-------
- diPLS-like prediction-aware latent alignment
- TOP-style orthogonal nuisance projection

Selection rule
--------------
Default:
    among candidates with AUROC >= (baseline_AUROC - max_auc_drop),
    choose the candidate with highest PR-AUC on DRIAMS-B.

Target C labels are NEVER used during hyperparameter selection.

Also computes domain AUROC before/after TOP projection to diagnose whether
reducing acquisition-domain separability is associated with AMR transfer.

Place next to nested_cv.py and engineering_transfer.py.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).parent))
from nested_cv import load_spectra, gather_rows

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

CFG_DOMAIN = dict(
    objective="binary",
    num_leaves=15,
    n_estimators=160,
    learning_rate=0.05,
    colsample_bytree=0.5,
    subsample=0.8,
    subsample_freq=1,
    verbose=-1,
    n_jobs=12,
)

CACHE = {}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def spectra(mdir, site):
    if site not in CACHE:
        CACHE[site] = load_spectra(mdir, site)
    return CACHE[site]


def load_task(lab, mdir, species, drug, site):
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

    return dict(df=sel.reset_index(drop=True), X=X, y=y)


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


def fit_source_svd(Xs, ncomp=120, seed=0):
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


def run_lgbm(Xs, ys, Xt, yt, seed=0):
    mdl = lgb.LGBMClassifier(**CFG, random_state=seed)
    mdl.fit(Xs, ys)
    p = mdl.predict_proba(Xt)[:, 1]
    return p, metrics(yt, p)


def domain_auc(Xs, Xt, seed=0, folds=3):
    X = np.vstack([Xs, Xt]).astype(np.float32)
    y = np.r_[
        np.zeros(len(Xs), dtype=int),
        np.ones(len(Xt), dtype=int)
    ]

    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=float)

    for f, (tr, te) in enumerate(skf.split(X, y)):
        mdl = lgb.LGBMClassifier(**CFG_DOMAIN, random_state=seed + f)
        mdl.fit(X[tr], y[tr])
        pred[te] = mdl.predict_proba(X[te])[:, 1]

    return float(roc_auc_score(y, pred))


# ---------------------------------------------------------------------
# diPLS-like
# ---------------------------------------------------------------------

def dipls_basis(Zs, ys, Zt, latent=8, lam=1.0, ridge=1e-3):
    y0 = ys.astype(float) - ys.mean()
    Z0 = Zs - Zs.mean(axis=0, keepdims=True)

    cy = (Z0.T @ y0.reshape(-1, 1)) / max(len(Z0) - 1, 1)
    Sy = cy @ cy.T

    dm = (Zs.mean(0) - Zt.mean(0)).reshape(-1, 1)
    Cs = np.cov(Zs, rowvar=False)
    Ct = np.cov(Zt, rowvar=False)
    Dc = Cs - Ct
    Sd = dm @ dm.T + (Dc @ Dc.T) / max(Zs.shape[1], 1)

    A = Sy + ridge * np.eye(Sy.shape[0])
    B = np.eye(Sy.shape[0]) + lam * Sd + ridge * np.eye(Sy.shape[0])

    vals, vecs = eigh(A, B)
    order = np.argsort(vals)[::-1]
    W = vecs[:, order[:max(1, min(latent, len(order)))]]
    return W.astype(np.float32)


def run_dipls(Xs, ys, Xt, yt, svdcomp=120, latent=8, lam=1.0, seed=0):
    mu, P = fit_source_svd(Xs, ncomp=svdcomp, seed=seed)
    Zs = transform_svd(Xs, mu, P)
    Zt = transform_svd(Xt, mu, P)

    sc = StandardScaler().fit(Zs)
    Zs = sc.transform(Zs).astype(np.float32)
    Zt = sc.transform(Zt).astype(np.float32)

    W = dipls_basis(Zs, ys, Zt, latent=latent, lam=lam)

    As = Zs @ W
    At = Zt @ W

    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(As, ys)
    p = clf.predict_proba(At)[:, 1]

    return p, metrics(yt, p)


# ---------------------------------------------------------------------
# TOP-style projection
# ---------------------------------------------------------------------

def nuisance_directions(Zs, Zt, k=5, seed=0):
    dm = (Zs.mean(0) - Zt.mean(0)).reshape(1, -1)
    Cdiff = np.cov(Zs, rowvar=False) - np.cov(Zt, rowvar=False)

    D = np.vstack([dm, Cdiff]).astype(np.float64)
    nc = max(1, min(k, min(D.shape) - 1))

    _, _, Vt = randomized_svd(
        D, n_components=nc, random_state=seed
    )
    return Vt.T.astype(np.float32)


def project_top(Z, V, rho):
    return (Z - rho * ((Z @ V) @ V.T)).astype(np.float32)


def run_top(
    Xs, ys, Xt, yt,
    svdcomp=120, k=5, rho=1.0, seed=0,
    compute_domain=True
):
    mu, P = fit_source_svd(Xs, ncomp=svdcomp, seed=seed)
    Zs = transform_svd(Xs, mu, P)
    Zt = transform_svd(Xt, mu, P)

    sc = StandardScaler().fit(Zs)
    Zs = sc.transform(Zs).astype(np.float32)
    Zt = sc.transform(Zt).astype(np.float32)

    dom_before = domain_auc(Zs, Zt, seed=seed) if compute_domain else np.nan

    V = nuisance_directions(Zs, Zt, k=k, seed=seed)
    Zsp = project_top(Zs, V, rho)
    Ztp = project_top(Zt, V, rho)

    dom_after = domain_auc(Zsp, Ztp, seed=seed) if compute_domain else np.nan

    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(Zsp, ys)
    p = clf.predict_proba(Ztp)[:, 1]

    met = metrics(yt, p)
    met["domain_auroc_before"] = dom_before
    met["domain_auroc_after"] = dom_after
    met["delta_domain_auroc"] = dom_after - dom_before

    return p, met


# ---------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------

def select_candidate(df, baseline_auc, max_auc_drop=0.03):
    eligible = df[df["auroc"] >= baseline_auc - max_auc_drop].copy()

    if eligible.empty:
        eligible = df.copy()

    eligible = eligible.sort_values(
        ["prauc", "auroc", "brier"],
        ascending=[False, False, True]
    )

    return eligible.iloc[0]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    ap.add_argument("--species", required=True)
    ap.add_argument("--drug", required=True)

    ap.add_argument("--dev-target", default="DRIAMS-B")
    ap.add_argument("--test-target", default="DRIAMS-C")

    ap.add_argument("--labels", default="outputs/driams_long.parquet")
    ap.add_argument("--matrices", default="matrices")
    ap.add_argument("--out", default="outputs/engineering_validation")

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-auc-drop", type=float, default=0.03)

    ap.add_argument("--dipls-svd", type=int, default=120)
    ap.add_argument("--dipls-latents", default="2,4,8,12,20,30")
    ap.add_argument("--dipls-lambdas", default="0,0.01,0.1,0.3,1,3,10,30,100")

    ap.add_argument("--top-svd", type=int, default=120)
    ap.add_argument("--top-ks", default="1,2,5,10,15,20,30,40,50")
    ap.add_argument("--top-rhos", default="0.25,0.5,0.75,1.0")

    args = ap.parse_args()

    mdir = Path(args.matrices)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lab = pd.read_parquet(args.labels)

    src = load_task(lab, mdir, args.species, args.drug, "DRIAMS-A")
    dev = load_task(lab, mdir, args.species, args.drug, args.dev_target)
    test = load_task(lab, mdir, args.species, args.drug, args.test_target)

    if src is None or dev is None or test is None:
        raise SystemExit("Source, development target, or test target unavailable.")

    if len(np.unique(dev["y"])) < 2 or len(np.unique(test["y"])) < 2:
        raise SystemExit("Development or test target is single-class.")

    print("=== ENGINEERING VALIDATION ===")
    print(
        f"{args.species}/{args.drug}\n"
        f"A source: n={len(src['y'])}, R={src['y'].mean():.3f}\n"
        f"{args.dev_target} dev: n={len(dev['y'])}, R={dev['y'].mean():.3f}\n"
        f"{args.test_target} test: n={len(test['y'])}, R={test['y'].mean():.3f}"
    )

    # -----------------------------------------------------------------
    # Baselines
    # -----------------------------------------------------------------
    _, dev_base = run_lgbm(
        src["X"], src["y"], dev["X"], dev["y"], seed=args.seed
    )
    _, test_base = run_lgbm(
        src["X"], src["y"], test["X"], test["y"], seed=args.seed
    )

    print(
        f"\nBaseline A->{args.dev_target}: "
        f"AUROC={dev_base['auroc']:.3f} PR={dev_base['prauc']:.3f}"
    )
    print(
        f"Baseline A->{args.test_target}: "
        f"AUROC={test_base['auroc']:.3f} PR={test_base['prauc']:.3f}"
    )

    # -----------------------------------------------------------------
    # diPLS grid on dev target
    # -----------------------------------------------------------------
    dipls_rows = []

    for latent in [int(x) for x in args.dipls_latents.split(",")]:
        for lam in [float(x) for x in args.dipls_lambdas.split(",")]:
            try:
                _, met = run_dipls(
                    src["X"], src["y"],
                    dev["X"], dev["y"],
                    svdcomp=args.dipls_svd,
                    latent=latent,
                    lam=lam,
                    seed=args.seed,
                )
                dipls_rows.append(dict(
                    method="dipls_like",
                    latent=latent,
                    lam=lam,
                    **met,
                ))
                print(
                    f"dev diPLS latent={latent:>2} lam={lam:>6}: "
                    f"AUROC={met['auroc']:.3f} PR={met['prauc']:.3f}"
                )
            except Exception as e:
                print(f"diPLS HATA latent={latent} lam={lam}: {e}")

    dipls_df = pd.DataFrame(dipls_rows)
    dipls_df.to_csv(
        out / f"dipls_dev_grid__{args.dev_target}.csv",
        index=False
    )

    best_dipls = select_candidate(
        dipls_df,
        baseline_auc=dev_base["auroc"],
        max_auc_drop=args.max_auc_drop,
    )

    print("\nSelected diPLS on dev:")
    print(best_dipls.to_string())

    # Locked test
    _, dipls_test = run_dipls(
        src["X"], src["y"],
        test["X"], test["y"],
        svdcomp=args.dipls_svd,
        latent=int(best_dipls["latent"]),
        lam=float(best_dipls["lam"]),
        seed=args.seed,
    )

    # -----------------------------------------------------------------
    # TOP grid on dev target
    # -----------------------------------------------------------------
    top_rows = []

    for k in [int(x) for x in args.top_ks.split(",")]:
        for rho in [float(x) for x in args.top_rhos.split(",")]:
            try:
                _, met = run_top(
                    src["X"], src["y"],
                    dev["X"], dev["y"],
                    svdcomp=args.top_svd,
                    k=k,
                    rho=rho,
                    seed=args.seed,
                    compute_domain=True,
                )
                top_rows.append(dict(
                    method="top_projection",
                    k=k,
                    rho=rho,
                    **met,
                ))
                print(
                    f"dev TOP k={k:>2} rho={rho:.2f}: "
                    f"AUROC={met['auroc']:.3f} PR={met['prauc']:.3f} "
                    f"domain={met['domain_auroc_before']:.3f}"
                    f"->{met['domain_auroc_after']:.3f}"
                )
            except Exception as e:
                print(f"TOP HATA k={k} rho={rho}: {e}")

    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(
        out / f"top_dev_grid__{args.dev_target}.csv",
        index=False
    )

    best_top = select_candidate(
        top_df,
        baseline_auc=dev_base["auroc"],
        max_auc_drop=args.max_auc_drop,
    )

    print("\nSelected TOP on dev:")
    print(best_top.to_string())

    # Locked test
    _, top_test = run_top(
        src["X"], src["y"],
        test["X"], test["y"],
        svdcomp=args.top_svd,
        k=int(best_top["k"]),
        rho=float(best_top["rho"]),
        seed=args.seed,
        compute_domain=True,
    )

    # -----------------------------------------------------------------
    # Locked summary
    # -----------------------------------------------------------------
    locked = pd.DataFrame([
        dict(
            method="lgbm_baseline",
            selected_on=args.dev_target,
            tested_on=args.test_target,
            auroc=test_base["auroc"],
            prauc=test_base["prauc"],
            brier=test_base["brier"],
            slope=test_base["slope"],
        ),
        dict(
            method="dipls_like_locked",
            selected_on=args.dev_target,
            tested_on=args.test_target,
            latent=int(best_dipls["latent"]),
            lam=float(best_dipls["lam"]),
            dev_auroc=float(best_dipls["auroc"]),
            dev_prauc=float(best_dipls["prauc"]),
            auroc=dipls_test["auroc"],
            prauc=dipls_test["prauc"],
            brier=dipls_test["brier"],
            slope=dipls_test["slope"],
        ),
        dict(
            method="top_projection_locked",
            selected_on=args.dev_target,
            tested_on=args.test_target,
            k=int(best_top["k"]),
            rho=float(best_top["rho"]),
            dev_auroc=float(best_top["auroc"]),
            dev_prauc=float(best_top["prauc"]),
            dev_domain_before=float(best_top["domain_auroc_before"]),
            dev_domain_after=float(best_top["domain_auroc_after"]),
            auroc=top_test["auroc"],
            prauc=top_test["prauc"],
            brier=top_test["brier"],
            slope=top_test["slope"],
            test_domain_before=top_test["domain_auroc_before"],
            test_domain_after=top_test["domain_auroc_after"],
        ),
    ])

    locked["delta_auroc_vs_baseline"] = locked["auroc"] - test_base["auroc"]
    locked["delta_prauc_vs_baseline"] = locked["prauc"] - test_base["prauc"]
    locked["delta_brier_vs_baseline"] = locked["brier"] - test_base["brier"]

    locked_path = out / (
        f"locked_summary__{args.species.replace(' ', '_')}__"
        f"{args.drug}__{args.dev_target}_to_{args.test_target}.csv"
    )
    locked.to_csv(locked_path, index=False)

    # Config / audit trail
    config = vars(args).copy()
    config["selection_rule"] = (
        "max PR-AUC among candidates with AUROC >= "
        f"baseline_dev_AUROC - {args.max_auc_drop}"
    )
    (out / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\n=== LOCKED EXTERNAL TEST ===")
    print(locked.to_string(index=False))
    print(f"\nSaved: {locked_path}")


if __name__ == "__main__":
    main()
