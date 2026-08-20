#!/usr/bin/env python3
"""
shift_mechanism.py
==================

Mechanistic analyses for cross-site MALDI-TOF AMR transfer in DRIAMS.

Designed to plug into the existing maldi-shift project and its nested_cv.py helpers:
    load_spectra(mdir, site)
    gather_rows(xs, row_indices)
    patient_map(root, site)
    group_key(codes, patient_map_dict, fallback)

Analyses
--------
1) temporal_site
   - Same-site temporal transfer (e.g. A 2017 -> A 2018)
   - Same-period cross-site transfer (e.g. A 2018 -> C 2018)
   - Uses acquisition_date, not folder/year labels.

2) ast_selection
   - Models whether the requested drug was tested.
   - Quantifies site-specific AST selection / verification policy.
   - Uses other-drug test indicators and, where available, other-drug RI labels.

3) support
   - Source-support score for every target isolate via k-NN distance in a source-fitted SVD space.
   - Reports AMR performance by high/medium/low support strata.
   - Produces source-vs-target domain AUROC.

4) conditional_shift
   - Measures source-target separability separately among resistant and susceptible isolates.
   - Quantifies whether P(X|Y=R) shifts more than P(X|Y=S).

5) lineage
   - Unsupervised within-species spectral clustering.
   - Compares cluster composition by site and resistance rate by cluster/site.
   - Evaluates transfer performance inside sufficiently populated spectral clusters.

6) density_weight
   - Cross-fitted domain-propensity / density-ratio weighting.
   - Reweights source isolates that look more target-like without altering spectra.

7) invariant_peaks
   - Finds m/z bins whose AMR association has consistent sign across sites.
   - Trains source LightGBM using only cross-site-stable bins.

8) selective
   - Risk/coverage style selective prediction using:
       a) source-support distance
       b) predictive confidence
       c) both
   - Useful when global transport fails but a transferable subset exists.

Outputs are CSV files under --out.

Important
---------
- This script is diagnostic first. It does not assume "site" == "instrument".
  Site/acquisition domain may include instrument, calibration, preparation,
  workflow, time, population, lineage composition, and AST policy.
- External target labels are used only for evaluation except where an analysis
  explicitly diagnoses conditional shift / lineage composition.
"""

import argparse
import json
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).parent))
from nested_cv import load_spectra, gather_rows, patient_map, group_key

warnings.filterwarnings("ignore")

SITES = ["DRIAMS-A", "DRIAMS-B", "DRIAMS-C", "DRIAMS-D"]
CACHE = {}

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
    n_estimators=180,
    learning_rate=0.05,
    colsample_bytree=0.5,
    subsample=0.8,
    subsample_freq=1,
    verbose=-1,
    n_jobs=12,
)


# ---------------------------------------------------------------------
# Core utilities
# ---------------------------------------------------------------------

def spectra(mdir, site):
    if site not in CACHE:
        CACHE[site] = load_spectra(mdir, site)
    return CACHE[site]


def safe_auc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, p))


def safe_prauc(y, p):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, p))


def cal_slope(y, p):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return np.nan
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        m = LogisticRegression(
            penalty=None, solver="lbfgs", max_iter=1000
        ).fit(logit, y)
        return float(m.coef_[0, 0])
    except Exception:
        return np.nan


def metrics(y, p):
    return dict(
        auroc=safe_auc(y, p),
        prauc=safe_prauc(y, p),
        brier=float(brier_score_loss(y, p)) if len(y) else np.nan,
        slope=cal_slope(y, p),
        n=int(len(y)),
        positives=int(np.sum(y)),
        prevalence=float(np.mean(y)) if len(y) else np.nan,
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

    xs, _, idx = spectra(mdir, site)
    sel = sel[sel.code.isin(idx.keys())].copy()
    if sel.empty:
        return None

    X = gather_rows(xs, [idx[c] for c in sel.code])
    y = sel.label_RI.to_numpy(dtype=int)

    try:
        pmap = patient_map(root, site)
        g = group_key(sel.code.to_numpy(), pmap, "patient")
    except Exception:
        g = sel.code.astype(str).to_numpy()

    return dict(df=sel.reset_index(drop=True), X=X, y=y, g=np.asarray(g))


def fit_source_svd(Xs, ncomp=100, seed=0):
    mu = Xs.mean(axis=0, keepdims=True)
    ncomp_eff = max(2, min(ncomp, min(Xs.shape) - 1))
    _, _, Vt = randomized_svd(
        Xs.astype(np.float64) - mu,
        n_components=ncomp_eff,
        random_state=seed,
    )
    P = Vt.T.astype(np.float32)
    return mu.astype(np.float32), P


def transform_svd(X, mu, P):
    return ((X - mu) @ P).astype(np.float32)


def grouped_internal_cv(X, y, g, seed=0, folds=5, sample_weight=None):
    rows = []
    splitter = StratifiedGroupKFold(
        folds, shuffle=True, random_state=seed
    )
    for fold, (tr, te) in enumerate(splitter.split(X, y, g)):
        m = lgb.LGBMClassifier(**CFG, random_state=seed + fold)
        kw = {}
        if sample_weight is not None:
            kw["sample_weight"] = sample_weight[tr]
        m.fit(X[tr], y[tr], **kw)
        p = m.predict_proba(X[te])[:, 1]
        r = metrics(y[te], p)
        r["fold"] = fold
        rows.append(r)
    return pd.DataFrame(rows)


def train_eval_source_target(Xs, ys, Xt, yt, seed=0, sample_weight=None):
    m = lgb.LGBMClassifier(**CFG, random_state=seed)
    kw = {}
    if sample_weight is not None:
        kw["sample_weight"] = sample_weight
    m.fit(Xs, ys, **kw)
    p = m.predict_proba(Xt)[:, 1]
    return m, p, metrics(yt, p)


def domain_auc(Xs, Xt, seed=0, folds=3):
    X = np.vstack([Xs, Xt]).astype(np.float32)
    y = np.r_[np.zeros(len(Xs), dtype=int), np.ones(len(Xt), dtype=int)]
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=np.float64)
    for f, (tr, te) in enumerate(skf.split(X, y)):
        m = lgb.LGBMClassifier(**CFG_DOMAIN, random_state=seed + f)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return float(roc_auc_score(y, pred))


def date_series(df):
    candidates = [
        "acquisition_date", "date", "measurement_date",
        "collection_date", "sample_date"
    ]
    for col in candidates:
        if col in df.columns:
            x = pd.to_datetime(df[col], errors="coerce")
            if x.notna().sum() > 0:
                return x, col
    raise ValueError(
        "No usable acquisition date column found. Tried: "
        + ", ".join(candidates)
    )


def bootstrap_ci(values, reps=5000, seed=0):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    b = rng.choice(x, size=(reps, len(x)), replace=True).mean(axis=1)
    return tuple(np.percentile(b, [2.5, 97.5]))


# ---------------------------------------------------------------------
# 1) Temporal vs site shift
# ---------------------------------------------------------------------

def analysis_temporal_site(lab, mdir, root, species, drug, out, seed=0):
    src = load_task(lab, mdir, root, species, drug, "DRIAMS-A")
    if src is None:
        raise ValueError("DRIAMS-A source task unavailable.")

    dates, date_col = date_series(src["df"])
    src["df"]["_date"] = dates
    years = sorted(src["df"]["_date"].dropna().dt.year.unique().tolist())

    rows = []

    # Same-site consecutive-year transfer
    for y0, y1 in zip(years[:-1], years[1:]):
        trix = np.where(src["df"]["_date"].dt.year.to_numpy() == y0)[0]
        teix = np.where(src["df"]["_date"].dt.year.to_numpy() == y1)[0]
        if len(trix) < 100 or len(teix) < 100:
            continue
        if len(np.unique(src["y"][trix])) < 2 or len(np.unique(src["y"][teix])) < 2:
            continue
        _, p, met = train_eval_source_target(
            src["X"][trix], src["y"][trix],
            src["X"][teix], src["y"][teix], seed=seed
        )
        rows.append(dict(
            comparison="temporal",
            source_site="DRIAMS-A",
            target_site="DRIAMS-A",
            source_period=str(y0),
            target_period=str(y1),
            date_column=date_col,
            **met,
        ))

    # Same-period A -> B/C/D; use overlapping calendar year(s)
    for target in SITES[1:]:
        tgt = load_task(lab, mdir, root, species, drug, target)
        if tgt is None or len(np.unique(tgt["y"])) < 2:
            continue
        td, tcol = date_series(tgt["df"])
        tgt["df"]["_date"] = td
        common_years = sorted(
            set(src["df"]["_date"].dropna().dt.year.unique())
            & set(tgt["df"]["_date"].dropna().dt.year.unique())
        )
        for yr in common_years:
            si = np.where(src["df"]["_date"].dt.year.to_numpy() == yr)[0]
            ti = np.where(tgt["df"]["_date"].dt.year.to_numpy() == yr)[0]
            if len(si) < 100 or len(ti) < 50:
                continue
            if len(np.unique(src["y"][si])) < 2 or len(np.unique(tgt["y"][ti])) < 2:
                continue
            _, p, met = train_eval_source_target(
                src["X"][si], src["y"][si],
                tgt["X"][ti], tgt["y"][ti], seed=seed
            )
            rows.append(dict(
                comparison="same_year_site",
                source_site="DRIAMS-A",
                target_site=target,
                source_period=str(yr),
                target_period=str(yr),
                date_column=f"{date_col}|{tcol}",
                **met,
            ))

    df = pd.DataFrame(rows)
    df.to_csv(out / "temporal_vs_site.csv", index=False)
    return df


# ---------------------------------------------------------------------
# 2) AST testing propensity / selection mechanism
# ---------------------------------------------------------------------

def build_ast_selection_table(lab, species, target_drug):
    """
    One row per isolate/code/site for the selected species.
    Outcome: whether target_drug was tested.
    Predictors:
      - site
      - other-drug tested indicators
      - other-drug RI labels where available
    """
    z = lab[(lab.species == species) & lab.has_spectrum].copy()
    if z.empty:
        raise ValueError("No rows for species.")

    keys = ["site", "code"]

    tested_wide = z.pivot_table(
        index=keys,
        columns="drug",
        values="tested",
        aggfunc="max",
        fill_value=False,
    ).astype(int)
    tested_wide.columns = [f"tested__{c}" for c in tested_wide.columns]

    # RI label only among tested observations
    z2 = z.copy()
    z2["_ri"] = np.where(z2["tested"], z2["label_RI"], np.nan)
    ri_wide = z2.pivot_table(
        index=keys,
        columns="drug",
        values="_ri",
        aggfunc="first",
    )
    ri_wide.columns = [f"ri__{c}" for c in ri_wide.columns]

    df = tested_wide.join(ri_wide, how="outer").reset_index()

    ycol = f"tested__{target_drug}"
    if ycol not in df.columns:
        raise ValueError(f"Target drug {target_drug!r} absent from tested matrix.")
    df["outcome_tested"] = df[ycol].astype(int)

    # Remove target-drug-derived predictors
    drop_cols = [ycol, f"ri__{target_drug}"]
    Xdf = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    return df, Xdf


def analysis_ast_selection(lab, species, drug, out, seed=0):
    df, Xdf = build_ast_selection_table(lab, species, drug)

    summary = (
        df.groupby("site")["outcome_tested"]
        .agg(["size", "sum", "mean"])
        .rename(columns={"size": "n_isolates", "sum": "n_tested", "mean": "tested_rate"})
        .reset_index()
    )
    summary.to_csv(out / "ast_selection_rates.csv", index=False)

    # One-hot site; numeric other AST predictors, missing RI -> -1 plus missing flags.
    y = df["outcome_tested"].to_numpy(dtype=int)
    feat = Xdf.drop(columns=["code"], errors="ignore").copy()
    feat = pd.get_dummies(feat, columns=["site"], drop_first=False)

    for c in feat.columns:
        if feat[c].dtype == bool:
            feat[c] = feat[c].astype(int)
        if feat[c].isna().any():
            feat[c + "__missing"] = feat[c].isna().astype(int)
            feat[c] = feat[c].fillna(-1)

    X = feat.to_numpy(dtype=np.float32)

    # Cross-validated propensity AUROC
    if len(np.unique(y)) < 2:
        perf = pd.DataFrame([dict(auroc=np.nan, prauc=np.nan, n=len(y))])
    else:
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        pred = np.zeros(len(y), dtype=float)
        for f, (tr, te) in enumerate(skf.split(X, y)):
            m = lgb.LGBMClassifier(**CFG_DOMAIN, random_state=seed + f)
            m.fit(X[tr], y[tr])
            pred[te] = m.predict_proba(X[te])[:, 1]
        perf = pd.DataFrame([dict(
            auroc=roc_auc_score(y, pred),
            prauc=average_precision_score(y, pred),
            n=len(y),
            prevalence=y.mean(),
            n_features=X.shape[1],
        )])

    perf.to_csv(out / "ast_selection_model.csv", index=False)

    # Site-specific pairwise test-rate differences
    pair_rows = []
    rates = summary.set_index("site")
    sites = rates.index.tolist()
    for i, a in enumerate(sites):
        for b in sites[i + 1:]:
            pair_rows.append(dict(
                site_a=a,
                site_b=b,
                tested_rate_a=rates.loc[a, "tested_rate"],
                tested_rate_b=rates.loc[b, "tested_rate"],
                absolute_difference=abs(
                    rates.loc[a, "tested_rate"] - rates.loc[b, "tested_rate"]
                ),
            ))
    pd.DataFrame(pair_rows).to_csv(out / "ast_selection_site_differences.csv", index=False)

    return summary, perf


# ---------------------------------------------------------------------
# 3) Common support / overlap
# ---------------------------------------------------------------------

def support_distances(Zs, Zt, k=10):
    k_eff = max(1, min(k, len(Zs)))
    nn = NearestNeighbors(n_neighbors=k_eff, metric="euclidean", n_jobs=-1)
    nn.fit(Zs)
    d, _ = nn.kneighbors(Zt)
    return d.mean(axis=1)


def analysis_support(lab, mdir, root, species, drug, target, out,
                     ncomp=100, k=10, seed=0):
    src = load_task(lab, mdir, root, species, drug, "DRIAMS-A")
    tgt = load_task(lab, mdir, root, species, drug, target)
    if src is None or tgt is None:
        raise ValueError("Source or target task unavailable.")

    mu, P = fit_source_svd(src["X"], ncomp=ncomp, seed=seed)
    Zs = transform_svd(src["X"], mu, P)
    Zt = transform_svd(tgt["X"], mu, P)

    # Base model in original 6000-d space
    model, p, met = train_eval_source_target(
        src["X"], src["y"], tgt["X"], tgt["y"], seed=seed
    )

    d = support_distances(Zs, Zt, k=k)
    q1, q2 = np.quantile(d, [1/3, 2/3])
    strata = np.where(d <= q1, "high_support",
             np.where(d <= q2, "medium_support", "low_support"))

    iso = pd.DataFrame({
        "code": tgt["df"]["code"].astype(str).to_numpy(),
        "site": target,
        "y": tgt["y"],
        "p": p,
        "support_distance": d,
        "support_stratum": strata,
    })
    iso.to_csv(out / f"support_isolates__{target}.csv", index=False)

    rows = [dict(stratum="all", **met)]
    for st in ["high_support", "medium_support", "low_support"]:
        ii = np.where(strata == st)[0]
        if len(ii) == 0:
            continue
        rows.append(dict(stratum=st, **metrics(tgt["y"][ii], p[ii])))

    dom = domain_auc(Zs, Zt, seed=seed)
    df = pd.DataFrame(rows)
    df["domain_auroc_svd"] = dom
    df["knn_k"] = k
    df["ncomp"] = Zs.shape[1]
    df.to_csv(out / f"support_summary__{target}.csv", index=False)
    return df, iso


# ---------------------------------------------------------------------
# 4) Conditional shift P(X|Y)
# ---------------------------------------------------------------------

def centroid_distance(A, B):
    return float(np.linalg.norm(A.mean(axis=0) - B.mean(axis=0)))


def analysis_conditional_shift(lab, mdir, root, species, drug, target, out,
                               ncomp=100, seed=0):
    src = load_task(lab, mdir, root, species, drug, "DRIAMS-A")
    tgt = load_task(lab, mdir, root, species, drug, target)
    if src is None or tgt is None:
        raise ValueError("Source or target unavailable.")

    mu, P = fit_source_svd(src["X"], ncomp=ncomp, seed=seed)
    Zs = transform_svd(src["X"], mu, P)
    Zt = transform_svd(tgt["X"], mu, P)

    rows = []
    for cls, name in [(0, "susceptible"), (1, "resistant")]:
        A = Zs[src["y"] == cls]
        B = Zt[tgt["y"] == cls]
        if len(A) < 40 or len(B) < 40:
            rows.append(dict(
                class_label=cls, class_name=name,
                n_source=len(A), n_target=len(B),
                domain_auroc=np.nan, centroid_distance=np.nan,
            ))
            continue
        rows.append(dict(
            class_label=cls,
            class_name=name,
            n_source=len(A),
            n_target=len(B),
            domain_auroc=domain_auc(A, B, seed=seed),
            centroid_distance=centroid_distance(A, B),
        ))

    df = pd.DataFrame(rows)
    df.to_csv(out / f"conditional_shift__{target}.csv", index=False)
    return df


# ---------------------------------------------------------------------
# 5) Spectral lineage / mixture shift
# ---------------------------------------------------------------------

def analysis_lineage(lab, mdir, root, species, drug, target, out,
                     ncomp=100, n_clusters=8, seed=0):
    src = load_task(lab, mdir, root, species, drug, "DRIAMS-A")
    tgt = load_task(lab, mdir, root, species, drug, target)
    if src is None or tgt is None:
        raise ValueError("Source or target unavailable.")

    # Source-fitted representation to avoid target-defined axes.
    mu, P = fit_source_svd(src["X"], ncomp=ncomp, seed=seed)
    Zs = transform_svd(src["X"], mu, P)
    Zt = transform_svd(tgt["X"], mu, P)

    Zall = np.vstack([Zs, Zt])
    scaler = StandardScaler().fit(Zall)
    Zall_s = scaler.transform(Zall)

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=seed,
        batch_size=512,
        n_init=20,
    )
    cl = km.fit_predict(Zall_s)
    cs = cl[:len(Zs)]
    ct = cl[len(Zs):]

    comp_rows = []
    for site, cvec, yvec in [
        ("DRIAMS-A", cs, src["y"]),
        (target, ct, tgt["y"]),
    ]:
        for k in range(n_clusters):
            ii = cvec == k
            comp_rows.append(dict(
                site=site,
                cluster=k,
                n=int(ii.sum()),
                fraction=float(ii.mean()),
                resistance_rate=float(yvec[ii].mean()) if ii.sum() else np.nan,
            ))
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(out / f"lineage_composition__{target}.csv", index=False)

    # Jensen-Shannon divergence of cluster composition
    pa = np.bincount(cs, minlength=n_clusters).astype(float) + 1e-9
    pb = np.bincount(ct, minlength=n_clusters).astype(float) + 1e-9
    pa /= pa.sum()
    pb /= pb.sum()
    m = 0.5 * (pa + pb)
    js = 0.5 * np.sum(pa * np.log(pa / m)) + 0.5 * np.sum(pb * np.log(pb / m))

    # Cluster-specific transfer
    transfer_rows = []
    for k in range(n_clusters):
        ia = np.where(cs == k)[0]
        ib = np.where(ct == k)[0]
        if len(ia) < 80 or len(ib) < 40:
            continue
        if len(np.unique(src["y"][ia])) < 2 or len(np.unique(tgt["y"][ib])) < 2:
            continue
        _, p, met = train_eval_source_target(
            src["X"][ia], src["y"][ia],
            tgt["X"][ib], tgt["y"][ib],
            seed=seed,
        )
        transfer_rows.append(dict(cluster=k, js_divergence=js, **met))

    trdf = pd.DataFrame(transfer_rows)
    trdf.to_csv(out / f"lineage_transfer__{target}.csv", index=False)
    pd.DataFrame([dict(
        target=target,
        n_clusters=n_clusters,
        js_divergence=js,
    )]).to_csv(out / f"lineage_js__{target}.csv", index=False)

    return comp, trdf


# ---------------------------------------------------------------------
# 6) Density-ratio / propensity weighting
# ---------------------------------------------------------------------

def crossfit_domain_probability(Zs, Zt, seed=0, folds=5):
    X = np.vstack([Zs, Zt]).astype(np.float32)
    d = np.r_[np.zeros(len(Zs), dtype=int), np.ones(len(Zt), dtype=int)]
    pred = np.zeros(len(d), dtype=float)

    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for f, (tr, te) in enumerate(skf.split(X, d)):
        m = lgb.LGBMClassifier(**CFG_DOMAIN, random_state=seed + f)
        m.fit(X[tr], d[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]

    return pred[:len(Zs)], pred[len(Zs):], roc_auc_score(d, pred)


def density_weights(q_source, n_source, n_target, clip=10.0):
    """
    If q(x)=P(domain=target | x) under pooled sample proportions:
        p_t(x)/p_s(x) ≈ [q/(1-q)] * [n_source/n_target]
    """
    q = np.clip(q_source, 1e-4, 1 - 1e-4)
    w = (q / (1 - q)) * (n_source / n_target)
    w = np.clip(w, 0, clip)
    # normalize mean weight to 1 for stable LightGBM optimization
    if np.mean(w) > 0:
        w = w / np.mean(w)
    return w.astype(np.float32)


def analysis_density_weight(lab, mdir, root, species, drug, target, out,
                            ncomp=100, clips=(2, 5, 10, 20), seed=0):
    src = load_task(lab, mdir, root, species, drug, "DRIAMS-A")
    tgt = load_task(lab, mdir, root, species, drug, target)
    if src is None or tgt is None:
        raise ValueError("Source or target unavailable.")

    mu, P = fit_source_svd(src["X"], ncomp=ncomp, seed=seed)
    Zs = transform_svd(src["X"], mu, P)
    Zt = transform_svd(tgt["X"], mu, P)

    qs, qt, dauc = crossfit_domain_probability(Zs, Zt, seed=seed)

    rows = []
    _, p0, m0 = train_eval_source_target(
        src["X"], src["y"], tgt["X"], tgt["y"], seed=seed
    )
    rows.append(dict(method="unweighted", clip=np.nan, domain_auroc=dauc, **m0))

    weight_rows = []
    for cl in clips:
        w = density_weights(qs, len(Zs), len(Zt), clip=cl)
        _, p, met = train_eval_source_target(
            src["X"], src["y"], tgt["X"], tgt["y"],
            seed=seed, sample_weight=w
        )
        ess = (w.sum() ** 2) / np.sum(w ** 2)
        rows.append(dict(
            method="density_ratio",
            clip=float(cl),
            domain_auroc=dauc,
            effective_sample_size=float(ess),
            weight_p99=float(np.quantile(w, 0.99)),
            **met,
        ))
        weight_rows.append(pd.DataFrame({
            "code": src["df"]["code"].astype(str),
            "target": target,
            "clip": cl,
            "q_target": qs,
            "weight": w,
            "y": src["y"],
        }))

    df = pd.DataFrame(rows)
    df.to_csv(out / f"density_weight__{target}.csv", index=False)
    if weight_rows:
        pd.concat(weight_rows, ignore_index=True).to_csv(
            out / f"density_weight_isolates__{target}.csv", index=False
        )
    return df


# ---------------------------------------------------------------------
# 7) Cross-site invariant peak selection
# ---------------------------------------------------------------------

def standardized_mean_diff(X, y):
    y = np.asarray(y)
    A = X[y == 1]
    B = X[y == 0]
    if len(A) < 5 or len(B) < 5:
        return np.full(X.shape[1], np.nan, dtype=np.float32)
    m1 = A.mean(axis=0)
    m0 = B.mean(axis=0)
    v1 = A.var(axis=0, ddof=1)
    v0 = B.var(axis=0, ddof=1)
    sp = np.sqrt(((len(A) - 1) * v1 + (len(B) - 1) * v0)
                 / max(len(A) + len(B) - 2, 1))
    return ((m1 - m0) / (sp + 1e-8)).astype(np.float32)


def analysis_invariant_peaks(lab, mdir, root, species, drug, target, out,
                             min_per_class=30, topk=100, seed=0):
    site_data = {}
    effects = {}

    for site in SITES:
        d = load_task(lab, mdir, root, species, drug, site)
        if d is None:
            continue
        if (d["y"] == 1).sum() < min_per_class or (d["y"] == 0).sum() < min_per_class:
            continue
        site_data[site] = d
        effects[site] = standardized_mean_diff(d["X"], d["y"])

    if "DRIAMS-A" not in site_data or target not in site_data:
        raise ValueError("Need source and target with enough observations per class.")

    sites = list(effects)
    E = np.vstack([effects[s] for s in sites])  # site x bins
    signs = np.sign(E)

    all_pos = np.all(signs > 0, axis=0)
    all_neg = np.all(signs < 0, axis=0)
    consistent = all_pos | all_neg
    med_abs = np.nanmedian(np.abs(E), axis=0)

    rank = np.argsort(np.where(consistent, med_abs, -np.inf))[::-1]
    rank = rank[np.isfinite(np.where(consistent[rank], med_abs[rank], np.nan))]
    selected = rank[:min(topk, len(rank))]

    # Approximate m/z from validated 3 Da/bin grid.
    mz = 2000.11 + 3.0 * np.arange(E.shape[1])

    peak_df = pd.DataFrame({
        "bin": np.arange(E.shape[1]),
        "mz_approx": mz,
        "consistent_sign": consistent,
        "median_abs_smd": med_abs,
    })
    for i, s in enumerate(sites):
        peak_df[f"smd__{s}"] = E[i]
    peak_df["selected"] = False
    peak_df.loc[selected, "selected"] = True
    peak_df.sort_values(["selected", "median_abs_smd"], ascending=[False, False]).to_csv(
        out / f"invariant_peaks__{target}.csv", index=False
    )

    src = site_data["DRIAMS-A"]
    tgt = site_data[target]

    rows = []
    # Full baseline
    _, p0, m0 = train_eval_source_target(
        src["X"], src["y"], tgt["X"], tgt["y"], seed=seed
    )
    rows.append(dict(feature_set="all_6000", n_features=src["X"].shape[1], **m0))

    if len(selected) >= 2:
        _, p1, m1 = train_eval_source_target(
            src["X"][:, selected], src["y"],
            tgt["X"][:, selected], tgt["y"],
            seed=seed
        )
        rows.append(dict(
            feature_set="cross_site_invariant",
            n_features=len(selected),
            **m1,
        ))

    df = pd.DataFrame(rows)
    df.to_csv(out / f"invariant_peak_performance__{target}.csv", index=False)
    return df, peak_df


# ---------------------------------------------------------------------
# 8) Selective prediction / abstention
# ---------------------------------------------------------------------

def ppv_npv(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    tp = np.sum((pred == 1) & (y == 1))
    tn = np.sum((pred == 0) & (y == 0))
    fp = np.sum((pred == 1) & (y == 0))
    fn = np.sum((pred == 0) & (y == 1))
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    return ppv, npv


def analysis_selective(lab, mdir, root, species, drug, target, out,
                       ncomp=100, k=10, seed=0):
    src = load_task(lab, mdir, root, species, drug, "DRIAMS-A")
    tgt = load_task(lab, mdir, root, species, drug, target)
    if src is None or tgt is None:
        raise ValueError("Source or target unavailable.")

    model, p, _ = train_eval_source_target(
        src["X"], src["y"], tgt["X"], tgt["y"], seed=seed
    )

    mu, P = fit_source_svd(src["X"], ncomp=ncomp, seed=seed)
    Zs = transform_svd(src["X"], mu, P)
    Zt = transform_svd(tgt["X"], mu, P)
    dist = support_distances(Zs, Zt, k=k)
    conf = np.abs(p - 0.5) * 2.0

    rows = []
    coverages = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    for rule in ["support", "confidence", "combined"]:
        if rule == "support":
            score = -dist  # larger is better
        elif rule == "confidence":
            score = conf
        else:
            # rank-normalized combination; avoids arbitrary scale mixing
            rs = pd.Series(-dist).rank(pct=True).to_numpy()
            rc = pd.Series(conf).rank(pct=True).to_numpy()
            score = 0.5 * rs + 0.5 * rc

        order = np.argsort(score)[::-1]
        for cov in coverages:
            n_keep = max(20, int(round(cov * len(order))))
            ii = order[:min(n_keep, len(order))]
            yy, pp = tgt["y"][ii], p[ii]
            met = metrics(yy, pp)
            ppv, npv = ppv_npv(yy, pp, threshold=0.5)
            rows.append(dict(
                rule=rule,
                requested_coverage=cov,
                actual_coverage=len(ii) / len(order),
                ppv_050=ppv,
                npv_050=npv,
                mean_support_distance=float(np.mean(dist[ii])),
                mean_confidence=float(np.mean(conf[ii])),
                **met,
            ))

    df = pd.DataFrame(rows)
    df.to_csv(out / f"selective_prediction__{target}.csv", index=False)
    return df


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--species", required=True)
    ap.add_argument("--drug", required=True)
    ap.add_argument("--target", default="DRIAMS-C")
    ap.add_argument("--labels", default="outputs/driams_long.parquet")
    ap.add_argument("--matrices", default="matrices")
    ap.add_argument("--root", default="~/data/DRIAMS")
    ap.add_argument("--out", default="outputs/shift_mechanism")
    ap.add_argument(
        "--analyses",
        default="temporal_site,ast_selection,support,conditional_shift,lineage,density_weight,invariant_peaks,selective",
        help="Comma-separated analyses or 'all'",
    )
    ap.add_argument("--ncomp", type=int, default=100)
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--clusters", type=int, default=8)
    ap.add_argument("--topk-peaks", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    mdir = Path(args.matrices)
    root = Path(args.root).expanduser()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lab = pd.read_parquet(args.labels)

    wanted = [x.strip() for x in args.analyses.split(",") if x.strip()]
    if "all" in wanted:
        wanted = [
            "temporal_site", "ast_selection", "support",
            "conditional_shift", "lineage", "density_weight",
            "invariant_peaks", "selective",
        ]

    meta = dict(
        species=args.species,
        drug=args.drug,
        target=args.target,
        labels=args.labels,
        matrices=args.matrices,
        ncomp=args.ncomp,
        knn_k=args.knn_k,
        clusters=args.clusters,
        topk_peaks=args.topk_peaks,
        seed=args.seed,
        analyses=wanted,
    )
    (out / "run_config.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"=== SHIFT MECHANISM ===")
    print(f"{args.species} / {args.drug} | target={args.target}")
    print(f"analyses: {', '.join(wanted)}", flush=True)

    failures = []

    for name in wanted:
        print(f"\n--- {name} ---", flush=True)
        try:
            if name == "temporal_site":
                df = analysis_temporal_site(
                    lab, mdir, root, args.species, args.drug, out, args.seed
                )
                print(df.to_string(index=False) if len(df) else "(no eligible comparisons)")

            elif name == "ast_selection":
                a, b = analysis_ast_selection(
                    lab, args.species, args.drug, out, args.seed
                )
                print(a.to_string(index=False))
                print(b.to_string(index=False))

            elif name == "support":
                df, iso = analysis_support(
                    lab, mdir, root, args.species, args.drug, args.target, out,
                    ncomp=args.ncomp, k=args.knn_k, seed=args.seed
                )
                print(df.to_string(index=False))

            elif name == "conditional_shift":
                df = analysis_conditional_shift(
                    lab, mdir, root, args.species, args.drug, args.target, out,
                    ncomp=args.ncomp, seed=args.seed
                )
                print(df.to_string(index=False))

            elif name == "lineage":
                comp, tr = analysis_lineage(
                    lab, mdir, root, args.species, args.drug, args.target, out,
                    ncomp=args.ncomp, n_clusters=args.clusters, seed=args.seed
                )
                print(comp.to_string(index=False))
                if len(tr):
                    print("\ncluster-specific transfer:")
                    print(tr.to_string(index=False))

            elif name == "density_weight":
                df = analysis_density_weight(
                    lab, mdir, root, args.species, args.drug, args.target, out,
                    ncomp=args.ncomp, seed=args.seed
                )
                print(df.to_string(index=False))

            elif name == "invariant_peaks":
                df, peaks = analysis_invariant_peaks(
                    lab, mdir, root, args.species, args.drug, args.target, out,
                    topk=args.topk_peaks, seed=args.seed
                )
                print(df.to_string(index=False))
                print("\nTop selected bins:")
                show = peaks[peaks.selected].head(20)
                cols = ["bin", "mz_approx", "median_abs_smd"]
                print(show[cols].to_string(index=False) if len(show) else "(none)")

            elif name == "selective":
                df = analysis_selective(
                    lab, mdir, root, args.species, args.drug, args.target, out,
                    ncomp=args.ncomp, k=args.knn_k, seed=args.seed
                )
                print(df.to_string(index=False))

            else:
                raise ValueError(f"Unknown analysis: {name}")

        except Exception as e:
            failures.append((name, repr(e)))
            print(f"HATA [{name}]: {e}", flush=True)

    if failures:
        pd.DataFrame(failures, columns=["analysis", "error"]).to_csv(
            out / "failures.csv", index=False
        )
        print("\n=== FAILURES ===")
        for name, err in failures:
            print(f"{name}: {err}")

    print(f"\nOutputs: {out}")


if __name__ == "__main__":
    main()
