#!/usr/bin/env python3
"""
local_transport.py
==================

Task-relevant local transportability analyses for cross-site MALDI-TOF AMR prediction.

Rationale
---------
Previous experiments can show that global acquisition-domain alignment is not
necessarily associated with better AMR transfer. This script therefore asks:

    "For a given target domain / target isolate, which source observations are
     actually informative for the AMR task?"

All adaptation steps are target-label-free. Target RI labels are used only for
final evaluation.

Implemented analyses
--------------------
1) selection_bias
   Within DRIAMS-A, predict whether the requested drug was tested using MALDI
   spectra only. This diagnoses whether the source AMR cohort is a spectrally
   selected subset of all isolates of that species.

2) class_conditional_weight
   Pseudo-class-conditional density-ratio weighting:
      - fit source-only AMR model
      - pseudo-label target with confidence thresholds
      - for source R, distinguish source-R from confident target-pseudo-R
      - for source S, distinguish source-S from confident target-pseudo-S
      - convert cross-fitted domain propensity into source sample weights
      - train weighted source LightGBM
   No target RI labels are used in weighting.

3) source_pruning
   Global target-likeness propensity is estimated from unlabeled target spectra.
   The most target-like source observations are retained. Also evaluates
   class-stratified pruning so resistant source samples are not accidentally lost.

4) local_knn
   Instance-specific source retrieval in a source-fitted SVD space:
      - weighted kNN posterior
      - optional class-balanced neighborhood construction
   This uses different source observations for each target isolate.

5) local_lgbm
   Optional query-specific local LightGBM. It trains one small model per target
   isolate using its nearest source neighborhood. Computationally heavier.
   Disabled unless explicitly requested.

6) selective
   Risk/coverage curves for each available prediction method using prediction
   confidence and source-support distance. No target-label-based threshold tuning.

Outputs
-------
CSV files under --out. Raw isolate-level target predictions are also saved.

Project dependency
------------------
Place this file next to nested_cv.py in:
    ~/projects/maldi-shift/

Expected helpers from nested_cv.py:
    load_spectra, gather_rows, patient_map, group_key
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

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
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

CFG_LOCAL = dict(
    objective="binary",
    num_leaves=15,
    n_estimators=120,
    learning_rate=0.05,
    colsample_bytree=0.5,
    subsample=0.9,
    subsample_freq=1,
    verbose=-1,
    n_jobs=1,
)


# ---------------------------------------------------------------------
# Utilities
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
    lo = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        m = LogisticRegression(
            penalty=None, solver="lbfgs", max_iter=1000
        ).fit(lo, y)
        return float(m.coef_[0, 0])
    except Exception:
        return np.nan


def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    return dict(
        auroc=safe_auc(y, p),
        prauc=safe_prauc(y, p),
        brier=float(brier_score_loss(y, p)) if len(y) else np.nan,
        slope=cal_slope(y, p),
        n=int(len(y)),
        positives=int(y.sum()),
        prevalence=float(y.mean()) if len(y) else np.nan,
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


def load_species_all(lab, mdir, species, site):
    """
    Load all available spectra for species/site regardless of whether a specific
    drug was tested. One row per isolate/code.
    """
    z = lab[
        (lab.species == species)
        & lab.has_spectrum
        & (lab.site == site)
    ][["site", "species", "code"]].drop_duplicates().copy()

    if z.empty:
        return None

    xs, _, idx = spectra(mdir, site)
    z = z[z.code.isin(idx.keys())].copy()
    if z.empty:
        return None

    X = gather_rows(xs, [idx[c] for c in z.code])
    return dict(df=z.reset_index(drop=True), X=X)


def fit_source_svd(Xs, ncomp=100, seed=0):
    mu = Xs.mean(axis=0, keepdims=True).astype(np.float32)
    nc = min(ncomp, min(Xs.shape) - 1)
    nc = max(2, nc)
    _, _, Vt = randomized_svd(
        Xs.astype(np.float64) - mu,
        n_components=nc,
        random_state=seed,
    )
    P = Vt.T.astype(np.float32)
    return mu, P


def transform_svd(X, mu, P):
    return ((X - mu) @ P).astype(np.float32)


def fit_baseline(Xs, ys, Xt, yt, seed=0, sample_weight=None):
    m = lgb.LGBMClassifier(**CFG, random_state=seed)
    kw = {}
    if sample_weight is not None:
        kw["sample_weight"] = np.asarray(sample_weight, dtype=np.float32)
    m.fit(Xs, ys, **kw)
    p = m.predict_proba(Xt)[:, 1]
    return m, p, metrics(yt, p)


def effective_sample_size(w):
    w = np.asarray(w, dtype=float)
    den = np.sum(w ** 2)
    return float((w.sum() ** 2) / den) if den > 0 else 0.0


def crossfit_domain_probability(Zs, Zt, seed=0, folds=5):
    """
    Cross-fitted P(domain=target|x) for source and target.
    """
    X = np.vstack([Zs, Zt]).astype(np.float32)
    d = np.r_[
        np.zeros(len(Zs), dtype=int),
        np.ones(len(Zt), dtype=int)
    ]
    pred = np.zeros(len(d), dtype=float)

    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for f, (tr, te) in enumerate(skf.split(X, d)):
        mdl = lgb.LGBMClassifier(**CFG_DOMAIN, random_state=seed + f)
        mdl.fit(X[tr], d[tr])
        pred[te] = mdl.predict_proba(X[te])[:, 1]

    return pred[:len(Zs)], pred[len(Zs):], float(roc_auc_score(d, pred))


def propensity_to_density_ratio(q_source, n_source, n_target, clip=10.0):
    """
    If q=P(target-domain|x) under pooled empirical domain priors:
      p_t(x)/p_s(x) ≈ [q/(1-q)] * [n_source/n_target]
    """
    q = np.clip(np.asarray(q_source), 1e-4, 1 - 1e-4)
    w = (q / (1 - q)) * (n_source / max(n_target, 1))
    w = np.clip(w, 0, clip)
    if w.mean() > 0:
        w /= w.mean()
    return w.astype(np.float32)


def support_distance(Zs, Zt, k=10):
    kk = max(1, min(k, len(Zs)))
    nn = NearestNeighbors(n_neighbors=kk, metric="euclidean", n_jobs=-1)
    nn.fit(Zs)
    d, ind = nn.kneighbors(Zt)
    return d.mean(axis=1), d, ind


def bootstrap_paired_diff(df, method_col="method", metric_col="prauc",
                          baseline="baseline", reps=5000, seed=0):
    """
    Used only when dataframe contains repeated runs with a 'rep' column.
    """
    if "rep" not in df.columns:
        return pd.DataFrame()
    base = df[df[method_col] == baseline][["rep", metric_col]].rename(
        columns={metric_col: "base_metric"}
    )
    rows = []
    rng = np.random.default_rng(seed)
    for meth, g in df[df[method_col] != baseline].groupby(method_col):
        m = g.merge(base, on="rep")
        d = (m[metric_col] - m["base_metric"]).to_numpy()
        if len(d) < 2:
            continue
        bs = rng.choice(d, (reps, len(d)), replace=True).mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append(dict(
            method=meth,
            mean_difference=float(d.mean()),
            ci_low=float(lo),
            ci_high=float(hi),
            significant=bool(lo > 0 or hi < 0),
            n_reps=len(d),
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 1) Within-source AST selection bias
# ---------------------------------------------------------------------

def analysis_selection_bias(
    lab, mdir, species, drug, out, site="DRIAMS-A",
    ncomp=100, seed=0, folds=5
):
    """
    Predict T = 1(drug tested) vs 0(not tested) from MALDI spectra within a
    single site/species. Site itself is NOT a predictor.

    This directly diagnoses:
        P(X | tested=1, site=A) != P(X | tested=0, site=A)
    """
    allsp = load_species_all(lab, mdir, species, site)
    if allsp is None:
        raise ValueError("No species spectra available.")

    # Build one tested flag per isolate
    z = lab[
        (lab.species == species)
        & (lab.site == site)
        & (lab.drug == drug)
    ][["code", "tested"]].copy()

    tested_map = z.groupby("code")["tested"].max().to_dict()
    y = np.array(
        [int(bool(tested_map.get(c, False))) for c in allsp["df"].code],
        dtype=int
    )

    if len(np.unique(y)) < 2:
        raise ValueError(
            f"{site}: drug-tested status is single-class for {species}/{drug}."
        )

    mu, P = fit_source_svd(allsp["X"], ncomp=ncomp, seed=seed)
    Z = transform_svd(allsp["X"], mu, P)

    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    pred = np.zeros(len(y), dtype=float)

    for f, (tr, te) in enumerate(skf.split(Z, y)):
        mdl = lgb.LGBMClassifier(**CFG_DOMAIN, random_state=seed + f)
        mdl.fit(Z[tr], y[tr])
        pred[te] = mdl.predict_proba(Z[te])[:, 1]

    met = metrics(y, pred)
    result = pd.DataFrame([dict(
        analysis="within_site_tested_vs_untested",
        site=site,
        species=species,
        drug=drug,
        ncomp=Z.shape[1],
        **met,
    )])
    result.to_csv(out / f"selection_bias__{site}.csv", index=False)

    raw = pd.DataFrame({
        "code": allsp["df"].code.astype(str),
        "tested": y,
        "p_tested": pred,
    })
    raw.to_csv(out / f"selection_bias_predictions__{site}.csv", index=False)

    return result


# ---------------------------------------------------------------------
# 2) Pseudo-class-conditional density weighting
# ---------------------------------------------------------------------

def class_conditional_weights(
    Xs, ys, Xt, source_model,
    ncomp=100, pseudo_low=0.20, pseudo_high=0.80,
    clip=10.0, seed=0
):
    """
    Target-label-free class-conditional density weighting.

    Target pseudo-S: p <= pseudo_low
    Target pseudo-R: p >= pseudo_high
    Ambiguous target observations do not enter class-specific domain models.

    Source class 0 is compared to confident pseudo-S target.
    Source class 1 is compared to confident pseudo-R target.
    """
    pt = source_model.predict_proba(Xt)[:, 1]

    pseudo_t = np.full(len(Xt), -1, dtype=int)
    pseudo_t[pt <= pseudo_low] = 0
    pseudo_t[pt >= pseudo_high] = 1

    w = np.ones(len(Xs), dtype=np.float32)
    diagnostics = []

    for cls in [0, 1]:
        si = np.where(ys == cls)[0]
        ti = np.where(pseudo_t == cls)[0]

        if len(si) < 40 or len(ti) < 40:
            diagnostics.append(dict(
                class_label=cls,
                n_source=len(si),
                n_target_pseudo=len(ti),
                domain_auroc=np.nan,
                ess=np.nan,
                applied=False,
            ))
            continue

        # Fit representation only on this source class.
        mu, P = fit_source_svd(Xs[si], ncomp=min(ncomp, 100), seed=seed + cls)
        Zs = transform_svd(Xs[si], mu, P)
        Zt = transform_svd(Xt[ti], mu, P)

        qs, _, dauc = crossfit_domain_probability(
            Zs, Zt, seed=seed + cls, folds=5
        )
        wc = propensity_to_density_ratio(
            qs, len(Zs), len(Zt), clip=clip
        )
        w[si] = wc

        diagnostics.append(dict(
            class_label=cls,
            n_source=len(si),
            n_target_pseudo=len(ti),
            domain_auroc=dauc,
            ess=effective_sample_size(wc),
            applied=True,
        ))

    # Normalize globally
    if w.mean() > 0:
        w /= w.mean()

    return w, pseudo_t, pt, pd.DataFrame(diagnostics)


def analysis_class_conditional_weight(
    lab, mdir, root, species, drug, target, out,
    ncomp=100, pseudo_lows=(0.10, 0.20, 0.30),
    pseudo_highs=(0.70, 0.80, 0.90),
    clips=(2, 5, 10, 20), seed=0
):
    src = load_task(lab, mdir, root, species, drug, "DRIAMS-A")
    tgt = load_task(lab, mdir, root, species, drug, target)
    if src is None or tgt is None:
        raise ValueError("Source or target task unavailable.")

    source_model = lgb.LGBMClassifier(**CFG, random_state=seed)
    source_model.fit(src["X"], src["y"])
    p0 = source_model.predict_proba(tgt["X"])[:, 1]

    rows = [dict(
        method="baseline",
        pseudo_low=np.nan,
        pseudo_high=np.nan,
        clip=np.nan,
        source_ess=float(len(src["y"])),
        **metrics(tgt["y"], p0),
    )]

    diag_frames = []
    weight_frames = []

    # Predefined grid; do not select the winner on the target test labels.
    for low, high in zip(pseudo_lows, pseudo_highs):
        if low >= high:
            continue
        for cl in clips:
            w, pseudo_t, pt, diag = class_conditional_weights(
                src["X"], src["y"], tgt["X"], source_model,
                ncomp=ncomp,
                pseudo_low=low,
                pseudo_high=high,
                clip=cl,
                seed=seed,
            )
            mdl = lgb.LGBMClassifier(**CFG, random_state=seed)
            mdl.fit(src["X"], src["y"], sample_weight=w)
            p = mdl.predict_proba(tgt["X"])[:, 1]

            rows.append(dict(
                method="pseudo_class_conditional_weight",
                pseudo_low=low,
                pseudo_high=high,
                clip=cl,
                source_ess=effective_sample_size(w),
                n_target_pseudo_S=int((pseudo_t == 0).sum()),
                n_target_pseudo_R=int((pseudo_t == 1).sum()),
                n_target_ambiguous=int((pseudo_t == -1).sum()),
                **metrics(tgt["y"], p),
            ))

            d = diag.copy()
            d["pseudo_low"] = low
            d["pseudo_high"] = high
            d["clip"] = cl
            diag_frames.append(d)

            weight_frames.append(pd.DataFrame({
                "code": src["df"].code.astype(str),
                "y": src["y"],
                "weight": w,
                "pseudo_low": low,
                "pseudo_high": high,
                "clip": cl,
            }))

    df = pd.DataFrame(rows)
    df.to_csv(out / f"class_conditional_weight__{target}.csv", index=False)

    if diag_frames:
        pd.concat(diag_frames, ignore_index=True).to_csv(
            out / f"class_conditional_weight_diagnostics__{target}.csv",
            index=False
        )
    if weight_frames:
        pd.concat(weight_frames, ignore_index=True).to_csv(
            out / f"class_conditional_source_weights__{target}.csv",
            index=False
        )

    return df


# ---------------------------------------------------------------------
# 3) Target-similar source pruning
# ---------------------------------------------------------------------

def analysis_source_pruning(
    lab, mdir, root, species, drug, target, out,
    ncomp=100, keep_fracs=(0.10, 0.20, 0.30, 0.50, 0.75),
    class_stratified=True, seed=0
):
    src = load_task(lab, mdir, root, species, drug, "DRIAMS-A")
    tgt = load_task(lab, mdir, root, species, drug, target)
    if src is None or tgt is None:
        raise ValueError("Source or target task unavailable.")

    mu, P = fit_source_svd(src["X"], ncomp=ncomp, seed=seed)
    Zs = transform_svd(src["X"], mu, P)
    Zt = transform_svd(tgt["X"], mu, P)

    qs, _, dauc = crossfit_domain_probability(Zs, Zt, seed=seed)

    _, p0, m0 = fit_baseline(
        src["X"], src["y"], tgt["X"], tgt["y"], seed=seed
    )

    rows = [dict(
        method="baseline",
        keep_fraction=1.0,
        n_source=len(src["y"]),
        positives_source=int(src["y"].sum()),
        source_prevalence=float(src["y"].mean()),
        domain_auroc=dauc,
        **m0,
    )]

    selected_frames = []

    for frac in keep_fracs:
        frac = float(frac)

        # Global pruning
        n_keep = max(100, int(round(frac * len(qs))))
        idx_keep = np.argsort(qs)[::-1][:min(n_keep, len(qs))]
        if len(np.unique(src["y"][idx_keep])) == 2:
            _, p, met = fit_baseline(
                src["X"][idx_keep], src["y"][idx_keep],
                tgt["X"], tgt["y"], seed=seed
            )
            rows.append(dict(
                method="global_target_like_pruning",
                keep_fraction=frac,
                n_source=len(idx_keep),
                positives_source=int(src["y"][idx_keep].sum()),
                source_prevalence=float(src["y"][idx_keep].mean()),
                domain_auroc=dauc,
                **met,
            ))
            selected_frames.append(pd.DataFrame({
                "code": src["df"].code.iloc[idx_keep].astype(str),
                "y": src["y"][idx_keep],
                "q_target": qs[idx_keep],
                "keep_fraction": frac,
                "method": "global_target_like_pruning",
            }))

        # Class-stratified pruning
        if class_stratified:
            parts = []
            for cls in [0, 1]:
                ii = np.where(src["y"] == cls)[0]
                n_c = max(20, int(round(frac * len(ii))))
                chosen = ii[np.argsort(qs[ii])[::-1][:min(n_c, len(ii))]]
                parts.append(chosen)
            idx2 = np.concatenate(parts)
            rng = np.random.default_rng(seed)
            rng.shuffle(idx2)

            if len(np.unique(src["y"][idx2])) == 2:
                _, p2, met2 = fit_baseline(
                    src["X"][idx2], src["y"][idx2],
                    tgt["X"], tgt["y"], seed=seed
                )
                rows.append(dict(
                    method="class_stratified_target_like_pruning",
                    keep_fraction=frac,
                    n_source=len(idx2),
                    positives_source=int(src["y"][idx2].sum()),
                    source_prevalence=float(src["y"][idx2].mean()),
                    domain_auroc=dauc,
                    **met2,
                ))
                selected_frames.append(pd.DataFrame({
                    "code": src["df"].code.iloc[idx2].astype(str),
                    "y": src["y"][idx2],
                    "q_target": qs[idx2],
                    "keep_fraction": frac,
                    "method": "class_stratified_target_like_pruning",
                }))

    df = pd.DataFrame(rows)
    df.to_csv(out / f"source_pruning__{target}.csv", index=False)

    if selected_frames:
        pd.concat(selected_frames, ignore_index=True).to_csv(
            out / f"source_pruning_selected__{target}.csv", index=False
        )

    return df


# ---------------------------------------------------------------------
# 4) Instance-specific local transfer: weighted kNN
# ---------------------------------------------------------------------

def weighted_knn_probability(
    Zs, ys, Zt, k=100, metric="euclidean", temperature=None
):
    """
    Weighted source-neighbor posterior for each target observation.
    """
    kk = max(5, min(int(k), len(Zs)))
    nn = NearestNeighbors(
        n_neighbors=kk, metric=metric, n_jobs=-1
    ).fit(Zs)
    d, ind = nn.kneighbors(Zt)

    if temperature is None:
        # robust global scale; avoid target-label tuning
        temperature = float(np.median(d[:, max(0, kk // 2 - 1)]) + 1e-8)

    W = np.exp(-d / max(temperature, 1e-8))
    Yn = ys[ind]
    p = (W * Yn).sum(axis=1) / np.maximum(W.sum(axis=1), 1e-12)

    return p.astype(float), d, ind, temperature


def class_balanced_knn_probability(
    Zs, ys, Zt, k_per_class=50, metric="euclidean"
):
    """
    Retrieve nearest susceptible and resistant source samples separately.
    Estimate class evidence from relative distances while preserving source prior.

    This avoids majority-class domination in ordinary kNN.
    """
    idx0 = np.where(ys == 0)[0]
    idx1 = np.where(ys == 1)[0]
    if len(idx0) < 5 or len(idx1) < 5:
        raise ValueError("Both source classes need at least 5 observations.")

    k0 = min(k_per_class, len(idx0))
    k1 = min(k_per_class, len(idx1))

    nn0 = NearestNeighbors(n_neighbors=k0, metric=metric, n_jobs=-1).fit(Zs[idx0])
    nn1 = NearestNeighbors(n_neighbors=k1, metric=metric, n_jobs=-1).fit(Zs[idx1])

    d0, i0 = nn0.kneighbors(Zt)
    d1, i1 = nn1.kneighbors(Zt)

    # robust temperature across both classes
    temp = float(np.median(np.c_[d0[:, -1], d1[:, -1]]) + 1e-8)

    e0 = np.exp(-d0 / temp).mean(axis=1)
    e1 = np.exp(-d1 / temp).mean(axis=1)

    prior1 = float(np.mean(ys))
    prior0 = 1.0 - prior1

    # evidence * source prior
    num = e1 * prior1
    den = num + e0 * prior0
    p = num / np.maximum(den, 1e-12)

    support = 0.5 * (d0.mean(axis=1) + d1.mean(axis=1))
    return p, support, d0, d1, temp


def analysis_local_knn(
    lab, mdir, root, species, drug, target, out,
    ncomp=100, ks=(25, 50, 100, 200, 400),
    class_balanced_ks=(10, 25, 50, 100), seed=0
):
    src = load_task(lab, mdir, root, species, drug, "DRIAMS-A")
    tgt = load_task(lab, mdir, root, species, drug, target)
    if src is None or tgt is None:
        raise ValueError("Source or target unavailable.")

    mu, P = fit_source_svd(src["X"], ncomp=ncomp, seed=seed)
    Zs = transform_svd(src["X"], mu, P)
    Zt = transform_svd(tgt["X"], mu, P)

    # Standardize using source only
    sc = StandardScaler().fit(Zs)
    Zs2 = sc.transform(Zs).astype(np.float32)
    Zt2 = sc.transform(Zt).astype(np.float32)

    _, p0, m0 = fit_baseline(
        src["X"], src["y"], tgt["X"], tgt["y"], seed=seed
    )

    rows = [dict(
        method="baseline_lgbm",
        k=np.nan,
        ncomp=ncomp,
        temperature=np.nan,
        **m0,
    )]

    pred_cols = {
        "code": tgt["df"].code.astype(str).to_numpy(),
        "y": tgt["y"],
        "p_baseline": p0,
    }

    # ordinary weighted kNN
    for k in ks:
        p, d, ind, temp = weighted_knn_probability(
            Zs2, src["y"], Zt2, k=k
        )
        met = metrics(tgt["y"], p)
        rows.append(dict(
            method="weighted_knn",
            k=int(k),
            ncomp=ncomp,
            temperature=temp,
            mean_support_distance=float(d.mean()),
            **met,
        ))
        pred_cols[f"p_knn_{k}"] = p
        pred_cols[f"support_knn_{k}"] = d.mean(axis=1)

    # class-balanced retrieval
    for kpc in class_balanced_ks:
        p, supp, d0, d1, temp = class_balanced_knn_probability(
            Zs2, src["y"], Zt2, k_per_class=kpc
        )
        met = metrics(tgt["y"], p)
        rows.append(dict(
            method="class_balanced_knn",
            k=int(kpc),
            ncomp=ncomp,
            temperature=temp,
            mean_support_distance=float(supp.mean()),
            **met,
        ))
        pred_cols[f"p_cbknn_{kpc}"] = p
        pred_cols[f"support_cbknn_{kpc}"] = supp

    df = pd.DataFrame(rows)
    df.to_csv(out / f"local_knn__{target}.csv", index=False)

    raw = pd.DataFrame(pred_cols)
    raw.to_csv(out / f"local_knn_predictions__{target}.csv", index=False)

    return df, raw


# ---------------------------------------------------------------------
# 5) Optional instance-specific local LightGBM
# ---------------------------------------------------------------------

def analysis_local_lgbm(
    lab, mdir, root, species, drug, target, out,
    ncomp=100, k=400, min_pos=20, seed=0, max_targets=None
):
    """
    Query-specific local LightGBM.

    For each target isolate:
      1) retrieve k nearest source observations in source-fitted SVD space
      2) fit a small LightGBM on that neighborhood
      3) predict only that target isolate

    This can be computationally expensive. Use --max-local-targets for a pilot.
    """
    src = load_task(lab, mdir, root, species, drug, "DRIAMS-A")
    tgt = load_task(lab, mdir, root, species, drug, target)
    if src is None or tgt is None:
        raise ValueError("Source or target unavailable.")

    mu, P = fit_source_svd(src["X"], ncomp=ncomp, seed=seed)
    Zs = transform_svd(src["X"], mu, P)
    Zt = transform_svd(tgt["X"], mu, P)

    sc = StandardScaler().fit(Zs)
    Zs2 = sc.transform(Zs).astype(np.float32)
    Zt2 = sc.transform(Zt).astype(np.float32)

    kk = min(k, len(Zs2))
    nn = NearestNeighbors(
        n_neighbors=kk, metric="euclidean", n_jobs=-1
    ).fit(Zs2)
    d, ind = nn.kneighbors(Zt2)

    n_eval = len(tgt["y"]) if max_targets is None else min(max_targets, len(tgt["y"]))
    p = np.full(len(tgt["y"]), np.nan, dtype=float)
    used_local = np.zeros(len(tgt["y"]), dtype=bool)

    global_model = lgb.LGBMClassifier(**CFG, random_state=seed).fit(
        src["X"], src["y"]
    )
    p_global = global_model.predict_proba(tgt["X"])[:, 1]

    for j in range(n_eval):
        ix = ind[j]
        yy = src["y"][ix]

        if len(np.unique(yy)) < 2 or int((yy == 1).sum()) < min_pos:
            p[j] = p_global[j]
            continue

        mdl = lgb.LGBMClassifier(**CFG_LOCAL, random_state=seed + j)
        mdl.fit(src["X"][ix], yy)
        p[j] = mdl.predict_proba(tgt["X"][j:j+1])[:, 1][0]
        used_local[j] = True

    # For non-evaluated targets, preserve global predictions
    p[~np.isfinite(p)] = p_global[~np.isfinite(p)]

    met = metrics(tgt["y"], p)
    summary = pd.DataFrame([dict(
        method="query_specific_local_lgbm",
        k=kk,
        n_local_models=int(used_local.sum()),
        fraction_local=float(used_local.mean()),
        **met,
    )])
    summary.to_csv(out / f"local_lgbm__{target}.csv", index=False)

    raw = pd.DataFrame({
        "code": tgt["df"].code.astype(str),
        "y": tgt["y"],
        "p_global": p_global,
        "p_local": p,
        "used_local": used_local,
        "support_distance": d.mean(axis=1),
    })
    raw.to_csv(out / f"local_lgbm_predictions__{target}.csv", index=False)

    return summary, raw


# ---------------------------------------------------------------------
# 6) Selective prediction from local transport outputs
# ---------------------------------------------------------------------

def selective_curve(y, p, support=None, method="", coverages=None):
    if coverages is None:
        coverages = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]

    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=float)
    conf = np.abs(p - 0.5) * 2.0

    rows = []

    scores = {"confidence": conf}
    if support is not None:
        support = np.asarray(support, dtype=float)
        rs = pd.Series(-support).rank(pct=True).to_numpy()
        rc = pd.Series(conf).rank(pct=True).to_numpy()
        scores["support"] = -support
        scores["combined"] = 0.5 * rs + 0.5 * rc

    for rule, score in scores.items():
        order = np.argsort(score)[::-1]
        for cov in coverages:
            nk = max(20, int(round(cov * len(order))))
            ii = order[:min(nk, len(order))]
            met = metrics(y[ii], p[ii])
            rows.append(dict(
                method=method,
                selection_rule=rule,
                requested_coverage=cov,
                actual_coverage=len(ii) / len(order),
                mean_confidence=float(conf[ii].mean()),
                mean_support=float(support[ii].mean()) if support is not None else np.nan,
                **met,
            ))
    return pd.DataFrame(rows)


def analysis_selective_local(out, target):
    """
    Consume local_knn_predictions__TARGET.csv generated by local_knn.
    Produce curves for all available prediction columns.
    """
    f = out / f"local_knn_predictions__{target}.csv"
    if not f.exists():
        raise ValueError(
            f"{f} not found. Run --analyses local_knn first or together."
        )

    z = pd.read_csv(f)
    y = z["y"].to_numpy(dtype=int)
    frames = []

    for c in z.columns:
        if not c.startswith("p_"):
            continue
        suffix = c[2:]

        support_col = None
        if suffix.startswith("knn_"):
            support_col = "support_" + suffix
        elif suffix.startswith("cbknn_"):
            support_col = "support_" + suffix

        supp = z[support_col].to_numpy() if support_col in z.columns else None
        frames.append(
            selective_curve(y, z[c].to_numpy(), support=supp, method=suffix)
        )

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(out / f"selective_local__{target}.csv", index=False)
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
    ap.add_argument("--out", default="outputs/local_transport")

    ap.add_argument(
        "--analyses",
        default="selection_bias,class_conditional_weight,source_pruning,local_knn,selective",
        help=(
            "Comma-separated: selection_bias,class_conditional_weight,"
            "source_pruning,local_knn,local_lgbm,selective or all"
        ),
    )

    ap.add_argument("--ncomp", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)

    # class-conditional weighting
    ap.add_argument("--pseudo-lows", default="0.10,0.20,0.30")
    ap.add_argument("--pseudo-highs", default="0.90,0.80,0.70")
    ap.add_argument("--clips", default="2,5,10,20")

    # pruning
    ap.add_argument("--keep-fracs", default="0.10,0.20,0.30,0.50,0.75")

    # kNN
    ap.add_argument("--knn-ks", default="25,50,100,200,400")
    ap.add_argument("--cbknn-ks", default="10,25,50,100")

    # local lgbm
    ap.add_argument("--local-k", type=int, default=400)
    ap.add_argument("--local-min-pos", type=int, default=20)
    ap.add_argument("--max-local-targets", type=int, default=None)

    args = ap.parse_args()

    mdir = Path(args.matrices)
    root = Path(args.root).expanduser()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    lab = pd.read_parquet(args.labels)

    analyses = [x.strip() for x in args.analyses.split(",") if x.strip()]
    if "all" in analyses:
        analyses = [
            "selection_bias",
            "class_conditional_weight",
            "source_pruning",
            "local_knn",
            "local_lgbm",
            "selective",
        ]

    lows = tuple(float(x) for x in args.pseudo_lows.split(","))
    highs = tuple(float(x) for x in args.pseudo_highs.split(","))
    clips = tuple(float(x) for x in args.clips.split(","))
    keep_fracs = tuple(float(x) for x in args.keep_fracs.split(","))
    knn_ks = tuple(int(x) for x in args.knn_ks.split(","))
    cbknn_ks = tuple(int(x) for x in args.cbknn_ks.split(","))

    config = vars(args).copy()
    config["analyses_resolved"] = analyses
    (out / "run_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=== LOCAL TRANSPORTABILITY ===")
    print(f"{args.species} / {args.drug} | source=DRIAMS-A | target={args.target}")
    print(f"analyses: {', '.join(analyses)}", flush=True)

    failures = []

    for name in analyses:
        print(f"\n--- {name} ---", flush=True)
        try:
            if name == "selection_bias":
                df = analysis_selection_bias(
                    lab, mdir, args.species, args.drug, out,
                    site="DRIAMS-A", ncomp=args.ncomp, seed=args.seed
                )
                print(df.to_string(index=False))

            elif name == "class_conditional_weight":
                df = analysis_class_conditional_weight(
                    lab, mdir, root, args.species, args.drug, args.target, out,
                    ncomp=args.ncomp,
                    pseudo_lows=lows,
                    pseudo_highs=highs,
                    clips=clips,
                    seed=args.seed,
                )
                print(df.to_string(index=False))

            elif name == "source_pruning":
                df = analysis_source_pruning(
                    lab, mdir, root, args.species, args.drug, args.target, out,
                    ncomp=args.ncomp,
                    keep_fracs=keep_fracs,
                    class_stratified=True,
                    seed=args.seed,
                )
                print(df.to_string(index=False))

            elif name == "local_knn":
                df, raw = analysis_local_knn(
                    lab, mdir, root, args.species, args.drug, args.target, out,
                    ncomp=args.ncomp,
                    ks=knn_ks,
                    class_balanced_ks=cbknn_ks,
                    seed=args.seed,
                )
                print(df.to_string(index=False))

            elif name == "local_lgbm":
                df, raw = analysis_local_lgbm(
                    lab, mdir, root, args.species, args.drug, args.target, out,
                    ncomp=args.ncomp,
                    k=args.local_k,
                    min_pos=args.local_min_pos,
                    seed=args.seed,
                    max_targets=args.max_local_targets,
                )
                print(df.to_string(index=False))

            elif name == "selective":
                df = analysis_selective_local(out, args.target)
                # concise console view: best PR-AUC within each method/rule
                show = (
                    df.sort_values("prauc", ascending=False)
                    .groupby(["method", "selection_rule"], as_index=False)
                    .head(1)
                )
                print(show.to_string(index=False))

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
