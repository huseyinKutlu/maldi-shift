#!/usr/bin/env python3
"""Fizik-bilgili artirma (domain randomization) + kontrol metrigi."""
import argparse, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).parent))
from nested_cv import load_spectra, gather_rows, patient_map, group_key

warnings.filterwarnings("ignore")
CFG = dict(objective="binary", num_leaves=31, n_estimators=300,
           learning_rate=0.05, colsample_bytree=0.3, subsample=0.8,
           subsample_freq=1, verbose=-1, n_jobs=12)
SITES = ["DRIAMS-A", "DRIAMS-B", "DRIAMS-C", "DRIAMS-D"]


def augment(X, rng, shift=5, scale=(0.8, 1.25), base=0.15, width=(0.0, 1.5),
            noise=0.03):
    n, d = X.shape
    Y = X.copy()
    sh = rng.integers(-shift, shift + 1, size=n)
    for i in range(n):
        if sh[i]:
            Y[i] = np.roll(Y[i], int(sh[i]))
    sc = rng.uniform(scale[0], scale[1], size=(n, 1)).astype(np.float32)
    Y *= sc
    t = np.linspace(0, 1, d, dtype=np.float32)
    amp = rng.uniform(0, base, size=(n, 1)).astype(np.float32)
    decay = rng.uniform(2.0, 8.0, size=(n, 1)).astype(np.float32)
    Y += amp * np.exp(-decay * t[None, :]) * Y.mean(1, keepdims=True)
    sg = rng.uniform(width[0], width[1], size=n)
    for i in range(n):
        if sg[i] > 0.05:
            Y[i] = gaussian_filter1d(Y[i], sg[i])
    Y += rng.normal(0, noise, size=Y.shape).astype(np.float32) * \
        Y.std(1, keepdims=True)
    return np.clip(Y, 0, None).astype(np.float32)


def make_aug(X, y, g, k, rng, **kw):
    if k <= 0:
        return X, y, g
    Xs, ys, gs = [X], [y], [g]
    for _ in range(k):
        Xs.append(augment(X, rng, **kw)); ys.append(y); gs.append(g)
    return np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)


def cal_slope(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lo = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(lo, y)
        return float(m.coef_[0][0])
    except Exception:
        return np.nan


def site_auc(X, lab, seed=0, folds=3, minpos=40):
    a = []
    for u in np.unique(lab):
        yy = (lab == u).astype(int)
        if yy.sum() < minpos or (1 - yy).sum() < minpos:
            continue
        sc = []
        for tr, te in StratifiedKFold(folds, shuffle=True,
                                      random_state=seed).split(X, yy):
            m = lgb.LGBMClassifier(**CFG, random_state=seed).fit(X[tr], yy[tr])
            sc.append(roc_auc_score(yy[te], m.predict_proba(X[te])[:, 1]))
        a.append(np.mean(sc))
    return float(np.mean(a)) if a else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--drug", required=True)
    ap.add_argument("--labels", default="outputs/driams_long.parquet")
    ap.add_argument("--matrices", default="matrices")
    ap.add_argument("--root", default="~/data/DRIAMS")
    ap.add_argument("--out", default="outputs/aug")
    ap.add_argument("--ks", default="0,2,5,10")
    ap.add_argument("--site-species", default="Escherichia coli")
    ap.add_argument("--site-n", type=int, default=500)
    a = ap.parse_args()

    mdir = Path(a.matrices); root = Path(a.root).expanduser()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_parquet(a.labels)
    rng = np.random.default_rng(0)

    base = lab[lab.has_spectrum][["site", "species", "code"]].drop_duplicates()
    Xl, Sl = [], []
    for s in SITES:
        cs = base[(base.species == a.site_species) & (base.site == s)].code.tolist()
        try:
            xs, c, idx = load_spectra(mdir, s)
        except FileNotFoundError:
            continue
        keep = [k for k in cs if k in idx]
        if len(keep) < 60: continue
        if len(keep) > a.site_n:
            keep = list(rng.choice(keep, a.site_n, replace=False))
        Xl.append(gather_rows(xs, [idx[k] for k in keep])); Sl += [s] * len(keep)
    Xsite = np.vstack(Xl); sites = np.array(Sl)
    print(f"merkez seti: {Xsite.shape[0]:,} spektrum", flush=True)

    sel = lab[(lab.species == a.species) & (lab.drug == a.drug) &
              lab.tested & lab.has_spectrum]
    tr = sel[sel.site == "DRIAMS-A"]
    xs, c, idx = load_spectra(mdir, "DRIAMS-A")
    tr = tr[tr.code.isin(idx.keys())]
    y = tr.label_RI.to_numpy(dtype=int)
    X = gather_rows(xs, [idx[cc] for cc in tr.code])
    g = group_key(tr.code.to_numpy(), patient_map(root, "DRIAMS-A"), "patient")
    print(f"{a.species}/{a.drug}: n={len(y):,} direnc={y.mean():.3f}", flush=True)

    Xe = {}
    for s in SITES[1:]:
        te = sel[sel.site == s]
        if te.empty: continue
        try:
            xs2, c2, idx2 = load_spectra(mdir, s)
        except FileNotFoundError:
            continue
        te = te[te.code.isin(idx2.keys())]
        if te.empty or te.label_RI.nunique() < 2: continue
        Xe[s] = (gather_rows(xs2, [idx2[cc] for cc in te.code]),
                 te.label_RI.to_numpy(dtype=int))
    print(f"  dis merkez: {', '.join(Xe)}", flush=True)

    rows = []
    for k in [int(v) for v in a.ks.split(",")]:
        r2 = np.random.default_rng(100 + k)
        Xa, ya, ga = make_aug(X, y, g, k, r2)
        ia, ip = [], []
        for tri, tei in StratifiedGroupKFold(5, shuffle=True,
                                             random_state=0).split(X, y, g):
            Xtr, ytr, _ = make_aug(X[tri], y[tri], g[tri], k,
                                   np.random.default_rng(200 + k))
            m = lgb.LGBMClassifier(**CFG, random_state=0).fit(Xtr, ytr)
            p = m.predict_proba(X[tei])[:, 1]
            ia.append(roc_auc_score(y[tei], p))
            ip.append(average_precision_score(y[tei], p))
        m = lgb.LGBMClassifier(**CFG, random_state=0).fit(Xa, ya)
        r = dict(k=k, n_egitim=len(ya), ic_auroc=round(float(np.mean(ia)), 3),
                 ic_prauc=round(float(np.mean(ip)), 3))
        for s, (Xs_, ys_) in Xe.items():
            p = m.predict_proba(Xs_)[:, 1]
            r[f"{s[-1]}_auroc"] = round(roc_auc_score(ys_, p), 3)
            r[f"{s[-1]}_prauc"] = round(average_precision_score(ys_, p), 3)
            r[f"{s[-1]}_egim"] = round(cal_slope(ys_, p), 2)
        Xsa, _, _ = make_aug(Xsite, np.zeros(len(sites), dtype=int),
                             np.arange(len(sites)), k,
                             np.random.default_rng(300 + k))
        lab_a = np.tile(sites, k + 1)
        r["merkez_auroc"] = round(site_auc(Xsa, lab_a), 3)
        rows.append(r)
        print(f"  k={k}: {r}", flush=True)

    df = pd.DataFrame(rows)
    tag = f"{a.species.replace(' ','_')}__{a.drug}"
    df.to_csv(out / f"{tag}.csv", index=False)
    print("\n=== FIZIK-BILGILI ARTIRMA ===")
    print(df.to_string(index=False))
    print(f"\nkayit: {out / (tag + '.csv')}")


if __name__ == "__main__":
    main()
