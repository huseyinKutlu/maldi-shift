#!/usr/bin/env python3
"""MALDI on isleme zinciri + kontrol metrigi (merkez AUROC vs AMR AUROC)."""
import argparse, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.signal import savgol_filter
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, str(Path(__file__).parent))
from nested_cv import load_spectra, gather_rows, patient_map, group_key

warnings.filterwarnings("ignore")
CFG = dict(objective="binary", num_leaves=31, n_estimators=300,
           learning_rate=0.05, colsample_bytree=0.3, verbose=-1, n_jobs=12)
SITES = ["DRIAMS-A", "DRIAMS-B", "DRIAMS-C", "DRIAMS-D"]


def st_sqrt(X):
    return np.sqrt(np.clip(X, 0, None)).astype(np.float32)


def st_smooth(X, w=11, po=3):
    return savgol_filter(X, w, po, axis=1).astype(np.float32)


def snip(X, iters=40):
    Y = np.log(np.log(np.sqrt(np.clip(X, 0, None) + 1) + 1) + 1)
    B = Y.copy(); n = Y.shape[1]
    for p in range(1, iters + 1):
        a = B[:, p:n - p]
        b = (B[:, :n - 2 * p] + B[:, 2 * p:]) / 2.0
        B[:, p:n - p] = np.minimum(a, b)
    base = (np.exp(np.exp(B) - 1) - 1) ** 2 - 1
    return np.clip(X - base, 0, None).astype(np.float32)


def st_tic(X):
    t = X.sum(axis=1, keepdims=True) + 1e-9
    return (X / t * X.shape[1]).astype(np.float32)


def st_align(X, ref=None, maxshift=5):
    if ref is None:
        ref = X.mean(axis=0)
    r = (ref - ref.mean()) / (ref.std() + 1e-9)
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        x = X[i]; z = (x - x.mean()) / (x.std() + 1e-9)
        best, bs = 0, -np.inf
        for s in range(-maxshift, maxshift + 1):
            v = float(np.roll(z, s) @ r)
            if v > bs:
                bs, best = v, s
        out[i] = np.roll(x, best)
    return out.astype(np.float32)


CHAIN = [("ham", None), ("sqrt", st_sqrt), ("smooth", st_smooth),
         ("snip", snip), ("tic", st_tic), ("align", st_align)]


def site_auc(X, lab, seed=0, folds=3, minpos=40):
    a = []
    for u in np.unique(lab):
        y = (lab == u).astype(int)
        if y.sum() < minpos or (1 - y).sum() < minpos:
            continue
        sc = []
        for tr, te in StratifiedKFold(folds, shuffle=True,
                                      random_state=seed).split(X, y):
            m = lgb.LGBMClassifier(**CFG, random_state=seed).fit(X[tr], y[tr])
            sc.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        a.append(np.mean(sc))
    return float(np.mean(a)) if a else np.nan


def amr_scores(X, y, g, Xe, seed=0, folds=5):
    ia, ip = [], []
    for tr, te in StratifiedGroupKFold(folds, shuffle=True,
                                       random_state=seed).split(X, y, g):
        m = lgb.LGBMClassifier(**CFG, random_state=seed).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        ia.append(roc_auc_score(y[te], p)); ip.append(average_precision_score(y[te], p))
    m = lgb.LGBMClassifier(**CFG, random_state=seed).fit(X, y)
    ext = {}
    for s, (Xs, ys) in Xe.items():
        p = m.predict_proba(Xs)[:, 1]
        ext[s] = (roc_auc_score(ys, p), average_precision_score(ys, p))
    return float(np.mean(ia)), float(np.mean(ip)), ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", default="Staphylococcus aureus")
    ap.add_argument("--drug", default="Oxacillin")
    ap.add_argument("--site-species", default="Escherichia coli")
    ap.add_argument("--site-n", type=int, default=600)
    ap.add_argument("--out", default="outputs/prep")
    a = ap.parse_args()

    mdir = Path("matrices"); root = Path("~/data/DRIAMS").expanduser()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_parquet("outputs/driams_long.parquet")
    rng = np.random.default_rng(0)

    base = lab[lab.has_spectrum][["site", "species", "code"]].drop_duplicates()
    Xs_l, S_l = [], []
    for s in SITES:
        cs = base[(base.species == a.site_species) & (base.site == s)].code.tolist()
        xs, c, idx = load_spectra(mdir, s)
        keep = [k for k in cs if k in idx]
        if len(keep) < 60:
            continue
        if len(keep) > a.site_n:
            keep = list(rng.choice(keep, a.site_n, replace=False))
        Xs_l.append(gather_rows(xs, [idx[k] for k in keep])); S_l += [s] * len(keep)
    Xsite = np.vstack(Xs_l); sites = np.array(S_l)
    print(f"merkez seti ({a.site_species}): {Xsite.shape[0]:,} spektrum", flush=True)

    sel = lab[(lab.species == a.species) & (lab.drug == a.drug) &
              lab.tested & lab.has_spectrum]
    tr = sel[sel.site == "DRIAMS-A"]
    xs, c, idx = load_spectra(mdir, "DRIAMS-A")
    tr = tr[tr.code.isin(idx.keys())]
    y = tr.label_RI.to_numpy(dtype=int)
    X = gather_rows(xs, [idx[cc] for cc in tr.code])
    g = group_key(tr.code.to_numpy(), patient_map(root, "DRIAMS-A"), "patient")
    Xe = {}
    for s in SITES[1:]:
        te = sel[sel.site == s]
        if te.empty: continue
        xs2, c2, idx2 = load_spectra(mdir, s)
        te = te[te.code.isin(idx2.keys())]
        if te.empty or te.label_RI.nunique() < 2: continue
        Xe[s] = (gather_rows(xs2, [idx2[cc] for cc in te.code]),
                 te.label_RI.to_numpy(dtype=int))
    print(f"AMR ({a.species}/{a.drug}): n={len(y):,} dis={list(Xe)}", flush=True)

    rows = []
    for name, fn in CHAIN:
        if fn is not None:
            if name == "align":
                ref = X.mean(axis=0)
                Xsite = st_align(Xsite, ref); X = st_align(X, ref)
                Xe = {k: (st_align(v[0], ref), v[1]) for k, v in Xe.items()}
            else:
                Xsite = fn(Xsite); X = fn(X)
                Xe = {k: (fn(v[0]), v[1]) for k, v in Xe.items()}
        sa = site_auc(Xsite, sites)
        ia, ip, ext = amr_scores(X, y, g, Xe)
        r = dict(adim=name, merkez_auroc=round(sa, 3),
                 amr_ic_auroc=round(ia, 3), amr_ic_prauc=round(ip, 3))
        for s, (aa, pp) in ext.items():
            r[f"{s[-1]}_auroc"] = round(aa, 3); r[f"{s[-1]}_prauc"] = round(pp, 3)
        rows.append(r)
        print(f"  {name:8s}: {r}", flush=True)

    df = pd.DataFrame(rows)
    tag = f"{a.species.replace(' ','_')}__{a.drug}"
    df.to_csv(out / f"{tag}.csv", index=False)
    print("\n=== ON ISLEME ZINCIRI ===")
    print(df.to_string(index=False))
    print(f"\nkayit: {out / (tag + '.csv')}")


if __name__ == "__main__":
    main()
