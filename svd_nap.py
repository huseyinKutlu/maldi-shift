#!/usr/bin/env python3
"""SVD ile boyut indirge -> NAP -> dogrusal vs agac model."""
import argparse, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
sys.path.insert(0, str(Path(__file__).parent))
from nested_cv import load_spectra, gather_rows, patient_map, group_key
from nap_transfer import build_nuisance

warnings.filterwarnings("ignore")
LGB = dict(objective="binary", num_leaves=31, n_estimators=300,
           learning_rate=0.05, colsample_bytree=0.3, subsample=0.8,
           subsample_freq=1, verbose=-1, n_jobs=12)
SITES = ["DRIAMS-A", "DRIAMS-B", "DRIAMS-C", "DRIAMS-D"]


def fit_svd(X, n_comp, seed=0):
    mu = X.mean(axis=0, keepdims=True)
    _, _, Vt = randomized_svd(X - mu, n_components=n_comp, random_state=seed)
    return mu, Vt


def proj(X, mu, Vt):
    return ((X - mu) @ Vt.T).astype(np.float32)


def nap_in_svd(Vt_svd, Vt_nap, k):
    if k <= 0:
        return None
    A = Vt_nap[:k] @ Vt_svd.T
    q, _ = np.linalg.qr(A.T)
    return q.T


def remove(Z, P):
    if P is None:
        return Z
    return (Z - (Z @ P.T) @ P).astype(np.float32)


def run(Z, y, g, Ze, model, seed=0, folds=5):
    def make():
        if model == "lgbm":
            return lgb.LGBMClassifier(**LGB, random_state=seed)
        return LogisticRegression(C=1.0, max_iter=3000)
    ia, ip = [], []
    for tr, te in StratifiedGroupKFold(folds, shuffle=True,
                                       random_state=seed).split(Z, y, g):
        sc = StandardScaler().fit(Z[tr])
        m = make().fit(sc.transform(Z[tr]), y[tr])
        p = m.predict_proba(sc.transform(Z[te]))[:, 1]
        ia.append(roc_auc_score(y[te], p)); ip.append(average_precision_score(y[te], p))
    sc = StandardScaler().fit(Z)
    m = make().fit(sc.transform(Z), y)
    ext = {}
    for s, (Zs, ys) in Ze.items():
        p = m.predict_proba(sc.transform(Zs))[:, 1]
        ext[s] = (roc_auc_score(ys, p), average_precision_score(ys, p))
    return float(np.mean(ia)), float(np.mean(ip)), ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--drug", required=True)
    ap.add_argument("--labels", default="outputs/driams_long.parquet")
    ap.add_argument("--matrices", default="matrices")
    ap.add_argument("--root", default="~/data/DRIAMS")
    ap.add_argument("--out", default="outputs/svdnap")
    ap.add_argument("--ncomp", type=int, default=200)
    ap.add_argument("--coranks", default="0,2,5,10,20,40")
    ap.add_argument("--models", default="lgbm,logreg")
    ap.add_argument("--min-sites", type=int, default=3)
    a = ap.parse_args()

    mdir = Path(a.matrices); root = Path(a.root).expanduser()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_parquet(a.labels)

    print("nuisance alt uzayi...", flush=True)
    Vt_nap, _ = build_nuisance(lab, mdir, min_sites=a.min_sites)

    sel = lab[(lab.species == a.species) & (lab.drug == a.drug) &
              lab.tested & lab.has_spectrum]
    tr = sel[sel.site == "DRIAMS-A"]
    xs, codes, idx = load_spectra(mdir, "DRIAMS-A")
    tr = tr[tr.code.isin(idx.keys())]
    y = tr.label_RI.to_numpy(dtype=int)
    X = gather_rows(xs, [idx[c] for c in tr.code])
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
        Xe[s] = (gather_rows(xs2, [idx2[c] for c in te.code]),
                 te.label_RI.to_numpy(dtype=int))
    print(f"  dis merkez: {', '.join(Xe)}", flush=True)

    print(f"SVD ({a.ncomp} bilesen) ...", flush=True)
    mu, Vt_svd = fit_svd(X, a.ncomp)
    Z = proj(X, mu, Vt_svd)
    Ze = {s: (proj(Xs, mu, Vt_svd), ys) for s, (Xs, ys) in Xe.items()}

    rows = []
    for k in [int(v) for v in a.coranks.split(",")]:
        P = nap_in_svd(Vt_svd, Vt_nap, k)
        Zk = remove(Z, P)
        Zek = {s: (remove(Zs, P), ys) for s, (Zs, ys) in Ze.items()}
        for model in a.models.split(","):
            ia, ip, ext = run(Zk, y, g, Zek, model)
            r = dict(model=model, corank=k, ic_auroc=round(ia, 3),
                     ic_prauc=round(ip, 3))
            for s, (aa, pp) in ext.items():
                r[f"{s[-1]}_auroc"] = round(aa, 3); r[f"{s[-1]}_prauc"] = round(pp, 3)
            rows.append(r)
            print(f"  {model:6s} k={k:>3}: {r}", flush=True)

    df = pd.DataFrame(rows)
    tag = f"{a.species.replace(' ','_')}__{a.drug}__d{a.ncomp}"
    df.to_csv(out / f"{tag}.csv", index=False)
    print("\n=== SVD + NAP ===")
    print(df.to_string(index=False))
    print(f"\nkayit: {out / (tag + '.csv')}")


if __name__ == "__main__":
    main()
