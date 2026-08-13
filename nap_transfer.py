#!/usr/bin/env python3
"""NAP (Nuisance Attribute Projection) + SNV ile merkezler arasi transfer."""
import argparse, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).parent))
from nested_cv import load_spectra, gather_rows, patient_map, group_key

warnings.filterwarnings("ignore")
CFG = dict(objective="binary", num_leaves=31, n_estimators=300,
           learning_rate=0.05, colsample_bytree=0.3, subsample=0.8,
           subsample_freq=1, verbose=-1, n_jobs=12)
SITES = ["DRIAMS-A", "DRIAMS-B", "DRIAMS-C", "DRIAMS-D"]


def snv(X):
    m = X.mean(axis=1, keepdims=True)
    s = X.std(axis=1, keepdims=True) + 1e-9
    return ((X - m) / s).astype(np.float32)


def build_nuisance(lab, mdir, max_per=400, min_n=60, min_sites=3):
    sub = lab[lab.has_spectrum][["site", "species", "code"]].drop_duplicates()
    cnt = sub.groupby(["species", "site"]).size().reset_index(name="n")
    cnt = cnt[cnt.n >= min_n]
    ok = cnt.groupby("species")["site"].nunique()
    species = ok[ok >= min_sites].index.tolist()
    print(f"  nuisance icin {len(species)} tur kullanilacak", flush=True)

    cache, rows = {}, []
    for sp in species:
        means = []
        for s in SITES:
            codes = sub[(sub.species == sp) & (sub.site == s)].code.tolist()
            if len(codes) < min_n:
                continue
            if s not in cache:
                try:
                    cache[s] = load_spectra(mdir, s)
                except FileNotFoundError:
                    continue
            xs, c, idx = cache[s]
            keep = [k for k in codes if k in idx][:max_per]
            if len(keep) < min_n:
                continue
            means.append(gather_rows(xs, [idx[k] for k in keep]).mean(axis=0))
        if len(means) < min_sites:
            continue
        M = np.vstack(means)
        rows.append(M - M.mean(axis=0, keepdims=True))
    if not rows:
        raise SystemExit("HATA: nuisance icin yeterli tur/merkez yok")
    D = np.vstack(rows).astype(np.float64)
    print(f"  nuisance matrisi: {D.shape}", flush=True)
    _, S, Vt = np.linalg.svd(D, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    print("  ilk 10 yonun varyans payi: " +
          ", ".join(f"{v:.3f}" for v in var[:10]), flush=True)
    return Vt, var


def project_out(X, Vt, k):
    if k <= 0:
        return X
    P = Vt[:k]
    return (X - (X @ P.T) @ P).astype(np.float32)


def cal_slope(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lo = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(lo, y)
        return float(m.coef_[0][0])
    except Exception:
        return np.nan


def evaluate(X, y, g, Xe, seed=0, folds=5):
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
        ext[s] = (roc_auc_score(ys, p), average_precision_score(ys, p),
                  brier_score_loss(ys, p), cal_slope(ys, p))
    return float(np.mean(ia)), float(np.mean(ip)), ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--drug", required=True)
    ap.add_argument("--labels", default="outputs/driams_long.parquet")
    ap.add_argument("--matrices", default="matrices")
    ap.add_argument("--root", default="~/data/DRIAMS")
    ap.add_argument("--out", default="outputs/nap")
    ap.add_argument("--coranks", default="0,1,2,3,5,8,12,20,30")
    ap.add_argument("--snv", action="store_true")
    ap.add_argument("--min-sites", type=int, default=3)
    a = ap.parse_args()

    mdir = Path(a.matrices); root = Path(a.root).expanduser()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_parquet(a.labels)

    print("nuisance alt uzayi kuruluyor...", flush=True)
    Vt, var = build_nuisance(lab, mdir, min_sites=a.min_sites)

    sel = lab[(lab.species == a.species) & (lab.drug == a.drug) &
              lab.tested & lab.has_spectrum]
    tr = sel[sel.site == "DRIAMS-A"]
    xs, codes, idx = load_spectra(mdir, "DRIAMS-A")
    tr = tr[tr.code.isin(idx.keys())]
    y = tr.label_RI.to_numpy(dtype=int)
    X0 = gather_rows(xs, [idx[c] for c in tr.code])
    g = group_key(tr.code.to_numpy(), patient_map(root, "DRIAMS-A"), "patient")
    print(f"{a.species}/{a.drug}: n={len(y):,} direnc={y.mean():.3f}", flush=True)

    Xe0 = {}
    for s in SITES[1:]:
        te = sel[sel.site == s]
        if te.empty: continue
        try:
            xs2, c2, idx2 = load_spectra(mdir, s)
        except FileNotFoundError:
            continue
        te = te[te.code.isin(idx2.keys())]
        if te.empty or te.label_RI.nunique() < 2: continue
        Xe0[s] = (gather_rows(xs2, [idx2[c] for c in te.code]),
                  te.label_RI.to_numpy(dtype=int))
    print(f"  dis merkez: {', '.join(Xe0)}", flush=True)

    if a.snv:
        X0 = snv(X0); Xe0 = {s: (snv(Xs), ys) for s, (Xs, ys) in Xe0.items()}
        print("  SNV uygulandi", flush=True)

    rows = []
    for k in [int(v) for v in a.coranks.split(",")]:
        X = project_out(X0, Vt, k)
        Xe = {s: (project_out(Xs, Vt, k), ys) for s, (Xs, ys) in Xe0.items()}
        ia, ip, ext = evaluate(X, y, g, Xe)
        r = dict(corank=k, ic_auroc=round(ia, 3), ic_prauc=round(ip, 3))
        for s, (aa, pp, bb, cs) in ext.items():
            r[f"{s[-1]}_auroc"] = round(aa, 3)
            r[f"{s[-1]}_prauc"] = round(pp, 3)
            r[f"{s[-1]}_egim"] = round(cs, 2)
        rows.append(r)
        print(f"  k={k:>3}: {r}", flush=True)

    df = pd.DataFrame(rows)
    tag = f"{a.species.replace(' ','_')}__{a.drug}" + ("__snv" if a.snv else "")
    df.to_csv(out / f"{tag}.csv", index=False)
    print("\n=== NAP CORANK TARAMASI ===")
    print(df.to_string(index=False))
    print(f"\nkayit: {out / (tag + '.csv')}")


if __name__ == "__main__":
    main()
