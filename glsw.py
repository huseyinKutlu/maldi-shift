#!/usr/bin/env python3
"""GLSW / EPO: tur-kosullu cihaz standardizasyonu (yumusak filtreleme)."""
import argparse, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, str(Path(__file__).parent))
from nested_cv import load_spectra, gather_rows, patient_map, group_key

warnings.filterwarnings("ignore")
CFG = dict(objective="binary", num_leaves=31, n_estimators=300,
           learning_rate=0.05, colsample_bytree=0.3, subsample=0.8,
           subsample_freq=1, verbose=-1, n_jobs=12)
SITES = ["DRIAMS-A", "DRIAMS-B", "DRIAMS-C", "DRIAMS-D"]


def build_D(lab, mdir, min_n=40, max_per=250, min_sites=2, rng=None):
    base = lab[lab.has_spectrum][["site", "species", "code"]].drop_duplicates()
    cnt = base.groupby(["species", "site"]).size().reset_index(name="n")
    cnt = cnt[cnt.n >= min_n]
    ok = cnt.groupby("species")["site"].nunique()
    species = ok[ok >= min_sites].index.tolist()
    print(f"  {len(species)} tur kullanilacak", flush=True)

    cache, rows = {}, []
    for sp in species:
        per_site = []
        for s in SITES:
            cs = base[(base.species == sp) & (base.site == s)].code.tolist()
            if len(cs) < min_n:
                continue
            if s not in cache:
                try:
                    cache[s] = load_spectra(mdir, s)
                except FileNotFoundError:
                    continue
            xs, c, idx = cache[s]
            keep = [k for k in cs if k in idx]
            if len(keep) < min_n:
                continue
            if len(keep) > max_per:
                keep = list(rng.choice(keep, max_per, replace=False))
            per_site.append(gather_rows(xs, [idx[k] for k in keep]))
        if len(per_site) < min_sites:
            continue
        gm = np.vstack(per_site).mean(axis=0)
        for M in per_site:
            rows.append(M - gm)
    D = np.vstack(rows).astype(np.float64)
    print(f"  D matrisi: {D.shape}", flush=True)
    return D


def eig_nuisance(D):
    C = (D.T @ D) / max(D.shape[0] - 1, 1)
    L, V = np.linalg.eigh(C)
    return np.clip(L, 0, None), V


def glsw(L, V, alpha):
    w = 1.0 / np.sqrt(1.0 + L / alpha)
    return (V * w) @ V.T


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
    ap.add_argument("--out", default="outputs/glsw")
    ap.add_argument("--alphas", default="inf,100,10,1,0.1,0.01")
    a = ap.parse_args()

    mdir = Path(a.matrices); root = Path(a.root).expanduser()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_parquet(a.labels)
    rng = np.random.default_rng(0)

    print("nuisance kovaryansi kuruluyor...", flush=True)
    D = build_D(lab, mdir, rng=rng)
    print("  ozayrisim...", flush=True)
    L, V = eig_nuisance(D)
    print("  ilk 5 ozdeger payi: " +
          ", ".join(f"{v:.3f}" for v in (L[::-1][:5] / L.sum())), flush=True)

    sel = lab[(lab.species == a.species) & (lab.drug == a.drug) &
              lab.tested & lab.has_spectrum]
    tr = sel[sel.site == "DRIAMS-A"]
    xs, c, idx = load_spectra(mdir, "DRIAMS-A")
    tr = tr[tr.code.isin(idx.keys())]
    y = tr.label_RI.to_numpy(dtype=int)
    X0 = gather_rows(xs, [idx[cc] for cc in tr.code])
    g = group_key(tr.code.to_numpy(), patient_map(root, "DRIAMS-A"), "patient")
    print(f"{a.species}/{a.drug}: n={len(y):,} direnc={y.mean():.3f}", flush=True)

    Xe0, Xsite0 = {}, {}
    base = lab[lab.has_spectrum][["site", "species", "code"]].drop_duplicates()
    for s in SITES[1:]:
        te = sel[sel.site == s]
        if te.empty: continue
        try:
            xs2, c2, idx2 = load_spectra(mdir, s)
        except FileNotFoundError:
            continue
        te = te[te.code.isin(idx2.keys())]
        if te.empty or te.label_RI.nunique() < 2: continue
        Xe0[s] = (gather_rows(xs2, [idx2[cc] for cc in te.code]),
                  te.label_RI.to_numpy(dtype=int))
    for s in SITES:
        cs = base[(base.species == "Escherichia coli") & (base.site == s)].code.tolist()
        try:
            xs3, c3, idx3 = load_spectra(mdir, s)
        except FileNotFoundError:
            continue
        keep = [k for k in cs if k in idx3][:500]
        if len(keep) >= 60:
            Xsite0[s] = gather_rows(xs3, [idx3[k] for k in keep])
    print(f"  dis merkez: {', '.join(Xe0)}", flush=True)

    rows = []
    for av in a.alphas.split(","):
        alpha = float("inf") if av == "inf" else float(av)
        W = None if np.isinf(alpha) else glsw(L, V, alpha).astype(np.float32)
        tf = (lambda Z: Z) if W is None else (lambda Z: (Z @ W).astype(np.float32))
        X = tf(X0)
        Xe = {s: (tf(v[0]), v[1]) for s, v in Xe0.items()}

        ia, ip = [], []
        for tri, tei in StratifiedGroupKFold(5, shuffle=True,
                                             random_state=0).split(X, y, g):
            m = lgb.LGBMClassifier(**CFG, random_state=0).fit(X[tri], y[tri])
            p = m.predict_proba(X[tei])[:, 1]
            ia.append(roc_auc_score(y[tei], p))
            ip.append(average_precision_score(y[tei], p))
        m = lgb.LGBMClassifier(**CFG, random_state=0).fit(X, y)
        r = dict(alpha=av, ic_auroc=round(float(np.mean(ia)), 3),
                 ic_prauc=round(float(np.mean(ip)), 3))
        for s, (Xs_, ys_) in Xe.items():
            p = m.predict_proba(Xs_)[:, 1]
            r[f"{s[-1]}_auroc"] = round(roc_auc_score(ys_, p), 3)
            r[f"{s[-1]}_prauc"] = round(average_precision_score(ys_, p), 3)
            r[f"{s[-1]}_egim"] = round(cal_slope(ys_, p), 2)
        Xl, Sl = [], []
        for s, Xv in Xsite0.items():
            Xl.append(tf(Xv)); Sl += [s] * len(Xv)
        r["merkez_auroc"] = round(site_auc(np.vstack(Xl), np.array(Sl)), 3)
        rows.append(r)
        print(f"  alpha={av:>6}: {r}", flush=True)

    df = pd.DataFrame(rows)
    tag = f"{a.species.replace(' ','_')}__{a.drug}"
    df.to_csv(out / f"{tag}.csv", index=False)
    print("\n=== GLSW / EPO ===")
    print(df.to_string(index=False))
    print(f"\nkayit: {out / (tag + '.csv')}")


if __name__ == "__main__":
    main()
