#!/usr/bin/env python3
"""Cihaz standardizasyonu: sozde-standart DS / PDS / prototip hizalama."""
import argparse, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, str(Path(__file__).parent))
from nested_cv import load_spectra, gather_rows, patient_map, group_key

warnings.filterwarnings("ignore")
CFG = dict(objective="binary", num_leaves=31, n_estimators=300,
           learning_rate=0.05, colsample_bytree=0.3, subsample=0.8,
           subsample_freq=1, verbose=-1, n_jobs=12)
SITES = ["DRIAMS-A", "DRIAMS-B", "DRIAMS-C", "DRIAMS-D"]
STD_SPECIES = ["Escherichia coli", "Staphylococcus aureus",
               "Klebsiella pneumoniae", "Pseudomonas aeruginosa",
               "Enterococcus faecalis", "Staphylococcus epidermidis"]


def std_matrix(lab, mdir, site, species, max_per=300, min_n=40, rng=None):
    base = lab[lab.has_spectrum][["site", "species", "code"]].drop_duplicates()
    xs, c, idx = load_spectra(mdir, site)
    rows, used = [], []
    for sp in species:
        cs = base[(base.species == sp) & (base.site == site)].code.tolist()
        keep = [k for k in cs if k in idx]
        if len(keep) < min_n:
            continue
        if len(keep) > max_per:
            keep = list(rng.choice(keep, max_per, replace=False))
        rows.append(gather_rows(xs, [idx[k] for k in keep]).mean(axis=0))
        used.append(sp)
    return (np.vstack(rows) if rows else None), used


def fit_proto(St, Ss):
    return dict(mt=St.mean(0), ms=Ss.mean(0),
                sdt=St.std(0) + 1e-9, sds=Ss.std(0) + 1e-9)


def apply_proto(X, P):
    return ((X - P["mt"]) / P["sdt"] * P["sds"] + P["ms"]).astype(np.float32)


def fit_ds(St, Ss, alpha=1.0):
    r = Ridge(alpha=alpha, fit_intercept=True).fit(St, Ss)
    return dict(F=r.coef_.T.astype(np.float32), b=r.intercept_.astype(np.float32))


def apply_ds(X, P):
    return (X @ P["F"] + P["b"]).astype(np.float32)


def fit_pds(St, Ss, win=25, alpha=1.0):
    d = St.shape[1]
    coefs, ints = [], np.zeros(d, dtype=np.float32)
    for j in range(d):
        lo, hi = max(0, j - win), min(d, j + win + 1)
        r = Ridge(alpha=alpha, fit_intercept=True).fit(St[:, lo:hi], Ss[:, j])
        coefs.append((lo, hi, r.coef_.astype(np.float32)))
        ints[j] = float(r.intercept_)
    return dict(coefs=coefs, b=ints)


def apply_pds(X, P):
    n, d = X.shape
    Y = np.empty((n, d), dtype=np.float32)
    for j, (lo, hi, w) in enumerate(P["coefs"]):
        Y[:, j] = X[:, lo:hi] @ w
    return (Y + P["b"]).astype(np.float32)


METHODS = {"none": (None, None), "proto": (fit_proto, apply_proto),
           "ds": (fit_ds, apply_ds), "pds": (fit_pds, apply_pds)}


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
    ap.add_argument("--out", default="outputs/stdz")
    ap.add_argument("--methods", default="none,proto,ds,pds")
    ap.add_argument("--pds-win", type=int, default=25)
    a = ap.parse_args()

    mdir = Path(a.matrices); root = Path(a.root).expanduser()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_parquet(a.labels)
    rng = np.random.default_rng(0)

    print("sozde-standart matrisleri...", flush=True)
    S = {}
    for s in SITES:
        M, used = std_matrix(lab, mdir, s, STD_SPECIES, rng=rng)
        if M is not None:
            S[s] = (M, used)
            print(f"  {s}: {M.shape[0]} tur", flush=True)

    sel = lab[(lab.species == a.species) & (lab.drug == a.drug) &
              lab.tested & lab.has_spectrum]
    tr = sel[sel.site == "DRIAMS-A"]
    xs, c, idx = load_spectra(mdir, "DRIAMS-A")
    tr = tr[tr.code.isin(idx.keys())]
    y = tr.label_RI.to_numpy(dtype=int)
    X = gather_rows(xs, [idx[cc] for cc in tr.code])
    g = group_key(tr.code.to_numpy(), patient_map(root, "DRIAMS-A"), "patient")
    print(f"{a.species}/{a.drug}: n={len(y):,} direnc={y.mean():.3f}", flush=True)

    Xe, Xsite_raw = {}, {}
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
        Xe[s] = (gather_rows(xs2, [idx2[cc] for cc in te.code]),
                 te.label_RI.to_numpy(dtype=int))
    for s in SITES:
        cs = base[(base.species == "Escherichia coli") & (base.site == s)].code.tolist()
        try:
            xs3, c3, idx3 = load_spectra(mdir, s)
        except FileNotFoundError:
            continue
        keep = [k for k in cs if k in idx3][:500]
        if len(keep) >= 60:
            Xsite_raw[s] = gather_rows(xs3, [idx3[k] for k in keep])
    print(f"  dis merkez: {', '.join(Xe)}", flush=True)

    model = lgb.LGBMClassifier(**CFG, random_state=0).fit(X, y)
    ic = []
    for tri, tei in StratifiedGroupKFold(5, shuffle=True,
                                         random_state=0).split(X, y, g):
        m = lgb.LGBMClassifier(**CFG, random_state=0).fit(X[tri], y[tri])
        p = m.predict_proba(X[tei])[:, 1]
        ic.append((roc_auc_score(y[tei], p), average_precision_score(y[tei], p)))
    IC = (round(float(np.mean([x[0] for x in ic])), 3),
          round(float(np.mean([x[1] for x in ic])), 3))
    print(f"  dahili: AUROC {IC[0]} PR-AUC {IC[1]}", flush=True)

    rows = []
    for meth in a.methods.split(","):
        fit, app = METHODS[meth]
        r = dict(yontem=meth, ic_auroc=IC[0], ic_prauc=IC[1])
        Xs_corr = {"DRIAMS-A": Xsite_raw.get("DRIAMS-A")}
        for s, (Xt, yt) in Xe.items():
            if fit is None:
                Xc = Xt; Xsc = Xsite_raw.get(s)
            else:
                kw = {"win": a.pds_win} if meth == "pds" else {}
                common = [i for i, u in enumerate(S[s][1]) if u in S["DRIAMS-A"][1]]
                idxA = [S["DRIAMS-A"][1].index(S[s][1][i]) for i in common]
                St = S[s][0][common]; Ss = S["DRIAMS-A"][0][idxA]
                P = fit(St, Ss, **kw)
                Xc = app(Xt, P)
                Xsc = app(Xsite_raw[s], P) if s in Xsite_raw else None
            p = model.predict_proba(Xc)[:, 1]
            r[f"{s[-1]}_auroc"] = round(roc_auc_score(yt, p), 3)
            r[f"{s[-1]}_prauc"] = round(average_precision_score(yt, p), 3)
            r[f"{s[-1]}_egim"] = round(cal_slope(yt, p), 2)
            if Xsc is not None:
                Xs_corr[s] = Xsc
        Xl, Sl = [], []
        for s, Xv in Xs_corr.items():
            if Xv is None: continue
            Xl.append(Xv); Sl += [s] * len(Xv)
        r["merkez_auroc"] = round(site_auc(np.vstack(Xl), np.array(Sl)), 3)
        rows.append(r)
        print(f"  {meth:6s}: {r}", flush=True)

    df = pd.DataFrame(rows)
    tag = f"{a.species.replace(' ','_')}__{a.drug}"
    df.to_csv(out / f"{tag}.csv", index=False)
    print("\n=== CIHAZ STANDARDIZASYONU ===")
    print(df.to_string(index=False))
    print(f"\nkayit: {out / (tag + '.csv')}")


if __name__ == "__main__":
    main()
