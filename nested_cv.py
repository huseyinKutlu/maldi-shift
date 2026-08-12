#!/usr/bin/env python3
"""DRIAMS: tur-ilac cifti icin nested CV + merkezler arasi dis dogrulama."""
import argparse, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb

warnings.filterwarnings("ignore")

GRID = [
    dict(num_leaves=15, min_child_samples=20, learning_rate=0.05, n_estimators=300),
    dict(num_leaves=31, min_child_samples=20, learning_rate=0.05, n_estimators=300),
    dict(num_leaves=15, min_child_samples=50, learning_rate=0.05, n_estimators=500),
    dict(num_leaves=31, min_child_samples=50, learning_rate=0.05, n_estimators=500),
    dict(num_leaves=63, min_child_samples=20, learning_rate=0.03, n_estimators=700),
]
BASE = dict(objective="binary", colsample_bytree=0.3, subsample=0.8,
            subsample_freq=1, verbose=-1, n_jobs=12)


def load_spectra(mdir, site):
    xs, cs = [], []
    for xp in sorted(mdir.glob(f"X_{site}*.npy")):
        tag = xp.stem[len("X_"):]
        cp = mdir / f"codes_{tag}.npy"
        if not cp.exists():
            continue
        xs.append(np.load(xp, mmap_mode="r"))
        cs.append(np.load(cp, allow_pickle=True))
    if not xs:
        raise FileNotFoundError(f"{site} icin matris yok ({mdir})")
    codes = np.concatenate(cs)
    return xs, codes, {c: i for i, c in enumerate(codes)}


def gather_rows(xs, rows):
    sizes = np.cumsum([0] + [x.shape[0] for x in xs])
    out = np.empty((len(rows), xs[0].shape[1]), dtype=np.float32)
    for k, r in enumerate(rows):
        j = np.searchsorted(sizes, r, side="right") - 1
        out[k] = xs[j][r - sizes[j]]
    return out


def patient_map(root, site):
    m = {}
    id_dir = root / site / "id"
    if not id_dir.is_dir():
        return m
    for yd in sorted(p for p in id_dir.iterdir() if p.is_dir()):
        for f in sorted(yd.glob("*_strat.csv")):
            if f.name.startswith("._"):
                continue
            try:
                df = pd.read_csv(f, dtype=str, encoding="latin-1",
                                 low_memory=False, usecols=["code", "patient_no"])
            except Exception:
                continue
            for c, p in zip(df["code"], df["patient_no"]):
                m[c] = p
    return m


def group_key(codes, pmap, mode):
    if mode == "isolate":
        return np.asarray(codes, dtype=object)
    out = []
    for c in codes:
        p = pmap.get(c)
        bad = (p is None) or (not isinstance(p, str)) or p.startswith("nan_") \
              or p in ("nan", "", "NA")
        out.append(c if bad else p)
    return np.asarray(out, dtype=object)


def cal_slope_intercept(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lo = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(lo, y)
        return float(m.coef_[0][0]), float(m.intercept_[0])
    except Exception:
        return np.nan, np.nan


def metrics(y, p):
    d = dict(n=int(len(y)), pos=int(y.sum()))
    two = len(np.unique(y)) > 1
    d["auroc"] = float(roc_auc_score(y, p)) if two else np.nan
    d["prauc"] = float(average_precision_score(y, p)) if two else np.nan
    d["brier"] = float(brier_score_loss(y, p))
    d["cal_slope"], d["cal_intercept"] = cal_slope_intercept(y, p)
    return d


def fit_best(Xtr, ytr, gtr, seed, folds=3):
    inner = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    best, best_s = None, -np.inf
    for cfg in GRID:
        sc = []
        for tr, va in inner.split(Xtr, ytr, gtr):
            if len(np.unique(ytr[va])) < 2:
                continue
            m = lgb.LGBMClassifier(**BASE, **cfg, random_state=seed)
            m.fit(Xtr[tr], ytr[tr])
            sc.append(roc_auc_score(ytr[va], m.predict_proba(Xtr[va])[:, 1]))
        s = np.mean(sc) if sc else -np.inf
        if s > best_s:
            best_s, best = s, cfg
    m = lgb.LGBMClassifier(**BASE, **best, random_state=seed)
    m.fit(Xtr, ytr)
    return m, best, best_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--drug", required=True)
    ap.add_argument("--train-site", default="DRIAMS-A")
    ap.add_argument("--test-sites", default="DRIAMS-B,DRIAMS-C,DRIAMS-D")
    ap.add_argument("--labels", default="outputs/driams_long.parquet")
    ap.add_argument("--matrices", default="matrices")
    ap.add_argument("--root", default="~/data/DRIAMS")
    ap.add_argument("--out", default="outputs/cv")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--mz-min", type=float, default=2000.0)
    ap.add_argument("--mz-max", type=float, default=20000.0)
    a = ap.parse_args()

    root = Path(a.root).expanduser()
    mdir = Path(a.matrices).expanduser()
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.species.replace(' ','_')}__{a.drug.replace(' ','_')}"

    print(f"=== {a.species} / {a.drug} ===", flush=True)
    lab = pd.read_parquet(a.labels)
    lab = lab[(lab.species == a.species) & (lab.drug == a.drug) &
              lab.tested & lab.has_spectrum]
    if lab.empty:
        raise SystemExit("Bu tur-ilac cifti icin etiket yok.")

    rows = []
    tr = lab[lab.site == a.train_site]
    xs, codes, idx = load_spectra(mdir, a.train_site)
    pmap = patient_map(root, a.train_site)
    tr = tr[tr.code.isin(idx.keys())]
    y = tr.label_RI.to_numpy(dtype=int)
    X = gather_rows(xs, [idx[c] for c in tr.code])
    b0=int(max(0,(a.mz_min-2000)//3)); b1=int(min(6000,(a.mz_max-2000)//3))
    X = X[:, b0:b1]
    print(f"  mz {a.mz_min:.0f}-{a.mz_max:.0f} -> bin {b0}:{b1} ({b1-b0} ozellik)")
    print(f"{a.train_site}: n={len(y):,} | direnc={y.mean():.3f}", flush=True)

    for mode in ["patient", "isolate"]:
        g = group_key(tr.code.to_numpy(), pmap, mode)
        print(f"  [{mode}] grup sayisi: {len(np.unique(g)):,}", flush=True)
        for seed in range(a.seeds):
            outer = StratifiedGroupKFold(n_splits=a.folds, shuffle=True,
                                         random_state=seed)
            for k, (tri, tei) in enumerate(outer.split(X, y, g)):
                m, cfg, isc = fit_best(X[tri], y[tri], g[tri], seed)
                r = metrics(y[tei], m.predict_proba(X[tei])[:, 1])
                r.update(dict(kind="internal", grouping=mode, seed=seed, fold=k,
                              site=a.train_site, cfg=json.dumps(cfg)))
                rows.append(r)
            d = [x for x in rows if x["grouping"] == mode and x["seed"] == seed]
            print(f"    tohum {seed}: AUROC {np.nanmean([x['auroc'] for x in d]):.3f}",
                  flush=True)

    m, cfg, isc = fit_best(X, y, group_key(tr.code.to_numpy(), pmap, "patient"), 0)
    print(f"  dis model konfig: {cfg} (ic AUROC {isc:.3f})", flush=True)
    for site in [s for s in a.test_sites.split(",") if s.strip()]:
        te = lab[lab.site == site]
        if te.empty:
            print(f"  {site}: etiket yok, atlandi"); continue
        try:
            xs2, c2, idx2 = load_spectra(mdir, site)
        except FileNotFoundError:
            print(f"  {site}: matris yok, atlandi"); continue
        te = te[te.code.isin(idx2.keys())]
        if te.empty or te.label_RI.nunique() < 2:
            print(f"  {site}: yetersiz, atlandi"); continue
        X2 = gather_rows(xs2, [idx2[c] for c in te.code])
        X2 = X2[:, b0:b1]
        y2 = te.label_RI.to_numpy(dtype=int)
        r = metrics(y2, m.predict_proba(X2)[:, 1])
        r.update(dict(kind="external", grouping="-", seed=0, fold=-1,
                      site=site, cfg=json.dumps(cfg)))
        rows.append(r)
        print(f"  {site}: n={r['n']:,} direnc={y2.mean():.3f} "
              f"AUROC={r['auroc']:.3f} PR-AUC={r['prauc']:.3f} "
              f"Brier={r['brier']:.3f} egim={r['cal_slope']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / f"{tag}.csv", index=False)
    print("\n=== OZET ===")
    print(df[df.kind == "internal"].groupby("grouping")[
        ["auroc", "prauc", "brier", "cal_slope"]].agg(["mean", "std"]).round(3).to_string())
    ex = df[df.kind == "external"]
    if len(ex):
        print()
        print(ex[["site", "n", "pos", "auroc", "prauc", "brier",
                  "cal_slope"]].round(3).to_string(index=False))
    print(f"\nkayit: {out / (tag + '.csv')}")


if __name__ == "__main__":
    main()

