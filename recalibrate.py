#!/usr/bin/env python3
"""Hedef merkezde yeniden kalibrasyon: kac etiketli ornek yeterli?"""
import argparse, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
sys.path.insert(0, str(Path(__file__).parent))
from nested_cv import load_spectra, gather_rows

warnings.filterwarnings("ignore")
CFG = dict(objective="binary", num_leaves=31, n_estimators=300,
           learning_rate=0.05, colsample_bytree=0.3, subsample=0.8,
           subsample_freq=1, verbose=-1, n_jobs=12)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def cal_slope_int(y, p):
    lo = logit(p).reshape(-1, 1)
    try:
        m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(lo, y)
        return float(m.coef_[0][0]), float(m.intercept_[0])
    except Exception:
        return np.nan, np.nan


def metrics(y, p):
    s, i = cal_slope_int(y, p)
    return dict(auroc=roc_auc_score(y, p), prauc=average_precision_score(y, p),
                brier=brier_score_loss(y, p), egim=s, kesisim=i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--drug", required=True)
    ap.add_argument("--labels", default="outputs/driams_long.parquet")
    ap.add_argument("--matrices", default="matrices")
    ap.add_argument("--out", default="outputs/recal")
    ap.add_argument("--ncal", default="25,50,100,200")
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()

    mdir = Path(a.matrices)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_parquet(a.labels)
    sel = lab[(lab.species == a.species) & (lab.drug == a.drug) &
              lab.tested & lab.has_spectrum]

    tr = sel[sel.site == "DRIAMS-A"]
    xs, codes, idx = load_spectra(mdir, "DRIAMS-A")
    tr = tr[tr.code.isin(idx.keys())]
    y = tr.label_RI.to_numpy(dtype=int)
    X = gather_rows(xs, [idx[c] for c in tr.code])
    print(f"{a.species}/{a.drug}: kaynak n={len(y):,} direnc={y.mean():.3f}",
          flush=True)
    model = lgb.LGBMClassifier(**CFG, random_state=0).fit(X, y)

    rows = []
    for site in ["DRIAMS-B", "DRIAMS-C", "DRIAMS-D"]:
        te = sel[sel.site == site]
        if te.empty: continue
        try:
            xs2, c2, idx2 = load_spectra(mdir, site)
        except FileNotFoundError:
            continue
        te = te[te.code.isin(idx2.keys())]
        if te.empty or te.label_RI.nunique() < 2: continue
        Xt = gather_rows(xs2, [idx2[c] for c in te.code])
        yt = te.label_RI.to_numpy(dtype=int)
        pt = model.predict_proba(Xt)[:, 1]
        print(f"\n{site}: n={len(yt):,} direnc={yt.mean():.3f}", flush=True)

        base = metrics(yt, pt)
        print(f"  ham      : AUROC {base['auroc']:.3f} PR-AUC {base['prauc']:.3f} "
              f"Brier {base['brier']:.3f} egim {base['egim']:.2f}", flush=True)
        rows.append(dict(site=site, yontem="ham", n_cal=0,
                         **{k: round(v, 3) for k, v in base.items()}))

        rng = np.random.default_rng(0)
        for n_cal in [int(v) for v in a.ncal.split(",")]:
            if n_cal >= len(yt) - 40:
                continue
            acc = {"platt": [], "isotonic": []}
            for rep in range(a.reps):
                perm = rng.permutation(len(yt))
                ci, ei = perm[:n_cal], perm[n_cal:]
                if len(np.unique(yt[ci])) < 2 or len(np.unique(yt[ei])) < 2:
                    continue
                lr = LogisticRegression(penalty=None, solver="lbfgs",
                                        max_iter=1000).fit(
                    logit(pt[ci]).reshape(-1, 1), yt[ci])
                pp = lr.predict_proba(logit(pt[ei]).reshape(-1, 1))[:, 1]
                acc["platt"].append(metrics(yt[ei], pp))
                try:
                    iso = IsotonicRegression(out_of_bounds="clip").fit(pt[ci], yt[ci])
                    pi = np.clip(iso.predict(pt[ei]), 1e-6, 1 - 1e-6)
                    acc["isotonic"].append(metrics(yt[ei], pi))
                except Exception:
                    pass
            for meth, lst in acc.items():
                if not lst: continue
                agg = {k: float(np.nanmean([d[k] for d in lst])) for k in lst[0]}
                print(f"  {meth:9s} n_cal={n_cal:>4}: AUROC {agg['auroc']:.3f} "
                      f"PR-AUC {agg['prauc']:.3f} Brier {agg['brier']:.3f} "
                      f"egim {agg['egim']:.2f}", flush=True)
                rows.append(dict(site=site, yontem=meth, n_cal=n_cal,
                                 **{k: round(v, 3) for k, v in agg.items()}))

    df = pd.DataFrame(rows)
    tag = f"{a.species.replace(' ','_')}__{a.drug}"
    df.to_csv(out / f"{tag}.csv", index=False)
    print("\n=== YENIDEN KALIBRASYON ===")
    print(df.to_string(index=False))
    print(f"\nkayit: {out / (tag + '.csv')}")


if __name__ == "__main__":
    main()
