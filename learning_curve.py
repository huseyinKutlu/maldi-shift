#!/usr/bin/env python3
"""Hedef-etiket ogrenme egrisi: yeni merkez icin kac lokal AST sonucu gerekir?"""
import argparse, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
sys.path.insert(0, str(Path(__file__).parent))
from nested_cv import load_spectra, gather_rows, patient_map, group_key

warnings.filterwarnings("ignore")
CFG = dict(objective="binary", num_leaves=31, n_estimators=300,
           learning_rate=0.05, colsample_bytree=0.3, subsample=0.8,
           subsample_freq=1, verbose=-1, n_jobs=12)
CFG_SMALL = dict(CFG, num_leaves=7, n_estimators=200, min_child_samples=5)
SITES = ["DRIAMS-A", "DRIAMS-B", "DRIAMS-C", "DRIAMS-D"]



LOCAL_GRID = [
    dict(num_leaves=3, n_estimators=100, min_child_samples=5, learning_rate=0.1),
    dict(num_leaves=7, n_estimators=200, min_child_samples=5, learning_rate=0.05),
    dict(num_leaves=15, n_estimators=300, min_child_samples=10, learning_rate=0.05),
    dict(num_leaves=31, n_estimators=300, min_child_samples=20, learning_rate=0.05),
]


def pick_local(Xc, yc, seed=0):
    """Kucuk ic CV ile lokal model hiperparametresi sec (adil karsilastirma)."""
    from sklearn.model_selection import StratifiedKFold
    best, bs = None, -1
    if yc.sum() >= 6 and len(yc) >= 30:
        for cfg in LOCAL_GRID:
            sc = []
            try:
                for t, v in StratifiedKFold(3, shuffle=True,
                                            random_state=seed).split(Xc, yc):
                    if len(np.unique(yc[v])) < 2:
                        continue
                    m = lgb.LGBMClassifier(**dict(CFG, **cfg),
                                           random_state=seed).fit(Xc[t], yc[t])
                    sc.append(roc_auc_score(yc[v], m.predict_proba(Xc[v])[:, 1]))
            except Exception:
                sc = []
            if sc and np.mean(sc) > bs:
                bs, best = np.mean(sc), cfg
    if best is None:
        best = LOCAL_GRID[1]
    return lgb.LGBMClassifier(**dict(CFG, **best), random_state=seed).fit(Xc, yc)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def cal_slope(y, p):
    try:
        m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        m.fit(logit(p).reshape(-1, 1), y)
        return float(m.coef_[0][0])
    except Exception:
        return np.nan


def score(y, p):
    return dict(auroc=roc_auc_score(y, p), prauc=average_precision_score(y, p),
                brier=brier_score_loss(y, p), egim=cal_slope(y, p))


def grouped_subsample(groups, y, n, rng):
    order = rng.permutation(np.unique(groups))
    take, cnt = [], 0
    for gg in order:
        ix = np.where(groups == gg)[0]
        take.append(ix); cnt += len(ix)
        if cnt >= n:
            break
    ix = np.concatenate(take)
    return ix if len(np.unique(y[ix])) > 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--drug", required=True)
    ap.add_argument("--target", default="DRIAMS-D")
    ap.add_argument("--labels", default="outputs/driams_long.parquet")
    ap.add_argument("--matrices", default="matrices")
    ap.add_argument("--root", default="~/data/DRIAMS")
    ap.add_argument("--out", default="outputs/curve")
    ap.add_argument("--ns", default="25,50,100,200,500,1000")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--test-frac", type=float, default=0.4)
    a = ap.parse_args()

    mdir = Path(a.matrices); root = Path(a.root).expanduser()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_parquet(a.labels)
    sel = lab[(lab.species == a.species) & (lab.drug == a.drug) &
              lab.tested & lab.has_spectrum]

    tr = sel[sel.site == "DRIAMS-A"]
    xs, codes, idx = load_spectra(mdir, "DRIAMS-A")
    tr = tr[tr.code.isin(idx.keys())]
    ys = tr.label_RI.to_numpy(dtype=int)
    Xs = gather_rows(xs, [idx[c] for c in tr.code])
    print(f"kaynak DRIAMS-A: n={len(ys):,} direnc={ys.mean():.3f}", flush=True)
    src = lgb.LGBMClassifier(**CFG, random_state=0).fit(Xs, ys)

    te = sel[sel.site == a.target]
    xs2, c2, idx2 = load_spectra(mdir, a.target)
    te = te[te.code.isin(idx2.keys())]
    yt = te.label_RI.to_numpy(dtype=int)
    Xt = gather_rows(xs2, [idx2[c] for c in te.code])
    gt = group_key(te.code.to_numpy(), patient_map(root, a.target), "patient")
    print(f"hedef {a.target}: n={len(yt):,} direnc={yt.mean():.3f} "
          f"grup={len(np.unique(gt)):,}", flush=True)

    ns = [int(v) for v in a.ns.split(",")]
    rows = []
    for rep in range(a.reps):
        rng = np.random.default_rng(rep)
        gsplit = StratifiedGroupKFold(int(round(1 / a.test_frac)), shuffle=True,
                                      random_state=rep)
        pool_ix, test_ix = next(iter(gsplit.split(Xt, yt, gt)))
        Xte, yte = Xt[test_ix], yt[test_ix]
        if len(np.unique(yte)) < 2:
            continue
        p0 = src.predict_proba(Xte)[:, 1]
        rows.append(dict(rep=rep, strateji="source", n=0,
                         **{k: round(v, 4) for k, v in score(yte, p0).items()}))

        Xp, yp, gp = Xt[pool_ix], yt[pool_ix], gt[pool_ix]
        for n in ns:
            if n > len(yp):
                continue
            ix = grouped_subsample(gp, yp, n, rng)
            if ix is None or yp[ix].sum() < 3:
                continue
            Xc, yc = Xp[ix], yp[ix]

            pc = src.predict_proba(Xc)[:, 1]
            lr = LogisticRegression(penalty=None, solver="lbfgs",
                                    max_iter=1000).fit(logit(pc).reshape(-1, 1), yc)
            p = lr.predict_proba(logit(p0).reshape(-1, 1))[:, 1]
            rows.append(dict(rep=rep, strateji="recal", n=len(yc),
                             **{k: round(v, 4) for k, v in score(yte, p).items()}))

            try:
                ft = lgb.LGBMClassifier(**dict(CFG_SMALL, n_estimators=100),
                                        random_state=0)
                ft.fit(Xc, yc, init_model=src.booster_)
                p = ft.predict_proba(Xte)[:, 1]
                rows.append(dict(rep=rep, strateji="finetune", n=len(yc),
                                 **{k: round(v, 4) for k, v in score(yte, p).items()}))
            except Exception:
                pass

            lo = pick_local(Xc, yc)
            p = lo.predict_proba(Xte)[:, 1]
            rows.append(dict(rep=rep, strateji="local", n=len(yc),
                             **{k: round(v, 4) for k, v in score(yte, p).items()}))

            po = lgb.LGBMClassifier(**CFG, random_state=0).fit(
                np.vstack([Xs, Xc]), np.concatenate([ys, yc]))
            p = po.predict_proba(Xte)[:, 1]
            rows.append(dict(rep=rep, strateji="pooled", n=len(yc),
                             **{k: round(v, 4) for k, v in score(yte, p).items()}))
        print(f"  tekrar {rep+1}/{a.reps} bitti (test n={len(yte)})", flush=True)

    df = pd.DataFrame(rows)
    tag = f"{a.species.replace(' ','_')}__{a.drug}__{a.target}"
    df.to_csv(out / f"{tag}_raw.csv", index=False)
    agg = (df.groupby(["strateji", "n"])[["auroc", "prauc", "brier", "egim"]]
           .agg(["mean", "std"]).round(3))
    agg.to_csv(out / f"{tag}.csv")
    print("\n=== OGRENME EGRISI ===")
    for met, dec in [("prauc", 3), ("auroc", 3), ("egim", 2)]:
        piv = df.pivot_table(index="n", columns="strateji", values=met,
                             aggfunc="mean").round(dec)
        print(f"\n{met.upper()}:"); print(piv.to_string())
    print(f"\nkayit: {out / (tag + '.csv')}")


if __name__ == "__main__":
    main()
