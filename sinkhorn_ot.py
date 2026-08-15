#!/usr/bin/env python3
"""Kosullu Sinkhorn OT — duzeltilmis surum (sizinti giderildi, etiketsiz egri)."""
import argparse, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, ot
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils.extmath import randomized_svd
sys.path.insert(0, str(Path(__file__).parent))
from nested_cv import load_spectra, gather_rows, patient_map, group_key

warnings.filterwarnings("ignore")
CFG = dict(objective="binary", num_leaves=31, n_estimators=300,
           learning_rate=0.05, colsample_bytree=0.3, subsample=0.8,
           subsample_freq=1, verbose=-1, n_jobs=12)
CFG_S = dict(CFG, n_estimators=120)
CACHE = {}


def spectra(mdir, site):
    if site not in CACHE:
        CACHE[site] = load_spectra(mdir, site)
    return CACHE[site]


def cal_slope(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lo = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(lo, y)
        return float(m.coef_[0][0])
    except Exception:
        return np.nan


def domain_auc(Xs, Xt, seed=0, folds=2, max_n=1000):
    r = np.random.default_rng(seed)
    a = Xs[r.choice(len(Xs), min(max_n, len(Xs)), replace=False)]
    b = Xt[r.choice(len(Xt), min(max_n, len(Xt)), replace=False)]
    X = np.vstack([a, b])
    y = np.r_[np.zeros(len(a)), np.ones(len(b))].astype(int)
    sc = []
    for tr, te in StratifiedKFold(folds, shuffle=True,
                                  random_state=seed).split(X, y):
        m = lgb.LGBMClassifier(**CFG_S, random_state=seed).fit(X[tr], y[tr])
        sc.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return float(np.mean(sc))


def cost_scale(A, B, n=400, seed=0):
    r = np.random.default_rng(seed)
    a = A[r.choice(len(A), min(n, len(A)), replace=False)]
    b = B[r.choice(len(B), min(n, len(B)), replace=False)]
    C = ot.dist(a.astype(np.float64), b.astype(np.float64))
    return float(np.median(C))


def fit_ot(Xs, ys, Xt, method, reg_rel, reg_cl, cscale, max_n=1200, seed=0):
    r = np.random.default_rng(seed)
    si = r.choice(len(Xs), min(max_n, len(Xs)), replace=False)
    ti = r.choice(len(Xt), min(max_n, len(Xt)), replace=False)
    A, B, ya = Xs[si].astype(np.float64), Xt[ti].astype(np.float64), ys[si]
    reg = reg_rel * cscale
    if method == "sinkhorn":
        T = ot.da.SinkhornTransport(reg_e=reg, max_iter=200)
        T.fit(Xs=A, Xt=B)
    elif method == "sinkhorn_l1l2":
        T = ot.da.SinkhornL1l2Transport(reg_e=reg, reg_cl=reg_cl,
                                        max_iter=10, max_inner_iter=100)
        T.fit(Xs=A, ys=ya, Xt=B)
    else:
        raise ValueError(method)
    return T


def boot_ci(v, B=5000, seed=0):
    v = np.asarray([x for x in v if np.isfinite(x)])
    if len(v) < 2:
        return (np.nan, np.nan, np.nan)
    r = np.random.default_rng(seed)
    bs = r.choice(v, (B, len(v)), replace=True).mean(axis=1)
    return (float(v.mean()), float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--drug", required=True)
    ap.add_argument("--target", default="DRIAMS-C")
    ap.add_argument("--labels", default="outputs/driams_long.parquet")
    ap.add_argument("--matrices", default="matrices")
    ap.add_argument("--root", default="~/data/DRIAMS")
    ap.add_argument("--out", default="outputs/ot_v2")
    ap.add_argument("--ncomp", type=int, default=150)
    ap.add_argument("--regs", default="0.05,0.5")
    ap.add_argument("--regcls", default="0.1,1.0")
    ap.add_argument("--nunlab", default="100,250,500,0")
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()

    mdir = Path(a.matrices); root = Path(a.root).expanduser()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_parquet(a.labels)

    sel = lab[(lab.species == a.species) & (lab.drug == a.drug) &
              lab.tested & lab.has_spectrum]
    tr = sel[sel.site == "DRIAMS-A"]
    xs, c, idx = spectra(mdir, "DRIAMS-A")
    tr = tr[tr.code.isin(idx.keys())]
    y = tr.label_RI.to_numpy(dtype=int)
    X0 = gather_rows(xs, [idx[cc] for cc in tr.code])
    te = sel[sel.site == a.target]
    xs2, c2, idx2 = spectra(mdir, a.target)
    te = te[te.code.isin(idx2.keys())]
    Xt0 = gather_rows(xs2, [idx2[cc] for cc in te.code])
    yt = te.label_RI.to_numpy(dtype=int)
    gt = group_key(te.code.to_numpy(), patient_map(root, a.target), "patient")
    print(f"{a.species}/{a.drug}: kaynak n={len(y):,} | hedef n={len(yt):,} "
          f"direnc={yt.mean():.3f}", flush=True)

    regs = [float(v) for v in a.regs.split(",")]
    regcls = [float(v) for v in a.regcls.split(",")]
    nus = [int(v) for v in a.nunlab.split(",")]
    rows = []

    for rep in range(a.reps):
        rng = np.random.default_rng(rep)
        gs = StratifiedGroupKFold(3, shuffle=True, random_state=rep)
        adapt_ix, test_ix = next(iter(gs.split(Xt0, yt, gt)))
        Xte_raw, yte = Xt0[test_ix], yt[test_ix]
        Xad_full = Xt0[adapt_ix]
        if len(np.unique(yte)) < 2:
            continue

        m = lgb.LGBMClassifier(**CFG, random_state=0).fit(X0, y)
        p = m.predict_proba(Xte_raw)[:, 1]
        rows.append(dict(rep=rep, yontem="raw6000", n_unlab=-1, reg=np.nan,
                         reg_cl=np.nan, auroc=roc_auc_score(yte, p),
                         prauc=average_precision_score(yte, p),
                         egim=cal_slope(yte, p),
                         dom_once=np.nan, dom_sonra=np.nan))

        for nu in nus:
            n_use = len(Xad_full) if nu == 0 else min(nu, len(Xad_full))
            Xad = Xad_full[rng.choice(len(Xad_full), n_use, replace=False)]

            mu = np.vstack([X0, Xad]).mean(0, keepdims=True)
            _, _, Vt = randomized_svd(np.vstack([X0, Xad]) - mu,
                                      n_components=a.ncomp, random_state=rep)
            P = Vt.T.astype(np.float32)
            Z0 = ((X0 - mu) @ P).astype(np.float32)
            Zad = ((Xad - mu) @ P).astype(np.float32)
            Zte = ((Xte_raw - mu) @ P).astype(np.float32)
            d_before = domain_auc(Z0, Zad, seed=rep)
            cs = cost_scale(Z0, Zad, seed=rep)

            m = lgb.LGBMClassifier(**CFG, random_state=0).fit(Z0, y)
            p = m.predict_proba(Zte)[:, 1]
            rows.append(dict(rep=rep, yontem="svd_none", n_unlab=n_use,
                             reg=np.nan, reg_cl=np.nan,
                             auroc=roc_auc_score(yte, p),
                             prauc=average_precision_score(yte, p),
                             egim=cal_slope(yte, p),
                             dom_once=d_before, dom_sonra=d_before))

            for meth in ["sinkhorn", "sinkhorn_l1l2"]:
                rcs = [np.nan] if meth == "sinkhorn" else regcls
                for reg in regs:
                    for rc in rcs:
                        try:
                            T = fit_ot(Z0, y, Zad, meth, reg, rc, cs, seed=rep)
                            Zs = T.transform(Xs=Z0.astype(np.float64)).astype(np.float32)
                            m = lgb.LGBMClassifier(**CFG, random_state=0).fit(Zs, y)
                            p = m.predict_proba(Zte)[:, 1]
                            rows.append(dict(rep=rep, yontem=meth, n_unlab=n_use,
                                             reg=reg, reg_cl=rc,
                                             auroc=roc_auc_score(yte, p),
                                             prauc=average_precision_score(yte, p),
                                             egim=cal_slope(yte, p),
                                             dom_once=d_before,
                                             dom_sonra=domain_auc(Zs, Zad, seed=rep)))
                        except Exception as e:
                            print(f"    {meth} nu={n_use} reg={reg} HATA: {e}",
                                  flush=True)
        print(f"  tekrar {rep+1}/{a.reps} (adapt={len(Xad_full)}, test={len(yte)})",
              flush=True)

    df = pd.DataFrame(rows)
    tag = f"{a.species.replace(' ','_')}__{a.drug}__{a.target}"
    df.to_csv(out / f"{tag}_raw.csv", index=False)

    print("\n=== TABAN CIZGILERI ===")
    for nm in ["raw6000", "svd_none"]:
        s = df[df.yontem == nm]
        if len(s):
            ci = boot_ci(s.prauc.values)
            print(f"  {nm:10s} PR-AUC {ci[0]:.3f} [{ci[1]:.3f}-{ci[2]:.3f}]")

    print("\n=== OT: ETIKETSIZ HEDEF SPEKTRUM EGRISI (PR-AUC) ===")
    piv = df[df.yontem.str.startswith("sinkhorn")].pivot_table(
        index="n_unlab", columns=["yontem", "reg", "reg_cl"],
        values="prauc", aggfunc="mean", dropna=False).round(3)
    print(piv.to_string())

    print("\n=== ALAN AYIRT EDILEBILIRLIGI (once -> sonra) ===")
    dd = df[df.yontem.str.startswith("sinkhorn")].groupby(
        ["yontem", "reg", "reg_cl", "n_unlab"], dropna=False)[
        ["dom_once", "dom_sonra"]].mean().round(3)
    print(dd.to_string())

    print("\n=== ESLESTIRILMIS FARK (OT - svd_none, ayni n_unlab) ===")
    b = df[df.yontem == "svd_none"][["rep", "n_unlab", "prauc"]].rename(
        columns={"prauc": "pr0"})
    for keys, gsub in df[df.yontem.str.startswith("sinkhorn")].groupby(
            ["yontem", "reg", "reg_cl", "n_unlab"], dropna=False):
        m = gsub.merge(b, on=["rep", "n_unlab"])
        if len(m) < 3:
            continue
        ci = boot_ci((m.prauc - m.pr0).values)
        sig = "EVET" if (np.isfinite(ci[1]) and (ci[1] > 0 or ci[2] < 0)) else "hayir"
        print(f"  {keys[0]:14s} reg={keys[1]} rc={keys[2]} nu={keys[3]:>4} | "
              f"dPR {ci[0]:+.3f} [{ci[1]:+.3f},{ci[2]:+.3f}] | {sig}")
    print(f"\nkayit: {out / (tag + '_raw.csv')}")


if __name__ == "__main__":
    main()
