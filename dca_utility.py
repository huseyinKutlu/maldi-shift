#!/usr/bin/env python3
"""Klinik fayda transferi: karar egrisi analizi + ampirik tedavi simulasyonu."""
import argparse, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).parent))
from nested_cv import load_spectra, gather_rows, patient_map, group_key

warnings.filterwarnings("ignore")
CFG = dict(objective="binary", num_leaves=31, n_estimators=300,
           learning_rate=0.05, colsample_bytree=0.3, subsample=0.8,
           subsample_freq=1, verbose=-1, n_jobs=12)
SITES = ["DRIAMS-A", "DRIAMS-B", "DRIAMS-C", "DRIAMS-D"]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def net_benefit(y, p, pt):
    pred = p >= pt
    n = len(y)
    tp = np.sum(pred & (y == 1)) / n
    fp = np.sum(pred & (y == 0)) / n
    return tp - fp * (pt / (1 - pt))


def nb_all(y, pt):
    prev = y.mean()
    return prev - (1 - prev) * (pt / (1 - pt))


def dca(y, p, thresholds):
    return np.array([net_benefit(y, p, t) for t in thresholds])


def empiric_sim(y, p, pt, cost_inappropriate=1.0, cost_broad=0.15):
    broad = p >= pt
    inappropriate = np.sum((~broad) & (y == 1)) / len(y)
    unnecessary = np.sum(broad & (y == 0)) / len(y)
    base_inappropriate = y.mean()
    return dict(esik=pt, uygunsuz=inappropriate, gereksiz_genis=unnecessary,
                uygunsuz_referans=base_inappropriate,
                mutlak_azalma=base_inappropriate - inappropriate,
                bagil_azalma=(base_inappropriate - inappropriate) /
                             max(base_inappropriate, 1e-9),
                agirlikli_maliyet=inappropriate * cost_inappropriate +
                                  unnecessary * cost_broad,
                referans_maliyet=base_inappropriate * cost_inappropriate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--drug", required=True)
    ap.add_argument("--labels", default="outputs/driams_long.parquet")
    ap.add_argument("--matrices", default="matrices")
    ap.add_argument("--root", default="~/data/DRIAMS")
    ap.add_argument("--out", default="outputs/dca")
    ap.add_argument("--ncal", type=int, default=200)
    ap.add_argument("--npool", type=int, default=500)
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()

    mdir = Path(a.matrices); root = Path(a.root).expanduser()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_parquet(a.labels)
    sel = lab[(lab.species == a.species) & (lab.drug == a.drug) &
              lab.tested & lab.has_spectrum]

    tr = sel[sel.site == "DRIAMS-A"]
    xs, c, idx = load_spectra(mdir, "DRIAMS-A")
    tr = tr[tr.code.isin(idx.keys())]
    ys = tr.label_RI.to_numpy(dtype=int)
    Xs = gather_rows(xs, [idx[cc] for cc in tr.code])
    gs = group_key(tr.code.to_numpy(), patient_map(root, "DRIAMS-A"), "patient")
    print(f"kaynak: n={len(ys):,} direnc={ys.mean():.3f}", flush=True)
    src = lgb.LGBMClassifier(**CFG, random_state=0).fit(Xs, ys)

    TH = np.arange(0.05, 0.51, 0.01)
    rows, sim_rows = [], []

    oof = np.zeros(len(ys))
    for tri, tei in StratifiedGroupKFold(5, shuffle=True,
                                         random_state=0).split(Xs, ys, gs):
        m = lgb.LGBMClassifier(**CFG, random_state=0).fit(Xs[tri], ys[tri])
        oof[tei] = m.predict_proba(Xs[tei])[:, 1]
    for t, nb in zip(TH, dca(ys, oof, TH)):
        rows.append(dict(site="DRIAMS-A", kol="model", esik=round(t, 3),
                         net_fayda=round(nb, 5)))
    for t in TH:
        rows.append(dict(site="DRIAMS-A", kol="hepsi", esik=round(t, 3),
                         net_fayda=round(nb_all(ys, t), 5)))
    print(f"  DRIAMS-A dahili net fayda @0.10: "
          f"{net_benefit(ys, oof, 0.10):.4f} (hepsi: {nb_all(ys, 0.10):.4f})",
          flush=True)

    for site in SITES[1:]:
        te = sel[sel.site == site]
        if te.empty: continue
        try:
            xs2, c2, idx2 = load_spectra(mdir, site)
        except FileNotFoundError:
            continue
        te = te[te.code.isin(idx2.keys())]
        if te.empty or te.label_RI.nunique() < 2: continue
        Xt = gather_rows(xs2, [idx2[cc] for cc in te.code])
        yt = te.label_RI.to_numpy(dtype=int)
        gt = group_key(te.code.to_numpy(), patient_map(root, site), "patient")
        print(f"\n{site}: n={len(yt):,} direnc={yt.mean():.3f}", flush=True)

        acc = {k: [] for k in ["ham", "kalibre", "havuz"]}
        sims = {k: [] for k in acc}
        for rep in range(a.reps):
            rng = np.random.default_rng(rep)
            gsp = StratifiedGroupKFold(3, shuffle=True, random_state=rep)
            pool_ix, test_ix = next(iter(gsp.split(Xt, yt, gt)))
            Xte, yte = Xt[test_ix], yt[test_ix]
            if len(np.unique(yte)) < 2: continue
            Xp, yp = Xt[pool_ix], yt[pool_ix]

            p_ham = src.predict_proba(Xte)[:, 1]
            acc["ham"].append(dca(yte, p_ham, TH))

            nc = min(a.ncal, len(yp))
            ci = rng.permutation(len(yp))[:nc]
            p_cal = p_ham
            if len(np.unique(yp[ci])) > 1:
                pc = src.predict_proba(Xp[ci])[:, 1]
                lr = LogisticRegression(penalty=None, solver="lbfgs",
                                        max_iter=1000).fit(
                    logit(pc).reshape(-1, 1), yp[ci])
                p_cal = lr.predict_proba(logit(p_ham).reshape(-1, 1))[:, 1]
                acc["kalibre"].append(dca(yte, p_cal, TH))

            npl = min(a.npool, len(yp))
            pi = rng.permutation(len(yp))[:npl]
            p_pool = p_ham
            if len(np.unique(yp[pi])) > 1:
                po = lgb.LGBMClassifier(**CFG, random_state=0).fit(
                    np.vstack([Xs, Xp[pi]]), np.concatenate([ys, yp[pi]]))
                p_pool = po.predict_proba(Xte)[:, 1]
                acc["havuz"].append(dca(yte, p_pool, TH))

            for nm, pp in [("ham", p_ham), ("kalibre", p_cal), ("havuz", p_pool)]:
                for t in [0.05, 0.10, 0.20]:
                    d = empiric_sim(yte, pp, t)
                    d.update(site=site, kol=nm, rep=rep)
                    sims[nm].append(d)

        for nm, lst in acc.items():
            if not lst: continue
            mean = np.mean(np.vstack(lst), axis=0)
            for t, nb in zip(TH, mean):
                rows.append(dict(site=site, kol=nm, esik=round(t, 3),
                                 net_fayda=round(float(nb), 5)))
            i10 = int(np.argmin(np.abs(TH - 0.10)))
            print(f"  {nm:8s} net fayda @0.10: {mean[i10]:.4f}", flush=True)
        for t in TH:
            rows.append(dict(site=site, kol="hepsi", esik=round(t, 3),
                             net_fayda=round(nb_all(yt, t), 5)))
        print(f"  {'hepsi':8s} net fayda @0.10: {nb_all(yt, 0.10):.4f}", flush=True)
        for nm, lst in sims.items():
            sim_rows += lst

    df = pd.DataFrame(rows)
    tag = f"{a.species.replace(' ','_')}__{a.drug}"
    df.to_csv(out / f"{tag}_dca.csv", index=False)
    sd = pd.DataFrame(sim_rows)
    if len(sd):
        sd.to_csv(out / f"{tag}_sim_raw.csv", index=False)
        agg = (sd.groupby(["site", "kol", "esik"])
               [["uygunsuz", "gereksiz_genis", "uygunsuz_referans",
                 "bagil_azalma", "agirlikli_maliyet", "referans_maliyet"]]
               .mean().round(4))
        agg.to_csv(out / f"{tag}_sim.csv")
        print("\n=== AMPIRIK TEDAVI SIMULASYONU ===")
        print(agg.to_string())
    print(f"\nkayit: {out}")


if __name__ == "__main__":
    main()
