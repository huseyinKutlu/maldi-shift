#!/usr/bin/env python3
"""DWT alt bant teshisi: hangi bant merkezi, hangi bant direnci tasiyor?"""
import argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd, pywt
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
sys.path.insert(0, str(Path(__file__).parent))
from nested_cv import load_spectra, gather_rows, patient_map, group_key

warnings.filterwarnings("ignore")
CFG = dict(objective="binary", num_leaves=31, n_estimators=300,
           learning_rate=0.05, colsample_bytree=0.3, subsample=0.8,
           subsample_freq=1, verbose=-1, n_jobs=12)


def subbands(X, wavelet="db4", level=5):
    co = pywt.wavedec(X, wavelet, level=level, axis=1)
    names = [f"cA{level}"] + [f"cD{level-i}" for i in range(level)]
    return dict(zip(names, [np.ascontiguousarray(c, dtype=np.float32) for c in co]))


def site_auc(Xb, sites, seed=0, folds=3):
    aucs = []
    for s in np.unique(sites):
        y = (sites == s).astype(int)
        if y.sum() < 30: continue
        sc = []
        for tr, te in StratifiedKFold(folds, shuffle=True,
                                      random_state=seed).split(Xb, y):
            m = lgb.LGBMClassifier(**CFG, random_state=seed).fit(Xb[tr], y[tr])
            sc.append(roc_auc_score(y[te], m.predict_proba(Xb[te])[:, 1]))
        aucs.append(np.mean(sc))
    return float(np.mean(aucs)) if aucs else np.nan


def res_scores(Xb, y, g, Xe, seed=0, folds=5):
    ins = []
    for tr, te in StratifiedGroupKFold(folds, shuffle=True,
                                       random_state=seed).split(Xb, y, g):
        m = lgb.LGBMClassifier(**CFG, random_state=seed).fit(Xb[tr], y[tr])
        p = m.predict_proba(Xb[te])[:, 1]
        ins.append((roc_auc_score(y[te], p), average_precision_score(y[te], p)))
    m = lgb.LGBMClassifier(**CFG, random_state=seed).fit(Xb, y)
    out = {}
    for site, (Xs, ys) in Xe.items():
        p = m.predict_proba(Xs)[:, 1]
        out[site] = (roc_auc_score(ys, p), average_precision_score(ys, p))
    a = np.mean([x[0] for x in ins]); b = np.mean([x[1] for x in ins])
    return a, b, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--drug", required=True)
    ap.add_argument("--labels", default="outputs/driams_long.parquet")
    ap.add_argument("--matrices", default="matrices")
    ap.add_argument("--root", default="~/data/DRIAMS")
    ap.add_argument("--out", default="outputs/dwt")
    ap.add_argument("--site-n", type=int, default=1500)
    ap.add_argument("--level", type=int, default=5)
    a = ap.parse_args()

    mdir = Path(a.matrices); root = Path(a.root).expanduser()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    print("merkez ornekleri yukleniyor...", flush=True)
    Xs_list, site_list = [], []
    for s in ["DRIAMS-A", "DRIAMS-B", "DRIAMS-C", "DRIAMS-D"]:
        try:
            xs, codes, idx = load_spectra(mdir, s)
        except FileNotFoundError:
            continue
        n = min(a.site_n, len(codes))
        rows = rng.choice(len(codes), n, replace=False)
        Xs_list.append(gather_rows(xs, sorted(rows)))
        site_list += [s] * n
    Xsite = np.vstack(Xs_list); sites = np.array(site_list)
    print(f"  {Xsite.shape[0]:,} spektrum, {len(np.unique(sites))} merkez", flush=True)

    lab = pd.read_parquet(a.labels)
    lab = lab[(lab.species == a.species) & (lab.drug == a.drug) &
              lab.tested & lab.has_spectrum]
    tr = lab[lab.site == "DRIAMS-A"]
    xs, codes, idx = load_spectra(mdir, "DRIAMS-A")
    tr = tr[tr.code.isin(idx.keys())]
    y = tr.label_RI.to_numpy(dtype=int)
    X = gather_rows(xs, [idx[c] for c in tr.code])
    g = group_key(tr.code.to_numpy(), patient_map(root, "DRIAMS-A"), "patient")
    print(f"{a.species}/{a.drug}: n={len(y):,} direnc={y.mean():.3f}", flush=True)

    Xe_raw = {}
    for site in ["DRIAMS-B", "DRIAMS-C", "DRIAMS-D"]:
        te = lab[lab.site == site]
        if te.empty: continue
        try:
            xs2, c2, idx2 = load_spectra(mdir, site)
        except FileNotFoundError:
            continue
        te = te[te.code.isin(idx2.keys())]
        if te.empty or te.label_RI.nunique() < 2: continue
        Xe_raw[site] = (gather_rows(xs2, [idx2[c] for c in te.code]),
                        te.label_RI.to_numpy(dtype=int))
    print(f"  dis merkez: {', '.join(Xe_raw)}", flush=True)

    SB_site = subbands(Xsite, level=a.level)
    SB_tr = subbands(X, level=a.level)
    SB_ext = {s: subbands(Xs, level=a.level) for s, (Xs, _) in Xe_raw.items()}

    rows = []
    for band in SB_tr:
        print(f"[{band}] {SB_tr[band].shape[1]} katsayi ...", flush=True)
        sa = site_auc(SB_site[band], sites)
        Xe = {s: (SB_ext[s][band], ye) for s, (_, ye) in Xe_raw.items()}
        ia, ip, ext = res_scores(SB_tr[band], y, g, Xe)
        r = dict(band=band, n_coef=SB_tr[band].shape[1],
                 merkez_auc=round(sa, 3), ic_auroc=round(ia, 3),
                 ic_prauc=round(ip, 3))
        for s, (aa, pp) in ext.items():
            r[f"{s[-1]}_auroc"] = round(aa, 3); r[f"{s[-1]}_prauc"] = round(pp, 3)
        rows.append(r)
        print("   ", r, flush=True)

    print("[TAM] referans ...", flush=True)
    sa = site_auc(Xsite, sites)
    ia, ip, ext = res_scores(X, y, g, Xe_raw)
    r = dict(band="TAM", n_coef=X.shape[1], merkez_auc=round(sa, 3),
             ic_auroc=round(ia, 3), ic_prauc=round(ip, 3))
    for s, (aa, pp) in ext.items():
        r[f"{s[-1]}_auroc"] = round(aa, 3); r[f"{s[-1]}_prauc"] = round(pp, 3)
    rows.append(r)

    df = pd.DataFrame(rows)
    tag = f"{a.species.replace(' ','_')}__{a.drug}"
    df.to_csv(out / f"{tag}.csv", index=False)
    print("\n=== ALT BANT TESHISI ===")
    print(df.to_string(index=False))
    print(f"\nkayit: {out / (tag + '.csv')}")


if __name__ == "__main__":
    main()
