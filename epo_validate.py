#!/usr/bin/env python3
"""EPO dogrulama: cok tohumlu, eslestirilmis fark + %95 GA."""
import argparse, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.utils.extmath import randomized_svd
sys.path.insert(0, str(Path(__file__).parent))
from nested_cv import load_spectra, gather_rows, patient_map, group_key

warnings.filterwarnings("ignore")
CFG = dict(objective="binary", num_leaves=31, n_estimators=300,
           learning_rate=0.05, colsample_bytree=0.3, subsample=0.8,
           subsample_freq=1, verbose=-1, n_jobs=12)
CFG_S = dict(CFG, n_estimators=120)
SITES = ["DRIAMS-A", "DRIAMS-B", "DRIAMS-C", "DRIAMS-D"]
CACHE = {}


def spectra(mdir, site):
    if site not in CACHE:
        CACHE[site] = load_spectra(mdir, site)
    return CACHE[site]


def collect(lab, mdir, species, sites, min_n=40, max_per=400, rng=None):
    base = lab[lab.has_spectrum][["site", "species", "code"]].drop_duplicates()
    out = {}
    for sp in species:
        for s in sites:
            cs = base[(base.species == sp) & (base.site == s)].code.tolist()
            try:
                xs, c, idx = spectra(mdir, s)
            except FileNotFoundError:
                continue
            keep = [k for k in cs if k in idx]
            if len(keep) < min_n:
                continue
            if len(keep) > max_per:
                keep = list(rng.choice(keep, max_per, replace=False))
            out[(sp, s)] = gather_rows(xs, [idx[k] for k in keep])
    return out


def build_D(pool, species, sites, n_boot, n_per, rng, min_sites=2):
    rows = []
    for sp in species:
        avail = [s for s in sites if (sp, s) in pool]
        if len(avail) < min_sites:
            continue
        for _ in range(n_boot):
            mus = []
            for s in avail:
                M = pool[(sp, s)]
                take = rng.choice(len(M), min(n_per, len(M)), replace=True)
                mus.append(M[take].mean(axis=0))
            gm = np.mean(mus, axis=0)
            for mu in mus:
                rows.append(mu - gm)
    return np.vstack(rows).astype(np.float64)


def basis(D, nc=120, seed=0):
    Dc = D - D.mean(axis=0, keepdims=True)
    k = min(nc, min(Dc.shape) - 1)
    _, S, Vt = randomized_svd(Dc, n_components=k, random_state=seed)
    return Vt.T


def epo(X, V, k, rho):
    if k <= 0 or rho <= 0:
        return X
    P = V[:, :k].astype(np.float32)
    return (X - rho * ((X @ P) @ P.T)).astype(np.float32)


def site_auc(X, lab, seed=0, folds=2, minpos=40):
    a = []
    for u in np.unique(lab):
        yy = (lab == u).astype(int)
        if yy.sum() < minpos or (1 - yy).sum() < minpos:
            continue
        sc = []
        for tr, te in StratifiedKFold(folds, shuffle=True,
                                      random_state=seed).split(X, yy):
            m = lgb.LGBMClassifier(**CFG_S, random_state=seed).fit(X[tr], yy[tr])
            sc.append(roc_auc_score(yy[te], m.predict_proba(X[te])[:, 1]))
        a.append(np.mean(sc))
    return float(np.mean(a)) if a else np.nan


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
    ap.add_argument("--out", default="outputs/epo_val")
    ap.add_argument("--configs", default="20:0.5,30:0.75,2:0.5,20:1.0")
    ap.add_argument("--protocol", default="B", choices=["A", "B"])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--unseen", default="Klebsiella pneumoniae")
    a = ap.parse_args()

    mdir = Path(a.matrices); root = Path(a.root).expanduser()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_parquet(a.labels)
    rng0 = np.random.default_rng(0)

    base = lab[lab.has_spectrum][["site", "species", "code"]].drop_duplicates()
    cnt = base.groupby(["species", "site"]).size().reset_index(name="n")
    cnt = cnt[cnt.n >= 40]
    ok = cnt.groupby("species")["site"].nunique()
    excl = {a.unseen, a.species}
    D_species = [s for s in ok[ok >= 2].index if s not in excl]
    d_sites = SITES if a.protocol == "A" else [s for s in SITES if s != a.target]
    print(f"Protokol {a.protocol} | D: {len(D_species)} tur, "
          f"{len(d_sites)} merkez", flush=True)

    pool = collect(lab, mdir, set(D_species) | excl, SITES, rng=rng0)

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

    Xd, Sd = [], []
    for s in SITES:
        if (a.unseen, s) in pool:
            M = pool[(a.unseen, s)][:300]
            Xd.append(M); Sd += [s] * len(M)
    Xd = np.vstack(Xd); Sd = np.array(Sd)

    cfgs = [(0, 0.0)] + [(int(x.split(":")[0]), float(x.split(":")[1]))
                         for x in a.configs.split(",")]
    rows = []
    for rep in range(a.reps):
        rng = np.random.default_rng(1000 + rep)
        D = build_D(pool, D_species, d_sites, n_boot=40, n_per=40, rng=rng)
        V = basis(D, seed=rep)
        gs = StratifiedGroupKFold(3, shuffle=True, random_state=rep)
        _, test_ix = next(iter(gs.split(Xt0, yt, gt)))
        Xte, yte = Xt0[test_ix], yt[test_ix]
        if len(np.unique(yte)) < 2:
            continue
        for k, rho in cfgs:
            Xa = epo(X0, V, k, rho)
            Xb = epo(Xte, V, k, rho)
            m = lgb.LGBMClassifier(**CFG, random_state=0).fit(Xa, y)
            p = m.predict_proba(Xb)[:, 1]
            rows.append(dict(rep=rep, k=k, rho=rho,
                             auroc=roc_auc_score(yte, p),
                             prauc=average_precision_score(yte, p),
                             site=site_auc(epo(Xd, V, k, rho), Sd, seed=rep)))
        print(f"  tekrar {rep+1}/{a.reps} (test n={len(yte)})", flush=True)

    df = pd.DataFrame(rows)
    tag = f"{a.species.replace(' ','_')}__{a.drug}__{a.target}__prot{a.protocol}"
    df.to_csv(out / f"{tag}_raw.csv", index=False)

    print("\n=== ORTALAMALAR (%95 GA) ===")
    for (k, rho), gsub in df.groupby(["k", "rho"]):
        pa = boot_ci(gsub.prauc.values); si = boot_ci(gsub.site.values)
        print(f"  k={k:>3} rho={rho:.2f} | PR-AUC {pa[0]:.3f} [{pa[1]:.3f}-{pa[2]:.3f}]"
              f" | site {si[0]:.4f} [{si[1]:.4f}-{si[2]:.4f}]")

    print("\n=== ESLESTIRILMIS FARK (ayar - taban), %95 GA ===")
    b = df[(df.k == 0)][["rep", "prauc", "site"]].rename(
        columns={"prauc": "pr0", "site": "si0"})
    res = []
    for (k, rho), gsub in df.groupby(["k", "rho"]):
        if k == 0:
            continue
        m = gsub.merge(b, on="rep")
        dpr = boot_ci((m.prauc - m.pr0).values)
        dsi = boot_ci((m.site - m.si0).values)
        sig = "EVET" if (np.isfinite(dpr[1]) and (dpr[1] > 0 or dpr[2] < 0)) else "hayir"
        res.append(dict(k=k, rho=rho,
                        dPR=f"{dpr[0]:+.3f} [{dpr[1]:+.3f},{dpr[2]:+.3f}]",
                        dSite=f"{dsi[0]:+.4f} [{dsi[1]:+.4f},{dsi[2]:+.4f}]",
                        anlamli=sig))
        print(f"  k={k:>3} rho={rho:.2f} | dPR {res[-1]['dPR']} | "
              f"dSite {res[-1]['dSite']} | anlamli: {sig}")
    pd.DataFrame(res).to_csv(out / f"{tag}_diff.csv", index=False)
    print(f"\nkayit: {out / (tag + '_diff.csv')}")


if __name__ == "__main__":
    main()
