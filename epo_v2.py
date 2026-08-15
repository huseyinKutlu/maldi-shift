#!/usr/bin/env python3
"""Bootstrap Species-Conditioned EPO (soft projection) — duzeltilmis surum."""
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
CFG_S = dict(CFG, n_estimators=150)
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


def build_D_boot(pool, species, sites, n_boot=40, n_per=40, min_sites=2, rng=None):
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


def nuisance_basis(D, n_components=120):
    from sklearn.utils.extmath import randomized_svd
    Dc = D - D.mean(axis=0, keepdims=True)
    nc = min(n_components, min(Dc.shape) - 1)
    U, S, Vt = randomized_svd(Dc, n_components=nc, random_state=0)
    L = (S ** 2) / max(Dc.shape[0] - 1, 1)
    return L, Vt.T


def soft_epo(X, V, k, rho):
    if k <= 0 or rho <= 0:
        return X
    P = V[:, :k].astype(np.float32)
    return (X - rho * ((X @ P) @ P.T)).astype(np.float32)


def cal_slope(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lo = np.log(p / (1 - p)).reshape(-1, 1)
    try:
        m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(lo, y)
        return float(m.coef_[0][0])
    except Exception:
        return np.nan


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--drug", required=True)
    ap.add_argument("--target", default="DRIAMS-C")
    ap.add_argument("--labels", default="outputs/driams_long.parquet")
    ap.add_argument("--matrices", default="matrices")
    ap.add_argument("--root", default="~/data/DRIAMS")
    ap.add_argument("--out", default="outputs/epo")
    ap.add_argument("--ks", default="1,2,3,5,10,20,30,50,100")
    ap.add_argument("--rhos", default="0.25,0.5,0.75,1.0")
    ap.add_argument("--seen", default="Escherichia coli")
    ap.add_argument("--unseen", default="Klebsiella pneumoniae")
    ap.add_argument("--n-boot", type=int, default=40)
    a = ap.parse_args()

    mdir = Path(a.matrices); root = Path(a.root).expanduser()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_parquet(a.labels)
    rng = np.random.default_rng(0)

    base = lab[lab.has_spectrum][["site", "species", "code"]].drop_duplicates()
    cnt = base.groupby(["species", "site"]).size().reset_index(name="n")
    cnt = cnt[cnt.n >= 40]
    ok = cnt.groupby("species")["site"].nunique()
    all_sp = list(ok[ok >= 2].index)
    excl = {a.unseen, a.species}
    D_species = [s for s in all_sp if s not in excl]
    print(f"D icin {len(D_species)} tur (dislanan: {', '.join(excl)})", flush=True)

    pool = collect(lab, mdir, set(D_species) | excl | {a.seen}, SITES, rng=rng)

    sel = lab[(lab.species == a.species) & (lab.drug == a.drug) &
              lab.tested & lab.has_spectrum]
    tr = sel[sel.site == "DRIAMS-A"]
    xs, c, idx = spectra(mdir, "DRIAMS-A")
    tr = tr[tr.code.isin(idx.keys())]
    y = tr.label_RI.to_numpy(dtype=int)
    X0 = gather_rows(xs, [idx[cc] for cc in tr.code])
    g = group_key(tr.code.to_numpy(), patient_map(root, "DRIAMS-A"), "patient")
    te = sel[sel.site == a.target]
    xs2, c2, idx2 = spectra(mdir, a.target)
    te = te[te.code.isin(idx2.keys())]
    if te.empty or te.label_RI.nunique() < 2:
        raise SystemExit(f"{a.target} icin yeterli etiket yok")
    Xt0 = gather_rows(xs2, [idx2[cc] for cc in te.code])
    yt = te.label_RI.to_numpy(dtype=int)
    print(f"{a.species}/{a.drug}: kaynak n={len(y):,} | hedef {a.target} n={len(yt):,}",
          flush=True)

    def diag_set(sp):
        Xl, Sl = [], []
        for s in SITES:
            if (sp, s) in pool:
                M = pool[(sp, s)][:400]
                Xl.append(M); Sl += [s] * len(M)
        return (np.vstack(Xl), np.array(Sl)) if len(Xl) >= 2 else (None, None)

    DIAG = {}
    for nm, sp in [("train_species", a.seen), ("heldout_species", a.unseen),
                   ("task_species", a.species)]:
        Xd, Sd = diag_set(sp)
        if Xd is not None:
            DIAG[nm] = (Xd, Sd)
            print(f"  teshis[{nm}] {sp}: {Xd.shape[0]:,} spektrum", flush=True)

    ks = [int(v) for v in a.ks.split(",")]
    rhos = [float(v) for v in a.rhos.split(",")]
    rows = []

    for proto, d_sites in [("A_transductive", SITES),
                           ("B_unseen_site", [s for s in SITES if s != a.target])]:
        print(f"\n--- Protokol {proto} (D merkezleri: {len(d_sites)}) ---", flush=True)
        D = build_D_boot(pool, D_species, d_sites, n_boot=a.n_boot, rng=rng)
        L, V = nuisance_basis(D)
        print(f"  D: {D.shape} | ilk 5 varyans payi: " +
              ", ".join(f"{v:.3f}" for v in (L[:5] / L.sum())) +
              f" | kumulatif@50: {(L[:50].sum()/L.sum()):.3f}", flush=True)

        grid = [(0, 0.0)] + [(k, r) for k in ks for r in rhos]
        for k, rho in grid:
            X = soft_epo(X0, V, k, rho)
            Xt = soft_epo(Xt0, V, k, rho)
            ia, ip = [], []
            for tri, tei in StratifiedGroupKFold(5, shuffle=True,
                                                 random_state=0).split(X, y, g):
                m = lgb.LGBMClassifier(**CFG, random_state=0).fit(X[tri], y[tri])
                p = m.predict_proba(X[tei])[:, 1]
                ia.append(roc_auc_score(y[tei], p))
                ip.append(average_precision_score(y[tei], p))
            m = lgb.LGBMClassifier(**CFG, random_state=0).fit(X, y)
            p = m.predict_proba(Xt)[:, 1]
            r = dict(protokol=proto, k=k, rho=rho,
                     ic_auroc=round(float(np.mean(ia)), 3),
                     ic_prauc=round(float(np.mean(ip)), 3),
                     dis_auroc=round(roc_auc_score(yt, p), 3),
                     dis_prauc=round(average_precision_score(yt, p), 3),
                     dis_egim=round(cal_slope(yt, p), 2))
            for nm, (Xd, Sd) in DIAG.items():
                r[f"site_{nm}"] = round(site_auc(soft_epo(Xd, V, k, rho), Sd), 3)
            rows.append(r)
            print(f"  k={k:>3} rho={rho:.2f}: dis_PR={r['dis_prauc']:.3f} "
                  f"ic_PR={r['ic_prauc']:.3f} site_heldout_species={r.get('site_heldout_species')}",
                  flush=True)

    df = pd.DataFrame(rows)
    tag = f"{a.species.replace(' ','_')}__{a.drug}__{a.target}"
    df.to_csv(out / f"{tag}.csv", index=False)
    print("\n=== BOOTSTRAP SPECIES-CONDITIONED EPO ===")
    print(df.to_string(index=False))

    print("\n=== TABAN CIZGISINI DOMINE EDEN AYARLAR ===")
    for proto, gsub in df.groupby("protokol"):
        b = gsub[gsub.k == 0].iloc[0]
        pf = gsub[(gsub.site_heldout_species < b.site_heldout_species) &
                  (gsub.dis_prauc >= b.dis_prauc)]
        print(f"{proto}: taban site={b.site_heldout_species} dis_PR={b.dis_prauc} | "
              f"Pareto'da {len(pf)} ayar")
        if len(pf):
            print(pf[["k", "rho", "site_heldout_species", "dis_prauc",
                      "ic_prauc"]].to_string(index=False))
    print(f"\nkayit: {out / (tag + '.csv')}")


if __name__ == "__main__":
    main()
