#!/usr/bin/env python3
"""Site-specific spectral signature ayristirmasi (5 katman)."""
import sys, warnings, json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
from nested_cv import load_spectra, gather_rows


def read_any(f):
    for enc in ('utf-8', 'latin-1'):
        try:
            return pd.read_csv(f, dtype=str, low_memory=False, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(f, dtype=str, low_memory=False, encoding='latin-1',
                       encoding_errors='replace')

warnings.filterwarnings("ignore")
CFG = dict(objective="binary", num_leaves=31, n_estimators=300,
           learning_rate=0.05, colsample_bytree=0.3, verbose=-1, n_jobs=12)
SITES = ["DRIAMS-A", "DRIAMS-B", "DRIAMS-C", "DRIAMS-D"]
SPECIES = ["Escherichia coli", "Staphylococcus aureus",
           "Klebsiella pneumoniae", "Pseudomonas aeruginosa"]
MAXN, MINN = 700, 60


def ova_auc(X, lab, seed=0, folds=3, minpos=40):
    aucs = {}
    for u in np.unique(lab):
        y = (lab == u).astype(int)
        if y.sum() < minpos or (1 - y).sum() < minpos:
            continue
        sc = []
        for tr, te in StratifiedKFold(folds, shuffle=True,
                                      random_state=seed).split(X, y):
            m = lgb.LGBMClassifier(**CFG, random_state=seed).fit(X[tr], y[tr])
            sc.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        aucs[str(u)] = float(np.mean(sc))
    return aucs


def meta_A(root):
    m = {}
    idd = root / "DRIAMS-A" / "id"
    for yd in sorted(p for p in idd.iterdir() if p.is_dir()):
        for f in sorted(yd.glob("*_strat.csv")):
            if f.name.startswith("._"):
                continue
            d = read_any(f)
            if "code" not in d.columns:
                continue
            cols = [c for c in ["code", "workstation"] if c in d.columns]
            for _, r in d[cols].iterrows():
                m[r["code"]] = (r.get("workstation", None), yd.name)
    return m


def main():
    mdir = Path("matrices"); root = Path("~/data/DRIAMS").expanduser()
    out = Path("outputs"); out.mkdir(exist_ok=True)
    lab = pd.read_parquet("outputs/driams_long.parquet")
    base = lab[lab.has_spectrum][["site", "species", "year", "code"]].drop_duplicates()
    res = lab[lab.tested & lab.has_spectrum][["site", "species", "drug",
                                              "code", "label_RI"]]
    cache = {}
    rng = np.random.default_rng(0)

    def load(site, codes):
        if site not in cache:
            cache[site] = load_spectra(mdir, site)
        xs, c, idx = cache[site]
        keep = [k for k in codes if k in idx]
        if len(keep) > MAXN:
            keep = list(rng.choice(keep, MAXN, replace=False))
        return (gather_rows(xs, [idx[k] for k in keep]), keep) if len(keep) >= MINN \
            else (None, [])

    rows = []

    def add(katman, kosul, aucs):
        if not aucs:
            return
        v = float(np.mean(list(aucs.values())))
        rows.append(dict(katman=katman, kosul=kosul, n_sinif=len(aucs),
                         makro_auroc=round(v, 3), detay=json.dumps(aucs)))
        print(f"  {katman:16s} {kosul:34s} AUROC {v:.3f} ({len(aucs)} sinif)",
              flush=True)

    print("\n[1] tur-kosullu merkez tahmini", flush=True)
    for sp in SPECIES:
        X, L = [], []
        for s in SITES:
            cs = base[(base.species == sp) & (base.site == s)].code.tolist()
            Xi, keep = load(s, cs)
            if Xi is None: continue
            X.append(Xi); L += [s] * len(keep)
        if len(X) < 2: continue
        add("1_tur", sp, ova_auc(np.vstack(X), np.array(L)))

    print("\n[2] tur + yil(2018) kosullu merkez tahmini", flush=True)
    for sp in SPECIES:
        X, L = [], []
        for s in SITES:
            cs = base[(base.species == sp) & (base.site == s) &
                      (base.year == "2018")].code.tolist()
            Xi, keep = load(s, cs)
            if Xi is None: continue
            X.append(Xi); L += [s] * len(keep)
        if len(X) < 2: continue
        add("2_tur_yil", f"{sp} | 2018", ova_auc(np.vstack(X), np.array(L)))

    print("\n[3] tur + direnc kosullu merkez tahmini", flush=True)
    pairs = [("Escherichia coli", "Ceftriaxone"),
             ("Escherichia coli", "Ciprofloxacin"),
             ("Staphylococcus aureus", "Oxacillin")]
    for sp, dr in pairs:
        sub = res[(res.species == sp) & (res.drug == dr)]
        for lv, nm in [(0.0, "duyarli"), (1.0, "direncli")]:
            X, L = [], []
            for s in SITES:
                cs = sub[(sub.site == s) & (sub.label_RI == lv)].code.tolist()
                Xi, keep = load(s, cs)
                if Xi is None: continue
                X.append(Xi); L += [s] * len(keep)
            if len(X) < 2: continue
            add("3_tur_direnc", f"{sp}/{dr} | {nm}",
                ova_auc(np.vstack(X), np.array(L), minpos=30))

    print("\n[4] DRIAMS-A ici, tur-kosullu workstation tahmini", flush=True)
    mA = meta_A(root)
    for sp in SPECIES:
        cs = base[(base.species == sp) & (base.site == "DRIAMS-A")].code.tolist()
        ws = {}
        for c in cs:
            w = mA.get(c, (None, None))[0]
            if isinstance(w, str) and w:
                ws.setdefault(w, []).append(c)
        ws = {k: v for k, v in ws.items() if len(v) >= MINN}
        if len(ws) < 2: continue
        X, L = [], []
        for w, cl in ws.items():
            Xi, keep = load("DRIAMS-A", cl)
            if Xi is None: continue
            X.append(Xi); L += [w] * len(keep)
        if len(X) < 2: continue
        add("4_workstation", sp, ova_auc(np.vstack(X), np.array(L), minpos=30))

    print("\n[5] DRIAMS-A ici, tur+workstation kosullu yil tahmini", flush=True)
    for sp in SPECIES[:2]:
        cs = base[(base.species == sp) & (base.site == "DRIAMS-A")].code.tolist()
        grp = {}
        for c in cs:
            w, y = mA.get(c, (None, None))
            if isinstance(w, str) and w and y:
                grp.setdefault(w, {}).setdefault(y, []).append(c)
        for w, yy in grp.items():
            yy = {k: v for k, v in yy.items() if len(v) >= MINN}
            if len(yy) < 2: continue
            X, L = [], []
            for y, cl in yy.items():
                Xi, keep = load("DRIAMS-A", cl)
                if Xi is None: continue
                X.append(Xi); L += [y] * len(keep)
            if len(X) < 2: continue
            add("5_yil", f"{sp} | {w}", ova_auc(np.vstack(X), np.array(L),
                                                minpos=30))

    df = pd.DataFrame(rows)
    df.to_csv(out / "site_decomposition.csv", index=False)
    print("\n=== OZET (katman ortalamasi) ===")
    print(df.groupby("katman")["makro_auroc"].agg(["mean", "min", "max", "size"])
          .round(3).to_string())
    print("\nkayit: outputs/site_decomposition.csv")


if __name__ == "__main__":
    main()
