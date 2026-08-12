#!/usr/bin/env python3
"""Ciftler arasi oznitelik atif karsilastirmasi (klonal ogrenme hipotezi)."""
import argparse, glob
from pathlib import Path
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr

import sys
sys.path.insert(0, str(Path(__file__).parent))
from nested_cv import load_spectra, gather_rows, patient_map, group_key, fit_best

BIN0, BINW = 2000.0, 3.0

def mz(b):
    return BIN0 + BINW * np.asarray(b)

def importance(species, drug, lab, mdir, root, site="DRIAMS-A"):
    sub = lab[(lab.species == species) & (lab.drug == drug) &
              lab.tested & lab.has_spectrum & (lab.site == site)]
    xs, codes, idx = load_spectra(mdir, site)
    sub = sub[sub.code.isin(idx.keys())]
    X = gather_rows(xs, [idx[c] for c in sub.code])
    y = sub.label_RI.to_numpy(dtype=int)
    g = group_key(sub.code.to_numpy(), patient_map(root, site), "patient")
    m, cfg, _ = fit_best(X, y, g, 0)
    imp = m.booster_.feature_importance(importance_type="gain")
    return imp / (imp.sum() + 1e-12), len(y), y.mean()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="outputs/driams_long.parquet")
    ap.add_argument("--matrices", default="matrices")
    ap.add_argument("--root", default="~/data/DRIAMS")
    ap.add_argument("--out", default="outputs/xai")
    a = ap.parse_args()

    lab = pd.read_parquet(a.labels)
    mdir = Path(a.matrices); root = Path(a.root).expanduser()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    pairs = [
        ("Escherichia coli", "Ciprofloxacin"),
        ("Escherichia coli", "Ceftriaxone"),
        ("Klebsiella pneumoniae", "Ceftriaxone"),
        ("Staphylococcus aureus", "Oxacillin"),
        ("Pseudomonas aeruginosa", "Ciprofloxacin"),
    ]

    I, meta = {}, []
    for sp, dr in pairs:
        k = f"{sp.split()[0][0]}.{sp.split()[1][:5]}-{dr[:4]}"
        print(f"egitiliyor: {sp} / {dr} ...", flush=True)
        imp, n, pos = importance(sp, dr, lab, mdir, root)
        I[k] = imp
        meta.append(dict(cift=k, n=n, direnc=round(pos, 3)))
        top = np.argsort(imp)[::-1][:10]
        print(f"  n={n:,} direnc={pos:.3f} | top-10 m/z: "
              f"{', '.join(f'{v:.0f}' for v in mz(top))}")

    df = pd.DataFrame(I)
    df.insert(0, "mz", mz(np.arange(len(df))))
    df.to_csv(out / "importance.csv", index=False)

    ks = list(I)
    print("\n=== ATIF BENZERLIGI: Spearman (tum binler) ===")
    S = pd.DataFrame(index=ks, columns=ks, dtype=float)
    for i in ks:
        for j in ks:
            S.loc[i, j] = spearmanr(I[i], I[j]).statistic
    print(S.round(3).to_string())

    print("\n=== ATIF BENZERLIGI: Jaccard (top-100 bin) ===")
    T = {k: set(np.argsort(v)[::-1][:100]) for k, v in I.items()}
    J = pd.DataFrame(index=ks, columns=ks, dtype=float)
    for i in ks:
        for j in ks:
            J.loc[i, j] = len(T[i] & T[j]) / len(T[i] | T[j])
    print(J.round(3).to_string())
    S.to_csv(out / "spearman.csv"); J.to_csv(out / "jaccard.csv")

    print("\n=== PSM-mec BOLGESI (2400-2440 Da) SIRALAMASI ===")
    lo, hi = int((2400 - BIN0) / BINW), int((2440 - BIN0) / BINW)
    for k, v in I.items():
        r = (-v).argsort().argsort()
        best = int(np.argmin(r[lo:hi + 1])) + lo
        print(f"  {k:22s} en iyi sira {int(r[best]):>5,}/6000  "
              f"(m/z {mz(best):.0f}, agirlik {v[best]:.5f})")

    print(f"\nkayit: {out}")

if __name__ == "__main__":
    main()

