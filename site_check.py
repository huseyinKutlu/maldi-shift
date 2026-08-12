#!/usr/bin/env python3
"""Tur sabitken merkez ne kadar ayirt ediliyor? (NAP oncesi kontrol)"""
import sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path(__file__).parent))
from nested_cv import load_spectra, gather_rows
warnings.filterwarnings("ignore")

CFG = dict(objective="binary", num_leaves=31, n_estimators=300,
           learning_rate=0.05, colsample_bytree=0.3, verbose=-1, n_jobs=12)
SITES = ["DRIAMS-A", "DRIAMS-B", "DRIAMS-C", "DRIAMS-D"]
MAXN = 800

def load_species(lab, sp):
    sub = lab[(lab.species == sp) & lab.has_spectrum][["site", "code"]].drop_duplicates()
    X, S = [], []
    for s in SITES:
        codes = sub.loc[sub.site == s, "code"].tolist()
        if not codes: continue
        try:
            xs, c, idx = load_spectra(Path("matrices"), s)
        except FileNotFoundError:
            continue
        keep = [c for c in codes if c in idx][:MAXN]
        if len(keep) < 60: continue
        X.append(gather_rows(xs, [idx[c] for c in keep])); S += [s] * len(keep)
    if len(X) < 2: return None, None
    return np.vstack(X), np.array(S)

def site_auc(X, S, seed=0):
    out = {}
    for s in np.unique(S):
        y = (S == s).astype(int)
        if y.sum() < 40 or (1 - y).sum() < 40: continue
        sc = []
        for tr, te in StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y):
            m = lgb.LGBMClassifier(**CFG, random_state=seed).fit(X[tr], y[tr])
            sc.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
        out[s] = float(np.mean(sc))
    return out

lab = pd.read_parquet("outputs/driams_long.parquet")
rows = []
for sp in ["Escherichia coli", "Staphylococcus aureus",
           "Klebsiella pneumoniae", "Pseudomonas aeruginosa"]:
    X, S = load_species(lab, sp)
    if X is None:
        print(f"{sp}: yetersiz, atlandi"); continue
    cnt = dict(zip(*np.unique(S, return_counts=True)))
    print(f"{sp}: {X.shape[0]:,} spektrum {cnt}", flush=True)
    au = site_auc(X, S)
    for s, v in au.items():
        print(f"   {s}: AUROC {v:.3f}")
        rows.append(dict(tur=sp, merkez=s, auroc=round(v, 3)))
    rows.append(dict(tur=sp, merkez="ORTALAMA",
                     auroc=round(float(np.mean(list(au.values()))), 3)))

df = pd.DataFrame(rows)
Path("outputs").mkdir(exist_ok=True)
df.to_csv("outputs/site_check.csv", index=False)
print("\n=== TUR-ICI MERKEZ AYIRT EDILEBILIRLIGI ===")
print(df[df.merkez == "ORTALAMA"].to_string(index=False))
print("\nkayit: outputs/site_check.csv")

