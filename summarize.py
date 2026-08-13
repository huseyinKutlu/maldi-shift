#!/usr/bin/env python3
"""Ogrenme egrilerinden bootstrap %95 GA ve ozet tablolar."""
import glob
from pathlib import Path
import numpy as np, pandas as pd

RNG = np.random.default_rng(0)
B = 5000


def ci(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)])
    if len(v) < 2:
        return (np.nan, np.nan, np.nan)
    bs = RNG.choice(v, (B, len(v)), replace=True).mean(axis=1)
    return (float(v.mean()), float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)))


def fmt(t):
    if not np.isfinite(t[0]):
        return "-"
    return f"{t[0]:.3f} [{t[1]:.3f}-{t[2]:.3f}]"


rows, diffs = [], []
for f in sorted(glob.glob("outputs/curve2/*_raw.csv")):
    d = pd.read_csv(f)
    name = Path(f).stem.replace("_raw", "")
    parts = name.split("__")
    sp, dr, tg = parts[0].replace("_", " "), parts[1], parts[2]
    src = d[d.strateji == "source"]
    for met in ["prauc", "auroc", "egim"]:
        base = ci(src[met].values)
        for strat in ["recal", "finetune", "local", "pooled"]:
            for n in sorted(d[d.strateji == strat]["n"].unique()):
                sub = d[(d.strateji == strat) & (d.n == n)]
                rows.append(dict(cift=f"{sp}/{dr}", hedef=tg, metrik=met,
                                 strateji=strat, n=int(n),
                                 deger=fmt(ci(sub[met].values)),
                                 kaynak=fmt(base)))
                if met == "prauc":
                    # eslesmis fark (ayni tekrar icinde)
                    m = sub.merge(src[["rep", met]], on="rep",
                                  suffixes=("", "_src"))
                    dv = (m[met] - m[f"{met}_src"]).values
                    t = ci(dv)
                    diffs.append(dict(cift=f"{sp}/{dr}", hedef=tg,
                                      strateji=strat, n=int(n),
                                      fark=fmt(t),
                                      anlamli="EVET" if (np.isfinite(t[1]) and
                                              (t[1] > 0 or t[2] < 0)) else "hayir"))

pd.DataFrame(rows).to_csv("outputs/ozet_egri_ci.csv", index=False)
df = pd.DataFrame(diffs)
df.to_csv("outputs/ozet_egri_fark.csv", index=False)

print("=== PR-AUC FARKI (strateji - kaynak), %95 GA ===\n")
for (c, h), g in df.groupby(["cift", "hedef"]):
    print(f"--- {c} -> {h} ---")
    p = g.pivot(index="n", columns="strateji", values="fark")
    print(p.to_string())
    q = g.pivot(index="n", columns="strateji", values="anlamli")
    print("anlamli:"); print(q.to_string()); print()

print("kayit: outputs/ozet_egri_ci.csv, outputs/ozet_egri_fark.csv")
