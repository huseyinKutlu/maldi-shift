#!/usr/bin/env python3
"""
DRIAMS envanter cikarici
Merkez x yil x tur x ilac kiriliminda n, direnc orani, eksiklik orani.
"""
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

ID_COLS = {"species", "code", "combined_code", "laboratory_species","workstation","patient_no","case_no","order_no","acquisition_date","acquisition_time"}
VALID_RSI = {"R", "I", "S"}

def find_sites(root):
    return [n for n in ["DRIAMS-A","DRIAMS-B","DRIAMS-C","DRIAMS-D"]
            if (root/n/"id").is_dir()]

def read_any(p):
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(p, low_memory=False, dtype=str, encoding=enc)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(p, low_memory=False, dtype=str, encoding="latin-1", encoding_errors="replace")

def load_site_ids(root, site):
    id_dir = root/site/"id"; frames=[]
    for year_dir in sorted(p for p in id_dir.iterdir() if p.is_dir()):
        year = year_dir.name
        for csv_path in sorted(year_dir.glob("*_strat.csv")) or sorted(year_dir.glob("*_clean.csv")):
            df = read_any(csv_path)
            df.columns = [str(c).strip() for c in df.columns]
            df = df.loc[:, [c for c in df.columns
                            if c and not c.lower().startswith("unnamed")]]
            if "code" not in df.columns or "species" not in df.columns:
                print(f"  UYARI: {csv_path} atlandi"); continue
            drug_cols = [c for c in df.columns if c not in ID_COLS]
            if not drug_cols: continue
            df["site"]=site; df["year"]=year
            frames.append(df.melt(id_vars=["site","year","code","species"],
                                  value_vars=drug_cols,
                                  var_name="drug", value_name="rsi"))
    if not frames:
        return pd.DataFrame(columns=["site","year","code","species","drug","rsi"])
    out = pd.concat(frames, ignore_index=True)
    out["rsi"] = out["rsi"].astype("string").str.strip().str.upper()
    return out

def spectrum_codes(root, site):
    bdir = root/site/"binned_6000"
    return {p.stem for p in bdir.rglob("*.txt")} if bdir.is_dir() else set()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-n", type=int, default=100)
    a = ap.parse_args()
    root = Path(a.root).expanduser().resolve()
    out = Path(a.out).expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)

    sites = find_sites(root)
    if not sites: print(f"HATA: {root} altinda DRIAMS-* yok."); sys.exit(1)
    print(f"Bulunan merkezler: {', '.join(sites)}\n")

    all_long=[]; spec_map={}
    for site in sites:
        print(f"[{site}] id dosyalari okunuyor...")
        lg = load_site_ids(root, site); codes = spectrum_codes(root, site)
        spec_map[site]=codes
        print(f"  izolat: {lg['code'].nunique() if len(lg) else 0:>7,}  |  "
              f"spektrum: {len(codes):>7,}  |  ham satir: {len(lg):>9,}")
        all_long.append(lg)
    long = pd.concat(all_long, ignore_index=True)

    long["has_spectrum"]=False
    for site, codes in spec_map.items():
        m = long["site"]==site
        long.loc[m,"has_spectrum"] = long.loc[m,"code"].isin(codes)

    long["tested"] = long["rsi"].fillna("").isin(VALID_RSI).to_numpy(dtype=bool)
    rsi = long["rsi"].fillna("")
    is_RI = rsi.isin(["R","I"]).to_numpy(dtype=bool)
    is_R  = (rsi=="R").to_numpy(dtype=bool)
    rs_only = rsi.isin(["R","S"]).to_numpy(dtype=bool)
    tested = long["tested"].to_numpy(dtype=bool)
    long["label_RI"] = np.where(tested, is_RI.astype(float), np.nan)
    long["label_R"]  = np.where(rs_only, is_R.astype(float), np.nan)

    print(f"\nToplam satir: {len(long):,}")
    print(f"Test edilmis (R/I/S): {int(long['tested'].sum()):,}")
    print(f"Spektrumu mevcut ve test edilmis: "
          f"{int((long['tested'] & long['has_spectrum']).sum()):,}")

    usable = long[long["tested"] & long["has_spectrum"]].copy()
    site_year = (usable.groupby(["site","year"], observed=True)
                 .agg(n_izolat=("code","nunique"), n_etiket=("rsi","size"),
                      n_tur=("species","nunique"), n_ilac=("drug","nunique"))
                 .reset_index())
    print("\n=== MERKEZ x YIL ===")
    print(site_year.to_string(index=False))

    pair = (usable.groupby(["site","species","drug"], observed=True)
            .agg(n=("label_RI","size"), n_direncli=("label_RI","sum"))
            .reset_index())
    pair["direnc_orani"] = pair["n_direncli"]/pair["n"]
    tot = (long[long["has_spectrum"]].groupby(["site","species"], observed=True)
           ["code"].nunique().rename("n_tur_toplam").reset_index())
    pair = pair.merge(tot, on=["site","species"], how="left")
    pair["eksiklik_orani"] = 1 - pair["n"]/pair["n_tur_toplam"]
    pair = pair.sort_values(["site","n"], ascending=[True,False])

    ok = pair[(pair["n"]>=a.min_n) &
              (pair["direnc_orani"].between(0.05,0.95))].copy()
    print(f"\n=== UYGUN CIFTLER (n >= {a.min_n}, direnc orani %5-95) ===")
    if len(ok):
        print(ok[["site","species","drug","n","direnc_orani","eksiklik_orani"]]
              .head(40).to_string(index=False,
              formatters={"direnc_orani":"{:.3f}".format,
                          "eksiklik_orani":"{:.3f}".format}))
    else:
        print("(esigi gecen cift yok)")

    if len(sites)>1:
        sets={s:set(zip(ok.loc[ok["site"]==s,"species"],
                        ok.loc[ok["site"]==s,"drug"])) for s in sites}
        rows=[{"kaynak":x,"hedef":y,"ortak_cift":len(sets[x]&sets[y])}
              for x in sites for y in sites]
        cross=pd.DataFrame(rows).pivot(index="kaynak",columns="hedef",
                                       values="ortak_cift")
        print("\n=== MERKEZLER ARASI ORTAK UYGUN CIFT ===")
        print(cross.to_string()); cross.to_csv(out/"ortak_ciftler.csv")

    try:
        long.to_parquet(out/"driams_long.parquet", index=False)
    except Exception as e:
        print(f"  (parquet yazilamadi: {e}; csv.gz)")
        long.to_csv(out/"driams_long.csv.gz", index=False)
    site_year.to_csv(out/"ozet_merkez_yil.csv", index=False)
    pair.to_csv(out/"ozet_tur_ilac.csv", index=False)
    ok.to_csv(out/"uygun_ciftler.csv", index=False)

    print(f"\nCikti klasoru: {out}")
    for f in sorted(out.iterdir()):
        print(f"  {f.name:28s} {f.stat().st_size/1e6:8.2f} MB")

if __name__ == "__main__":
    main()
