#!/usr/bin/env python3
"""binned_6000 .txt -> .npy matris (bellek dostu, partili)."""
import argparse, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

N_BINS = 6000

def read_one(path):
    try:
        v = np.loadtxt(path, skiprows=1, usecols=1, dtype=np.float32)
    except Exception:
        return None
    return v if v.shape == (N_BINS,) else None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--site", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--year", default=None)
    ap.add_argument("--batch", type=int, default=2000)
    a = ap.parse_args()

    root = Path(a.root).expanduser().resolve()
    out = Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    bdir = root / a.site / "binned_6000"
    if a.year: bdir = bdir / a.year
    files = [f for f in sorted(bdir.rglob("*.txt"))
             if not f.name.startswith("._")]
    if not files:
        print(f"HATA: {bdir} altinda .txt yok"); sys.exit(1)
    n = len(files)
    tag = f"_{a.year}" if a.year else ""
    print(f"{a.site}: {n:,} dosya | {a.workers} islemci | parti {a.batch:,}")

    xp = out / f"X_{a.site}{tag}.npy"
    X = np.lib.format.open_memmap(xp, mode="w+", dtype=np.float32,
                                  shape=(n, N_BINS))
    codes, bad, w = [], [], 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for s in range(0, n, a.batch):
            chunk = files[s:s + a.batch]
            for f, v in zip(chunk, ex.map(read_one, chunk, chunksize=32)):
                if v is None:
                    bad.append(f.name); continue
                X[w] = v; w += 1
                codes.append(f.stem)
            X.flush()
            print(f"  {min(s+a.batch, n):,}/{n:,}", flush=True)

    del X
    if w < n:
        full = np.load(xp, mmap_mode="r")
        np.save(out / f"X_{a.site}_tmp.npy", np.asarray(full[:w]))
        del full
        (out / f"X_{a.site}_tmp.npy").replace(xp)

    np.save(out / f"codes_{a.site}{tag}.npy",
            np.array(codes, dtype=object), allow_pickle=True)
    print(f"  matris: ({w}, {N_BINS})  ({w*N_BINS*4/1e6:.1f} MB)")
    print(f"  bozuk/atlanan: {len(bad)}")
    if bad[:3]:
        print(f"  ornek: {bad[:3]}")

if __name__ == "__main__":
    main()
