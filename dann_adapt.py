#!/usr/bin/env python3
"""Dogrusal olmayan alan uyarlama: 1D-CNN + {source, BN-adapt, CORAL, DANN}"""
import argparse, warnings, sys, copy
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, str(Path(__file__).parent))
from nested_cv import load_spectra, gather_rows, patient_map, group_key

warnings.filterwarnings("ignore")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SITES = ["DRIAMS-A", "DRIAMS-B", "DRIAMS-C", "DRIAMS-D"]


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)
    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


class Net(nn.Module):
    def __init__(self, feat=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1, 32, 9, stride=2, padding=4), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 9, stride=2, padding=4), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 9, stride=2, padding=4), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 128, 9, stride=2, padding=4), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(4), nn.Flatten(), nn.Linear(128 * 4, feat),
            nn.BatchNorm1d(feat), nn.ReLU())
        self.clf = nn.Sequential(nn.Dropout(0.3), nn.Linear(feat, 1))
        self.dom = nn.Sequential(nn.Linear(feat, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x, lam=0.0, dom=False):
        f = self.enc(x.unsqueeze(1))
        if dom:
            return self.clf(f).squeeze(1), self.dom(GRL.apply(f, lam)).squeeze(1), f
        return self.clf(f).squeeze(1), f


def coral_loss(fs, ft):
    d = fs.size(1)
    cs = torch.cov(fs.T) if fs.size(0) > 1 else torch.zeros(d, d, device=fs.device)
    ct = torch.cov(ft.T) if ft.size(0) > 1 else torch.zeros(d, d, device=ft.device)
    return ((cs - ct) ** 2).sum() / (4 * d * d)


def mmd_loss(fs, ft, sigmas=(1.0, 2.0, 4.0, 8.0)):
    def k(a, b):
        d2 = torch.cdist(a, b) ** 2
        return sum(torch.exp(-d2 / (2 * s ** 2)) for s in sigmas)
    return k(fs, fs).mean() + k(ft, ft).mean() - 2 * k(fs, ft).mean()


def train(Xs, ys, Xt_un, method, epochs=20, bs=128, seed=0, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    m, s = Xs.mean(0, keepdims=True), Xs.std(0, keepdims=True) + 1e-8
    Zs = torch.tensor((Xs - m) / s); Ys = torch.tensor(ys, dtype=torch.float32)
    ds = DataLoader(TensorDataset(Zs, Ys), batch_size=bs, shuffle=True, drop_last=True)
    dt = None
    if Xt_un is not None and len(Xt_un):
        Zt = torch.tensor((Xt_un - m) / s)
        dt = DataLoader(TensorDataset(Zt), batch_size=bs, shuffle=True, drop_last=True)

    net = Net().to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    pos = max(float(ys.mean()), 1e-6)
    lossf = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((1 - pos) / pos, device=DEV))
    bce = nn.BCEWithLogitsLoss()
    for ep in range(epochs):
        net.train()
        it = iter(dt) if dt is not None else None
        p = ep / max(epochs - 1, 1)
        lam = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        for xb, yb in ds:
            xb, yb = xb.to(DEV), yb.to(DEV)
            opt.zero_grad()
            xt = None
            if it is not None:
                try: xt = next(it)[0]
                except StopIteration:
                    it = iter(dt); xt = next(it)[0]
                xt = xt.to(DEV)
            if method == "dann" and xt is not None:
                lo, d1, _ = net(xb, lam, dom=True)
                _, d2, _ = net(xt, lam, dom=True)
                loss = lossf(lo, yb) + bce(d1, torch.ones_like(d1)) \
                                     + bce(d2, torch.zeros_like(d2))
            elif method == "coral" and xt is not None:
                lo, fs = net(xb); _, ft = net(xt)
                loss = lossf(lo, yb) + 1.0 * coral_loss(fs, ft)
            elif method == "mmd" and xt is not None:
                lo, fs = net(xb); _, ft = net(xt)
                loss = lossf(lo, yb) + 1.0 * mmd_loss(fs, ft)
            else:
                lo, _ = net(xb)
                loss = lossf(lo, yb)
            loss.backward(); opt.step()
    return net, (m, s)


@torch.no_grad()
def predict(net, X, ms, bn_adapt=False):
    m, s = ms
    Z = torch.tensor((X - m) / s)
    if bn_adapt:
        net = copy.deepcopy(net); net.train()
        for mod in net.modules():
            if isinstance(mod, nn.BatchNorm1d):
                mod.reset_running_stats(); mod.momentum = None
        for i in range(0, len(Z), 256):
            net(Z[i:i + 256].to(DEV))
    net.eval()
    out = []
    for i in range(0, len(Z), 512):
        lo, _ = net(Z[i:i + 512].to(DEV))
        out.append(torch.sigmoid(lo).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--drug", required=True)
    ap.add_argument("--labels", default="outputs/driams_long.parquet")
    ap.add_argument("--matrices", default="matrices")
    ap.add_argument("--root", default="~/data/DRIAMS")
    ap.add_argument("--out", default="outputs/dann")
    ap.add_argument("--methods", default="source,bnadapt,coral,mmd,dann")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--unlab", type=int, default=2000)
    a = ap.parse_args()

    print(f"cihaz: {DEV}", flush=True)
    mdir = Path(a.matrices); root = Path(a.root).expanduser()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_parquet(a.labels)
    sel = lab[(lab.species == a.species) & (lab.drug == a.drug) &
              lab.tested & lab.has_spectrum]

    tr = sel[sel.site == "DRIAMS-A"]
    xs, c, idx = load_spectra(mdir, "DRIAMS-A")
    tr = tr[tr.code.isin(idx.keys())]
    y = tr.label_RI.to_numpy(dtype=int)
    X = gather_rows(xs, [idx[cc] for cc in tr.code]).astype(np.float32)
    g = group_key(tr.code.to_numpy(), patient_map(root, "DRIAMS-A"), "patient")
    print(f"{a.species}/{a.drug}: n={len(y):,} direnc={y.mean():.3f}", flush=True)

    Xe, Xun = {}, {}
    rng = np.random.default_rng(0)
    for s in SITES[1:]:
        te = sel[sel.site == s]
        if te.empty: continue
        try:
            xs2, c2, idx2 = load_spectra(mdir, s)
        except FileNotFoundError:
            continue
        te = te[te.code.isin(idx2.keys())]
        if te.empty or te.label_RI.nunique() < 2: continue
        Xe[s] = (gather_rows(xs2, [idx2[cc] for cc in te.code]).astype(np.float32),
                 te.label_RI.to_numpy(dtype=int))
        nrow = min(a.unlab, len(c2))
        Xun[s] = gather_rows(xs2, sorted(rng.choice(len(c2), nrow,
                                                    replace=False))).astype(np.float32)
    print(f"  dis merkez: {', '.join(Xe)}", flush=True)

    print("dahili CV (kaynak-only) ...", flush=True)
    ia, ip = [], []
    for tri, tei in StratifiedGroupKFold(a.folds, shuffle=True,
                                         random_state=0).split(X, y, g):
        net, ms = train(X[tri], y[tri], None, "source", epochs=a.epochs)
        p = predict(net, X[tei], ms)
        ia.append(roc_auc_score(y[tei], p))
        ip.append(average_precision_score(y[tei], p))
    IC = (round(float(np.mean(ia)), 3), round(float(np.mean(ip)), 3))
    print(f"  dahili: AUROC {IC[0]} PR-AUC {IC[1]}", flush=True)

    rows = []
    for method in a.methods.split(","):
        r = dict(method=method, ic_auroc=IC[0], ic_prauc=IC[1])
        for s, (Xt, yt) in Xe.items():
            un = Xun.get(s) if method in ("dann", "coral", "mmd") else None
            net, ms = train(X, y, un, method, epochs=a.epochs)
            p = predict(net, Xt, ms, bn_adapt=(method == "bnadapt"))
            r[f"{s[-1]}_auroc"] = round(roc_auc_score(yt, p), 3)
            r[f"{s[-1]}_prauc"] = round(average_precision_score(yt, p), 3)
        rows.append(r)
        print(f"  {method:8s}: {r}", flush=True)

    df = pd.DataFrame(rows)
    tag = f"{a.species.replace(' ','_')}__{a.drug}"
    df.to_csv(out / f"{tag}.csv", index=False)
    print("\n=== DOGRUSAL OLMAYAN ALAN UYARLAMA ===")
    print(df.to_string(index=False))
    print(f"\nkayit: {out / (tag + '.csv')}")


if __name__ == "__main__":
    main()
