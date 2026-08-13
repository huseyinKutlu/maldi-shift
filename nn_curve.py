#!/usr/bin/env python3
"""Sinir agi ile hedef-etiket ogrenme egrisi (fine-tuning kollari dahil)."""
import argparse, warnings, sys, copy
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).parent))
from nested_cv import load_spectra, gather_rows, patient_map, group_key

warnings.filterwarnings("ignore")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


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

    def forward(self, x):
        return self.clf(self.enc(x.unsqueeze(1))).squeeze(1)


def fit(X, y, ms=None, net=None, epochs=25, lr=1e-3, bs=64, seed=0,
        freeze_enc=False):
    torch.manual_seed(seed); np.random.seed(seed)
    if ms is None:
        ms = (X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-8)
    m, s = ms
    Z = torch.tensor((X - m) / s); Y = torch.tensor(y, dtype=torch.float32)
    bs = min(bs, max(8, len(y) // 4))
    dl = DataLoader(TensorDataset(Z, Y), batch_size=bs, shuffle=True,
                    drop_last=len(y) > 2 * bs)
    net = Net().to(DEV) if net is None else copy.deepcopy(net)
    if freeze_enc:
        for p in net.enc.parameters():
            p.requires_grad = False
        params = list(net.clf.parameters())
    else:
        params = list(net.parameters())
    opt = torch.optim.AdamW([p for p in params if p.requires_grad], lr=lr,
                            weight_decay=1e-4)
    pos = max(float(y.mean()), 1e-6)
    lf = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((1 - pos) / pos, device=DEV))
    for _ in range(epochs):
        net.train()
        if freeze_enc:
            for mod in net.enc.modules():
                if isinstance(mod, nn.BatchNorm1d):
                    mod.eval()
        for xb, yb in dl:
            opt.zero_grad()
            loss = lf(net(xb.to(DEV)), yb.to(DEV))
            loss.backward(); opt.step()
    return net, ms


@torch.no_grad()
def pred(net, X, ms):
    m, s = ms
    Z = torch.tensor((X - m) / s)
    net.eval()
    out = []
    for i in range(0, len(Z), 512):
        out.append(torch.sigmoid(net(Z[i:i + 512].to(DEV))).cpu().numpy())
    return np.concatenate(out)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sc(y, p):
    return dict(auroc=roc_auc_score(y, p), prauc=average_precision_score(y, p))


def gsub(groups, y, n, rng):
    order = rng.permutation(np.unique(groups))
    take, cnt = [], 0
    for gg in order:
        ix = np.where(groups == gg)[0]
        take.append(ix); cnt += len(ix)
        if cnt >= n:
            break
    ix = np.concatenate(take)
    return ix if len(np.unique(y[ix])) > 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True)
    ap.add_argument("--drug", required=True)
    ap.add_argument("--target", default="DRIAMS-D")
    ap.add_argument("--labels", default="outputs/driams_long.parquet")
    ap.add_argument("--matrices", default="matrices")
    ap.add_argument("--root", default="~/data/DRIAMS")
    ap.add_argument("--out", default="outputs/nncurve")
    ap.add_argument("--ns", default="25,50,100,200,500")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=25)
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
    ys = tr.label_RI.to_numpy(dtype=int)
    Xs = gather_rows(xs, [idx[cc] for cc in tr.code]).astype(np.float32)
    print(f"kaynak: n={len(ys):,} direnc={ys.mean():.3f}", flush=True)
    src, ms = fit(Xs, ys, epochs=a.epochs)

    te = sel[sel.site == a.target]
    xs2, c2, idx2 = load_spectra(mdir, a.target)
    te = te[te.code.isin(idx2.keys())]
    yt = te.label_RI.to_numpy(dtype=int)
    Xt = gather_rows(xs2, [idx2[cc] for cc in te.code]).astype(np.float32)
    gt = group_key(te.code.to_numpy(), patient_map(root, a.target), "patient")
    print(f"hedef {a.target}: n={len(yt):,} direnc={yt.mean():.3f}", flush=True)

    ns = [int(v) for v in a.ns.split(",")]
    rows = []
    for rep in range(a.reps):
        rng = np.random.default_rng(rep)
        gs = StratifiedGroupKFold(3, shuffle=True, random_state=rep)
        pool_ix, test_ix = next(iter(gs.split(Xt, yt, gt)))
        Xte, yte = Xt[test_ix], yt[test_ix]
        if len(np.unique(yte)) < 2:
            continue
        p0 = pred(src, Xte, ms)
        rows.append(dict(rep=rep, kol="source", n=0,
                         **{k: round(v, 4) for k, v in sc(yte, p0).items()}))
        Xp, yp, gp = Xt[pool_ix], yt[pool_ix], gt[pool_ix]

        for n in ns:
            if n > len(yp):
                continue
            ix = gsub(gp, yp, n, rng)
            if ix is None or yp[ix].sum() < 3:
                continue
            Xc, yc = Xp[ix], yp[ix]

            pc = pred(src, Xc, ms)
            lr_ = LogisticRegression(penalty=None, solver="lbfgs",
                                     max_iter=1000).fit(logit(pc).reshape(-1, 1), yc)
            p = lr_.predict_proba(logit(p0).reshape(-1, 1))[:, 1]
            rows.append(dict(rep=rep, kol="recal", n=len(yc),
                             **{k: round(v, 4) for k, v in sc(yte, p).items()}))

            nh, _ = fit(Xc, yc, ms=ms, net=src, epochs=30, lr=1e-3,
                        freeze_enc=True, seed=rep)
            rows.append(dict(rep=rep, kol="head", n=len(yc),
                             **{k: round(v, 4)
                                for k, v in sc(yte, pred(nh, Xte, ms)).items()}))

            nf, _ = fit(Xc, yc, ms=ms, net=src, epochs=15, lr=1e-4, seed=rep)
            rows.append(dict(rep=rep, kol="full", n=len(yc),
                             **{k: round(v, 4)
                                for k, v in sc(yte, pred(nf, Xte, ms)).items()}))

            nl, msl = fit(Xc, yc, epochs=a.epochs, seed=rep)
            rows.append(dict(rep=rep, kol="local", n=len(yc),
                             **{k: round(v, 4)
                                for k, v in sc(yte, pred(nl, Xte, msl)).items()}))

            npo, mso = fit(np.vstack([Xs, Xc]), np.concatenate([ys, yc]),
                           epochs=a.epochs, seed=rep)
            rows.append(dict(rep=rep, kol="pooled", n=len(yc),
                             **{k: round(v, 4)
                                for k, v in sc(yte, pred(npo, Xte, mso)).items()}))
        print(f"  tekrar {rep+1}/{a.reps} bitti", flush=True)

    df = pd.DataFrame(rows)
    tag = f"{a.species.replace(' ','_')}__{a.drug}__{a.target}"
    df.to_csv(out / f"{tag}_raw.csv", index=False)
    print("\n=== CNN OGRENME EGRISI ===")
    for met in ["prauc", "auroc"]:
        piv = df.pivot_table(index="n", columns="kol", values=met,
                             aggfunc="mean").round(3)
        print(f"\n{met.upper()}:"); print(piv.to_string())
    print(f"\nkayit: {out / (tag + '_raw.csv')}")


if __name__ == "__main__":
    main()
