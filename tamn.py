#!/usr/bin/env python3
"""Target-Adaptive MALDI Network: DSBN + residual adapter + target head."""
import argparse, warnings, sys, copy
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, str(Path(__file__).parent))
from nested_cv import load_spectra, gather_rows, patient_map, group_key

warnings.filterwarnings("ignore")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SITES = ["DRIAMS-A", "DRIAMS-B", "DRIAMS-C", "DRIAMS-D"]


class DSBN(nn.Module):
    def __init__(self, ch, ndom=2):
        super().__init__()
        self.bns = nn.ModuleList([nn.BatchNorm1d(ch) for _ in range(ndom)])
        self.dom = 0

    def forward(self, x):
        return self.bns[self.dom](x)


def set_domain(net, d):
    for m in net.modules():
        if isinstance(m, DSBN):
            m.dom = d


class Encoder(nn.Module):
    def __init__(self, feat=128, ndom=2):
        super().__init__()
        self.c1 = nn.Conv1d(1, 32, 9, stride=2, padding=4); self.b1 = DSBN(32, ndom)
        self.c2 = nn.Conv1d(32, 64, 9, stride=2, padding=4); self.b2 = DSBN(64, ndom)
        self.c3 = nn.Conv1d(64, 128, 9, stride=2, padding=4); self.b3 = DSBN(128, ndom)
        self.c4 = nn.Conv1d(128, 128, 9, stride=2, padding=4); self.b4 = DSBN(128, ndom)
        self.pool = nn.AdaptiveAvgPool1d(4)
        self.fc = nn.Linear(128 * 4, feat); self.b5 = DSBN(feat, ndom)
        self.act = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.act(self.b1(self.c1(x)))
        x = self.act(self.b2(self.c2(x)))
        x = self.act(self.b3(self.c3(x)))
        x = self.act(self.b4(self.c4(x)))
        x = self.pool(x).flatten(1)
        return self.act(self.b5(self.fc(x)))


class Adapter(nn.Module):
    def __init__(self, feat=128, bottleneck=32, alpha=1.0):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(feat, bottleneck), nn.ReLU(),
                               nn.Linear(bottleneck, feat))
        self.alpha = alpha
        nn.init.zeros_(self.f[-1].weight); nn.init.zeros_(self.f[-1].bias)

    def forward(self, z):
        return z + self.alpha * self.f(z)


class TAMN(nn.Module):
    def __init__(self, feat=128, ndom=2):
        super().__init__()
        self.enc = Encoder(feat, ndom)
        self.adapt = nn.ModuleList([nn.Identity(), Adapter(feat)])
        self.head = nn.ModuleList([nn.Linear(feat, 1), nn.Linear(feat, 1)])
        self.drop = nn.Dropout(0.3)
        self.dom = 0

    def forward(self, x):
        z = self.enc(x)
        z = self.adapt[self.dom](z)
        return self.head[self.dom](self.drop(z)).squeeze(1)


def _loader(X, y, m, s, bs, shuffle=True):
    Z = torch.tensor((X - m) / s)
    Y = torch.tensor(y, dtype=torch.float32)
    bs = min(bs, max(8, len(y) // 4))
    return DataLoader(TensorDataset(Z, Y), batch_size=bs, shuffle=shuffle,
                      drop_last=len(y) > 2 * bs)


def bce(y):
    pos = max(float(y.mean()), 1e-6)
    return nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor((1 - pos) / pos, device=DEV))


def train_source(X, y, epochs=25, lr=1e-3, bs=128, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    m, s = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-8
    dl = _loader(X, y, m, s, bs)
    net = TAMN().to(DEV); net.dom = 0; set_domain(net, 0)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    lf = bce(y)
    for _ in range(epochs):
        net.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = lf(net(xb.to(DEV)), yb.to(DEV))
            loss.backward(); opt.step()
    return net, (m, s)


def train_adapter(src, X, y, ms, epochs=40, lr=3e-3, bs=32, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    net = copy.deepcopy(src); net.dom = 1; set_domain(net, 1)
    for mod in net.enc.modules():
        if isinstance(mod, DSBN):
            mod.bns[1].load_state_dict(mod.bns[0].state_dict())
    net.head[1].load_state_dict(net.head[0].state_dict())
    for p in net.parameters():
        p.requires_grad = False
    trainable = []
    for mod in net.enc.modules():
        if isinstance(mod, DSBN):
            for p in mod.bns[1].parameters():
                p.requires_grad = True; trainable.append(p)
    for p in net.adapt[1].parameters():
        p.requires_grad = True; trainable.append(p)
    for p in net.head[1].parameters():
        p.requires_grad = True; trainable.append(p)
    m, s = ms
    dl = _loader(X, y, m, s, bs)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    lf = bce(y)
    for _ in range(epochs):
        net.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = lf(net(xb.to(DEV)), yb.to(DEV))
            loss.backward(); opt.step()
    return net


def train_full(src, X, y, ms, epochs=15, lr=1e-4, bs=32, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    net = copy.deepcopy(src); net.dom = 0; set_domain(net, 0)
    for p in net.parameters():
        p.requires_grad = True
    m, s = ms
    dl = _loader(X, y, m, s, bs)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    lf = bce(y)
    for _ in range(epochs):
        net.train()
        for xb, yb in dl:
            opt.zero_grad()
            loss = lf(net(xb.to(DEV)), yb.to(DEV))
            loss.backward(); opt.step()
    return net


@torch.no_grad()
def pred(net, X, ms, dom=None):
    if dom is not None:
        net.dom = dom; set_domain(net, dom)
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


def cal_slope(y, p):
    try:
        m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        m.fit(logit(p).reshape(-1, 1), y)
        return float(m.coef_[0][0])
    except Exception:
        return np.nan


def sc(y, p):
    return dict(auroc=roc_auc_score(y, p), prauc=average_precision_score(y, p),
                brier=brier_score_loss(y, p), egim=cal_slope(y, p))


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
    ap.add_argument("--out", default="outputs/tamn")
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

    Xl, yl = [], []
    for s in SITES:
        if s == a.target:
            continue
        sub = sel[sel.site == s]
        if sub.empty:
            continue
        try:
            xs, c, idx = load_spectra(mdir, s)
        except FileNotFoundError:
            continue
        sub = sub[sub.code.isin(idx.keys())]
        if sub.empty:
            continue
        Xl.append(gather_rows(xs, [idx[cc] for cc in sub.code]).astype(np.float32))
        yl.append(sub.label_RI.to_numpy(dtype=int))
    Xs_, ys_ = np.vstack(Xl), np.concatenate(yl)
    print(f"kaynak (hedef haric): n={len(ys_):,} direnc={ys_.mean():.3f}", flush=True)
    src, ms = train_source(Xs_, ys_, epochs=a.epochs)

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
        p0 = pred(src, Xte, ms, dom=0)
        rows.append(dict(rep=rep, kol="direct", n=0,
                         **{k: round(v, 4) for k, v in sc(yte, p0).items()}))
        Xp, yp, gp = Xt[pool_ix], yt[pool_ix], gt[pool_ix]

        for n in ns:
            if n > len(yp):
                continue
            ix = gsub(gp, yp, n, rng)
            if ix is None or yp[ix].sum() < 3:
                continue
            Xc, yc = Xp[ix], yp[ix]

            pc = pred(src, Xc, ms, dom=0)
            lr_ = LogisticRegression(penalty=None, solver="lbfgs",
                                     max_iter=1000).fit(logit(pc).reshape(-1, 1), yc)
            p = lr_.predict_proba(logit(p0).reshape(-1, 1))[:, 1]
            rows.append(dict(rep=rep, kol="calib", n=len(yc),
                             **{k: round(v, 4) for k, v in sc(yte, p).items()}))

            na = train_adapter(src, Xc, yc, ms, seed=rep)
            rows.append(dict(rep=rep, kol="adapter", n=len(yc),
                             **{k: round(v, 4)
                                for k, v in sc(yte, pred(na, Xte, ms, dom=1)).items()}))

            nf = train_full(src, Xc, yc, ms, seed=rep)
            rows.append(dict(rep=rep, kol="fullft", n=len(yc),
                             **{k: round(v, 4)
                                for k, v in sc(yte, pred(nf, Xte, ms, dom=0)).items()}))

            nl, msl = train_source(Xc, yc, epochs=a.epochs, seed=rep)
            rows.append(dict(rep=rep, kol="local", n=len(yc),
                             **{k: round(v, 4)
                                for k, v in sc(yte, pred(nl, Xte, msl, dom=0)).items()}))
        print(f"  tekrar {rep+1}/{a.reps} bitti", flush=True)

    df = pd.DataFrame(rows)
    tag = f"{a.species.replace(' ','_')}__{a.drug}__{a.target}"
    df.to_csv(out / f"{tag}_raw.csv", index=False)
    print("\n=== TAMN: DSBN + ADAPTER ===")
    for met in ["prauc", "auroc", "egim"]:
        piv = df.pivot_table(index="n", columns="kol", values=met,
                             aggfunc="mean").round(3)
        print(f"\n{met.upper()}:"); print(piv.to_string())
    print(f"\nkayit: {out / (tag + '_raw.csv')}")


if __name__ == "__main__":
    main()
