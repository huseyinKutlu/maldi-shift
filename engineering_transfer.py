#!/usr/bin/env python3
"""Engineering-inspired transfer methods for MALDI-TOF AMR."""

import argparse, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.linalg import eigh
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).parent))
from nested_cv import load_spectra, gather_rows, patient_map, group_key

warnings.filterwarnings("ignore")

CFG = dict(objective="binary", num_leaves=31, n_estimators=300,
           learning_rate=0.05, colsample_bytree=0.3, subsample=0.8,
           subsample_freq=1, verbose=-1, n_jobs=12)

CACHE = {}

def spectra(mdir, site):
    if site not in CACHE:
        CACHE[site] = load_spectra(mdir, site)
    return CACHE[site]

def cal_slope(y, p):
    if len(np.unique(y)) < 2:
        return np.nan
    p = np.clip(np.asarray(p), 1e-6, 1-1e-6)
    lo = np.log(p/(1-p)).reshape(-1,1)
    try:
        m = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000).fit(lo,y)
        return float(m.coef_[0,0])
    except Exception:
        return np.nan

def metrics(y,p):
    return dict(
        auroc=float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan,
        prauc=float(average_precision_score(y,p)) if len(np.unique(y))>1 else np.nan,
        brier=float(brier_score_loss(y,p)),
        slope=cal_slope(y,p),
        n=int(len(y)), positives=int(np.sum(y)), prevalence=float(np.mean(y))
    )

def load_task(lab, mdir, root, species, drug, site):
    sel = lab[(lab.species==species)&(lab.drug==drug)&lab.tested&
              lab.has_spectrum&(lab.site==site)].copy()
    if sel.empty: return None
    xs,_,idx = spectra(mdir,site)
    sel = sel[sel.code.isin(idx.keys())].copy()
    if sel.empty: return None
    X = gather_rows(xs,[idx[c] for c in sel.code]).astype(np.float32)
    y = sel.label_RI.to_numpy(dtype=int)
    return dict(df=sel.reset_index(drop=True), X=X, y=y)

def fit_svd(X,ncomp=120,seed=0):
    mu = X.mean(0,keepdims=True).astype(np.float32)
    nc=max(2,min(ncomp,min(X.shape)-1))
    _,_,Vt = randomized_svd(X.astype(np.float64)-mu,n_components=nc,random_state=seed)
    return mu,Vt.T.astype(np.float32)

def tx(X,mu,P):
    return ((X-mu)@P).astype(np.float32)

def run_lgbm(Xs,ys,Xt,yt,seed=0):
    m=lgb.LGBMClassifier(**CFG,random_state=seed).fit(Xs,ys)
    p=m.predict_proba(Xt)[:,1]
    return p,metrics(yt,p)

def run_plsda(Xs,ys,Xt,yt,ncomp=10):
    sc=StandardScaler().fit(Xs)
    A,B=sc.transform(Xs),sc.transform(Xt)
    nc=max(2,min(ncomp,A.shape[1],len(A)-1))
    m=PLSRegression(n_components=nc,scale=False).fit(A,ys.astype(float))
    p=np.clip(m.predict(B).ravel(),1e-6,1-1e-6)
    return p,metrics(yt,p)

def dipls_like_basis(Zs,ys,Zt,ncomp=8,lam=1.0,ridge=1e-3):
    y0=ys.astype(float)-ys.mean()
    Z0=Zs-Zs.mean(0,keepdims=True)
    cy=(Z0.T@y0.reshape(-1,1))/max(len(Z0)-1,1)
    Sy=cy@cy.T
    dm=(Zs.mean(0)-Zt.mean(0)).reshape(-1,1)
    Cs=np.cov(Zs,rowvar=False); Ct=np.cov(Zt,rowvar=False)
    Dc=Cs-Ct
    Sd=dm@dm.T+(Dc@Dc.T)/max(Zs.shape[1],1)
    A=Sy+ridge*np.eye(Sy.shape[0])
    B=np.eye(Sy.shape[0])+lam*Sd+ridge*np.eye(Sy.shape[0])
    vals,vecs=eigh(A,B)
    order=np.argsort(vals)[::-1]
    return vecs[:,order[:max(1,min(ncomp,len(order)))]].astype(np.float32)

def run_dipls_like(Xs,ys,Xt,yt,svdcomp=120,latent=8,lam=1.0,seed=0):
    mu,P=fit_svd(Xs,svdcomp,seed)
    Zs,Zt=tx(Xs,mu,P),tx(Xt,mu,P)
    sc=StandardScaler().fit(Zs)
    Zs,Zt=sc.transform(Zs),sc.transform(Zt)
    W=dipls_like_basis(Zs,ys,Zt,latent,lam)
    As,At=Zs@W,Zt@W
    clf=LogisticRegression(max_iter=2000,solver="lbfgs").fit(As,ys)
    p=clf.predict_proba(At)[:,1]
    return p,metrics(yt,p)

def jda_projection(Zs,Zt,ys,pseudo,confmask,dim=30,lam=1.0):
    X=np.vstack([Zs,Zt]).T
    ns,nt=len(Zs),len(Zt); n=ns+nt
    e=np.r_[np.ones(ns)/ns,-np.ones(nt)/nt].reshape(-1,1)
    M=e@e.T
    for c in [0,1]:
        es=np.zeros(ns); et=np.zeros(nt)
        si=np.where(ys==c)[0]; ti=np.where((pseudo==c)&confmask)[0]
        if len(si) and len(ti):
            es[si]=1/len(si); et[ti]=-1/len(ti)
            ec=np.r_[es,et].reshape(-1,1); M+=ec@ec.T
    M/=np.linalg.norm(M,"fro")+1e-12
    H=np.eye(n)-np.ones((n,n))/n
    A=X@M@X.T+lam*np.eye(X.shape[0])
    B=X@H@X.T+1e-6*np.eye(X.shape[0])
    vals,vecs=eigh(A,B)
    order=np.argsort(vals)
    return vecs[:,order[:max(2,min(dim,len(order)))]].astype(np.float32)

def run_jda(Xs,ys,Xt,yt,svdcomp=100,dim=30,lam=1.0,conf=0.8,iters=5,seed=0):
    mu,P=fit_svd(Xs,svdcomp,seed)
    Zs,Zt=tx(Xs,mu,P),tx(Xt,mu,P)
    sc=StandardScaler().fit(Zs)
    Zs,Zt=sc.transform(Zs),sc.transform(Zt)
    clf=LogisticRegression(max_iter=2000,solver="lbfgs").fit(Zs,ys)
    pt=clf.predict_proba(Zt)[:,1]
    nconf=0
    for _ in range(iters):
        pseudo=(pt>=0.5).astype(int)
        mask=(pt>=conf)|(pt<=1-conf)
        nconf=int(mask.sum())
        W=jda_projection(Zs,Zt,ys,pseudo,mask,dim,lam)
        As,At=Zs@W,Zt@W
        clf=LogisticRegression(max_iter=2000,solver="lbfgs").fit(As,ys)
        pt=clf.predict_proba(At)[:,1]
    return pt,metrics(yt,pt),nconf

def nuisance_dirs(Zs,Zt,k=5):
    dm=(Zs.mean(0)-Zt.mean(0)).reshape(1,-1)
    D=np.vstack([dm,np.cov(Zs,rowvar=False)-np.cov(Zt,rowvar=False)]).astype(np.float64)
    nc=max(1,min(k,min(D.shape)-1))
    _,_,Vt=randomized_svd(D,n_components=nc,random_state=0)
    return Vt.T.astype(np.float32)

def run_top(Xs,ys,Xt,yt,svdcomp=120,k=5,rho=1.0,seed=0):
    mu,P=fit_svd(Xs,svdcomp,seed)
    Zs,Zt=tx(Xs,mu,P),tx(Xt,mu,P)
    sc=StandardScaler().fit(Zs)
    Zs,Zt=sc.transform(Zs),sc.transform(Zt)
    V=nuisance_dirs(Zs,Zt,k)
    Zsp=Zs-rho*((Zs@V)@V.T)
    Ztp=Zt-rho*((Zt@V)@V.T)
    clf=LogisticRegression(max_iter=2000,solver="lbfgs").fit(Zsp,ys)
    p=clf.predict_proba(Ztp)[:,1]
    return p,metrics(yt,p)

def date_series(df):
    for c in ["acquisition_date","date","measurement_date","collection_date","sample_date"]:
        if c in df.columns:
            s=pd.to_datetime(df[c],errors="coerce")
            if s.notna().sum()>0: return s,c
    raise ValueError("No usable acquisition date column found")

def run_temporal_ensemble(src,Xt,yt,ncomp=80,seed=0):
    dates,col=date_series(src["df"])
    years=sorted(dates.dropna().dt.year.unique())
    mu,P=fit_svd(src["X"],ncomp,seed)
    Zs,Zt=tx(src["X"],mu,P),tx(Xt,mu,P)
    sc=StandardScaler().fit(Zs)
    Zs,Zt=sc.transform(Zs),sc.transform(Zt)
    tc=Zt.mean(0)
    preds=[]; ds=[]; yrs=[]
    for yr in years:
        ix=np.where(dates.dt.year.to_numpy()==yr)[0]
        if len(ix)<200 or len(np.unique(src["y"][ix]))<2: continue
        m=lgb.LGBMClassifier(**CFG,random_state=seed+int(yr)).fit(src["X"][ix],src["y"][ix])
        preds.append(m.predict_proba(Xt)[:,1])
        ds.append(np.linalg.norm(Zs[ix].mean(0)-tc))
        yrs.append(int(yr))
    if len(preds)<2: raise ValueError("Not enough eligible source years")
    d=np.asarray(ds); scale=np.median(d)+1e-8
    w=np.exp(-d/scale); w/=w.sum()
    p=np.sum(w[:,None]*np.vstack(preds),axis=0)
    return p,metrics(yt,p),dict(years="|".join(map(str,yrs)),
                                weights="|".join(f"{x:.4f}" for x in w),
                                date_column=col)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--species",required=True)
    ap.add_argument("--drug",required=True)
    ap.add_argument("--target",default="DRIAMS-C")
    ap.add_argument("--labels",default="outputs/driams_long.parquet")
    ap.add_argument("--matrices",default="matrices")
    ap.add_argument("--root",default="~/data/DRIAMS")
    ap.add_argument("--out",default="outputs/engineering_transfer")
    ap.add_argument("--methods",default="lgbm_baseline,plsda,dipls_like,jda_conf,top_projection,temporal_ensemble")
    ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--pls-components",default="5,10,20")
    ap.add_argument("--dipls-latent",default="4,8,12")
    ap.add_argument("--dipls-lambdas",default="0.1,1,10")
    ap.add_argument("--jda-dims",default="10,30")
    ap.add_argument("--jda-lambdas",default="0.1,1")
    ap.add_argument("--jda-conf",default="0.7,0.8,0.9")
    ap.add_argument("--top-ks",default="1,2,5,10,20")
    ap.add_argument("--top-rhos",default="0.25,0.5,0.75,1.0")
    a=ap.parse_args()

    mdir=Path(a.matrices); root=Path(a.root).expanduser()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    lab=pd.read_parquet(a.labels)

    src=load_task(lab,mdir,root,a.species,a.drug,"DRIAMS-A")
    tgt=load_task(lab,mdir,root,a.species,a.drug,a.target)
    if src is None or tgt is None: raise SystemExit("Source/target unavailable")

    methods=[x.strip() for x in a.methods.split(",") if x.strip()]
    rows=[]

    print("=== ENGINEERING TRANSFER ===")
    print(f"{a.species}/{a.drug}: A n={len(src['y'])}, target {a.target} n={len(tgt['y'])}")

    if "lgbm_baseline" in methods:
        _,m=run_lgbm(src["X"],src["y"],tgt["X"],tgt["y"],a.seed)
        rows.append(dict(method="lgbm_baseline",**m))
        print("baseline",m)

    if "plsda" in methods:
        for nc in map(int,a.pls_components.split(",")):
            try:
                _,m=run_plsda(src["X"],src["y"],tgt["X"],tgt["y"],nc)
                rows.append(dict(method="plsda",ncomp=nc,**m))
                print("plsda",nc,m)
            except Exception as e: print("PLS HATA",nc,e)

    if "dipls_like" in methods:
        for latent in map(int,a.dipls_latent.split(",")):
            for lam in map(float,a.dipls_lambdas.split(",")):
                try:
                    _,m=run_dipls_like(src["X"],src["y"],tgt["X"],tgt["y"],120,latent,lam,a.seed)
                    rows.append(dict(method="dipls_like",latent=latent,lam=lam,**m))
                    print("dipls",latent,lam,m)
                except Exception as e: print("diPLS HATA",latent,lam,e)

    if "jda_conf" in methods:
        for dim in map(int,a.jda_dims.split(",")):
            for lam in map(float,a.jda_lambdas.split(",")):
                for conf in map(float,a.jda_conf.split(",")):
                    try:
                        _,m,nc=run_jda(src["X"],src["y"],tgt["X"],tgt["y"],100,dim,lam,conf,5,a.seed)
                        rows.append(dict(method="jda_conf",dim=dim,lam=lam,conf=conf,n_conf_target=nc,**m))
                        print("jda",dim,lam,conf,m,"nconf",nc)
                    except Exception as e: print("JDA HATA",dim,lam,conf,e)

    if "top_projection" in methods:
        for k in map(int,a.top_ks.split(",")):
            for rho in map(float,a.top_rhos.split(",")):
                try:
                    _,m=run_top(src["X"],src["y"],tgt["X"],tgt["y"],120,k,rho,a.seed)
                    rows.append(dict(method="top_projection",k=k,rho=rho,**m))
                    print("top",k,rho,m)
                except Exception as e: print("TOP HATA",k,rho,e)

    if "temporal_ensemble" in methods:
        try:
            _,m,info=run_temporal_ensemble(src,tgt["X"],tgt["y"],80,a.seed)
            rows.append(dict(method="temporal_ensemble",**info,**m))
            print("temporal",m,info)
        except Exception as e: print("TEMPORAL HATA",e)

    df=pd.DataFrame(rows)
    tag=f"{a.species.replace(' ','_')}__{a.drug}__{a.target}"
    fp=out/f"{tag}.csv"
    df.to_csv(fp,index=False)

    print("\n=== SUMMARY ===")
    if len(df):
        cols=[c for c in ["method","ncomp","latent","lam","dim","conf","k","rho","auroc","prauc","brier","slope"] if c in df.columns]
        print(df.sort_values("prauc",ascending=False)[cols].to_string(index=False))
    print("\nSaved:",fp)

if __name__=="__main__":
    main()
