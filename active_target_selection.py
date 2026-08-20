#!/usr/bin/env python3
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).parent))
from nested_cv import load_spectra, gather_rows, patient_map, group_key

warnings.filterwarnings("ignore")

CFG=dict(objective="binary",num_leaves=31,n_estimators=300,learning_rate=0.05,
         colsample_bytree=0.3,subsample=0.8,subsample_freq=1,verbose=-1,n_jobs=12)
CACHE={}

def spectra(mdir,site):
    if site not in CACHE: CACHE[site]=load_spectra(mdir,site)
    return CACHE[site]

def load_task(lab,mdir,root,species,drug,site):
    sel=lab[(lab.species==species)&(lab.drug==drug)&lab.tested&lab.has_spectrum&(lab.site==site)].copy()
    if sel.empty:return None
    xs,_,idx=spectra(mdir,site)
    sel=sel[sel.code.isin(idx.keys())].copy()
    if sel.empty:return None
    X=gather_rows(xs,[idx[c] for c in sel.code]).astype(np.float32)
    y=sel.label_RI.to_numpy(dtype=int)
    try:
        pmap=patient_map(root,site); g=group_key(sel.code.to_numpy(),pmap,"patient")
    except Exception:
        g=sel.code.astype(str).to_numpy()
    return dict(df=sel.reset_index(drop=True),X=X,y=y,g=np.asarray(g))

def cal_slope(y,p):
    if len(np.unique(y))<2:return np.nan
    p=np.clip(p,1e-6,1-1e-6); x=np.log(p/(1-p)).reshape(-1,1)
    try:
        m=LogisticRegression(penalty=None,solver="lbfgs",max_iter=1000).fit(x,y)
        return float(m.coef_[0,0])
    except:return np.nan

def metrics(y,p):
    return dict(auroc=float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan,
                prauc=float(average_precision_score(y,p)) if len(np.unique(y))>1 else np.nan,
                brier=float(brier_score_loss(y,p)),slope=cal_slope(y,p),
                n=len(y),positives=int(np.sum(y)),prevalence=float(np.mean(y)))

def fit_predict(Xtr,ytr,Xte,seed=0):
    m=lgb.LGBMClassifier(**CFG,random_state=seed).fit(Xtr,ytr)
    return m,m.predict_proba(Xte)[:,1]

def fit_svd(X,ncomp=100,seed=0):
    mu=X.mean(0,keepdims=True).astype(np.float32)
    nc=max(2,min(ncomp,min(X.shape)-1))
    _,_,Vt=randomized_svd(X.astype(np.float64)-mu,n_components=nc,random_state=seed)
    return mu,Vt.T.astype(np.float32)

def tx(X,mu,P): return ((X-mu)@P).astype(np.float32)

def choose_split(y,g,seed=0,test_fraction=0.5):
    n_splits=max(2,min(5,int(round(1/test_fraction))))
    sg=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    best=None; err0=np.inf
    for tr,te in sg.split(np.zeros(len(y)),y,g):
        if len(np.unique(y[tr]))<2 or len(np.unique(y[te]))<2: continue
        err=abs(len(te)/len(y)-test_fraction)
        if err<err0: best=(tr,te); err0=err
    if best is None: raise ValueError("Could not create valid target split")
    return best

def norm01(x):
    x=np.asarray(x,float); lo,hi=x.min(),x.max()
    return np.zeros_like(x) if hi-lo<1e-12 else (x-lo)/(hi-lo)

def kennard_stone(Z,budget):
    n=len(Z); budget=min(budget,n)
    if budget>=n:return np.arange(n)
    cent=Z.mean(0); first=int(np.argmax(np.linalg.norm(Z-cent,axis=1)))
    d1=np.linalg.norm(Z-Z[first],axis=1); second=int(np.argmax(d1))
    sel=[first] + ([second] if budget>1 and second!=first else [])
    chosen=np.zeros(n,bool); chosen[sel]=True
    md=np.linalg.norm(Z-Z[first],axis=1)
    if len(sel)>1: md=np.minimum(md,np.linalg.norm(Z-Z[second],axis=1))
    md[chosen]=-np.inf
    while len(sel)<budget:
        j=int(np.argmax(md)); sel.append(j); chosen[j]=True
        md=np.minimum(md,np.linalg.norm(Z-Z[j],axis=1)); md[chosen]=-np.inf
    return np.array(sel)

def uncertainty(p,budget):
    return np.argsort(-np.abs(p-0.5))[:min(budget,len(p))]

def domain_distance(Zs,Za,budget,k=10):
    kk=max(1,min(k,len(Zs))); nn=NearestNeighbors(n_neighbors=kk,n_jobs=-1).fit(Zs)
    d,_=nn.kneighbors(Za); s=d.mean(1)
    return np.argsort(s)[::-1][:min(budget,len(s))]

def hybrid(Z,p,budget,alpha=0.5):
    n=len(Z); budget=min(budget,n)
    if budget>=n:return np.arange(n)
    u=norm01(1-2*np.abs(p-0.5))
    first=int(np.argmax(u)); sel=[first]
    chosen=np.zeros(n,bool); chosen[first]=True
    md=np.linalg.norm(Z-Z[first],axis=1)
    while len(sel)<budget:
        div=norm01(md); score=alpha*u+(1-alpha)*div; score[chosen]=-np.inf
        j=int(np.argmax(score)); sel.append(j); chosen[j]=True
        md=np.minimum(md,np.linalg.norm(Z-Z[j],axis=1))
    return np.array(sel)

def boot_ci(x,reps=5000,seed=0):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)<2:return (np.nan,np.nan)
    rng=np.random.default_rng(seed)
    b=rng.choice(x,(reps,len(x)),replace=True).mean(1)
    return tuple(np.percentile(b,[2.5,97.5]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--species",required=True); ap.add_argument("--drug",required=True)
    ap.add_argument("--source",default="DRIAMS-A"); ap.add_argument("--target",default="DRIAMS-C")
    ap.add_argument("--labels",default="outputs/driams_long.parquet"); ap.add_argument("--matrices",default="matrices")
    ap.add_argument("--root",default="~/data/DRIAMS"); ap.add_argument("--out",default="outputs/active_target_selection")
    ap.add_argument("--budgets",default="10,20,30,50,75,100"); ap.add_argument("--reps",type=int,default=20)
    ap.add_argument("--test-fraction",type=float,default=0.5); ap.add_argument("--ncomp",type=int,default=100)
    ap.add_argument("--hybrid-alpha",type=float,default=0.5); ap.add_argument("--domain-k",type=int,default=10)
    ap.add_argument("--bootstrap-reps",type=int,default=5000)
    a=ap.parse_args()

    mdir=Path(a.matrices); root=Path(a.root).expanduser(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    lab=pd.read_parquet(a.labels)
    src=load_task(lab,mdir,root,a.species,a.drug,a.source); tgt=load_task(lab,mdir,root,a.species,a.drug,a.target)
    if src is None or tgt is None: raise SystemExit("Source/target unavailable")
    budgets=list(map(int,a.budgets.split(","))); rows=[]
    print("=== ACTIVE TARGET SELECTION ===")

    for rep in range(a.reps):
        ai,ti=choose_split(tgt["y"],tgt["g"],rep,a.test_fraction)
        Xa,ya=tgt["X"][ai],tgt["y"][ai]; Xte,yte=tgt["X"][ti],tgt["y"][ti]
        sm,pb=fit_predict(src["X"],src["y"],Xte,rep)
        rows.append(dict(rep=rep,strategy="source_only",budget_n=0,selected_pos=0,selected_pos_rate=0.0,**metrics(yte,pb)))
        pa=sm.predict_proba(Xa)[:,1]
        mu,P=fit_svd(src["X"],a.ncomp,rep); Zs,Za=tx(src["X"],mu,P),tx(Xa,mu,P)
        sc=StandardScaler().fit(Zs); Zs=sc.transform(Zs); Za=sc.transform(Za)
        rng=np.random.default_rng(rep)
        for b in budgets:
            if b>len(Xa):continue
            sels={
                "random":rng.choice(len(Xa),b,replace=False),
                "kennard_stone":kennard_stone(Za,b),
                "uncertainty":uncertainty(pa,b),
                "domain_distance":domain_distance(Zs,Za,b,a.domain_k),
                "hybrid":hybrid(Za,pa,b,a.hybrid_alpha),
            }
            for st,ix in sels.items():
                Xb,yb=Xa[ix],ya[ix]
                _,p=fit_predict(np.vstack([src["X"],Xb]),np.r_[src["y"],yb],Xte,rep)
                rows.append(dict(rep=rep,strategy=st,budget_n=b,selected_pos=int(yb.sum()),
                                 selected_pos_rate=float(yb.mean()),**metrics(yte,p)))
        print(f"rep {rep+1}/{a.reps} complete",flush=True)

    raw=pd.DataFrame(rows)
    tag=f"{a.species.replace(' ','_')}__{a.drug}__{a.source}_to_{a.target}"
    raw.to_csv(out/f"{tag}__raw.csv",index=False)

    agg=[]
    for (st,b),g in raw[raw.strategy!="source_only"].groupby(["strategy","budget_n"]):
        r=dict(strategy=st,budget_n=b,n_reps=len(g),mean_selected_pos=g.selected_pos.mean(),
               mean_selected_pos_rate=g.selected_pos_rate.mean())
        for met in ["auroc","prauc","brier","slope"]:
            v=g[met].to_numpy(float); r[f"{met}_mean"]=np.nanmean(v)
            lo,hi=boot_ci(v,a.bootstrap_reps,0); r[f"{met}_ci_low"]=lo; r[f"{met}_ci_high"]=hi
        agg.append(r)
    agg=pd.DataFrame(agg); agg.to_csv(out/f"{tag}__summary.csv",index=False)

    base=raw[raw.strategy=="source_only"][["rep","auroc","prauc","brier"]].rename(
        columns={"auroc":"ba","prauc":"bp","brier":"bb"})
    cmp=raw[raw.strategy!="source_only"].merge(base,on="rep")
    cmp["d_auroc"]=cmp.auroc-cmp.ba; cmp["d_prauc"]=cmp.prauc-cmp.bp; cmp["d_brier"]=cmp.brier-cmp.bb
    dr=[]
    for (st,b),g in cmp.groupby(["strategy","budget_n"]):
        r=dict(strategy=st,budget_n=b,n_reps=len(g),selected_pos_mean=g.selected_pos.mean())
        for c in ["d_auroc","d_prauc","d_brier"]:
            v=g[c].to_numpy(float); r[f"{c}_mean"]=np.nanmean(v); lo,hi=boot_ci(v,a.bootstrap_reps,1)
            r[f"{c}_ci_low"]=lo; r[f"{c}_ci_high"]=hi
        dr.append(r)
    dr=pd.DataFrame(dr); dr.to_csv(out/f"{tag}__delta_vs_source.csv",index=False)
    (out/"run_config.json").write_text(json.dumps(vars(a),indent=2),encoding="utf-8")
    print("\n=== SUMMARY ===\n",agg.sort_values(["budget_n","prauc_mean"],ascending=[True,False]).to_string(index=False))
    print("\n=== DELTA VS SOURCE-ONLY ===\n",dr.sort_values(["budget_n","d_prauc_mean"],ascending=[True,False]).to_string(index=False))
    print("\nOutputs:",out)

if __name__=="__main__":
    main()
