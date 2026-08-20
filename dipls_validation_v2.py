#!/usr/bin/env python3
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd

sys.path.insert(0, str(Path(__file__).parent))
from nested_cv import load_spectra, gather_rows
warnings.filterwarnings("ignore")

CFG=dict(objective="binary",num_leaves=31,n_estimators=300,learning_rate=0.05,
         colsample_bytree=0.3,subsample=0.8,subsample_freq=1,verbose=-1,n_jobs=12)
CACHE={}

def spectra(mdir,site):
    if site not in CACHE: CACHE[site]=load_spectra(mdir,site)
    return CACHE[site]

def load_task(lab,mdir,species,drug,site):
    sel=lab[(lab.species==species)&(lab.drug==drug)&lab.tested&lab.has_spectrum&(lab.site==site)].copy()
    if sel.empty:return None
    xs,_,idx=spectra(mdir,site)
    sel=sel[sel.code.isin(idx.keys())].copy()
    if sel.empty:return None
    X=gather_rows(xs,[idx[c] for c in sel.code]).astype(np.float32)
    return dict(df=sel.reset_index(drop=True),X=X,y=sel.label_RI.to_numpy(dtype=int))

def cal_slope(y,p):
    if len(np.unique(y))<2:return np.nan
    p=np.clip(p,1e-6,1-1e-6); x=np.log(p/(1-p)).reshape(-1,1)
    try:
        m=LogisticRegression(penalty=None,solver="lbfgs",max_iter=1000).fit(x,y)
        return float(m.coef_[0,0])
    except:return np.nan

def cal_intercept(y,p):
    if len(np.unique(y))<2:return np.nan
    p=np.clip(p,1e-6,1-1e-6); x=np.log(p/(1-p)).reshape(-1,1)
    try:
        m=LogisticRegression(penalty=None,solver="lbfgs",max_iter=1000).fit(x,y)
        return float(m.intercept_[0])
    except:return np.nan

def metrics(y,p):
    return dict(auroc=float(roc_auc_score(y,p)),prauc=float(average_precision_score(y,p)),
                brier=float(brier_score_loss(y,p)),slope=cal_slope(y,p),intercept=cal_intercept(y,p),
                n=len(y),positives=int(np.sum(y)),prevalence=float(np.mean(y)))

def fit_svd(X,ncomp=120,seed=0):
    mu=X.mean(0,keepdims=True).astype(np.float32)
    nc=max(2,min(ncomp,min(X.shape)-1))
    _,_,Vt=randomized_svd(X.astype(np.float64)-mu,n_components=nc,random_state=seed)
    return mu,Vt.T.astype(np.float32)

def tx(X,mu,P): return ((X-mu)@P).astype(np.float32)

def run_lgbm(Xs,ys,Xt,seed=0):
    m=lgb.LGBMClassifier(**CFG,random_state=seed).fit(Xs,ys)
    return m.predict_proba(Xt)[:,1]

def dipls_basis(Zs,ys,Zt,latent=2,lam=10.,ridge=1e-3):
    y0=ys.astype(float)-ys.mean(); Z0=Zs-Zs.mean(0,keepdims=True)
    cy=(Z0.T@y0.reshape(-1,1))/max(len(Z0)-1,1); Sy=cy@cy.T
    dm=(Zs.mean(0)-Zt.mean(0)).reshape(-1,1)
    Dc=np.cov(Zs,rowvar=False)-np.cov(Zt,rowvar=False)
    Sd=dm@dm.T+(Dc@Dc.T)/max(Zs.shape[1],1)
    A=Sy+ridge*np.eye(Sy.shape[0]); B=np.eye(Sy.shape[0])+lam*Sd+ridge*np.eye(Sy.shape[0])
    vals,vecs=eigh(A,B); order=np.argsort(vals)[::-1]
    return vecs[:,order[:max(1,min(latent,len(order)))]].astype(np.float32)

def dipls_predict(Xs,ys,Xadapt,Xpred,svdcomp=120,latent=2,lam=10.,seed=0):
    mu,P=fit_svd(Xs,svdcomp,seed)
    Zs,Za,Zp=tx(Xs,mu,P),tx(Xadapt,mu,P),tx(Xpred,mu,P)
    sc=StandardScaler().fit(Zs)
    Zs,Za,Zp=sc.transform(Zs),sc.transform(Za),sc.transform(Zp)
    W=dipls_basis(Zs,ys,Za,latent,lam)
    m=LogisticRegression(max_iter=2000,solver="lbfgs").fit(Zs@W,ys)
    return m.predict_proba(Zp@W)[:,1]

def fit_platt(y,p):
    p=np.clip(p,1e-6,1-1e-6); x=np.log(p/(1-p)).reshape(-1,1)
    return LogisticRegression(penalty=None,solver="lbfgs",max_iter=2000).fit(x,y)

def apply_platt(m,p):
    p=np.clip(p,1e-6,1-1e-6); x=np.log(p/(1-p)).reshape(-1,1)
    return m.predict_proba(x)[:,1]

def select_candidate(df,base_auc,max_drop):
    thr=base_auc-max_drop
    e=df[df.auroc>=thr].copy()
    if e.empty:return None,thr
    e=e.sort_values(["prauc","auroc","brier"],ascending=[False,False,True])
    return e.iloc[0],thr

def strat_boot_idx(y,rng):
    out=[]
    for c in np.unique(y):
        ii=np.where(y==c)[0]; out.append(rng.choice(ii,len(ii),replace=True))
    idx=np.concatenate(out); rng.shuffle(idx); return idx

def paired_bootstrap(y,preds,reps=10000,seed=0):
    rng=np.random.default_rng(seed); names=list(preds)
    pts={n:metrics(y,preds[n]) for n in names}
    boot={n:{m:[] for m in ["auroc","prauc","brier"]} for n in names}
    for _ in range(reps):
        idx=strat_boot_idx(y,rng); yy=y[idx]
        for n in names:
            pp=preds[n][idx]
            boot[n]["auroc"].append(roc_auc_score(yy,pp))
            boot[n]["prauc"].append(average_precision_score(yy,pp))
            boot[n]["brier"].append(brier_score_loss(yy,pp))
    rows=[]
    for n in names:
        for met in ["auroc","prauc","brier"]:
            v=np.asarray(boot[n][met]); lo,hi=np.percentile(v,[2.5,97.5])
            r=dict(model=n,metric=met,estimate=pts[n][met],ci_low=lo,ci_high=hi)
            if n!="baseline":
                b=np.asarray(boot["baseline"][met]); d=v-b; dlo,dhi=np.percentile(d,[2.5,97.5])
                r.update(delta_vs_baseline=pts[n][met]-pts["baseline"][met],
                         delta_ci_low=dlo,delta_ci_high=dhi,
                         delta_significant=bool(dlo>0 or dhi<0))
            rows.append(r)
    return pd.DataFrame(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--species",required=True); ap.add_argument("--drug",required=True)
    ap.add_argument("--dev-target",default="DRIAMS-B"); ap.add_argument("--test-target",default="DRIAMS-C")
    ap.add_argument("--labels",default="outputs/driams_long.parquet"); ap.add_argument("--matrices",default="matrices")
    ap.add_argument("--out",default="outputs/dipls_validation_v2"); ap.add_argument("--seed",type=int,default=0)
    ap.add_argument("--svdcomp",type=int,default=120)
    ap.add_argument("--latents",default="2,4,8,12,20,30")
    ap.add_argument("--lambdas",default="0,0.01,0.1,0.3,1,3,10,30,100")
    ap.add_argument("--max-auc-drop",type=float,default=0.03)
    ap.add_argument("--fallback-latent",type=int,default=2); ap.add_argument("--fallback-lambda",type=float,default=10.)
    ap.add_argument("--bootstrap-reps",type=int,default=10000)
    a=ap.parse_args()

    mdir=Path(a.matrices); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    lab=pd.read_parquet(a.labels)
    src=load_task(lab,mdir,a.species,a.drug,"DRIAMS-A")
    dev=load_task(lab,mdir,a.species,a.drug,a.dev_target)
    test=load_task(lab,mdir,a.species,a.drug,a.test_target)
    if src is None or dev is None or test is None: raise SystemExit("Source/dev/test unavailable")

    pbd=run_lgbm(src["X"],src["y"],dev["X"],a.seed); pbt=run_lgbm(src["X"],src["y"],test["X"],a.seed)
    mbd,mbt=metrics(dev["y"],pbd),metrics(test["y"],pbt)
    print("=== diPLS VALIDATION V2 ===")
    print(f"Baseline A->{a.dev_target}: AUROC={mbd['auroc']:.3f} PR={mbd['prauc']:.3f} Brier={mbd['brier']:.3f}")
    print(f"Baseline A->{a.test_target}: AUROC={mbt['auroc']:.3f} PR={mbt['prauc']:.3f} Brier={mbt['brier']:.3f}")

    rows=[]
    for lat in map(int,a.latents.split(",")):
        for lam in map(float,a.lambdas.split(",")):
            p=dipls_predict(src["X"],src["y"],dev["X"],dev["X"],a.svdcomp,lat,lam,a.seed)
            m=metrics(dev["y"],p); rows.append(dict(latent=lat,lam=lam,**m))
            print(f"dev latent={lat:>2} lam={lam:>6}: AUROC={m['auroc']:.3f} PR={m['prauc']:.3f} Brier={m['brier']:.3f}")
    grid=pd.DataFrame(rows); grid.to_csv(out/f"dev_grid__{a.dev_target}.csv",index=False)

    best,thr=select_candidate(grid,mbd["auroc"],a.max_auc_drop)
    if best is None:
        status="no_eligible_candidate"; lat=a.fallback_latent; lam=a.fallback_lambda
        print(f"NO ELIGIBLE CANDIDATE: required AUROC >= {thr:.3f}")
        print(f"Sensitivity fallback only: latent={lat}, lambda={lam}")
    else:
        status="eligible_selected"; lat=int(best.latent); lam=float(best.lam)
        print("Selected candidate:\n",best.to_string())

    pdv=dipls_predict(src["X"],src["y"],dev["X"],dev["X"],a.svdcomp,lat,lam,a.seed)
    pdt=dipls_predict(src["X"],src["y"],test["X"],test["X"],a.svdcomp,lat,lam,a.seed)
    pl=fit_platt(dev["y"],pdv); pcal=apply_platt(pl,pdt)

    mr,mc=metrics(test["y"],pdt),metrics(test["y"],pcal)
    summary=pd.DataFrame([
        dict(model="baseline",selection_status="not_applicable",latent=np.nan,lam=np.nan,**mbt),
        dict(model="dipls_raw",selection_status=status,latent=lat,lam=lam,**mr),
        dict(model="dipls_platt_B_to_C",selection_status=status,latent=lat,lam=lam,
             platt_intercept=float(pl.intercept_[0]),platt_slope=float(pl.coef_[0,0]),**mc)
    ])
    summary["delta_auroc_vs_baseline"]=summary.auroc-mbt["auroc"]
    summary["delta_prauc_vs_baseline"]=summary.prauc-mbt["prauc"]
    summary["delta_brier_vs_baseline"]=summary.brier-mbt["brier"]

    boot=paired_bootstrap(test["y"],{"baseline":pbt,"dipls_raw":pdt,"dipls_platt_B_to_C":pcal},
                          reps=a.bootstrap_reps,seed=a.seed)
    pred=pd.DataFrame({"code":test["df"].code.astype(str),"y":test["y"],
                       "p_baseline":pbt,"p_dipls_raw":pdt,"p_dipls_platt":pcal})
    tag=f"{a.species.replace(' ','_')}__{a.drug}__{a.dev_target}_to_{a.test_target}"
    summary.to_csv(out/f"summary__{tag}.csv",index=False)
    boot.to_csv(out/f"paired_bootstrap__{a.test_target}.csv",index=False)
    pred.to_csv(out/f"predictions__{a.test_target}.csv",index=False)
    (out/"run_config.json").write_text(json.dumps({**vars(a),"selection_status":status,"threshold":thr,
                                                    "locked_latent":lat,"locked_lambda":lam},indent=2),encoding="utf-8")
    print("\n=== TEST SUMMARY ===\n",summary.to_string(index=False))
    print("\n=== PAIRED BOOTSTRAP ===\n",boot.to_string(index=False))
    print("\nOutputs:",out)

if __name__=="__main__":
    main()
