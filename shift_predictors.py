#!/usr/bin/env python3
"""Explain when Hybrid active target selection helps across MALDI transfer tasks."""
import argparse, json, math, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils.extmath import randomized_svd
try:
    import statsmodels.api as sm
    HAVE_SM=True
except Exception:
    HAVE_SM=False
sys.path.insert(0,str(Path(__file__).parent))
from nested_cv import load_spectra, gather_rows
warnings.filterwarnings('ignore')

CACHE={}
CFG_DOMAIN=dict(objective='binary',num_leaves=15,n_estimators=160,learning_rate=0.05,
                colsample_bytree=0.5,subsample=0.8,subsample_freq=1,verbose=-1,n_jobs=12)
CFG_TASK=dict(objective='binary',num_leaves=31,n_estimators=300,learning_rate=0.05,
              colsample_bytree=0.3,subsample=0.8,subsample_freq=1,verbose=-1,n_jobs=12)

DEPLOY=[
'domain_auc_svd','centroid_distance_svd','covariance_distance_svd',
'source_to_target_nn_distance','target_to_source_nn_distance',
'source_internal_diversity','target_internal_diversity','diversity_ratio',
'source_n','target_n','source_positive_rate_model','target_positive_rate_model',
'source_prediction_entropy','target_prediction_entropy','entropy_shift',
'source_prediction_mean','target_prediction_mean','prediction_mean_shift',
'source_prediction_sd','target_prediction_sd','prediction_sd_shift',
'source_only_auroc_mean','source_only_prauc_mean','source_only_brier_mean']
LABEL=[
'source_prevalence','target_prevalence','prevalence_absolute_shift','prevalence_ratio',
'source_positives','target_positives']

def spectra(mdir,site):
    if site not in CACHE: CACHE[site]=load_spectra(mdir,site)
    return CACHE[site]

def load_task(lab,mdir,species,drug,site):
    z=lab[(lab.species==species)&(lab.drug==drug)&lab.tested&lab.has_spectrum&(lab.site==site)].copy()
    if z.empty:return None
    try: xs,_,idx=spectra(mdir,site)
    except FileNotFoundError:return None
    z=z[z.code.isin(idx.keys())].copy()
    if z.empty:return None
    X=gather_rows(xs,[idx[c] for c in z.code]).astype(np.float32)
    return dict(X=X,y=z.label_RI.to_numpy(dtype=int))

def fit_svd(X,ncomp=100,seed=0):
    mu=X.mean(0,keepdims=True).astype(np.float32); nc=max(2,min(ncomp,min(X.shape)-1))
    _,_,Vt=randomized_svd(X.astype(np.float64)-mu,n_components=nc,random_state=seed)
    return mu,Vt.T.astype(np.float32)

def tx(X,mu,P): return ((X-mu)@P).astype(np.float32)

def domain_auc(Zs,Zt,seed=0):
    X=np.vstack([Zs,Zt]).astype(np.float32); y=np.r_[np.zeros(len(Zs),int),np.ones(len(Zt),int)]
    p=np.zeros(len(y)); sk=StratifiedKFold(3,shuffle=True,random_state=seed)
    for f,(tr,te) in enumerate(sk.split(X,y)):
        m=lgb.LGBMClassifier(**CFG_DOMAIN,random_state=seed+f).fit(X[tr],y[tr])
        p[te]=m.predict_proba(X[te])[:,1]
    return float(roc_auc_score(y,p))

def entropy(p):
    p=np.clip(np.asarray(p,float),1e-6,1-1e-6)
    return -(p*np.log(p)+(1-p)*np.log(1-p))

def knn_mean(A,B,k=10):
    kk=max(1,min(k,len(B))); d,_=NearestNeighbors(n_neighbors=kk,n_jobs=-1).fit(B).kneighbors(A)
    return float(d.mean())

def internal_div(Z,k=10):
    if len(Z)<3:return np.nan
    kk=max(2,min(k+1,len(Z))); d,_=NearestNeighbors(n_neighbors=kk,n_jobs=-1).fit(Z).kneighbors(Z)
    return float(d[:,1:].mean())

def cov_dist(A,B):
    Ca,Cb=np.cov(A,rowvar=False),np.cov(B,rowvar=False)
    return float(np.linalg.norm(Ca-Cb,'fro')/(np.linalg.norm(Ca,'fro')+np.linalg.norm(Cb,'fro')+1e-12))

def ratio(a,b): return float(a/b) if abs(b)>1e-12 else np.nan

def compute_predictors(src,tgt,ncomp=100,k=10):
    Xs,ys,Xt,yt=src['X'],src['y'],tgt['X'],tgt['y']
    mu,P=fit_svd(Xs,ncomp,0); Zs,Zt=tx(Xs,mu,P),tx(Xt,mu,P)
    sc=StandardScaler().fit(Zs); Zs=sc.transform(Zs); Zt=sc.transform(Zt)
    m=lgb.LGBMClassifier(**CFG_TASK,random_state=0).fit(Xs,ys)
    ps,pt=m.predict_proba(Xs)[:,1],m.predict_proba(Xt)[:,1]
    ds,dt=internal_div(Zs,k),internal_div(Zt,k)
    return dict(
        source_n=len(ys),target_n=len(yt),source_positives=int(ys.sum()),target_positives=int(yt.sum()),
        source_prevalence=float(ys.mean()),target_prevalence=float(yt.mean()),
        prevalence_absolute_shift=float(abs(ys.mean()-yt.mean())),prevalence_ratio=ratio(yt.mean(),ys.mean()),
        domain_auc_svd=domain_auc(Zs,Zt),
        centroid_distance_svd=float(np.linalg.norm(Zs.mean(0)-Zt.mean(0))/math.sqrt(Zs.shape[1])),
        covariance_distance_svd=cov_dist(Zs,Zt),
        source_to_target_nn_distance=knn_mean(Zs,Zt,k),target_to_source_nn_distance=knn_mean(Zt,Zs,k),
        source_internal_diversity=ds,target_internal_diversity=dt,diversity_ratio=ratio(dt,ds),
        source_positive_rate_model=float((ps>=0.5).mean()),target_positive_rate_model=float((pt>=0.5).mean()),
        source_prediction_entropy=float(entropy(ps).mean()),target_prediction_entropy=float(entropy(pt).mean()),
        entropy_shift=float(entropy(pt).mean()-entropy(ps).mean()),
        source_prediction_mean=float(ps.mean()),target_prediction_mean=float(pt.mean()),prediction_mean_shift=float(pt.mean()-ps.mean()),
        source_prediction_sd=float(ps.std()),target_prediction_sd=float(pt.std()),prediction_sd_shift=float(pt.std()-ps.std()))

def boot_ci(x,reps=5000,seed=0):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if len(x)<2:return np.nan,np.nan
    rng=np.random.default_rng(seed); b=rng.choice(x,(reps,len(x)),replace=True).mean(1)
    return tuple(np.percentile(b,[2.5,97.5]))

def corr_table(df,preds,outcome='delta_prauc'):
    rows=[]; subsets=[('pooled',df)]+[(f'budget_{int(b)}',g) for b,g in df.groupby('budget_n')]
    for name,g in subsets:
        for p in preds:
            if p not in g:continue
            x,y=g[p].to_numpy(float),g[outcome].to_numpy(float); ok=np.isfinite(x)&np.isfinite(y)
            if ok.sum()<5:continue
            sr,sp=spearmanr(x[ok],y[ok]); pr,pp=pearsonr(x[ok],y[ok])
            rows.append(dict(subset=name,predictor=p,outcome=outcome,n=int(ok.sum()),
                             spearman_rho=sr,spearman_p=sp,pearson_r=pr,pearson_p=pp))
    return pd.DataFrame(rows)

def univ(df,preds,outcome='delta_prauc'):
    rows=[]
    for p in preds:
        if p not in df:continue
        z=df[[p,outcome]].replace([np.inf,-np.inf],np.nan).dropna()
        if len(z)<8:continue
        x=z[p].to_numpy(float); y=z[outcome].to_numpy(float); xs=(x-x.mean())/(x.std()+1e-12)
        if HAVE_SM:
            try:
                md=sm.OLS(y,sm.add_constant(xs)).fit(cov_type='HC3')
                rows.append(dict(predictor=p,n=len(z),beta_per_1sd=md.params[1],se_hc3=md.bse[1],p_hc3=md.pvalues[1],r_squared=md.rsquared))
            except Exception:pass
        else:
            beta=np.cov(xs,y)[0,1]/(np.var(xs)+1e-12); pred=y.mean()+beta*xs
            r2=1-np.sum((y-pred)**2)/(np.sum((y-y.mean())**2)+1e-12)
            rows.append(dict(predictor=p,n=len(z),beta_per_1sd=beta,se_hc3=np.nan,p_hc3=np.nan,r_squared=r2))
    return pd.DataFrame(rows)

def ridge_logo(df,preds):
    available=[p for p in preds if p in df]; z=df[available+['delta_prauc','transfer_id']].replace([np.inf,-np.inf],np.nan).copy()
    for p in available:z[p]=z[p].fillna(z[p].median())
    z=z.dropna(subset=['delta_prauc','transfer_id']); X=z[available].to_numpy(float); y=z.delta_prauc.to_numpy(float); g=z.transfer_id.to_numpy()
    if len(np.unique(g))<5:return pd.DataFrame(),pd.DataFrame()
    logo=LeaveOneGroupOut(); rows=[]
    for a in [0.01,0.1,1,10,100]:
        yt,yp=[],[]
        for tr,te in logo.split(X,y,g):
            m=Pipeline([('s',StandardScaler()),('r',Ridge(alpha=a))]).fit(X[tr],y[tr]); yt.extend(y[te]); yp.extend(m.predict(X[te]))
        rows.append(dict(alpha=a,logo_rmse=math.sqrt(mean_squared_error(yt,yp)),logo_pearson_r=pearsonr(yt,yp)[0]))
    cv=pd.DataFrame(rows).sort_values('logo_rmse'); a=float(cv.iloc[0].alpha)
    m=Pipeline([('s',StandardScaler()),('r',Ridge(alpha=a))]).fit(X,y); co=m.named_steps['r'].coef_
    coef=pd.DataFrame(dict(predictor=available,ridge_coefficient=co,abs_coefficient=np.abs(co),alpha=a)).sort_values('abs_coefficient',ascending=False)
    return cv,coef

def median_rules(df,preds):
    agg={p:(p,'first') for p in preds if p in df}; agg['delta_prauc']=('delta_prauc','mean')
    t=df.groupby('transfer_id').agg(**agg).reset_index(); rows=[]
    for p in preds:
        if p not in t:continue
        z=t[[p,'delta_prauc']].replace([np.inf,-np.inf],np.nan).dropna()
        if len(z)<8:continue
        med=float(z[p].median()); lo=z[z[p]<=med].delta_prauc.to_numpy(); hi=z[z[p]>med].delta_prauc.to_numpy()
        if len(lo)<3 or len(hi)<3:continue
        _,pv=mannwhitneyu(hi,lo,alternative='two-sided')
        rows.append(dict(predictor=p,median_cut=med,n_low=len(lo),n_high=len(hi),mean_delta_low=lo.mean(),mean_delta_high=hi.mean(),high_minus_low=hi.mean()-lo.mean(),mannwhitney_p=pv))
    return pd.DataFrame(rows).sort_values('high_minus_low',ascending=False)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--labels',default='outputs/driams_long.parquet'); ap.add_argument('--matrices',default='matrices')
    ap.add_argument('--input-dir',default='outputs/multitask_active_validation'); ap.add_argument('--out',default='outputs/shift_predictors')
    ap.add_argument('--ncomp',type=int,default=100); ap.add_argument('--knn-k',type=int,default=10); a=ap.parse_args()
    mdir=Path(a.matrices); inp=Path(a.input_dir); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    lab=pd.read_parquet(a.labels); raw=pd.read_csv(inp/'raw_results.csv'); paired=pd.read_csv(inp/'paired_hybrid_vs_random.csv')
    tasks=raw[['source','target','species','drug']].drop_duplicates().reset_index(drop=True)
    print('=== SHIFT PREDICTORS ==='); print('Transfers:',len(tasks))
    rows=[]
    for i,r in tasks.iterrows():
        print(f"[{i+1}/{len(tasks)}] {r.species} / {r.drug} | {r.source}->{r.target}",flush=True)
        s=load_task(lab,mdir,r.species,r.drug,r.source); t=load_task(lab,mdir,r.species,r.drug,r.target)
        if s is None or t is None:continue
        try:p=compute_predictors(s,t,a.ncomp,a.knn_k)
        except Exception as e:
            print('  ERROR',e,flush=True); continue
        rows.append(dict(source=r.source,target=r.target,species=r.species,drug=r.drug,
                         transfer_id=f'{r.source}->{r.target}|{r.species}|{r.drug}',**p))
    pred=pd.DataFrame(rows)
    base=raw[raw.strategy=='source_only'].groupby(['source','target','species','drug']).agg(
        source_only_auroc_mean=('auroc','mean'),source_only_prauc_mean=('prauc','mean'),source_only_brier_mean=('brier','mean')).reset_index()
    pred=pred.merge(base,on=['source','target','species','drug'],how='left'); pred.to_csv(out/'transfer_predictors.csv',index=False)

    def getmetric(name,new):
        return paired[paired.metric==name][['source','target','species','drug','budget_n','delta_mean']].rename(columns={'delta_mean':new})
    an=getmetric('prauc','delta_prauc').merge(getmetric('auroc','delta_auroc'),on=['source','target','species','drug','budget_n'],how='left')
    an=an.merge(getmetric('brier','delta_brier'),on=['source','target','species','drug','budget_n'],how='left')
    an=an.merge(getmetric('selected_pos','delta_selected_pos'),on=['source','target','species','drug','budget_n'],how='left')
    an=an.merge(pred,on=['source','target','species','drug'],how='left'); an.to_csv(out/'analysis_table.csv',index=False)

    dc=corr_table(an,DEPLOY); lc=corr_table(an,LABEL); dc.to_csv(out/'correlations_deployment_predictors.csv',index=False); lc.to_csv(out/'correlations_label_derived.csv',index=False)
    agg={p:(p,'first') for p in DEPLOY+LABEL if p in an}; agg['delta_prauc']=('delta_prauc','mean')
    tm=an.groupby('transfer_id').agg(**agg).reset_index(); ols=univ(tm,DEPLOY+LABEL)
    if not ols.empty:ols=ols.sort_values(['p_hc3','r_squared'],ascending=[True,False],na_position='last')
    ols.to_csv(out/'univariate_regression.csv',index=False)
    cv,coef=ridge_logo(an,DEPLOY); cv.to_csv(out/'ridge_logo_cv.csv',index=False); coef.to_csv(out/'ridge_coefficients.csv',index=False)
    rules=median_rules(an,DEPLOY+LABEL); rules.to_csv(out/'median_split_rules.csv',index=False)
    ranked=dc[dc.subset=='pooled'].copy(); ranked['abs_spearman_rho']=ranked.spearman_rho.abs(); ranked=ranked.sort_values('abs_spearman_rho',ascending=False)
    ranked.to_csv(out/'ranked_deployment_predictors.csv',index=False)
    (out/'run_config.json').write_text(json.dumps(vars(a),indent=2),encoding='utf-8')

    print('\n=== TOP DEPLOYMENT PREDICTORS: POOLED SPEARMAN ===')
    print(ranked[['predictor','n','spearman_rho','spearman_p','pearson_r','pearson_p']].head(15).to_string(index=False))
    print('\n=== UNIVARIATE REGRESSION ===')
    print(ols[['predictor','n','beta_per_1sd','se_hc3','p_hc3','r_squared']].head(15).to_string(index=False) if not ols.empty else '(none)')
    print('\n=== RIDGE LOGO CV ==='); print(cv.to_string(index=False) if not cv.empty else '(none)')
    print('\n=== TOP RIDGE COEFFICIENTS ==='); print(coef[['predictor','ridge_coefficient','abs_coefficient','alpha']].head(15).to_string(index=False) if not coef.empty else '(none)')
    print('\n=== MEDIAN-SPLIT RULES ==='); print(rules[['predictor','median_cut','mean_delta_low','mean_delta_high','high_minus_low','mannwhitney_p']].head(15).to_string(index=False) if not rules.empty else '(none)')
    print('\nOutputs:',out)

if __name__=='__main__':main()
