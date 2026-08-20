#!/usr/bin/env python3
import argparse, json, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, accuracy_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PRIMARY_PREDICTORS = [
    "domain_auc_svd",
    "centroid_distance_svd",
    "covariance_distance_svd",
    "source_to_target_nn_distance",
    "source_internal_diversity",
    "source_only_auroc_mean",
]

def bootstrap_corr(x,y,kind="spearman",reps=10000,seed=0):
    x=np.asarray(x,float); y=np.asarray(y,float)
    ok=np.isfinite(x)&np.isfinite(y); x=x[ok]; y=y[ok]
    if len(x)<5: return dict(n=len(x),estimate=np.nan,ci_low=np.nan,ci_high=np.nan,p_value=np.nan)
    est,p=(spearmanr(x,y) if kind=="spearman" else pearsonr(x,y))
    rng=np.random.default_rng(seed); vals=[]
    for _ in range(reps):
        idx=rng.choice(len(x),len(x),replace=True)
        try:
            r=(spearmanr(x[idx],y[idx])[0] if kind=="spearman" else pearsonr(x[idx],y[idx])[0])
        except: continue
        if np.isfinite(r): vals.append(r)
    lo,hi=(np.percentile(vals,[2.5,97.5]) if len(vals)>=100 else (np.nan,np.nan))
    return dict(n=len(x),estimate=float(est),ci_low=float(lo),ci_high=float(hi),p_value=float(p))

def make_transfer_level_table(paired,predictors,budgets):
    keys=["source","target","species","drug"]
    def agg_metric(metric,name):
        z=paired[(paired.metric==metric)&(paired.budget_n.isin(budgets))]
        return z.groupby(keys).agg(**{name:("delta_mean","mean")}).reset_index()
    p=paired[(paired.metric=="prauc")&(paired.budget_n.isin(budgets))]
    out=p.groupby(keys).agg(
        mean_delta_prauc=("delta_mean","mean"),
        median_delta_prauc=("delta_mean","median"),
        min_delta_prauc=("delta_mean","min"),
        max_delta_prauc=("delta_mean","max"),
        positive_budget_fraction=("delta_mean",lambda x: np.mean(np.asarray(x)>0)),
        significant_positive_budget_fraction=("significant",lambda x: np.mean(np.asarray(x,dtype=bool))),
    ).reset_index()
    out=out.merge(agg_metric("auroc","mean_delta_auroc"),on=keys,how="left")
    out=out.merge(agg_metric("brier","mean_delta_brier"),on=keys,how="left")
    out=out.merge(predictors,on=keys,how="left")
    out["transfer_id"]=out.source.astype(str)+"->"+out.target.astype(str)+"|"+out.species.astype(str)+"|"+out.drug.astype(str)
    return out

def confirmatory_correlations(df,predictors,outcome,reps):
    rows=[]
    for i,pred in enumerate(predictors):
        if pred not in df.columns: continue
        s=bootstrap_corr(df[pred],df[outcome],"spearman",reps,100+i)
        q=bootstrap_corr(df[pred],df[outcome],"pearson",reps,500+i)
        rows.append(dict(predictor=pred,outcome=outcome,n=s["n"],
                         spearman_rho=s["estimate"],spearman_ci_low=s["ci_low"],spearman_ci_high=s["ci_high"],spearman_p=s["p_value"],
                         pearson_r=q["estimate"],pearson_ci_low=q["ci_low"],pearson_ci_high=q["ci_high"],pearson_p=q["p_value"]))
    return pd.DataFrame(rows)

def median_split_rules(df,predictors,outcome):
    rows=[]
    for pred in predictors:
        if pred not in df.columns: continue
        z=df[[pred,outcome]].replace([np.inf,-np.inf],np.nan).dropna()
        if len(z)<8: continue
        cut=float(z[pred].median())
        lo=z[z[pred]<=cut][outcome].to_numpy(float); hi=z[z[pred]>cut][outcome].to_numpy(float)
        if len(lo)<3 or len(hi)<3: continue
        try: p=mannwhitneyu(hi,lo,alternative="two-sided")[1]
        except: p=np.nan
        rows.append(dict(predictor=pred,median_cut=cut,n_low=len(lo),n_high=len(hi),
                         mean_delta_low=float(lo.mean()),mean_delta_high=float(hi.mean()),
                         high_minus_low=float(hi.mean()-lo.mean()),mannwhitney_p=float(p)))
    return pd.DataFrame(rows)

def prepare_matrix(df,predictors,outcome):
    avail=[p for p in predictors if p in df.columns]
    z=df[["transfer_id",outcome]+avail].replace([np.inf,-np.inf],np.nan).copy()
    for p in avail: z[p]=z[p].fillna(z[p].median())
    z=z.dropna(subset=[outcome]).reset_index(drop=True)
    return z,avail,z[avail].to_numpy(float),z[outcome].to_numpy(float)

def logo_ridge(df,predictors,outcome,alphas):
    z,avail,X,y=prepare_matrix(df,predictors,outcome)
    rows=[]; n=len(z)
    for a in alphas:
        pred=np.zeros(n)
        for i in range(n):
            tr=np.arange(n)!=i
            pipe=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=a))])
            pipe.fit(X[tr],y[tr]); pred[i]=pipe.predict(X[[i]])[0]
        rmse=math.sqrt(mean_squared_error(y,pred))
        rows.append(dict(alpha=a,logo_rmse=rmse,
                         logo_pearson_r=float(pearsonr(y,pred)[0]),
                         logo_spearman_rho=float(spearmanr(y,pred)[0])))
    s=pd.DataFrame(rows).sort_values("logo_rmse")
    best=float(s.iloc[0].alpha)
    pred=np.zeros(n)
    for i in range(n):
        tr=np.arange(n)!=i
        pipe=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=best))])
        pipe.fit(X[tr],y[tr]); pred[i]=pipe.predict(X[[i]])[0]
    p=z[["transfer_id",outcome]].copy()
    p["predicted_delta_prauc"]=pred; p["residual"]=p[outcome]-pred; p["alpha"]=best
    return s,p

def permutation_logo(df,predictors,outcome,alpha,reps=5000,seed=0):
    z,avail,X,y=prepare_matrix(df,predictors,outcome); n=len(z)
    def corr(yt):
        pred=np.zeros(n)
        for i in range(n):
            tr=np.arange(n)!=i
            pipe=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=alpha))])
            pipe.fit(X[tr],yt[tr]); pred[i]=pipe.predict(X[[i]])[0]
        return pearsonr(yt,pred)[0]
    obs=corr(y); rng=np.random.default_rng(seed); null=[]
    for _ in range(reps):
        yp=rng.permutation(y); null.append(corr(yp))
    null=np.asarray(null,float)
    p=(1+np.sum(null>=obs))/(1+len(null)); lo,hi=np.percentile(null,[2.5,97.5])
    return pd.DataFrame([dict(observed_logo_pearson_r=float(obs),permutation_p=float(p),
                              null_ci_low=float(lo),null_ci_high=float(hi),permutations=len(null),alpha=alpha)])

def best_threshold_train(x,y):
    x=np.asarray(x,float); ybin=(np.asarray(y,float)>0).astype(int); u=np.unique(x[np.isfinite(x)])
    if len(u)<2:return None
    best=None
    for d in ["le","ge"]:
        for t in (u[:-1]+u[1:])/2:
            pred=(x<=t).astype(int) if d=="le" else (x>=t).astype(int)
            bal=balanced_accuracy_score(ybin,pred)
            if best is None or bal>best["bal"]: best=dict(threshold=float(t),direction=d,bal=float(bal))
    return best

def logo_threshold(df,predictors,outcome):
    rows=[]; sums=[]
    for pred in predictors:
        if pred not in df.columns: continue
        z=df[["transfer_id",pred,outcome]].replace([np.inf,-np.inf],np.nan).dropna().reset_index(drop=True)
        if len(z)<8: continue
        preds=[]; th=[]; dr=[]
        for i in range(len(z)):
            tr=np.arange(len(z))!=i
            rule=best_threshold_train(z.loc[tr,pred].to_numpy(float),z.loc[tr,outcome].to_numpy(float))
            if rule is None: preds.append(np.nan); th.append(np.nan); dr.append(""); continue
            xv=float(z.loc[i,pred]); yp=int(xv<=rule["threshold"]) if rule["direction"]=="le" else int(xv>=rule["threshold"])
            preds.append(yp); th.append(rule["threshold"]); dr.append(rule["direction"])
        ytrue=(z[outcome].to_numpy(float)>0).astype(int); pa=np.asarray(preds,float); ok=np.isfinite(pa)
        sums.append(dict(predictor=pred,n=int(ok.sum()),accuracy=float(accuracy_score(ytrue[ok],pa[ok])),
                         balanced_accuracy=float(balanced_accuracy_score(ytrue[ok],pa[ok])),
                         mean_threshold=float(np.nanmean(th)),
                         modal_direction=("le" if dr.count("le")>=dr.count("ge") else "ge")))
        for i in range(len(z)):
            rows.append(dict(predictor=pred,transfer_id=z.loc[i,"transfer_id"],true_benefit=int(ytrue[i]),
                             predicted_benefit=(int(preds[i]) if np.isfinite(preds[i]) else np.nan),
                             threshold=th[i],direction=dr[i],predictor_value=float(z.loc[i,pred]),
                             observed_delta_prauc=float(z.loc[i,outcome])))
    return pd.DataFrame(rows),pd.DataFrame(sums)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--shift-dir",default="outputs/shift_predictors")
    ap.add_argument("--multitask-dir",default="outputs/multitask_active_validation")
    ap.add_argument("--out",default="outputs/shift_predictors_validate")
    ap.add_argument("--budgets",default="20,30,50")
    ap.add_argument("--bootstrap-reps",type=int,default=10000)
    ap.add_argument("--permutation-reps",type=int,default=5000)
    a=ap.parse_args()

    sd=Path(a.shift_dir); md=Path(a.multitask_dir); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    preds=pd.read_csv(sd/"transfer_predictors.csv")
    paired=pd.read_csv(md/"paired_hybrid_vs_random.csv")
    budgets=list(map(int,a.budgets.split(",")))
    tr=make_transfer_level_table(paired,preds,budgets); tr.to_csv(out/"transfer_level_table.csv",index=False)

    corr=confirmatory_correlations(tr,PRIMARY_PREDICTORS,"mean_delta_prauc",a.bootstrap_reps)
    corr["abs_rho"]=corr.spearman_rho.abs(); corr=corr.sort_values("abs_rho",ascending=False)
    corr.to_csv(out/"confirmatory_correlations.csv",index=False)

    rules=median_split_rules(tr,PRIMARY_PREDICTORS,"mean_delta_prauc")
    rules.to_csv(out/"median_split_rules.csv",index=False)

    ridge_sum,ridge_pred=logo_ridge(tr,PRIMARY_PREDICTORS,"mean_delta_prauc",[0.01,0.1,1,10,100])
    ridge_sum.to_csv(out/"ridge_logo_summary.csv",index=False)
    ridge_pred.to_csv(out/"ridge_logo_predictions.csv",index=False)

    best=float(ridge_sum.iloc[0].alpha)
    perm=permutation_logo(tr,PRIMARY_PREDICTORS,"mean_delta_prauc",best,a.permutation_reps,0)
    perm.to_csv(out/"ridge_permutation.csv",index=False)

    tp,ts=logo_threshold(tr,PRIMARY_PREDICTORS,"mean_delta_prauc")
    tp.to_csv(out/"threshold_logo_predictions.csv",index=False)
    ts=ts.sort_values("balanced_accuracy",ascending=False); ts.to_csv(out/"threshold_rule_summary.csv",index=False)

    (out/"run_config.json").write_text(json.dumps({**vars(a),"primary_predictors":PRIMARY_PREDICTORS},indent=2),encoding="utf-8")

    print("=== SHIFT PREDICTORS VALIDATE ===")
    print(f"Independent transfers: {len(tr)}")
    print(f"Budgets averaged: {budgets}")
    print("\n=== TRANSFER-LEVEL CONFIRMATORY CORRELATIONS ===")
    print(corr[["predictor","n","spearman_rho","spearman_ci_low","spearman_ci_high","spearman_p",
                "pearson_r","pearson_ci_low","pearson_ci_high"]].to_string(index=False))
    print("\n=== MEDIAN SPLIT RULES ===")
    print(rules[["predictor","median_cut","mean_delta_low","mean_delta_high","high_minus_low","mannwhitney_p"]].to_string(index=False))
    print("\n=== RIDGE LOGO SUMMARY ===")
    print(ridge_sum.to_string(index=False))
    print("\n=== RIDGE PERMUTATION ===")
    print(perm.to_string(index=False))
    print("\n=== THRESHOLD RULE SUMMARY ===")
    print(ts.to_string(index=False))
    print("\nOutputs:",out)

if __name__=="__main__":
    main()
