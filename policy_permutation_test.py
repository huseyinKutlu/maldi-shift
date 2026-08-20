#!/usr/bin/env python3
"""
Permutation test for leakage-free selective adaptation policy.
"""

import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PREDICTORS = [
    "domain_auc_svd",
    "centroid_distance_svd",
    "covariance_distance_svd",
    "source_to_target_nn_distance",
    "source_internal_diversity",
]

def build_transfer_table(raw, paired, predictors, budget):
    keys = ["source", "target", "species", "drug"]
    z = raw[raw.strategy.isin(["random","hybrid"]) & (raw.budget_n == budget)].copy()
    perf = z.groupby(keys + ["strategy"]).agg(
        auroc=("auroc","mean"),
        prauc=("prauc","mean"),
        brier=("brier","mean"),
    ).reset_index()
    wide = perf.pivot_table(index=keys, columns="strategy", values=["auroc","prauc","brier"])
    wide.columns = [f"{m}_{s}" for m,s in wide.columns]
    wide = wide.reset_index()

    hp = paired[(paired.metric=="prauc") & (paired.budget_n==budget)][keys+["delta_mean"]].rename(
        columns={"delta_mean":"delta_prauc_hybrid_random"}
    )
    out = wide.merge(hp,on=keys,how="inner").merge(predictors,on=keys,how="left")
    out["transfer_id"] = (
        out.source.astype(str)+"->"+out.target.astype(str)+"|"+
        out.species.astype(str)+"|"+out.drug.astype(str)
    )
    return out.reset_index(drop=True)

def prepare_X(df):
    avail=[p for p in PREDICTORS if p in df.columns]
    X=df[avail].copy()
    for p in avail:
        X[p]=X[p].fillna(X[p].median())
    return X.to_numpy(float)

def choose_alpha(X,y,alphas):
    n=len(y)
    best_a,best_rmse=None,np.inf
    for a in alphas:
        pred=np.zeros(n)
        for i in range(n):
            tr=np.arange(n)!=i
            pipe=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=a))])
            pipe.fit(X[tr],y[tr])
            pred[i]=pipe.predict(X[[i]])[0]
        rmse=math.sqrt(mean_squared_error(y,pred))
        if rmse<best_rmse:
            best_a,best_rmse=float(a),rmse
    return best_a

def outer_logo_policy(df, y_learning, alphas):
    X=prepare_X(df)
    true_delta=df["delta_prauc_hybrid_random"].to_numpy(float)
    n=len(df)
    pred_delta=np.zeros(n)

    for i in range(n):
        tr=np.arange(n)!=i
        alpha=choose_alpha(X[tr], y_learning[tr], alphas)
        pipe=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=alpha))])
        pipe.fit(X[tr], y_learning[tr])
        pred_delta[i]=pipe.predict(X[[i]])[0]

    choose_hybrid=pred_delta>0
    pr_r=df["prauc_random"].to_numpy(float)
    pr_h=df["prauc_hybrid"].to_numpy(float)
    pr_policy=np.where(choose_hybrid, pr_h, pr_r)
    pr_oracle=np.maximum(pr_r,pr_h)
    regret=pr_oracle-pr_policy

    return {
        "mean_policy_prauc":float(pr_policy.mean()),
        "mean_gain_vs_random":float((pr_policy-pr_r).mean()),
        "mean_gain_vs_hybrid":float((pr_policy-pr_h).mean()),
        "sign_accuracy":float(np.mean(choose_hybrid==(true_delta>0))),
        "mean_regret":float(regret.mean()),
        "zero_regret_fraction":float(np.mean(regret==0)),
    }

def p_greater(null,obs):
    null=np.asarray(null,float)
    return float((1+np.sum(null>=obs))/(1+len(null)))

def p_less(null,obs):
    null=np.asarray(null,float)
    return float((1+np.sum(null<=obs))/(1+len(null)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--multitask-dir",default="outputs/multitask_active_validation_final")
    ap.add_argument("--shift-dir",default="outputs/shift_predictors")
    ap.add_argument("--out",default="outputs/policy_permutation_test")
    ap.add_argument("--budgets",default="20,30,50")
    ap.add_argument("--permutations",type=int,default=10000)
    ap.add_argument("--alphas",default="0.01,0.1,1,10,100")
    ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args()

    md=Path(a.multitask_dir); sd=Path(a.shift_dir); out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)

    raw=pd.read_csv(md/"raw_results.csv")
    paired=pd.read_csv(md/"paired_hybrid_vs_random.csv")
    predictors=pd.read_csv(sd/"transfer_predictors.csv")
    budgets=[int(x) for x in a.budgets.split(",")]
    alphas=[float(x) for x in a.alphas.split(",")]
    rng=np.random.default_rng(a.seed)

    obs_rows=[]; raw_rows=[]; sum_rows=[]

    print("=== POLICY PERMUTATION TEST ===")
    print("Budgets:",budgets)
    print("Permutations:",a.permutations)

    for budget in budgets:
        print(f"\n--- Budget {budget} ---",flush=True)
        df=build_transfer_table(raw,paired,predictors,budget)
        true_y=df["delta_prauc_hybrid_random"].to_numpy(float)
        obs=outer_logo_policy(df,true_y,alphas)
        obs_rows.append(dict(budget_n=budget,n_transfers=len(df),**obs))
        print(f"Observed: ΔPR(Random)={obs['mean_gain_vs_random']:+.6f}, sign_acc={obs['sign_accuracy']:.3f}, regret={obs['mean_regret']:.6f}")

        rows=[]
        for b in range(a.permutations):
            yp=rng.permutation(true_y)
            res=outer_logo_policy(df,yp,alphas)
            row=dict(budget_n=budget,permutation=b,**res)
            rows.append(row)
            if (b+1)%500==0:
                print(f"  permutation {b+1}/{a.permutations}",flush=True)

        bp=pd.DataFrame(rows)
        raw_rows.extend(rows)

        tests=[
            ("mean_gain_vs_random","greater"),
            ("mean_gain_vs_hybrid","greater"),
            ("sign_accuracy","greater"),
            ("mean_regret","less"),
            ("zero_regret_fraction","greater"),
        ]
        for metric,direction in tests:
            null=bp[metric].to_numpy(float)
            observed=obs[metric]
            p=p_greater(null,observed) if direction=="greater" else p_less(null,observed)
            sum_rows.append(dict(
                budget_n=budget,metric=metric,observed=observed,
                null_mean=float(null.mean()),null_sd=float(null.std(ddof=1)),
                null_ci_low=float(np.percentile(null,2.5)),
                null_ci_high=float(np.percentile(null,97.5)),
                empirical_p=p,direction=direction
            ))

    obs_df=pd.DataFrame(obs_rows)
    raw_df=pd.DataFrame(raw_rows)
    sum_df=pd.DataFrame(sum_rows)

    obs_df.to_csv(out/"observed_summary.csv",index=False)
    raw_df.to_csv(out/"permutation_raw.csv",index=False)
    sum_df.to_csv(out/"permutation_summary.csv",index=False)
    (out/"run_config.json").write_text(json.dumps({
        **vars(a),
        "predictors":PREDICTORS,
        "permutation_target":"delta_prauc_hybrid_random across transfer tasks",
        "evaluation":"Observed random/hybrid PR-AUC retained; only the policy-learning target is permuted."
    },indent=2),encoding="utf-8")

    print("\n=== OBSERVED SUMMARY ===")
    print(obs_df.to_string(index=False))
    print("\n=== PERMUTATION SUMMARY ===")
    print(sum_df[["budget_n","metric","observed","null_mean","null_sd","null_ci_low","null_ci_high","empirical_p"]].to_string(index=False))
    print("\nOutputs:",out)

if __name__=="__main__":
    main()
