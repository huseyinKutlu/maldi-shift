#!/usr/bin/env python3
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
    keys = ["source","target","species","drug"]
    z = raw[raw.strategy.isin(["random","hybrid"]) & (raw.budget_n == budget)].copy()
    perf = z.groupby(keys+["strategy"]).agg(
        auroc=("auroc","mean"), prauc=("prauc","mean"), brier=("brier","mean")
    ).reset_index()
    wide = perf.pivot_table(index=keys, columns="strategy", values=["auroc","prauc","brier"])
    wide.columns = [f"{m}_{s}" for m,s in wide.columns]
    wide = wide.reset_index()
    hp = paired[(paired.metric=="prauc") & (paired.budget_n==budget)][keys+["delta_mean"]].rename(
        columns={"delta_mean":"delta_prauc_hybrid_random"}
    )
    out = wide.merge(hp,on=keys,how="inner").merge(predictors,on=keys,how="left")
    out["transfer_id"] = out.source.astype(str)+"->"+out.target.astype(str)+"|"+out.species.astype(str)+"|"+out.drug.astype(str)
    return out.reset_index(drop=True)

def prepare_X(df):
    avail = [p for p in PREDICTORS if p in df.columns]
    X = df[avail].copy()
    for p in avail:
        X[p] = X[p].fillna(X[p].median())
    return X.to_numpy(float)

def choose_alpha_inner_logo(X,y,alphas):
    n=len(y)
    best_alpha=None
    best_rmse=np.inf
    for a in alphas:
        pred=np.zeros(n)
        for i in range(n):
            tr=np.arange(n)!=i
            pipe=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=a))])
            pipe.fit(X[tr],y[tr])
            pred[i]=pipe.predict(X[[i]])[0]
        rmse=math.sqrt(mean_squared_error(y,pred))
        if rmse<best_rmse:
            best_rmse=rmse
            best_alpha=float(a)
    return best_alpha

def nested_policy(df,alphas):
    X=prepare_X(df)
    y=df["delta_prauc_hybrid_random"].to_numpy(float)
    n=len(df)
    pred_delta=np.zeros(n)
    for i in range(n):
        tr=np.arange(n)!=i
        a=choose_alpha_inner_logo(X[tr],y[tr],alphas)
        pipe=Pipeline([("scale",StandardScaler()),("ridge",Ridge(alpha=a))])
        pipe.fit(X[tr],y[tr])
        pred_delta[i]=pipe.predict(X[[i]])[0]
    choose_h=pred_delta>0
    pr_r=df["prauc_random"].to_numpy(float)
    pr_h=df["prauc_hybrid"].to_numpy(float)
    pr_p=np.where(choose_h,pr_h,pr_r)
    pr_o=np.maximum(pr_r,pr_h)
    regret=pr_o-pr_p
    return dict(
        n_transfers=n,
        mean_random_prauc=float(pr_r.mean()),
        mean_hybrid_prauc=float(pr_h.mean()),
        mean_policy_prauc=float(pr_p.mean()),
        mean_oracle_prauc=float(pr_o.mean()),
        mean_gain_vs_random=float((pr_p-pr_r).mean()),
        mean_gain_vs_hybrid=float((pr_p-pr_h).mean()),
        sign_accuracy=float(np.mean(choose_h==(y>0))),
        mean_regret=float(regret.mean()),
        zero_regret_fraction=float(np.mean(regret==0)),
    )

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--multitask-dir",default="outputs/multitask_active_validation_final")
    ap.add_argument("--shift-dir",default="outputs/shift_predictors")
    ap.add_argument("--out",default="outputs/leave_one_species_out_policy")
    ap.add_argument("--budgets",default="20,30,50")
    ap.add_argument("--alphas",default="0.01,0.1,1,10,100")
    a=ap.parse_args()

    md=Path(a.multitask_dir); sd=Path(a.shift_dir); out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(md/"raw_results.csv")
    paired=pd.read_csv(md/"paired_hybrid_vs_random.csv")
    predictors=pd.read_csv(sd/"transfer_predictors.csv")
    budgets=[int(x) for x in a.budgets.split(",")]
    alphas=[float(x) for x in a.alphas.split(",")]

    full_rows=[]; loo_rows=[]
    print("=== LEAVE-ONE-SPECIES-OUT POLICY ===")
    print("Budgets:",budgets)

    for budget in budgets:
        print(f"\n--- Budget {budget} ---")
        df=build_transfer_table(raw,paired,predictors,budget)
        full=nested_policy(df,alphas)
        full_rows.append(dict(budget_n=budget,excluded_species="NONE",**full))
        print(f"Full data: n={full['n_transfers']}, Policy-Random={full['mean_gain_vs_random']:+.6f}, sign_acc={full['sign_accuracy']:.3f}")
        for sp in sorted(df.species.dropna().unique()):
            sub=df[df.species!=sp].reset_index(drop=True)
            if len(sub)<6:
                continue
            res=nested_policy(sub,alphas)
            loo_rows.append(dict(
                budget_n=budget,excluded_species=sp,
                excluded_n=int((df.species==sp).sum()),
                remaining_n=len(sub),
                full_mean_gain_vs_random=full["mean_gain_vs_random"],
                delta_from_full=res["mean_gain_vs_random"]-full["mean_gain_vs_random"],
                **res
            ))
            print(f"  exclude {sp:28s} | remain={len(sub):2d} | Policy-Random={res['mean_gain_vs_random']:+.6f} | sign_acc={res['sign_accuracy']:.3f}")

    full_df=pd.DataFrame(full_rows)
    loo_df=pd.DataFrame(loo_rows)
    full_df.to_csv(out/"full_data_summary.csv",index=False)
    loo_df.to_csv(out/"leave_one_species_out_results.csv",index=False)

    stability=[]
    for budget,g in loo_df.groupby("budget_n"):
        gains=g["mean_gain_vs_random"].to_numpy(float)
        stability.append(dict(
            budget_n=int(budget),
            n_species_exclusions=len(g),
            all_positive=bool(np.all(gains>0)),
            positive_fraction=float(np.mean(gains>0)),
            minimum_gain=float(gains.min()),
            maximum_gain=float(gains.max()),
            mean_gain=float(gains.mean()),
            median_gain=float(np.median(gains)),
            min_sign_accuracy=float(g.sign_accuracy.min()),
            mean_sign_accuracy=float(g.sign_accuracy.mean())
        ))
    stab_df=pd.DataFrame(stability)
    stab_df.to_csv(out/"direction_stability_summary.csv",index=False)
    (out/"run_config.json").write_text(json.dumps({
        **vars(a),"predictors":PREDICTORS,
        "primary_sensitivity_question":"Does Policy-Random remain positive after excluding each species?"
    },indent=2),encoding="utf-8")

    print("\n=== FULL DATA SUMMARY ===")
    print(full_df[["budget_n","n_transfers","mean_gain_vs_random","mean_gain_vs_hybrid","sign_accuracy","mean_regret"]].to_string(index=False))
    print("\n=== LEAVE-ONE-SPECIES-OUT RESULTS ===")
    print(loo_df[["budget_n","excluded_species","excluded_n","remaining_n","mean_gain_vs_random","delta_from_full","sign_accuracy","mean_regret"]].to_string(index=False))
    print("\n=== DIRECTION STABILITY SUMMARY ===")
    print(stab_df.to_string(index=False))
    print("\nOutputs:",out)

if __name__=="__main__":
    main()
