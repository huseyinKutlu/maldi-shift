#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

def boot_diff(x,y,reps=10000,seed=0):
    x=np.asarray(x,float); y=np.asarray(y,float)
    ok=np.isfinite(x)&np.isfinite(y); d=x[ok]-y[ok]
    if len(d)<2:
        return np.nan,np.nan,np.nan,False,len(d)
    rng=np.random.default_rng(seed)
    bs=rng.choice(d,(reps,len(d)),replace=True).mean(1)
    lo,hi=np.percentile(bs,[2.5,97.5])
    return float(d.mean()),float(lo),float(hi),bool(lo>0 or hi<0),len(d)

def compare(raw,s1,b1,s2,b2,reps=10000,seed=0):
    a=raw[(raw.strategy==s1)&(raw.budget_n==b1)][["rep","auroc","prauc","brier","selected_pos"]]
    b=raw[(raw.strategy==s2)&(raw.budget_n==b2)][["rep","auroc","prauc","brier","selected_pos"]]
    m=a.merge(b,on="rep",suffixes=("_a","_b"))
    rows=[]
    for met in ["auroc","prauc","brier","selected_pos"]:
        d,lo,hi,sig,n=boot_diff(m[f"{met}_a"],m[f"{met}_b"],reps,seed)
        rows.append(dict(strategy_a=s1,budget_a=b1,strategy_b=s2,budget_b=b2,
                         metric=met,mean_a=m[f"{met}_a"].mean(),mean_b=m[f"{met}_b"].mean(),
                         mean_diff=d,ci_low=lo,ci_high=hi,significant=sig,n_pairs=n))
    return pd.DataFrame(rows)

def summary(raw):
    rows=[]
    z=raw[raw.strategy!="source_only"]
    for (s,b),g in z.groupby(["strategy","budget_n"]):
        rows.append(dict(strategy=s,budget_n=int(b),n_reps=len(g),
                         auroc_mean=g.auroc.mean(),prauc_mean=g.prauc.mean(),
                         brier_mean=g.brier.mean(),selected_pos_mean=g.selected_pos.mean(),
                         selected_pos_rate_mean=g.selected_pos_rate.mean()))
    return pd.DataFrame(rows)

def random_equivalent(summ,target):
    r=summ[summ.strategy=="random"].sort_values("budget_n")
    x=r.budget_n.to_numpy(float); y=np.maximum.accumulate(r.prauc_mean.to_numpy(float))
    if target<=y[0]: return float(x[0])
    if target>y[-1]: return np.nan
    for i in range(1,len(x)):
        if y[i]>=target:
            if abs(y[i]-y[i-1])<1e-12: return float(x[i])
            t=(target-y[i-1])/(y[i]-y[i-1])
            return float(x[i-1]+t*(x[i]-x[i-1]))
    return np.nan

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--raw",default="outputs/active_target_selection/Staphylococcus_aureus__Oxacillin__DRIAMS-A_to_DRIAMS-C__raw.csv")
    ap.add_argument("--out",default="outputs/active_selection_validation")
    ap.add_argument("--bootstrap-reps",type=int,default=10000)
    a=ap.parse_args()

    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    raw=pd.read_csv(a.raw); summ=summary(raw)
    summ.to_csv(out/"strategy_learning_curves.csv",index=False)

    budgets=[10,20,30,50,75,100]
    matched=pd.concat([compare(raw,"hybrid",b,"random",b,a.bootstrap_reps,b) for b in budgets],ignore_index=True)
    matched.to_csv(out/"hybrid_vs_random_matched_budget.csv",index=False)

    pairs=[(10,50),(10,75),(10,100),(20,50),(20,75),(20,100),(30,50),(30,75),(30,100)]
    head=pd.concat([compare(raw,"hybrid",hb,"random",rb,a.bootstrap_reps,1000+hb+rb) for hb,rb in pairs],ignore_index=True)
    head.to_csv(out/"headline_comparisons.csv",index=False)

    eff=[]
    h=summ[summ.strategy=="hybrid"].sort_values("budget_n")
    for _,r in h.iterrows():
        req=random_equivalent(summ,r.prauc_mean)
        eff.append(dict(hybrid_budget=int(r.budget_n),hybrid_prauc=r.prauc_mean,
                        hybrid_auroc=r.auroc_mean,hybrid_selected_pos=r.selected_pos_mean,
                        random_budget_needed_for_same_prauc=req,
                        label_efficiency_ratio=(req/r.budget_n if np.isfinite(req) else np.nan)))
    eff=pd.DataFrame(eff); eff.to_csv(out/"label_efficiency.csv",index=False)

    best=[]
    for b,g in summ.groupby("budget_n"):
        r=g.sort_values(["prauc_mean","auroc_mean","brier_mean"],ascending=[False,False,True]).iloc[0]
        best.append(dict(budget_n=int(b),best_strategy=r.strategy,auroc=r.auroc_mean,
                         prauc=r.prauc_mean,brier=r.brier_mean,selected_pos=r.selected_pos_mean))
    best=pd.DataFrame(best); best.to_csv(out/"best_strategy_by_budget.csv",index=False)

    print("=== ACTIVE SELECTION VALIDATION ===")
    print("\n=== BEST STRATEGY BY BUDGET ===")
    print(best.to_string(index=False))
    print("\n=== HYBRID VS RANDOM: MATCHED BUDGET PR-AUC ===")
    print(matched[matched.metric=="prauc"][["budget_a","mean_a","mean_b","mean_diff","ci_low","ci_high","significant"]].to_string(index=False))
    print("\n=== HEADLINE: HYBRID VS LARGER RANDOM BUDGETS (PR-AUC) ===")
    print(head[head.metric=="prauc"][["budget_a","budget_b","mean_a","mean_b","mean_diff","ci_low","ci_high","significant"]].to_string(index=False))
    print("\n=== LABEL EFFICIENCY ===")
    print(eff.to_string(index=False))
    (out/"run_config.json").write_text(json.dumps(vars(a),indent=2),encoding="utf-8")
    print("\nOutputs:",out)

if __name__=="__main__":
    main()
