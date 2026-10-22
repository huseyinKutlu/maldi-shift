#!/usr/bin/env python3
"""
revision_analyses.py
====================

Additional analyses requested in the major revision of antibiotics-4609218.
Place this file in the repository root (next to nested_cv.py,
multitask_active_validation.py and leave_one_cluster_out_policy.py).

Subcommands
-----------
selfcheck   Reproduce Table 5 from the 72-task LOCO policy predictions (sanity check).
comparator  R1-3  Descriptor-free "Training-Average" comparator          -> Table S18
dev         R1-1  Policy results without the 20-task development subset   -> Table S19b
blocks      R1-2  Sign-flip with species blocks (clusters sharing isolates)
losbo       R1-2  Nested leave-one-SPECIES-BLOCK-out re-evaluation of the policy
overlap     R1-2  Shared source/target isolates between clusters           -> Table S20
            R2-6  + patient-identifier availability per site
duplicates  R1-4  Antibiotic pairs with (near-)identical labels (derived phenotypes)
descset     R1-1  Fixed 5 vs geometry-8 vs 21 label-free vs nested top-k selection
dedup       R2-7  Re-run Table 5 / Table 3 without duplicated target phenotypes
budget-merge      Combine sharded budget runs (--shard/--nshards)
budget      R1-4  Distinct isolates needed per species at a given budget  -> Table S21
            R1-5  (--errors) false-susceptible / false-resistant rates   -> Table S22

Inputs
------
--policy        LOCO policy_predictions.csv of the 72-TASK run
                (columns: budget_n, cluster_id, source, target, species, drug,
                 prauc_random, prauc_hybrid, prauc_policy, policy_choice, ...)
--multitask-dir 72-task multitask_active_validation output folder
                (raw_results.csv, paired_hybrid_vs_random.csv, task_inventory.csv)
--shift-dir     72-task shift-descriptor folder (transfer_predictors.csv)
--dev-tasks     CSV listing the 20 development tasks (source,target,species,drug);
                the 20-task raw_results.csv of the development run also works.
                If omitted, the subset is reconstructed from task_inventory.csv
                (eligible tasks sorted by target positives, then target n; first 20).

Inference
---------
All p values are two-sided sign-flip p values obtained by complete enumeration of
2^k sign assignments of k unit-level differences (statistic = unweighted mean of
the unit-level differences), identical to Table 5 of the manuscript.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEYS = ["source", "target", "species", "drug"]
EPS = 1e-12


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def signflip(d):
    """Return (mean, two-sided enumerated p, n_units, n_nonzero, min_attainable_p)."""
    d = np.asarray(d, dtype=float)
    d = d[np.isfinite(d)]
    k = len(d)
    if k == 0:
        return np.nan, np.nan, 0, 0, np.nan
    if k > 22:
        raise ValueError("too many units for complete enumeration")
    obs = float(d.mean())
    signs = np.array(list(itertools.product([1.0, -1.0], repeat=k)))
    stat = (signs * d).mean(axis=1)
    p = float(np.mean(np.abs(stat) >= abs(obs) - EPS))
    nz = int(np.sum(np.abs(d) > EPS))
    pmin = 2.0 / 2 ** nz if nz > 0 else 1.0
    return obs, p, k, nz, pmin


def cluster_of(df):
    return df["source"].astype(str) + "->" + df["target"].astype(str) + "|" + df["species"].astype(str)


def load_policy(path):
    p = pd.read_csv(path)
    need = {"budget_n", "source", "target", "species", "drug",
            "prauc_random", "prauc_hybrid", "prauc_policy"}
    miss = need - set(p.columns)
    if miss:
        raise SystemExit(f"{path} is missing columns: {sorted(miss)}")
    if "cluster_id" not in p.columns:
        p["cluster_id"] = cluster_of(p)
    return p


def cluster_diffs(df, a, b, unit="cluster_id"):
    """Equal-unit estimand: mean over tasks within unit of (a - b)."""
    z = df.assign(_d=df[a] - df[b])
    return z.groupby(unit)["_d"].mean()


def block_diffs(df, a, b, block="species"):
    """Clusters averaged within block (equal-cluster), then one value per block."""
    cm = df.assign(_d=df[a] - df[b]).groupby(["cluster_id"]).agg(
        _d=("_d", "mean"), blk=(block, "first"))
    return cm.groupby("blk")["_d"].mean()


def compare_rows(df, label, budget, unit="cluster_id"):
    rows = []
    for a, b, name in [("prauc_policy", "prauc_random", "Policy - Random"),
                       ("prauc_policy", "prauc_hybrid", "Policy - Hybrid")]:
        if a not in df or b not in df:
            continue
        d = cluster_diffs(df, a, b, unit)
        m, p, k, nz, pmin = signflip(d.values)
        rows.append(dict(analysis=label, budget_n=budget, comparison=name,
                         n_tasks=len(df), n_units=k, n_nonzero_units=nz,
                         equal_unit_delta=round(m, 5),
                         task_weighted_delta=round(float((df[a] - df[b]).mean()), 5),
                         units_positive=int((d > EPS).sum()),
                         units_negative=int((d < -EPS).sum()),
                         signflip_p=round(p, 4), min_attainable_p=round(pmin, 4)))
    return rows


def out_dir(path):
    o = Path(path)
    o.mkdir(parents=True, exist_ok=True)
    return o


def read_dev_tasks(dev_path, inventory_path):
    if dev_path:
        d = pd.read_csv(dev_path)
        d = d[KEYS].drop_duplicates()
        print(f"Development tasks read from {dev_path}: {len(d)}")
        return d
    if not inventory_path or not Path(inventory_path).exists():
        raise SystemExit("Give --dev-tasks or --inventory (task_inventory.csv of the 72-task run).")
    inv = pd.read_csv(inventory_path)
    inv = inv[inv["eligible"].astype(bool)]
    inv = inv.sort_values(["positives_target", "n_target"], ascending=[False, False])
    d = inv.head(20)[KEYS]
    print("Development tasks RECONSTRUCTED from task_inventory.csv (verify!):")
    print(d.to_string(index=False))
    return d


# ----------------------------------------------------------------------------
# selfcheck: reproduce Table 5
# ----------------------------------------------------------------------------
def cmd_selfcheck(a):
    pol = load_policy(a.policy)
    rows = []
    for b, g in pol.groupby("budget_n"):
        rows += compare_rows(g, "all tasks", int(b))
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))
    print("\nCompare with Table 5: Policy-Random +0.0084/+0.0078/+0.0048, p 0.0645/0.0117/0.0449;"
          " Policy-Hybrid 0.0000/+0.0015/-0.0012, p 1.000/0.125/1.000")
    t.to_csv(out_dir(a.out) / "selfcheck_table5.csv", index=False)


# ----------------------------------------------------------------------------
# A1 comparator
# ----------------------------------------------------------------------------
def cmd_comparator(a):
    pol = load_policy(a.policy)
    o = out_dir(a.out)
    fold_rows, task_rows, summ = [], [], []
    for b, g in pol.groupby("budget_n"):
        g = g.copy()
        g["dHR"] = g["prauc_hybrid"] - g["prauc_random"]
        cl = g.groupby("cluster_id").agg(dHR=("dHR", "mean"), n=("dHR", "size"))
        for c in cl.index:
            tr = cl.drop(index=c)
            tr_tasks = g[g.cluster_id != c]
            m_eq = float(tr["dHR"].mean())
            m_tw = float(tr_tasks["dHR"].mean())
            ch_eq = "hybrid" if m_eq > 0 else "random"
            ch_tw = "hybrid" if m_tw > 0 else "random"
            fold_rows.append(dict(budget_n=int(b), held_out_cluster=c, n_tasks=int(cl.loc[c, "n"]),
                                  train_mean_dHR_equal_cluster=round(m_eq, 5),
                                  train_mean_dHR_task_weighted=round(m_tw, 5),
                                  comparator_choice=ch_eq, comparator_choice_task_weighted=ch_tw))
            sel = g.cluster_id == c
            g.loc[sel, "prauc_comparator"] = g.loc[sel, f"prauc_{ch_eq}"]
            g.loc[sel, "prauc_comparator_tw"] = g.loc[sel, f"prauc_{ch_tw}"]
            g.loc[sel, "comparator_choice"] = ch_eq
        task_rows.append(g)
        for col, lab in [("prauc_comparator", "equal-cluster training mean"),
                         ("prauc_comparator_tw", "task-weighted training mean")]:
            for x, y, name in [("prauc_policy", col, "Policy - Comparator"),
                               (col, "prauc_random", "Comparator - Random"),
                               (col, "prauc_hybrid", "Comparator - Hybrid")]:
                m, p, k, nz, pmin = signflip(cluster_diffs(g, x, y).values)
                summ.append(dict(budget_n=int(b), comparator=lab, comparison=name,
                                 equal_cluster_delta=round(m, 5), n_clusters=k,
                                 n_nonzero=nz, signflip_p=round(p, 4),
                                 min_attainable_p=round(pmin, 4)))
    folds = pd.DataFrame(fold_rows)
    summary = pd.DataFrame(summ)
    folds.to_csv(o / "TableS18_comparator_folds.csv", index=False)
    summary.to_csv(o / "TableS18_comparator_summary.csv", index=False)
    pd.concat(task_rows).to_csv(o / "comparator_task_level.csv", index=False)
    print(folds.groupby("budget_n")[["comparator_choice", "comparator_choice_task_weighted"]]
          .agg(lambda s: dict(s.value_counts())).to_string())
    print(folds.groupby("budget_n")[["train_mean_dHR_equal_cluster", "train_mean_dHR_task_weighted"]]
          .agg(["min", "max"]).to_string())
    print(summary.to_string(index=False))


# ----------------------------------------------------------------------------
# A2 development overlap
# ----------------------------------------------------------------------------
def cmd_dev(a):
    pol = load_policy(a.policy)
    dev = read_dev_tasks(a.dev_tasks, a.inventory)
    dev["is_dev"] = True
    pol = pol.merge(dev, on=KEYS, how="left")
    pol["is_dev"] = pol["is_dev"].fillna(False).astype(bool)
    n_dev = pol.drop_duplicates(KEYS)["is_dev"].sum()
    if n_dev != len(dev):
        print(f"WARNING: only {n_dev} of {len(dev)} development tasks matched the policy file.")
    dev_clusters = set(pol.loc[pol.is_dev, "cluster_id"])
    pol["cluster_in_dev"] = pol["cluster_id"].isin(dev_clusters)

    inv = (pol.drop_duplicates(KEYS).groupby("cluster_id")
           .agg(n_tasks=("drug", "size"), n_dev_tasks=("is_dev", "sum")).reset_index())
    rows = []
    for b, g in pol.groupby("budget_n"):
        rows += compare_rows(g, "all tasks", int(b))
        rows += compare_rows(g[~g.is_dev], "excluding development tasks", int(b))
        rows += compare_rows(g[~g.cluster_in_dev], "clusters with no development task", int(b))
    t = pd.DataFrame(rows)
    o = out_dir(a.out)
    inv.to_csv(o / "dev_overlap_cluster_inventory.csv", index=False)
    t.to_csv(o / "TableS19b_dev_exclusion.csv", index=False)
    print(inv.to_string(index=False))
    print(t.to_string(index=False))


# ----------------------------------------------------------------------------
# A3a species-block sign flip
# ----------------------------------------------------------------------------
def cmd_blocks(a):
    pol = load_policy(a.policy)
    rows = []
    for b, g in pol.groupby("budget_n"):
        for x, y, name in [("prauc_policy", "prauc_random", "Policy - Random"),
                           ("prauc_policy", "prauc_hybrid", "Policy - Hybrid")]:
            for unit, lab in [("cluster", "12 clusters (as Table 5)"),
                              ("block_eq", "species blocks, clusters weighted equally within block"),
                              ("block_task", "species blocks, tasks pooled within block")]:
                if unit == "cluster":
                    d = cluster_diffs(g, x, y)
                elif unit == "block_eq":
                    d = block_diffs(g, x, y, a.block)
                else:
                    d = cluster_diffs(g, x, y, unit=a.block)
                m, p, k, nz, pmin = signflip(d.values)
                rows.append(dict(budget_n=int(b), comparison=name, unit=lab, n_units=k,
                                 n_nonzero=nz, delta=round(m, 5), signflip_p=round(p, 4),
                                 min_attainable_p=round(pmin, 4)))
    t = pd.DataFrame(rows)
    t.to_csv(out_dir(a.out) / "species_block_signflip.csv", index=False)
    print(t.to_string(index=False))


# ----------------------------------------------------------------------------
# A3b nested leave-one-species-block-out
# ----------------------------------------------------------------------------
def cmd_losbo(a):
    sys.path.insert(0, str(Path(__file__).parent))
    import leave_one_cluster_out_policy as L

    md, sd = Path(a.multitask_dir), Path(a.shift_dir)
    raw = pd.read_csv(md / "raw_results.csv")
    paired = pd.read_csv(md / "paired_hybrid_vs_random.csv")
    preds = pd.read_csv(sd / "transfer_predictors.csv")
    alphas = [float(x) for x in a.alphas.split(",")]
    allp, rows = [], []
    for b in [int(x) for x in a.budgets.split(",")]:
        table = L.build_transfer_table(raw, paired, preds, b)
        table["orig_cluster"] = table["cluster_id"]
        table["cluster_id"] = table[a.block].astype(str)       # outer + inner unit = block
        print(f"Budget {b}: {len(table)} tasks, {table['orig_cluster'].nunique()} clusters, "
              f"{table['cluster_id'].nunique()} blocks")
        pred, _, _ = L.evaluate_budget(table, b, alphas)
        pred = pred.rename(columns={"cluster_id": "block"})
        pred["cluster_id"] = cluster_of(pred)
        allp.append(pred)
        rows += compare_rows(pred, "LOSBO, clusters", b)
        rows += compare_rows(pred.assign(blk=pred["block"]), "LOSBO, species blocks", b, unit="blk")
    pred = pd.concat(allp)
    t = pd.DataFrame(rows)
    o = out_dir(a.out)
    pred.to_csv(o / "losbo_policy_predictions.csv", index=False)
    t.to_csv(o / "losbo_summary.csv", index=False)
    print(t.to_string(index=False))


# ----------------------------------------------------------------------------
# label-table helpers (overlap / duplicates / budget)
# ----------------------------------------------------------------------------
def eligible_tasks(a):
    inv = pd.read_csv(Path(a.multitask_dir) / "task_inventory.csv")
    inv = inv[inv["eligible"].astype(bool)].copy()
    inv["cluster_id"] = cluster_of(inv)
    print(f"Eligible tasks: {len(inv)} in {inv.cluster_id.nunique()} clusters")
    return inv


def usable_labels(a):
    lab = pd.read_parquet(a.labels)
    z = lab[lab.tested & lab.has_spectrum][["site", "species", "drug", "code", "label_RI"]].copy()
    if a.matrices:
        sys.path.insert(0, str(Path(__file__).parent))
        from nested_cv import load_spectra
        keep = []
        for site in z.site.unique():
            try:
                _, codes, _ = load_spectra(Path(a.matrices), site)
                keep.append(z[(z.site == site) & z.code.isin(set(codes))])
            except FileNotFoundError:
                pass
        z = pd.concat(keep)
    return z


def task_codes(z, site, species, drug):
    return set(z[(z.site == site) & (z.species == species) & (z.drug == drug)].code)


# ----------------------------------------------------------------------------
# A3c overlap matrix + patient-identifier availability
# ----------------------------------------------------------------------------
def cmd_overlap(a):
    inv = eligible_tasks(a)
    z = usable_labels(a)
    src, tgt = {}, {}
    for c, g in inv.groupby("cluster_id"):
        s, t, sp = g.iloc[0][["source", "target", "species"]]
        src[c] = (s, set().union(*[task_codes(z, s, sp, d) for d in g.drug]))
        tgt[c] = (t, set().union(*[task_codes(z, t, sp, d) for d in g.drug]))
    rows = []
    cl = sorted(src)
    for i, c1 in enumerate(cl):
        for c2 in cl[i + 1:]:
            def shared(x, y):
                return len(x[1] & y[1]) if x[0] == y[0] else 0
            r = dict(cluster_1=c1, cluster_2=c2,
                     shared_source=shared(src[c1], src[c2]),
                     shared_target=shared(tgt[c1], tgt[c2]),
                     target1_as_source2=shared(tgt[c1], src[c2]),
                     source1_as_target2=shared(src[c1], tgt[c2]))
            r["any_shared"] = any(v > 0 for k, v in r.items() if k.startswith(("shared", "target1", "source1")))
            rows.append(r)
    t = pd.DataFrame(rows)
    sizes = pd.DataFrame([dict(cluster_id=c, n_source_isolates=len(src[c][1]),
                               n_target_isolates=len(tgt[c][1])) for c in cl])
    o = out_dir(a.out)
    t.to_csv(o / "TableS20_cluster_overlap.csv", index=False)
    sizes.to_csv(o / "TableS20_cluster_sizes.csv", index=False)
    print(t[t.any_shared].to_string(index=False))
    linked = set(t.loc[t.any_shared, "cluster_1"]) | set(t.loc[t.any_shared, "cluster_2"])
    print(f"\nClusters linked to at least one other cluster: {len(linked)} of {len(cl)}")

    if a.root:
        sys.path.insert(0, str(Path(__file__).parent))
        from nested_cv import patient_map
        prow = []
        for site in sorted(z.site.unique()):
            pm = patient_map(Path(a.root).expanduser(), site)
            codes = z[z.site == site].code.unique()
            ok = [c for c in codes if isinstance(pm.get(c), str) and not pm.get(c).startswith("nan_")
                  and pm.get(c) not in ("nan", "", "NA")]
            prow.append(dict(site=site, n_isolates=len(codes), n_with_patient_id=len(ok),
                             fraction_with_patient_id=round(len(ok) / max(1, len(codes)), 3),
                             n_patients=len({pm[c] for c in ok})))
        pt = pd.DataFrame(prow)
        pt.to_csv(o / "patient_id_availability.csv", index=False)
        print("\nPatient-identifier availability (sets the wording of 'patient-aware'):")
        print(pt.to_string(index=False))


# ----------------------------------------------------------------------------
# duplicates: derived phenotypes
# ----------------------------------------------------------------------------
def cmd_duplicates(a):
    inv = eligible_tasks(a)
    z = usable_labels(a)
    dup_codes = z.duplicated(["site", "species", "drug", "code"]).sum()
    print(f"Duplicated (site, species, drug, code) rows: {dup_codes}")
    rows = []
    for c, g in inv.groupby("cluster_id"):
        s, t, sp = g.iloc[0][["source", "target", "species"]]
        drugs = sorted(g.drug)
        for role, site in [("source", s), ("target", t)]:
            w = (z[(z.site == site) & (z.species == sp) & z.drug.isin(drugs)]
                 .drop_duplicates(["code", "drug"]).pivot(index="code", columns="drug", values="label_RI"))
            for d1, d2 in itertools.combinations(drugs, 2):
                if d1 not in w or d2 not in w:
                    continue
                both = w[[d1, d2]].dropna()
                if both.empty:
                    continue
                agree = float((both[d1] == both[d2]).mean())
                rows.append(dict(cluster_id=c, role=role, site=site, drug_1=d1, drug_2=d2,
                                 n_common=len(both), n_only_1=int(w[d1].notna().sum() - len(both)),
                                 n_only_2=int(w[d2].notna().sum() - len(both)),
                                 agreement=round(agree, 4),
                                 discordant=int((both[d1] != both[d2]).sum())))
    t = pd.DataFrame(rows).sort_values("agreement", ascending=False)
    o = out_dir(a.out)
    t.to_csv(o / "label_identity_pairs.csv", index=False)
    hi = t[t.agreement >= a.min_agreement]
    print(f"\nDrug pairs with agreement >= {a.min_agreement}:")
    print(hi.to_string(index=False))
    # effective number of distinct target phenotypes per cluster (identical target labels collapsed)
    eff = []
    for c, g in inv.groupby("cluster_id"):
        drugs = set(g.drug)
        h = hi[(hi.cluster_id == c) & (hi.role == "target") & (hi.discordant == 0)
               & (hi.n_only_1 == 0) & (hi.n_only_2 == 0)]
        parent = {d: d for d in drugs}

        def find(x):
            while parent[x] != x:
                x = parent[x]
            return x
        for _, r in h.iterrows():
            parent[find(r.drug_1)] = find(r.drug_2)
        eff.append(dict(cluster_id=c, n_tasks=len(drugs),
                        n_distinct_target_phenotypes=len({find(d) for d in drugs})))
    e = pd.DataFrame(eff)
    e.to_csv(o / "distinct_phenotypes_per_cluster.csv", index=False)
    print(e.to_string(index=False))
    print(f"Total tasks {e.n_tasks.sum()} -> distinct target phenotypes {e.n_distinct_target_phenotypes.sum()}")


# ----------------------------------------------------------------------------
# A4/A6 budget accounting (+ optional error rates)
# ----------------------------------------------------------------------------
def cmd_budget(a):
    sys.path.insert(0, str(Path(__file__).parent))
    import multitask_active_validation as M
    from sklearn.preprocessing import StandardScaler

    if a.threads:
        M.CFG["n_jobs"] = a.threads
    md = Path(a.multitask_dir)
    raw = pd.read_csv(md / "raw_results.csv", usecols=KEYS + ["task_id"]).drop_duplicates()
    raw = raw.sort_values("task_id").reset_index(drop=True)
    if a.nshards > 1:
        raw = raw[raw.index % a.nshards == a.shard]
        print(f"Shard {a.shard}/{a.nshards}: {len(raw)} tasks, {M.CFG['n_jobs']} threads", flush=True)
    lab = pd.read_parquet(a.labels)
    mdir, root = Path(a.matrices), Path(a.root).expanduser()
    budgets_run = [int(x) for x in a.budgets_in_run.split(",")]
    B = a.budget
    thresholds = [float(x) for x in a.thresholds.split(",")]

    policy = None
    if a.errors:
        if not a.policy:
            raise SystemExit("--errors needs --policy (task-level policy choices)")
        policy = load_policy(a.policy)
        policy = policy[policy.budget_n == B].set_index(KEYS)["policy_choice"]

    sel_rows, err_rows = [], []
    for _, r in raw.sort_values("task_id").iterrows():
        tid = int(r.task_id)
        src = M.load_task(lab, mdir, root, r.species, r.drug, r.source)
        tgt = M.load_task(lab, mdir, root, r.species, r.drug, r.target)
        if src is None or tgt is None:
            print("skip", tuple(r[KEYS]))
            continue
        print(f"[{tid}] {r.species} / {r.drug} | {r.source}->{r.target}", flush=True)
        codes = tgt["df"]["code"].to_numpy()
        for rep in range(a.reps):
            adapt_ix, test_ix = M.choose_target_split(tgt["y"], tgt["g"], seed=rep, test_fraction=0.5)
            Xa, ya = tgt["X"][adapt_ix], tgt["y"][adapt_ix]
            Xte, yte = tgt["X"][test_ix], tgt["y"][test_ix]
            src_model, _ = M.fit_predict(src["X"], src["y"], Xte[:2], seed=rep)
            p_adapt = src_model.predict_proba(Xa)[:, 1]
            mu, P = M.fit_source_svd(src["X"], ncomp=100, seed=rep)
            Zs, Za = M.transform_svd(src["X"], mu, P), M.transform_svd(Xa, mu, P)
            sc = StandardScaler().fit(Zs)
            Za = sc.transform(Za).astype(np.float32)
            rng = np.random.default_rng(tid * 100000 + rep)
            ix_random = None
            for bb in budgets_run:              # replay the draws exactly as in the run
                if bb > len(Xa):
                    continue
                ixr = rng.choice(len(Xa), size=bb, replace=False)
                if bb == B:
                    ix_random = ixr
            ix_hybrid = M.select_hybrid(Za, p_adapt, budget=B, alpha=0.5)
            for strat, ix in [("random", ix_random), ("hybrid", ix_hybrid)]:
                sel_rows.append(dict(task_id=tid, **{k: r[k] for k in KEYS}, rep=rep, strategy=strat,
                                     codes="|".join(map(str, codes[adapt_ix][ix])),
                                     n_selected_positive=int(ya[ix].sum())))
            if a.errors:
                choice = policy.get(tuple(r[KEYS]), None)
                fits = {}
                for strat, ix in [("random", ix_random), ("hybrid", ix_hybrid)]:
                    _, p = M.fit_predict(np.vstack([src["X"], Xa[ix]]), np.r_[src["y"], ya[ix]],
                                         Xte, seed=rep)
                    fits[strat] = p
                if choice in fits:
                    fits["policy"] = fits[choice]
                npos, nneg = int(yte.sum()), int((1 - yte).sum())
                for strat, p in fits.items():
                    for th in thresholds:
                        pred = (p >= th).astype(int)
                        fn = int(((pred == 0) & (yte == 1)).sum())
                        fp = int(((pred == 1) & (yte == 0)).sum())
                        err_rows.append(dict(task_id=tid, **{k: r[k] for k in KEYS}, rep=rep,
                                             strategy=strat, threshold=th,
                                             very_major_error_rate=fn / max(1, npos),
                                             major_error_rate=fp / max(1, nneg),
                                             n_resistant=npos, n_susceptible=nneg))
    o = out_dir(a.out)
    sfx = f"_shard{a.shard}" if a.nshards > 1 else ""
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(o / f"selected_isolates_b{B}{sfx}.csv", index=False)
    err = pd.DataFrame(err_rows) if a.errors else None
    if err is not None:
        err.to_csv(o / f"error_rates_raw_b{B}{sfx}.csv", index=False)
    if a.nshards > 1:
        print(f"Shard {a.shard} finished. Run 'budget-merge' after all shards.")
        return
    summarise_budget(sel, err, B, o)


def summarise_budget(sel, err, B, o):
    sel = sel.copy()
    sel["cluster_id"] = cluster_of(sel)
    acc = []
    for (c, rep, strat), g in sel.groupby(["cluster_id", "rep", "strategy"]):
        sets = [set(str(s).split("|")) for s in g.codes]
        acc.append(dict(cluster_id=c, rep=rep, strategy=strat, n_agents=len(sets),
                        labels_requested=B * len(sets),
                        distinct_isolates=len(set().union(*sets))))
    acc = pd.DataFrame(acc)
    s21 = (acc.groupby(["cluster_id", "strategy", "n_agents", "labels_requested"])["distinct_isolates"]
           .agg(median="median", min="min", max="max").reset_index())
    s21.to_csv(o / f"TableS21_budget_accounting_b{B}.csv", index=False)
    print(s21.to_string(index=False))
    if err is not None and len(err):
        task = err.groupby(KEYS + ["strategy", "threshold"])[["very_major_error_rate", "major_error_rate"]].mean().reset_index()
        s22 = (task.groupby(["strategy", "threshold"])[["very_major_error_rate", "major_error_rate"]]
               .describe(percentiles=[.25, .5, .75]).round(3))
        s22.to_csv(o / f"TableS22_error_rates_b{B}.csv")
        print(s22.to_string())


def cmd_budget_merge(a):
    o = Path(a.out)
    B = a.budget
    sel_files = sorted(o.glob(f"selected_isolates_b{B}_shard*.csv"))
    err_files = sorted(o.glob(f"error_rates_raw_b{B}_shard*.csv"))
    if not sel_files:
        raise SystemExit(f"No shard files found in {o}")
    print(f"Merging {len(sel_files)} selection shards and {len(err_files)} error-rate shards")
    sel = pd.concat([pd.read_csv(f) for f in sel_files], ignore_index=True)
    err = pd.concat([pd.read_csv(f) for f in err_files], ignore_index=True) if err_files else None
    print(f"Tasks: {sel.task_id.nunique()} | reps: {sel.rep.nunique()}")
    sel.to_csv(o / f"selected_isolates_b{B}.csv", index=False)
    if err is not None:
        err.to_csv(o / f"error_rates_raw_b{B}.csv", index=False)
    summarise_budget(sel, err, B, o)


# ----------------------------------------------------------------------------
# dedup: sensitivity to derived (identical) target phenotypes
# ----------------------------------------------------------------------------
def cmd_dedup(a):
    from scipy.stats import spearmanr
    pairs = pd.read_csv(Path(a.out) / "label_identity_pairs.csv")
    ident = pairs[(pairs.role == "target") & (pairs.discordant == 0)
                  & (pairs.n_only_1 == 0) & (pairs.n_only_2 == 0)]
    drop = set()
    for c, g in ident.groupby("cluster_id"):
        parent = {}

        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                x = parent[x]
            return x
        for _, r in g.iterrows():
            r1, r2 = find(r.drug_1), find(r.drug_2)
            if r1 != r2:
                parent[max(r1, r2)] = min(r1, r2)
        for d in list(parent):
            if find(d) != d:
                drop.add((c, d))
    print(f"Tasks removed as duplicated target phenotypes: {len(drop)}")
    for c, d in sorted(drop):
        print("  ", c, "|", d)

    pol = load_policy(a.policy)
    keep = ~pd.Series(list(zip(pol.cluster_id, pol.drug))).isin(drop).to_numpy()
    rows = []
    for b, g in pol.groupby("budget_n"):
        rows += compare_rows(g, "all tasks", int(b))
        rows += compare_rows(g[keep[g.index]], "distinct target phenotypes only", int(b))
    t = pd.DataFrame(rows)
    print(t.to_string(index=False))

    preds = pd.read_csv(Path(a.shift_dir) / "transfer_predictors.csv")
    paired = pd.read_csv(Path(a.multitask_dir) / "paired_hybrid_vs_random.csv")
    y = (paired[paired.metric == "prauc"].groupby(KEYS)["delta_mean"].mean()
         .rename("mean_delta_prauc").reset_index())
    tab = preds.merge(y, on=KEYS)
    tab["cluster_id"] = cluster_of(tab)
    tab["dup"] = pd.Series(list(zip(tab.cluster_id, tab.drug))).isin(drop).to_numpy()
    descr = ["domain_auc_svd", "source_to_target_nn_distance", "covariance_distance_svd",
             "centroid_distance_svd", "source_internal_diversity"]
    crow = []
    for d in descr:
        for lab, z in [("all", tab), ("distinct phenotypes", tab[~tab.dup])]:
            rho, p = spearmanr(z[d], z["mean_delta_prauc"])
            crow.append(dict(descriptor=d, set=lab, n=len(z), spearman_rho=round(rho, 3),
                             transfer_level_p=round(p, 4)))
    ct = pd.DataFrame(crow)
    print(ct.to_string(index=False))
    o = out_dir(a.out)
    t.to_csv(o / "dedup_signflip.csv", index=False)
    ct.to_csv(o / "dedup_descriptor_correlations.csv", index=False)


# ----------------------------------------------------------------------------
# descset: descriptor-set comparison (R1-1) — fixed 5 vs geometry-8 vs all
# label-free candidates vs descriptor selection nested inside every fold
# ----------------------------------------------------------------------------
GEOMETRY8 = ["domain_auc_svd", "centroid_distance_svd", "covariance_distance_svd",
             "source_to_target_nn_distance", "target_to_source_nn_distance",
             "source_internal_diversity", "target_internal_diversity", "diversity_ratio"]
MODEL_OUTPUT11 = ["source_positive_rate_model", "target_positive_rate_model",
                  "source_prediction_entropy", "target_prediction_entropy", "entropy_shift",
                  "source_prediction_mean", "target_prediction_mean", "prediction_mean_shift",
                  "source_prediction_sd", "target_prediction_sd", "prediction_sd_shift"]
SIZE2 = ["source_n", "target_n"]
LABEL_REQUIRING3 = ["source_only_auroc_mean", "source_only_prauc_mean", "source_only_brier_mean"]
LABEL_FREE21 = GEOMETRY8 + MODEL_OUTPUT11 + SIZE2


def cmd_descset(a):
    from scipy.stats import spearmanr
    sys.path.insert(0, str(Path(__file__).parent))
    import leave_one_cluster_out_policy as L

    md, sd = Path(a.multitask_dir), Path(a.shift_dir)
    raw = pd.read_csv(md / "raw_results.csv")
    paired = pd.read_csv(md / "paired_hybrid_vs_random.csv")
    preds = pd.read_csv(sd / "transfer_predictors.csv")
    missing = [p for p in LABEL_FREE21 if p not in preds.columns]
    if missing:
        raise SystemExit(f"transfer_predictors.csv lacks: {missing}")
    alphas = [float(x) for x in a.alphas.split(",")]
    budgets = [int(x) for x in a.budgets.split(",")]
    primary = list(L.PREDICTORS)
    orig_fit = L.fit_weighted_ridge
    K = a.k

    def select_topk(train, pool):
        y = train["delta_prauc_hybrid_random"].to_numpy(float)
        sc = {}
        for p in pool:
            x = train[p].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() > 3 and np.unique(x[ok]).size > 1:
                r = spearmanr(x[ok], y[ok])[0]
                if np.isfinite(r):
                    sc[p] = abs(r)
        return sorted(sc, key=sc.get, reverse=True)[:K]

    def fit_selected(train, test, predictors, alpha):
        sel = select_topk(train, predictors)   # selection on THIS training set only
        return orig_fit(train, test, sel, alpha)

    sets = [("primary 5 (fixed)", primary, False),
            ("geometry 8 (fixed)", GEOMETRY8, False),
            ("all 21 label-free (fixed)", LABEL_FREE21, False),
            (f"top-{K} of 21, selected inside every fold", LABEL_FREE21, True)]
    allpred, rows, selrows, fitrows = [], [], [], []
    try:
        for name, plist, nested in sets:
            L.PREDICTORS = list(plist)
            L.fit_weighted_ridge = fit_selected if nested else orig_fit
            for b in budgets:
                table = L.build_transfer_table(raw, paired, preds, b)
                pred, alpha_tab, coef_tab = L.evaluate_budget(table, b, alphas)
                pred = pred.copy()
                pred["descriptor_set"] = name
                if "cluster_id" not in pred.columns:
                    pred["cluster_id"] = cluster_of(pred)
                allpred.append(pred)
                rows += [dict(descriptor_set=name, **r) for r in compare_rows(pred, name, b)]
                o, pr = pred["observed_delta_prauc_hybrid_random"], pred["predicted_delta_prauc_hybrid_random"]
                fitrows.append(dict(descriptor_set=name, budget_n=b,
                                    spearman_pred_obs=round(float(spearmanr(pr, o)[0]), 3),
                                    pearson_pred_obs=round(float(np.corrcoef(pr, o)[0, 1]), 3),
                                    n_hybrid_choices=int((pred.policy_choice == "hybrid").sum()),
                                    alphas=dict(pd.Series(alpha_tab["selected_alpha"]).value_counts())))
                if nested and isinstance(coef_tab, pd.DataFrame) and len(coef_tab):
                    freq = coef_tab.groupby("predictor")["outer_held_cluster"].nunique()
                    nf = coef_tab["outer_held_cluster"].nunique()
                    for p_, f in freq.items():
                        selrows.append(dict(budget_n=b, predictor=p_, folds_selected=int(f),
                                            n_folds=int(nf), in_primary5=p_ in primary))
    finally:
        L.PREDICTORS = primary
        L.fit_weighted_ridge = orig_fit

    allp = pd.concat(allpred, ignore_index=True)
    cmp_rows = []
    base = allp[allp.descriptor_set == sets[0][0]]
    for name, _, _ in sets[1:]:
        other = allp[allp.descriptor_set == name]
        for b in budgets:
            m = other[other.budget_n == b].merge(
                base[base.budget_n == b][KEYS + ["prauc_policy"]].rename(columns={"prauc_policy": "prauc_primary"}),
                on=KEYS)
            d = cluster_diffs(m, "prauc_policy", "prauc_primary")
            mm, p, k_, nz, pmin = signflip(d.values)
            cmp_rows.append(dict(descriptor_set=name, budget_n=b, vs="primary 5",
                                 equal_cluster_delta=round(mm, 5), n_nonzero=nz,
                                 signflip_p=round(p, 4), min_attainable_p=round(pmin, 4)))
    o = out_dir(a.out)
    allp.to_csv(o / "descset_policy_predictions.csv", index=False)
    t = pd.DataFrame(rows)
    t.to_csv(o / "descset_summary.csv", index=False)
    pd.DataFrame(fitrows).to_csv(o / "descset_fit.csv", index=False)
    pd.DataFrame(cmp_rows).to_csv(o / "descset_vs_primary.csv", index=False)
    print("\n=== POLICY EFFECT BY DESCRIPTOR SET (equal-cluster; sign-flip) ===")
    print(t[["descriptor_set", "budget_n", "comparison", "equal_unit_delta", "signflip_p",
             "units_positive", "units_negative"]].to_string(index=False))
    print("\n=== OUT-OF-CLUSTER PREDICTION AND CHOICES ===")
    print(pd.DataFrame(fitrows).to_string(index=False))
    print("\n=== EACH SET MINUS PRIMARY-5 POLICY ===")
    print(pd.DataFrame(cmp_rows).to_string(index=False))
    if selrows:
        s_ = pd.DataFrame(selrows).sort_values(["budget_n", "folds_selected"], ascending=[True, False])
        s_.to_csv(o / "descset_selection_frequency.csv", index=False)
        print(f"\n=== NESTED TOP-{K} SELECTION FREQUENCY (outer folds) ===")
        print(s_.to_string(index=False))


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, policy=False, mt=False, shift=False, labels=False):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--out", default="outputs/revision")
        if policy:
            p.add_argument("--policy", required=True)
        if mt:
            p.add_argument("--multitask-dir", required=True)
        if shift:
            p.add_argument("--shift-dir", required=True)
        if labels:
            p.add_argument("--labels", default="outputs/driams_long.parquet")
            p.add_argument("--matrices", default="matrices")
            p.add_argument("--root", default="~/data/DRIAMS")
        return p

    add("selfcheck", cmd_selfcheck, policy=True)
    add("comparator", cmd_comparator, policy=True)
    p = add("dev", cmd_dev, policy=True)
    p.add_argument("--dev-tasks", default=None)
    p.add_argument("--inventory", default=None)
    p = add("blocks", cmd_blocks, policy=True)
    p.add_argument("--block", default="species")
    p = add("losbo", cmd_losbo, mt=True, shift=True)
    p.add_argument("--block", default="species")
    p.add_argument("--budgets", default="20,30,50")
    p.add_argument("--alphas", default="0.01,0.1,1,10,100")
    add("overlap", cmd_overlap, mt=True, labels=True)
    p = add("duplicates", cmd_duplicates, mt=True, labels=True)
    p.add_argument("--min-agreement", type=float, default=0.99)
    p = add("budget", cmd_budget, mt=True, labels=True)
    p.add_argument("--budget", type=int, default=30)
    p.add_argument("--budgets-in-run", default="20,30,50",
                   help="budgets of the original run, in order (needed to replay random draws)")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--errors", action="store_true")
    p.add_argument("--policy", default=None)
    p.add_argument("--thresholds", default="0.1,0.5")
    p.add_argument("--threads", type=int, default=0, help="LightGBM threads per process (0 = keep 12)")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p = add("budget-merge", cmd_budget_merge)
    p.add_argument("--budget", type=int, default=30)
    p = add("dedup", cmd_dedup, policy=True, mt=True, shift=True)
    p = add("descset", cmd_descset, mt=True, shift=True)
    p.add_argument("--budgets", default="20,30,50")
    p.add_argument("--alphas", default="0.01,0.1,1,10,100")
    p.add_argument("--k", type=int, default=5)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
