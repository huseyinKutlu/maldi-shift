# maldi-shift

Analysis code for the study *Unlabeled Spectral Shift Informs Target-Label Acquisition Strategy Selection for Cross-Site MALDI-TOF Antimicrobial Resistance Prediction*.

The repository covers three connected questions. Does a resistance-prediction model trained at one laboratory transfer to another? If not, can unsupervised harmonisation repair it? And when a small number of local susceptibility results can be obtained, how should the isolates be chosen?

---

## Data

The study uses [DRIAMS](https://doi.org/10.5061/dryad.bzkh1899q), a public collection of MALDI-TOF mass spectra linked to antimicrobial susceptibility results from four Swiss laboratories.

Spectra are represented as 6,000 intensity bins spanning approximately 2,000.11 to 19,999.69 Da (about 3 Da per bin).

Raw DRIAMS archives, the derived `matrices/` directory (~8 GB of `.npy` files) and `outputs/driams_long.parquet` (~350 MB) are excluded from version control. Run the two preparation scripts below to regenerate them.

---

## Reproducing the manuscript

### Step 1 — Prepare the data

```bash
python inventory.py --root ~/data/DRIAMS --out outputs --min-n 100
python build_matrix.py --site DRIAMS-A --year 2018        # repeat per site/year
```

`inventory.py` writes `outputs/driams_long.parquet`, the label table every downstream script reads. `build_matrix.py` converts binned spectra into `.npy` matrices; the `--year` flag exists because writing several gigabytes in one pass can crash WSL2.

### Step 2 — Main analyses

| Script | Produces | Manuscript element |
|---|---|---|
| `selective_adaptation_policy_v2.py` | Policy, Random, Hybrid and oracle PR-AUC at 20/30/50 labels; direction accuracy; regret | **Table 3**, Figure 5A — the main transfer-level policy analysis |
| `cluster_aware_policy_bootstrap.py` | Cluster-aware confidence intervals for Policy−Random and Policy−Hybrid | **Table 4**, Figure 5B |
| `policy_permutation_test.py` | 10,000-permutation transfer-level null | **Table 4** p values (0.0125, 0.0269, 0.0219) |
| `shift_predictors.py` + `shift_predictors_validate.py` | Spearman correlations, ridge LOGO, permutation | **Table 2**, Figure 3 |
| `final_results.py` | Consolidates the above into manuscript-ready tables | Cross-check for all reported numbers |

### Step 3 — Robustness and stress tests

| Script | Produces | Manuscript element |
|---|---|---|
| `leave_one_species_out_policy.py` | 15 species-exclusion scenarios | Section 2.6 — all 15 positive |
| `policy_cluster_permutation.py` | Cluster-level null distributions | Section 2.6 — p = 0.052–0.106 |
| `leave_one_cluster_out_policy.py` | Nested LOCO with inner alpha selection | **Section 2.7**, Methods 4.8 |
| `loco_policy_audit.py` | Per-cluster descriptor-space audit | **Supplementary Table S11** |

The LOCO analysis is the stress test where the evidence weakens; its outputs are reported in full rather than summarised, so that the aggregate figures in the main text can be traced to their components.

### Step 4 — Acquisition strategies

| Script | Role |
|---|---|
| `active_target_selection.py` | Implements Random, Hybrid, uncertainty-only, Kennard–Stone and domain-distance acquisition |
| `active_selection_validation.py` | Single-task validation with matched-budget comparisons and label-efficiency curves |
| `multitask_active_validation.py` | 50-repetition validation across all 20 tasks — this is what overturned the single-task Hybrid result |
| `target_label_budget.py` | Label-budget learning curves |

---

## The adaptation benchmark

These scripts produced the negative results reported in Section 2.2 and the Supplementary Materials. None consistently restored cross-site transportability.

| Family | Scripts |
|---|---|
| Spectral selection and preprocessing | `dwt_diag.py`, `prep_chain.py`, `augment_dr.py` |
| Nuisance-subspace projection | `nap_transfer.py`, `svd_nap.py`, `glsw.py`, `epo_v2.py`, `epo_validate.py` |
| Instrument standardisation | `standardize.py` (pseudo-standard DS/PDS) |
| Distribution alignment | `dann_adapt.py` (DANN, CORAL, MMD, BN-adapt), `sinkhorn_ot.py` |
| Representation adaptation | `tamn.py` (DSBN + residual adapter), `nn_curve.py` |
| Engineering transfer | `engineering_transfer.py`, `engineering_validation.py`, `dipls_validation_v2.py`, `local_transport.py` |

`shift_mechanism.py` and `shift_mechanism_v2.py` examine why these failed — including the finding that reducing source–target separability was neither necessary nor sufficient for better transfer.

---

## Diagnostics

| Script | Question |
|---|---|
| `site_check.py` | Can a classifier identify the site when species is held fixed? (0.997–0.999) |
| `site_decomp.py` | Does that hold when year, resistance status or workstation are also fixed? |
| `xai_compare.py` | Does the model attend to the PSM-mec region near 2,408 Da? |
| `nested_cv.py` | Shared utilities — spectrum loading, patient grouping, nested CV |
| `recalibrate.py` | Platt versus isotonic recalibration at varying target-label counts |
| `dca_utility.py` | Decision-curve analysis and empiric-therapy simulation |
| `summarize.py` | Bootstrap confidence intervals for the learning curves |

---

## A note on prototypes

`selective_adaptation_policy.py` was the first version of the policy. It included source-only target AUROC among its inputs, which is estimated from target labels and therefore introduces indirect leakage into a decision meant to precede labelling. It was discarded and is described as such in the manuscript. The file itself was lost during a workspace transfer and is not in this repository; `selective_adaptation_policy_v2.py` is the version behind every reported result.

Some scripts exist in more than one version (`shift_mechanism.py` / `_v2.py`, `epo_v2.py`, `sinkhorn_ot.py`). Where versions differ, the manuscript reports the later one, and the earlier file is retained so that the correction can be inspected.

---

## Environment

```
Python 3.11
numpy, pandas, scipy
scikit-learn, lightgbm
torch (CNN and adapter experiments)
POT (optimal transport)
PyWavelets (wavelet subband analysis)
```

Random seeds are fixed in each script; per-run parameters are written to `run_config.json` inside the corresponding output directory.

---

## Output layout

Each script writes to its own subdirectory under `outputs/`. Files ending in `_raw.csv` hold per-repetition results; `_summary.csv` files hold aggregates. `outputs/final_results/` contains the consolidated tables used in the manuscript.

---

## Citation

If you use this code, please cite the manuscript (details to follow) and the DRIAMS dataset:

> Weis C, Cuénod A, Rieck B, et al. Direct antimicrobial resistance prediction from clinical MALDI-TOF mass spectra using machine learning. *Nat Med* 2022;28:164–174. doi:10.1038/s41591-021-01619-9

---

## License

MIT
