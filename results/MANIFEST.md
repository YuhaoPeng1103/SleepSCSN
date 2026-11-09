# Results manifest

Every number in the manuscript is produced from these files. Feature caches
(`*.npz`) are omitted; they are rebuilt on demand by the scripts and are not
inputs to any reported number.

## Analysis outputs

| Folder / file | Contents | Producer |
|---|---|---|
| `scsn_final/` | SCSN matrix on batch-normalised checkpoints; one JSON per (cohort, model, fold, seed), holding every arm of the run | `scripts/pilot_cssn4_fast.py` via `scripts/run_scsn_final.sh` |
| `scsn_gn_manual/` | The same on group-normalised checkpoints | `scripts/pilot_cssn4_fast.py` via `scripts/run_gn_parallel.sh` |
| `adabn_final/` | Layerwise AdaBN and SCSN stacked on AdaBN | `scripts/adabn_final.py` via `scripts/run_adabn_final.sh` |
| `tent/` | Tent baseline (entropy minimisation over BN affine parameters) | `scripts/tent_baseline.py` via `scripts/run_tent.sh` |
| `scsn_lams/` | SCSN rerun with λ_s ∈ {1, 100}, same checkpoints and caches | `scripts/pilot_cssn4_fast.py --shrink` |
| `scsn_cross/` | Cross-cohort transfer from Sleep-EDF to HMC | `scripts/cross_cohort_scsn.py` |
| `merged/` | Per-run tables merged across folders, keyed by dataset and arm | `scripts/final_stats.py` |
| `final_stats.json` | Arm means and every paired contrast (mean, 95% CI, d_z, Wilcoxon p, signs) | `scripts/final_stats.py` |
| `lam_s_summary.json` | Shrinkage sweep summary | `scripts/lam_s_summary.py` |

## Trained checkpoints and run outputs

| Folder | Contents |
|---|---|
| `main/` | Sleep-EDF batch-normalised checkpoints, training JSON, per-run outputs |
| `hmc/batch/` | HMC batch-normalised counterpart |
| `probe_norm/group_all/` | Sleep-EDF group-normalised counterpart |
| `hmc/group/` | HMC group-normalised counterpart |

Each run directory is `results/<root>/<model>/fold<f>_seed<s>/` and contains
`<model>_fold<f>_seed<s>.pt` (the checkpoint), the training JSON, the
calibration and test arrays, and the conformal analyses. The arms in
`scsn_final`, `scsn_gn_manual` and `adabn_final` are all computed from these
checkpoints, so the whole table can be rebuilt without retraining.

## Mapping to the manuscript

`paper2/EXPERIMENTS.md` gives the same mapping at the level of individual
tables and figures, and `paper2/SUBMISSION.md` records the final package.
