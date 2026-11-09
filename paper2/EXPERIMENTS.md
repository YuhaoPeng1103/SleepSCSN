# Paper 2 — protocol and provenance

Every number in `tex/` comes from a JSON file under `results/`. This file
records which script wrote which file, so that a table can be traced back to a
run without reading the manuscript.

## Cohorts and runs

| Cohort | Backbones | Seeds | Runs per arm |
|---|---|---|---|
| Sleep-EDF-20 (39 recordings, 20 subjects) | EEGNet, SleepCNN | 3 (EEGNet), 1 (SleepCNN) | 15 (EEGNet), 5 (SleepCNN) |
| HMC (151 recordings) | EEGNet, SleepCNN | 2 | 10 |

Split inside a fold: 20% of subjects held out for test, a disjoint 20% for
conformal calibration, 15% of the remaining subjects for early stopping, the
rest for training. Adam, lr 1e-3, weight decay 1e-4, batch 128, at most 40
epochs, patience 8, no class weights.

## Arms

| Arm | Inference-time operation | Applies to |
|---|---|---|
| `bn` | cohort statistics, no correction | BN and GN checkpoints |
| `marginal` | subtract the recording's mean shift at the representation | BN and GN |
| `cond_full` | SCSN with the soft posterior, λ=1, λ_s=10 | BN and GN |
| `cond_full_lam*` | interpolated family, λ ∈ {0.25, 0.5, 0.75} | BN |
| `cond_r1/2/4/8` | SCSN restricted to the top-r subject directions | BN |
| `cond_r4_orth` | shift projected off the classifier's row space | BN |
| `cond_hard` | SCSN with one-hot pseudo-labels | BN |
| `cond_mismatch` | another recording's shifts | BN |
| `adabn_layerwise` | re-estimate every BN layer on the recording | BN only |
| `tent_sgd/1/5` | Tent: entropy minimisation over BN affine parameters | BN only |

AdaBN and Tent are undefined on group-normalised checkpoints: there is no
batch statistic and no affine normalisation parameter to re-estimate.

## Files

| Quantity | Producer | Output |
|---|---|---|
| SCSN matrix (BN) | `scripts/pilot_cssn4_fast.py` via `scripts/run_scsn_final.sh` | `results/scsn_final/<tag>_<model>_fold*_seed*.json` |
| SCSN matrix (GN) | `scripts/pilot_cssn4_fast.py` via `scripts/run_gn_parallel.sh` | `results/scsn_gn_manual/<tag>_<model>_fold*_seed*.json` |
| layerwise AdaBN, AdaBN+SCSN | `scripts/adabn_final.py` via `scripts/run_adabn_final.sh` | `results/adabn_final/<tag>_<model>_fold*_seed*.json` |
| Tent baseline | `scripts/tent_baseline.py` via `scripts/run_tent.sh` | `results/tent/*_fold*_seed*.json` |
| cross-cohort transfer | `scripts/cross_cohort_scsn.py` | `results/scsn_cross/eegnet.json` |
| shrinkage sensitivity ($\lambda_s=1,100$) | `scripts/pilot_cssn4_fast.py --shrink` | `results/scsn_lams/*_lams*.json` |
| shrinkage summary | `scripts/lam_s_summary.py` | `results/lam_s_summary.json` |
| merged runs and paired statistics | `scripts/final_stats.py` | `results/merged/<tag>_<model>.json`, `results/final_stats.json` |
| LaTeX tables | `scripts/make_tables.py` | `tex/tables/*.tex` |
| figures | `scripts/make_figures.py` | `figures/*.pdf`, `*.png` |

Run the last three with the project environment (`D:/Anaconda/envs/sleepcp`)
for the figure step.

## Definitions

- **Probe.** Multinomial logistic regression from the penultimate
  representation to the patient index, three-fold cross-validated over epochs,
  with a fixed capped subsample (5,000 epochs; 3,000 for the 256-dimensional
  SleepCNN features). Chance is `1/|test subjects|` — 0.25 on Sleep-EDF and
  ≈0.032 on HMC.
- **Within-stage probe.** The same probe refitted on the epochs of one stage
  and averaged over stages with at least 50 epochs. It uses the true stage to
  form the groups; it is a diagnostic, not part of any arm.
- **Coverage.** Fraction of test epochs whose conformal set contains the true
  stage at α = 0.1, split conformal with the APS score.
- **Worst-patient coverage.** Minimum of per-patient coverage within a run.
- **Coverage SD.** Standard deviation of per-patient coverage (ddof = 1).
- **Review workload.** Fraction of epochs whose set holds more than one stage.
- **Paired statistics.** Mean of per-run differences, 95% CI from the t
  distribution over runs, Cohen's d_z, two-sided Wilcoxon signed-rank, sign
  count over folds.

## Reproducing the headline comparison

```bash
python scripts/final_stats.py
python scripts/make_tables.py
python scripts/make_figures.py
cd paper2/tex && latexmk -pdf main.tex
```

`results/final_stats.json` holds the per-dataset arm means (`datasets`), the
paired contrasts (`pairs`, keyed `<tag>:<a>_vs_<b>`), and the per-run values
they were computed from is in `results/merged/`.
