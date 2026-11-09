# SleepSCSN

Code, fitted checkpoints, and released result summaries for

> **Stage-conditional shift normalization for subject-robust EEG sleep
> staging**

The paper separates patient-dependent variation in a frozen sleep stager into
a recording-wide offset and a stage-conditional residual. A within-stage
subject probe is used to measure the residual that ordinary global
normalization leaves behind. Stage-conditional shift normalization (SCSN)
then estimates one first-moment shift per sleep stage from the model's soft
posterior and removes the posterior-weighted shift in closed form. SCSN
requires no target labels, optimizer, or weight updates and contains the
global mean correction as a special case.

## Headline results

- Sleep-EDF EEGNet: the within-stage subject probe falls from 0.738 to 0.465
  under SCSN.
- HMC EEGNet: the within-stage probe falls from 0.527 to 0.209 and accuracy
  rises from 0.672 to 0.690.
- On group-normalized checkpoints, where the conventional AdaBN replacement is
  undefined, SCSN improves accuracy significantly in three of four settings
  and improves worst-patient conformal coverage over the uncorrected model in
  all four.
- Cross-cohort transfer shows the precondition: with a weak target posterior,
  SCSN approaches the global correction rather than the oracle-label result.

The complete paired statistics are in `results/final_stats.json`; the
protocol and provenance map is in `paper2/EXPERIMENTS.md`.

## Repository layout

```text
src/sleepcp/        models, data loading, conformal prediction, metrics
scripts/            preprocessing, training, adaptation, statistics, figures
results/            per-run JSON, checkpoints, merged results, summary JSON
paper2/             manuscript, figures, highlights, protocol, submission data
tests/              unit tests for conformal and temporal procedures
data/README.md      public-cohort download and preprocessing instructions
```

Raw and processed recordings are not redistributed. Both cohorts are public on
PhysioNet:

- Sleep-EDF Expanded: <https://physionet.org/content/sleep-edfx/1.0.0/>
- HMC Sleep Staging Database v1.1: <https://physionet.org/content/hmc-sleep-staging/1.1.0/>

## Environment

Python 3.11 was used for the reported runs. Exact package versions are in
`requirements-lock.txt`.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
python scripts/check_environment.py
```

## Reproduce the paper

1. Download and preprocess the two public cohorts as described in
   `data/README.md`.
2. Train each batch-normalized backbone:

```bash
DATA=data/processed/sleep-edf-20 ROOT=results/main \
    bash scripts/run_matrix.sh "eegnet sleepcnn" "0 1 2"

bash scripts/run_hmc_grid.sh batch eegnet 2
bash scripts/run_hmc_grid.sh batch sleepcnn 2
```

3. Evaluate the inference-time arms on the fitted checkpoints:

```bash
bash scripts/run_scsn_final.sh
bash scripts/run_gn_parallel.sh
bash scripts/run_adabn_final.sh
bash scripts/run_tent.sh
```

4. Recompute paired statistics, tables, and vector figures:

```bash
python scripts/final_stats.py
python scripts/lam_s_summary.py
python scripts/make_tables.py
python scripts/make_figures.py
```

5. Build the manuscript:

```bash
cd paper2/tex
latexmk -pdf main.tex
```

The released checkpoints under `results/main`, `results/hmc`, and
`results/probe_norm` allow the inference-time arms to be recomputed without
retraining. Feature caches (`*.npz`) are omitted because they can be rebuilt
from the public recordings.

## Tests

```bash
pytest -q
```

The suite covers split-conformal quantiles, nonconformity scores, forward
backward inference, and the dynamic-programming sequence routines.

## Citation

Citation metadata is provided in `CITATION.cff`.

## License

The source code and released result files are distributed under the MIT
License. The Sleep-EDF and HMC databases retain their original PhysioNet
licenses and are not redistributed here.
