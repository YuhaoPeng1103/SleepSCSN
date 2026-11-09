# Paper 2 project package

This archive contains the complete project for the manuscript

> **Stage-conditional shift normalization for
> subject-robust EEG sleep staging**

submitted to *Neurocomputing*: the manuscript source, the code that produces
every number and figure, the per-run results, and the trained checkpoints.
It is the same tree that was used to build the submitted PDF.

## Contents

| Path | Contents |
|---|---|
| `paper2/` | manuscript (`tex/`, `figures/`, `EXPERIMENTS.md`, `SUBMISSION.md`) |
| `scripts/` | data preparation, training, every analysis, statistics, tables and figures |
| `src/sleepcp/` | library: data loading, models, training, conformal prediction, metrics |
| `tests/` | unit tests (`pytest -q` from the repository root) |
| `results/` | per-run JSON, paired statistics, trained checkpoints; see `results/MANIFEST.md` |
| `data/README.md` | how to obtain and preprocess the two public cohorts |
| `requirements.txt`, `requirements-lock.txt` | environment |
| `CITATION.cff` | citation metadata |

## Not included

- **Raw and processed recordings** (about 42 GB). Both cohorts are public on
  PhysioNet; `data/README.md` gives the download and preprocessing commands.
- **Feature caches** (`*.npz`, about 3.2 GB). They are rebuilt on demand by
  `scripts/pilot_cssn4_fast.py`; no reported number depends on them.
- **Logs and exploratory runs** from earlier iterations of the project. The
  final runs are the ones under `results/` listed in `results/MANIFEST.md`.

## Environment

Python ≥ 3.10 with the packages in `requirements.txt`; the exact versions used
for the reported runs are in `requirements-lock.txt` (CUDA 12.8 wheels for
torch; a CPU-only machine can install plain `torch`/`torchvision` instead).
The experiments were run in a conda environment named `sleepcp`.

Install and test:

```bash
pip install -r requirements.txt
pip install -e .
pytest -q
```

## Reproducing

The short version is in `README.md` under "Reproducing paper 2", and
`paper2/EXPERIMENTS.md` maps every table and figure to the script that wrote
it. In order: preprocess the cohorts, train the backbones, run the
inference-time arms, aggregate with `scripts/final_stats.py`, regenerate tables
and figures, then build the manuscript with `latexmk -pdf main.tex` in
`paper2/tex`.

The checkpoints in `results/main`, `results/hmc`, `results/probe_norm` and
`results/hmc/group` let every arm be recomputed without retraining.

## Licence and citation

Code is released for review and reuse with the manuscript. Sleep-EDF is
distributed under ODC-BY 1.0 and the HMC database under CC-BY 4.0 by PhysioNet;
neither is redistributed here. Citation metadata is in `CITATION.cff`.
