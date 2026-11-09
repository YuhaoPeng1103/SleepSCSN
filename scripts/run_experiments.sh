#!/usr/bin/env bash
# Full experiment matrix: 5 subject-wise folds x seeds x backbones.
#
# Usage:
#   bash scripts/run_experiments.sh              # EEGNet, 5 folds, seed 0
#   bash scripts/run_experiments.sh "eegnet sleepcnn" "0 1 2"
#
# Each (model, fold, seed) produces one run directory under
# results/main/<model>/fold<f>_seed<s>/ with the training JSON, calibration and
# test probabilities, and the three conformal analyses. Re-running skips
# directories whose final outputs already exist, so the script is resumable.
set -euo pipefail

PY="${PY:-python}"
MODELS="${1:-eegnet}"
SEEDS="${2:-0}"
DATA="data/processed/sleep-edf-20"
ROOT="results/main"

if [[ ! -d "$DATA" ]]; then
  echo "missing processed data: $DATA"
  exit 1
fi

for model in $MODELS; do
  for seed in $SEEDS; do
    for fold in 0 1 2 3 4; do
      RUN="$ROOT/$model/fold${fold}_seed${seed}"
      mkdir -p "$RUN"
      STEM="${model}_fold${fold}_seed${seed}"
      CAL="$RUN/${STEM}_cal.npz"
      TEST="$RUN/${STEM}_test.npz"

      if [[ -s "$CAL" && -s "$TEST" ]]; then
        echo "== skip training $model fold=$fold seed=$seed =="
      else
        echo "== train $model fold=$fold seed=$seed =="
        "$PY" scripts/train_eegnet.py \
          --data-dir "$DATA" \
          --out-dir "$RUN" \
          --model "$model" \
          --fold "$fold" \
          --seed "$seed" \
          --epochs 40 \
          --batch-size 128 \
          --no-class-weights
      fi

      if [[ -s "$RUN/sacp.json" && -s "$RUN/sacp_smooth.json" && -s "$RUN/sacp_raps.json" ]]; then
        echo "== skip sacp =="
      else
        "$PY" scripts/run_sacp.py \
          --cal "$CAL" --test "$TEST" \
          --out "$RUN/sacp.json" \
          --alpha 0.1 \
          --budget 0 10 20 40 80 160 320 \
          --k 3 5 10
        "$PY" scripts/run_sacp.py \
          --cal "$CAL" --test "$TEST" \
          --out "$RUN/sacp_smooth.json" \
          --alpha 0.1 \
          --budget 0 10 20 40 80 160 320 \
          --k 3 5 10 \
          --smooth
        "$PY" scripts/run_sacp.py \
          --cal "$CAL" --test "$TEST" \
          --out "$RUN/sacp_raps.json" \
          --alpha 0.1 \
          --budget 0 10 20 40 80 160 320 \
          --k 3 5 10 \
          --score raps
      fi

      if [[ -s "$RUN/temporal_cp.json" ]]; then
        echo "== skip temporal =="
      else
        "$PY" scripts/run_temporal_cp.py \
          --cal "$CAL" --test "$TEST" \
          --out "$RUN/temporal_cp.json" \
          --alpha 0.1 \
          --block-length 10 20 30 \
          --beta 0 0.5 1.0 2.0
      fi

      if [[ -s "$RUN/cp_diagnostic.json" ]]; then
        echo "== skip diagnostic =="
      else
        "$PY" scripts/run_cp_diagnostic.py \
          --cal "$CAL" --test "$TEST" \
          --out "$RUN/cp_diagnostic.json" \
          --alpha 0.1 \
          --n-repeats 50
      fi

      if [[ -s "$RUN/guarantee_check.json" ]]; then
        echo "== skip guarantee check =="
      else
        "$PY" scripts/run_guarantee_check.py \
          --cal "$CAL" --test "$TEST" \
          --out "$RUN/guarantee_check.json" \
          --alpha 0.1 \
          --block-length 5 10 20 30 \
          --beta 0 0.5 1.0 \
          --repeats 100
      fi

      # Input-level shift study: one run is enough, it is a property of the
      # model rather than of the split, and it re-predicts the whole test set
      # once per corruption level.
      if [[ "$fold" == "0" && "$seed" == "0" ]]; then
        if [[ -s "$RUN/shift.json" ]]; then
          echo "== skip shift =="
        else
          "$PY" scripts/run_shift.py \
            --data-dir "$DATA" \
            --run-json "$RUN/${STEM}.json" \
            --checkpoint "$RUN/${STEM}.pt" \
            --cal "$CAL" \
            --out "$RUN/shift.json"
        fi
      fi
    done
  done
done

"$PY" scripts/aggregate.py --root "$ROOT" --out "$ROOT/summary"

# Figures: one representative night for the hypnogram, then the summary plots.
EPOCH_NPZ="$ROOT/figures/epoch_sets.npz"
if [[ ! -s "$EPOCH_NPZ" ]]; then
  REF_TEST="$ROOT/eegnet/fold0_seed0/eegnet_fold0_seed0_test.npz"
  REF_CAL="$ROOT/eegnet/fold0_seed0/eegnet_fold0_seed0_cal.npz"
  REF_JSON="$ROOT/eegnet/fold0_seed0/sacp.json"
  if [[ -s "$REF_TEST" && -s "$REF_JSON" ]]; then
    SUBJ=$("$PY" -c "import numpy as np; print(np.load(r'$REF_TEST')['subjects'].astype(str)[0])")
    "$PY" scripts/export_epoch_sets.py \
      --cal "$REF_CAL" --test "$REF_TEST" --sacp-json "$REF_JSON" \
      --subject "$SUBJ" --budget 40 --start 0 --out "$EPOCH_NPZ"
  fi
fi

"$PY" scripts/make_figures.py --root "$ROOT" --out paper/figures --epoch-sets "$EPOCH_NPZ"
