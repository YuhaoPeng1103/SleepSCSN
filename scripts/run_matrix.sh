#!/usr/bin/env bash
# Full experiment matrix over a processed cohort, parameterised by environment.
#
# Usage:
#   DATA=data/processed/sleep-edf-78 ROOT=results/se78 \
#     bash scripts/run_matrix.sh "eegnet sleepcnn" "0 1 2"
#
# Each (model, fold, seed) produces one run directory under
# $ROOT/<model>/fold<f>_seed<s>/ with the training JSON, the calibration and
# test probabilities, and the conformal analyses. Re-running skips work whose
# outputs already exist, so the script is resumable.
set -euo pipefail

PY="${PY:-python}"
MODELS="${1:-eegnet}"
SEEDS="${2:-0}"
DATA="${DATA:-data/processed/sleep-edf-20}"
ROOT="${ROOT:-results/main}"
FOLDS="${FOLDS:-0 1 2 3 4}"
# 100 repeats of the block-level guarantee check costs several minutes per run
# and only tightens a standard error that is already in the third decimal.
REPEATS="${REPEATS:-30}"

if [[ ! -d "$DATA" ]]; then
  echo "missing processed data: $DATA"
  exit 1
fi

# Two concurrent invocations on one ROOT race on the same output paths: both
# see no calibration file while the other is still training, both train, and a
# run directory can end up holding one process's checkpoint with the other's
# analyses. Refuse to start rather than produce a silently inconsistent matrix.
LOCK="$ROOT/.run_matrix.lock"
mkdir -p "$ROOT"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another run_matrix.sh is already writing to $ROOT" >&2
  echo "if that process is gone, remove the stale lock: rm -rf '$LOCK'" >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

for model in $MODELS; do
  for seed in $SEEDS; do
    for fold in $FOLDS; do
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

      if [[ -s "$RUN/subject_aci.json" ]]; then
        echo "== skip subject aci =="
      else
        "$PY" scripts/run_subject_aci.py \
          --cal "$CAL" --test "$TEST" \
          --out "$RUN/subject_aci.json" \
          --alpha 0.1 \
          --gamma0 0.005 0.01 0.02 0.05 0.1
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
          --repeats "$REPEATS"
      fi

      # Input-level shift study: a property of the model and the split, not of
      # the fold, so one representative run is enough.
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
echo "== matrix done: $ROOT =="
