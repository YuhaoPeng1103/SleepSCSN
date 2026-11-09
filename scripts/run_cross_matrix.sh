#!/usr/bin/env bash
# Conformal analyses on the cross-cohort runs produced by train_cross_cohort.py.
#
# Each cross run already contains its calibration/test probabilities, so this
# script only reruns the analyses the paper reports on them. It is resumable:
# anything whose output exists is skipped.
#
# Usage:
#   bash scripts/run_cross_matrix.sh "sc2st st2sc" "eegnet sleepcnn"
set -euo pipefail

PY="${PY:-python}"
DIRECTIONS="${1:-sc2st st2sc}"
MODELS="${2:-eegnet sleepcnn}"
SEED="${SEED:-0}"
ROOT="${ROOT:-results/cross}"

for direction in $DIRECTIONS; do
  for model in $MODELS; do
    RUN="$ROOT/$direction/$model/seed$SEED"
    STEM="${model}_cross_seed${SEED}"
    CAL="$RUN/${STEM}_cal.npz"
    TEST="$RUN/${STEM}_test.npz"
    if [[ ! -s "$CAL" || ! -s "$TEST" ]]; then
      echo "== skip $direction/$model: no probabilities =="
      continue
    fi

    if [[ -s "$RUN/temporal_cp.json" ]]; then
      echo "== skip temporal $direction/$model =="
    else
      "$PY" scripts/run_temporal_cp.py \
        --cal "$CAL" --test "$TEST" \
        --out "$RUN/temporal_cp.json" \
        --alpha 0.1 --block-length 10 20 30 --beta 0 0.5 1.0 2.0
    fi

    if [[ -s "$RUN/subject_aci.json" ]]; then
      echo "== skip subject aci $direction/$model =="
    else
      "$PY" scripts/run_subject_aci.py \
        --cal "$CAL" --test "$TEST" \
        --out "$RUN/subject_aci.json" \
        --alpha 0.1 --gamma0 0.005 0.01 0.02 0.05 0.1
    fi

    if [[ -s "$RUN/sacp.json" ]]; then
      echo "== skip sacp $direction/$model =="
    else
      "$PY" scripts/run_sacp.py \
        --cal "$CAL" --test "$TEST" \
        --out "$RUN/sacp.json" \
        --alpha 0.1 --budget 0 10 20 40 80 160 320 --k 3 5 10
    fi
  done
done

for direction in $DIRECTIONS; do
  if compgen -G "$ROOT/$direction/*/seed*" > /dev/null; then
    "$PY" scripts/aggregate.py --root "$ROOT/$direction" --out "$ROOT/$direction/summary"
  fi
done
echo "== cross matrix done =="
