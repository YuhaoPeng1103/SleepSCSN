#!/usr/bin/env bash
# Queue the experiments that follow the 20-subject matrix, in the order the
# paper needs them:
#
#   1. sleepcnn cross-cohort runs in both directions (the cross-cohort table
#      currently has eegnet only, and the paper claims conclusions are not
#      tied to the backbone);
#   2. the 78-subject cohort matrix at seed 0 (statistical power for the
#      worst-patient statistics);
#   3. a re-aggregation of every results root.
#
# The script waits for the 20-subject matrix to print its completion line, so
# it can be launched before that job finishes without contending for the GPU.
set -euo pipefail

PY="${PY:-python}"
MAIN_LOG="${MAIN_LOG:-logs/matrix20_clean.log}"

echo "== waiting for the 20-subject matrix =="
for _ in $(seq 1 720); do
  if grep -q "== matrix done: results/main ==" "$MAIN_LOG" 2>/dev/null; then
    break
  fi
  sleep 30
done
if ! grep -q "== matrix done: results/main ==" "$MAIN_LOG" 2>/dev/null; then
  echo "warning: did not see the completion line; continuing anyway"
fi

for pair in "sc2st sleep-edf-20 sleep-telemetry" "st2sc sleep-telemetry sleep-edf-20"; do
  set -- $pair
  direction="$1"; train="$2"; test="$3"
  run="results/cross/$direction/sleepcnn/seed0"
  stem="sleepcnn_cross_seed0"
  mkdir -p "$run"
  if [[ -s "$run/${stem}_cal.npz" && -s "$run/${stem}_test.npz" ]]; then
    echo "== skip cross training $direction/sleepcnn =="
  else
    echo "== cross training $direction/sleepcnn =="
    "$PY" scripts/train_cross_cohort.py \
      --train-dir "data/processed/$train" \
      --test-dir "data/processed/$test" \
      --out-dir "$run" \
      --model sleepcnn \
      --seed 0 \
      --epochs 40 \
      --batch-size 128 \
      --no-class-weights
  fi
done

bash scripts/run_cross_matrix.sh "sc2st st2sc" "sleepcnn"

echo "== 78-subject matrix =="
DATA=data/processed/sleep-edf-78 ROOT=results/se78 FOLDS="0 1 2 3 4" \
  bash scripts/run_matrix.sh "eegnet sleepcnn" "0"

echo "== re-aggregating =="
"$PY" scripts/aggregate.py --root results/main --out results/main/summary
"$PY" scripts/aggregate.py --root results/se78 --out results/se78/summary
for direction in sc2st st2sc; do
  "$PY" scripts/aggregate.py --root "results/cross/$direction" --out "results/cross/$direction/summary"
done

echo "== ACI step-schedule check =="
bash scripts/run_aci_schedule_check.sh results/main results/schedule/main

echo "== paired tests, tables, figures =="
"$PY" scripts/paired_tests.py --root results/main --out results/main/summary/paired.json
"$PY" scripts/make_tables.py --summary results/main/summary
"$PY" scripts/make_figures.py --root results/main \
  --epoch-sets paper/figures/hypnogram_SC409.npz
echo "== followups done =="
