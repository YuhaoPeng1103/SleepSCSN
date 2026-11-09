#!/usr/bin/env bash
# Finish the SleepCNN replication on HMC: group arm, its diagnostics, the
# AdaBN evaluation of the batch arm, both probes and the aggregation.
#
# Restartable. The first block waits for a run that was still training when
# the previous scheduler was stopped, so that the grid does not start a second
# copy of it.
#
#   bash scripts/run_hmc_sleepcnn_finish.sh 2>&1 | tee logs/hmc_sleepcnn_finish.log
set -u
PY=D:/Anaconda/envs/sleepcp/python.exe
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

echo "########## wait for fold0_seed0 of the group arm"
for i in $(seq 1 60); do
  if ls results/hmc/group/sleepcnn/fold0_seed0/*.json >/dev/null 2>&1; then
    echo "fold0_seed0 present"
    break
  fi
  sleep 60
done

echo "########## group grid"
bash scripts/run_hmc_grid.sh group sleepcnn 2

echo "########## group conformal diagnostics"
for run in results/hmc/group/sleepcnn/fold*_seed*; do
  [ -d "$run" ] || continue
  cal=$(ls "${run}"/*_cal.npz 2>/dev/null | head -1)
  test=$(ls "${run}"/*_test.npz 2>/dev/null | head -1)
  [ -n "$cal" ] && [ -n "$test" ] || continue
  [ -f "${run}/cp_diagnostic.json" ] && continue
  echo "--- ${run}"
  $PY -u -I scripts/run_cp_diagnostic.py --cal "$cal" --test "$test" \
    --out "${run}/cp_diagnostic.json" --n-repeats 20
done

echo "########## SleepCNN AdaBN on HMC"
if [ ! -f results/norm/adabn_hmc_sleepcnn.json ]; then
  $PY -u -I scripts/adapt_bn_stats.py --root results/hmc/batch \
    --data-dir data/processed/hmc --model sleepcnn --device cuda \
    --out results/norm/adabn_hmc_sleepcnn.json
fi

echo "########## SleepCNN probes on HMC"
for arm in batch group; do
  out="results/hmc/${arm}/summary/probe_sleepcnn.json"
  [ -f "$out" ] && { echo "skip $out"; continue; }
  $PY -u -I scripts/probe_subject_identity.py --root "results/hmc/${arm}" \
    --data-dir data/processed/hmc --model sleepcnn --device cuda --out "$out"
done

echo "########## aggregate the SleepCNN replication"
$PY -I scripts/aggregate_hmc.py --model sleepcnn \
  --adabn results/norm/adabn_hmc_sleepcnn.json \
  --out results/norm/arms_hmc_sleepcnn.json

echo "########## done"
