#!/usr/bin/env bash
# Layerwise AdaBN with the final probe protocol, and SCSN stacked on it.
set -e
cd "$(dirname "$0")/.."
mkdir -p results/adabn_final
PY="${PY:-python}"

emit () {  # tag model fold seed root data probe [extra]
  local tag=$1 model=$2 fold=$3 seed=$4 root=$5 data=$6 probe=$7 extra=${8:-}
  local ckpt="${root}/${model}/fold${fold}_seed${seed}/${model}_fold${fold}_seed${seed}.pt"
  [ -f "$ckpt" ] || return
  local out="results/adabn_final/${tag}_${model}_fold${fold}_seed${seed}.json"
  [ -f "$out" ] && return
  echo "$PY -u scripts/adabn_final.py --model $model --fold $fold --seed $seed --root $root --data-dir $data --probe-max $probe $extra --out $out"
}

PHASE="${1:-all}"

# Sleep-EDF: small recording matrix, one at a time while other jobs are running.
if [ "$PHASE" = "all" ] || [ "$PHASE" = "edf" ]; then
{
  for fold in 0 1 2 3 4; do
    for seed in 0 1 2; do
      emit edf_bn eegnet "$fold" "$seed" results/main data/processed/sleep-edf-20 5000
    done
  done
  for fold in 0 1 2 3 4; do
    emit edf_bn sleepcnn "$fold" 0 results/main data/processed/sleep-edf-20 3000
  done
} | xargs -P 1 -I CMD bash -c "cd '$PWD' && CMD"
fi

# HMC: one at a time, the recording matrix is large.
if [ "$PHASE" = "all" ] || [ "$PHASE" = "hmc" ]; then
{
  for fold in 0 1 2 3 4; do
    for seed in 0 1; do
      emit hmc_bn eegnet "$fold" "$seed" results/hmc/batch data/processed/hmc 5000 "--skip-stacked --device cuda"
      emit hmc_bn sleepcnn "$fold" "$seed" results/hmc/batch data/processed/hmc 3000 "--skip-stacked --device cuda"
    done
  done
} | xargs -P 1 -I CMD bash -c "cd '$PWD' && CMD"
fi

echo ADABN_FINAL_DONE
