#!/usr/bin/env bash
# Parallel GroupNorm phase, written to a staging directory so that it cannot
# race with the sequential matrix runner.
set -e
cd "$(dirname "$0")/.."
mkdir -p results/scsn_gn_manual/cache
RED4="bn,marginal,cond_full,cond_r4"

emit () {  # tag model fold seed root data probe
  local tag=$1 model=$2 fold=$3 seed=$4 root=$5 data=$6 probe=$7
  local ckpt="${root}/${model}/fold${fold}_seed${seed}/${model}_fold${fold}_seed${seed}.pt"
  [ -f "$ckpt" ] || return
  local out="results/scsn_gn_manual/${tag}_${model}_fold${fold}_seed${seed}.json"
  [ -f "$out" ] && return
  echo "python scripts/pilot_cssn4_fast.py --model $model --fold $fold --seed $seed --rank 4 --root $root --data-dir $data --probe-max $probe --cache results/scsn_gn_manual/cache/${tag}/${model}_fold${fold}_seed${seed}_features.npz --arms $RED4 --out $out"
}

{
  for fold in 0 1 2 3 4; do
    for seed in 0 1 2; do
      emit edf_gn eegnet "$fold" "$seed" results/probe_norm/group_all data/processed/sleep-edf-20 5000
    done
  done
} | xargs -P 2 -I CMD bash -c "cd '$PWD' && CMD"

{
  for fold in 0 1 2 3 4; do
    emit edf_gn sleepcnn "$fold" 0 results/probe_norm/group_all data/processed/sleep-edf-20 3000
  done
} | xargs -P 2 -I CMD bash -c "cd '$PWD' && CMD"

{
  for fold in 0 1 2 3 4; do
    for seed in 0 1; do
      emit hmc_gn eegnet "$fold" "$seed" results/hmc/group data/processed/hmc 5000
    done
  done
} | xargs -P 2 -I CMD bash -c "cd '$PWD' && CMD"

{
  for fold in 0 1 2 3 4; do
    for seed in 0 1; do
      emit hmc_gn sleepcnn "$fold" "$seed" results/hmc/group data/processed/hmc 3000
    done
  done
} | xargs -P 2 -I CMD bash -c "cd '$PWD' && CMD"

echo GN_PARALLEL_DONE
