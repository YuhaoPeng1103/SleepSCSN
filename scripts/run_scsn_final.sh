#!/usr/bin/env bash
# Final SCSN matrix. Every run writes one JSON under results/scsn_final/.
# Caches are reused where they already exist, and written otherwise.
set -e
cd "$(dirname "$0")/.."
mkdir -p results/scsn_final results/scsn_final/cache

FULL="bn,marginal,cond_full,cond_full_lam0.25,cond_full_lam0.5,cond_full_lam0.75,cond_r1,cond_r2,cond_r4,cond_r8,cond_r4_orth,cond_hard,cond_mismatch,marginal_mismatch"
RED6="bn,marginal,cond_full,cond_r4,cond_full_lam0.5,cond_hard"
RED4="bn,marginal,cond_full,cond_r4"

run () {  # model fold seed root data cache arms probe tag
  local model=$1 fold=$2 seed=$3 root=$4 data=$5 cache=$6 arms=$7 probe=$8 tag=$9
  local out="results/scsn_final/${tag}_${model}_fold${fold}_seed${seed}.json"
  local ckpt="${root}/${model}/fold${fold}_seed${seed}/${model}_fold${fold}_seed${seed}.pt"
  if [ ! -f "$ckpt" ]; then echo "skip missing $ckpt"; return; fi
  if [ -f "$out" ]; then echo "skip $out"; return; fi
  echo "=== $tag $model fold$fold seed$seed ==="
  python scripts/pilot_cssn4_fast.py --model "$model" --fold "$fold" --seed "$seed" \
    --rank 4 --root "$root" --data-dir "$data" --probe-max "$probe" \
    --cache "$cache" --arms "$arms" --out "$out"
}

phase () {  # tag model seeds root data cache-dir arms probe
  local tag=$1 model=$2 seeds=$3 root=$4 data=$5 cache_dir=$6 arms=$7 probe=$8
  for fold in 0 1 2 3 4; do
    for seed in $seeds; do
      local cache="${cache_dir}/${tag}/${model}_fold${fold}_seed${seed}_features.npz"
      if [ "$tag" = "hmc_bn" ]; then
        cache="results/pilot_cssn/hmc_${model}_fold${fold}_seed${seed}_features.npz"
      elif [ "$tag" = "edf_bn" ]; then
        cache="results/pilot_cssn/${model}_fold${fold}_seed${seed}_features.npz"
      fi
      run "$model" "$fold" "$seed" "$root" "$data" "$cache" "$arms" "$probe" "$tag"
    done
  done
}

echo "PHASE edf_bn_eegnet"
phase edf_bn eegnet "0 1 2" results/main data/processed/sleep-edf-20 results/scsn_final/cache "$FULL" 5000
echo "PHASE hmc_bn_eegnet"
phase hmc_bn eegnet "0 1" results/hmc/batch data/processed/hmc results/scsn_final/cache "$FULL" 5000
echo "PHASE edf_bn_sleepcnn"
phase edf_bn sleepcnn "0" results/main data/processed/sleep-edf-20 results/scsn_final/cache "$RED6" 3000
echo "PHASE hmc_bn_sleepcnn"
phase hmc_bn sleepcnn "0 1" results/hmc/batch data/processed/hmc results/scsn_final/cache "$RED4" 3000
echo "PHASE edf_gn_eegnet"
phase edf_gn eegnet "0 1 2" results/probe_norm/group_all data/processed/sleep-edf-20 results/scsn_final/cache "$RED4" 5000
echo "PHASE edf_gn_sleepcnn"
phase edf_gn sleepcnn "0" results/probe_norm/group_all data/processed/sleep-edf-20 results/scsn_final/cache "$RED4" 3000
echo "PHASE hmc_gn_eegnet"
phase hmc_gn eegnet "0 1" results/hmc/group data/processed/hmc results/scsn_final/cache "$RED4" 5000
echo "PHASE hmc_gn_sleepcnn"
phase hmc_gn sleepcnn "0 1" results/hmc/group data/processed/hmc results/scsn_final/cache "$RED4" 3000
echo SCSN_FINAL_DONE
