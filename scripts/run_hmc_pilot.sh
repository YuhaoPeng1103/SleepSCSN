#!/usr/bin/env bash
# HMC external validation for the conditional-normalisation pilot.
# Uses cached penultimate features under results/pilot_cssn/hmc_*.
set -e
cd "$(dirname "$0")/.."
mkdir -p results/pilot_cssn2
for fold in 0 1 2 3 4; do
  for seed in 0 1; do
    stem="hmc_eegnet_fold${fold}_seed${seed}"
    out="results/pilot_cssn2/${stem}.json"
    if [ -f "$out" ]; then echo "skip $out"; continue; fi
    echo "=== $stem ==="
    python scripts/pilot_cssn2.py \
      --model eegnet --fold "$fold" --seed "$seed" --rank 4 \
      --root results/hmc/batch --data-dir data/processed/hmc \
      --probe-max 5000 \
      --cache "results/pilot_cssn/${stem}_features.npz" \
      --arms bn,marginal,cond_full,cond_hard,cond_full_auto,cond_hard_auto,cond_full_lam0.5,cond_r4 \
      --out "$out"
  done
done
echo DONE
