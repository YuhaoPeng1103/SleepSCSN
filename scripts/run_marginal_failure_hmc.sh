#!/usr/bin/env bash
# Per-recording failure analysis of marginal adaptation on every HMC run.
set -e
cd "$(dirname "$0")/.."
for f in results/pilot_cssn2/hmc_eegnet_fold*_seed*.json; do
  stem=$(basename "$f" .json)
  fold=$(echo "$stem" | sed 's/.*_fold\([0-9]*\)_seed.*/\1/')
  seed=$(echo "$stem" | sed 's/.*_seed\([0-9]*\)$/\1/')
  out="results/pilot_cssn2/marginal_failure_${stem}.json"
  if [ -f "$out" ]; then echo "skip $out"; continue; fi
  python scripts/analyze_marginal_failure.py \
    --cache "results/pilot_cssn/${stem}_features.npz" \
    --run "results/hmc/batch/eegnet/fold${fold}_seed${seed}/eegnet_fold${fold}_seed${seed}.pt" \
    --out "$out" > /dev/null
  echo "wrote $out"
done
