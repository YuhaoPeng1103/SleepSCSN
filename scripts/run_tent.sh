#!/usr/bin/env bash
# Tent baseline on every run of the SCSN comparison.
set -e
cd "$(dirname "$0")/.."
mkdir -p results/tent
PY="${PY:-python}"
DEV="${DEV:-cpu}"

for fold in 0 1 2 3 4; do
  for seed in 0 1 2; do
    out="results/tent/eegnet_fold${fold}_seed${seed}.json"
    if [ -f "$out" ]; then echo "skip $out"; continue; fi
    echo "=== edf eegnet fold$fold seed$seed ==="
    "$PY" scripts/tent_baseline.py --model eegnet --fold "$fold" --seed "$seed" \
      --device "$DEV" --probe-max 5000 --out "$out"
  done
done

for fold in 0 1 2 3 4; do
  for seed in 0 1; do
    out="results/tent/hmc_eegnet_fold${fold}_seed${seed}.json"
    if [ -f "$out" ]; then echo "skip $out"; continue; fi
    echo "=== hmc eegnet fold$fold seed$seed ==="
    "$PY" scripts/tent_baseline.py --model eegnet --fold "$fold" --seed "$seed" \
      --root results/hmc/batch --data-dir data/processed/hmc --device "$DEV" \
      --probe-max 5000 --out "$out"
  done
done

for fold in 0 1 2 3 4; do
  out="results/tent/sleepcnn_fold${fold}_seed0.json"
  if [ -f "$out" ]; then echo "skip $out"; continue; fi
  echo "=== edf sleepcnn fold$fold ==="
  "$PY" scripts/tent_baseline.py --model sleepcnn --fold "$fold" --seed 0 \
    --device "$DEV" --probe-max 3000 --out "$out"
done

echo TENT_DONE
