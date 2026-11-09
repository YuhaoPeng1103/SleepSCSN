#!/usr/bin/env bash
# Train the external-cohort grid on the HMC database.
#   ./scripts/run_hmc_grid.sh <norm> <model> <n_seeds>
# Seeds are the outer loop so that stopping early still leaves every fold
# covered. Runs whose training JSON already exists are skipped, so the grid
# can be restarted.
set -u
PY=D:/Anaconda/envs/sleepcp/python.exe
NORM=${1:-batch}
MODEL=${2:-eegnet}
SEEDS=${3:-2}

for ((seed=0; seed<SEEDS; seed++)); do
  for fold in 0 1 2 3 4; do
    out="results/hmc/${NORM}/${MODEL}/fold${fold}_seed${seed}"
    if ls "${out}"/*.json >/dev/null 2>&1; then
      echo "skip ${out}"
      continue
    fi
    echo "=== ${out} ==="
    $PY -u -I scripts/train_eegnet.py --data-dir data/processed/hmc \
      --out-dir "${out}" --model "${MODEL}" --norm "${NORM}" \
      --fold "${fold}" --seed "${seed}" --epochs 40 --batch-size 128 \
      --no-class-weights --device cuda 2>&1 | tail -6
  done
done
echo "grid done: ${NORM} ${MODEL}"
