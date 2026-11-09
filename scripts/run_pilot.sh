#!/usr/bin/env bash
# End-to-end pilot: preprocess Sleep-EDF-20, train EEGNet, run CP diagnostics.
#
# Usage:
#   bash scripts/run_pilot.sh
#
# Assumes the raw EDF files are in data/sleep-edf/sleep-cassette.
set -euo pipefail

RAW="data/sleep-edf/sleep-cassette"
PROCESSED="data/processed/sleep-edf-20"
RUN="results/pilot_eegnet_fold0_seed0"

if [[ ! -d "$RAW" ]]; then
  echo "raw data not found: $RAW"
  echo "run: bash scripts/download_sleep_edf.sh 20"
  exit 1
fi

mkdir -p "$PROCESSED" "$RUN"

if ! compgen -G "$PROCESSED/*.npz" > /dev/null; then
  echo "== preprocessing =="
  python scripts/prepare_sleep_edf.py \
    --data-dir "$RAW" \
    --out-dir "$PROCESSED" \
    --subset sleep-edf-20 \
    --channel Fpz-Cz \
    --resample 100 \
    --normalize recording
else
  echo "== preprocessing already done =="
fi

echo "== training EEGNet =="
python scripts/train_eegnet.py \
  --data-dir "$PROCESSED" \
  --out-dir "$RUN" \
  --fold 0 \
  --seed 0 \
  --epochs 30 \
  --batch-size 128 \
  --no-class-weights

CAL="$RUN/eegnet_fold0_seed0_cal.npz"
TEST="$RUN/eegnet_fold0_seed0_test.npz"

echo "== CP diagnostic =="
python scripts/run_cp_diagnostic.py \
  --cal "$CAL" --test "$TEST" \
  --out "$RUN/cp_diagnostic.json" \
  --alpha 0.1 \
  --n-repeats 50

echo "== subject-adaptive CP =="
python scripts/run_sacp.py \
  --cal "$CAL" --test "$TEST" \
  --out "$RUN/sacp.json" \
  --alpha 0.1 \
  --budget 0 10 20 40 80 160 320 \
  --k 3 5 10

echo "== temporal CP =="
python scripts/run_temporal_cp.py \
  --cal "$CAL" --test "$TEST" \
  --out "$RUN/temporal_cp.json" \
  --alpha 0.1 \
  --block-length 10 20 30 \
  --beta 0 0.5 1.0 2.0

echo "== guarantee check (Theorem 1) =="
python scripts/run_guarantee_check.py \
  --cal "$CAL" --test "$TEST" \
  --out "$RUN/guarantee_check.json" \
  --alpha 0.1 \
  --block-length 5 10 20 30 \
  --beta 0 0.5 1.0 \
  --repeats 100

echo "== done: $RUN =="
