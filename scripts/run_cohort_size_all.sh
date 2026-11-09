#!/usr/bin/env bash
# Run the calibration-cohort-size sweep (scripts/run_cohort_size.py) over every
# run directory under a results root, then re-aggregate so that the summary
# carries the sweep and Table tab:cohort-size can be rebuilt.
#
# Usage:
#   bash scripts/run_cohort_size_all.sh                     # results/main
#   bash scripts/run_cohort_size_all.sh results/se78        # 78-subject cohort
#
# The sweep is deterministic given the stored calibration/test probabilities,
# so existing cohort_size.json files are skipped and the script is resumable.
set -euo pipefail

PY="${PY:-python}"
ROOT="${1:-results/main}"

if [[ ! -d "$ROOT" ]]; then
  echo "missing results root: $ROOT"
  exit 1
fi

n_run=0
n_skip=0
while IFS= read -r -d '' cal; do
  run="$(dirname "$cal")"
  backbone="$(basename "$(dirname "$run")")"
  stem="$(basename "$run")"
  test="$run/${backbone}_${stem}_test.npz"
  if [[ ! -s "$test" ]]; then
    continue
  fi
  if [[ -s "$run/cohort_size.json" ]]; then
    n_skip=$((n_skip + 1))
    continue
  fi
  echo "== cohort size $backbone/$stem =="
  "$PY" scripts/run_cohort_size.py \
    --cal "$cal" --test "$test" \
    --out "$run/cohort_size.json" \
    --alpha 0.1 \
    --repeats 50
  n_run=$((n_run + 1))
done < <(find "$ROOT" -mindepth 3 -maxdepth 3 -name '*_cal.npz' -print0 | sort -z)

echo "== cohort size: ran $n_run, skipped $n_skip =="

# Aggregate only if the run directories carry per-run JSON; aggregate.py picks
# up cohort_size.json alongside the other per-run files.
"$PY" scripts/aggregate.py --root "$ROOT" --out "$ROOT/summary"
echo "== cohort size done: $ROOT =="
