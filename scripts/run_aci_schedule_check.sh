#!/usr/bin/env bash
# Re-run per-patient ACI with a step schedule that meets the square-summability
# condition, on the fitted models of the main matrix.
#
# The paper's default schedule is gamma0 / sqrt(t + 2), which adapts quickly but
# has sum_t gamma_t^2 = infinity and therefore falls outside the condition under
# which the long-run ACI bound is proved. This check repeats the identical
# experiment with gamma0 / (t + 2)^0.6 (sum diverges, squares converge) so that
# the reported behaviour cannot be attributed to the schedule choice.
#
# Usage: bash scripts/run_aci_schedule_check.sh [results/main] [results/schedule/main]
set -euo pipefail

PY=${PY:-python}
SRC=${1:-results/main}
OUT=${2:-results/schedule/main}

for run in "$SRC"/*/*/; do
  model=$(basename "$(dirname "$run")")
  name=$(basename "$run")
  cal="$run/${model}_${name}_cal.npz"
  test="$run/${model}_${name}_test.npz"
  [[ -f "$cal" && -f "$test" ]] || continue
  dest="$OUT/$model/$name"
  mkdir -p "$dest"
  if [[ -s "$dest/subject_aci.json" ]]; then
    echo "== skip $model/$name =="
    continue
  fi
  echo "== aci (poly) $model/$name =="
  "$PY" scripts/run_subject_aci.py \
    --cal "$cal" --test "$test" \
    --out "$dest/subject_aci.json" \
    --alpha 0.1 \
    --gamma0 0.005 0.01 0.02 0.05 0.1 \
    --schedule poly --poly-power 0.6
done

"$PY" scripts/aggregate.py --root "$OUT" --out "$OUT/summary"
