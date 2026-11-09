#!/usr/bin/env bash
# Everything that still has to be computed for the normalisation paper.
# Restartable: each stage skips outputs that already exist.
#
# Order matters. The cheap Sleep-EDF measurements come first so that the
# manuscript can be updated while the multi-hour external-cohort grid runs.
#
#   ./scripts/run_paper2_pipeline.sh 2>&1 | tee logs/paper2_pipeline.log
set -u
PY=D:/Anaconda/envs/sleepcp/python.exe
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

echo "########## 0. Sleep-EDF AdaBN (both backbones)"
for m in eegnet sleepcnn; do
  if [ ! -f "results/norm/adabn_${m}.json" ]; then
    $PY -u -I scripts/adapt_bn_stats.py --root results/main --model $m \
      --device cuda --out "results/norm/adabn_${m}.json"
  fi
done

echo "########## 1. Sleep-EDF layerwise identity"
for root in results/main results/probe_norm/group_all; do
  tag=$(basename "$root")
  for m in eegnet sleepcnn; do
    out="results/norm/layerwise/${tag}_${m}.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    $PY -u -I scripts/layerwise_identity.py --root "$root" --model "$m" \
      --device cuda --out "$out"
  done
done

echo "########## 2. Cross-cohort transfer to HMC"
for m in eegnet sleepcnn; do
  for norm in batch group; do
    case $norm in
      batch) src=results/main ;;
      group) src=results/probe_norm/group_all ;;
    esac
    out="results/norm/cross_cohort/${m}_${norm}.json"
    [ -f "$out" ] && { echo "skip $out"; continue; }
    $PY -u -I scripts/cross_cohort.py --model "$m" --norm "$norm" \
      --source "${src}/${m}/fold0_seed0" --target data/processed/hmc \
      --device cuda --out "$out"
  done
done

echo "########## 3. HMC grids"
bash scripts/run_hmc_grid.sh batch eegnet 2
bash scripts/run_hmc_grid.sh group eegnet 2

echo "########## 4. HMC conformal diagnostics"
for arm in batch group; do
  for run in results/hmc/${arm}/eegnet/fold*_seed*; do
    [ -d "$run" ] || continue
    cal=$(ls "${run}"/*_cal.npz 2>/dev/null | head -1)
    test=$(ls "${run}"/*_test.npz 2>/dev/null | head -1)
    [ -n "$cal" ] && [ -n "$test" ] || continue
    [ -f "${run}/cp_diagnostic.json" ] && continue
    echo "--- ${run}"
    $PY -u -I scripts/run_cp_diagnostic.py --cal "$cal" --test "$test" \
      --out "${run}/cp_diagnostic.json" --n-repeats 20
  done
done

echo "########## 5. HMC AdaBN"
if [ ! -f results/norm/adabn_hmc_eegnet.json ]; then
  $PY -u -I scripts/adapt_bn_stats.py --root results/hmc/batch \
    --data-dir data/processed/hmc --model eegnet --device cuda \
    --out results/norm/adabn_hmc_eegnet.json
fi

echo "########## 6. HMC probes"
for arm in batch group; do
  out="results/hmc/${arm}/summary/probe_eegnet.json"
  [ -f "$out" ] && { echo "skip $out"; continue; }
  $PY -u -I scripts/probe_subject_identity.py --root "results/hmc/${arm}" \
    --data-dir data/processed/hmc --model eegnet --device cuda --out "$out"
done

echo "########## 7. aggregate the external cohort"
$PY -I scripts/aggregate_hmc.py

echo "########## 8. matched vs mismatched statistics control"
for m in eegnet sleepcnn; do
  out="results/norm/adabn_control_${m}.json"
  [ -f "$out" ] && { echo "skip $out"; continue; }
  $PY -u -I scripts/ablate_matched_stats.py --root results/main --model "$m" \
    --device cuda --out "$out"
done

echo "########## 9. SleepCNN on HMC (batch arm and AdaBN)"
bash scripts/run_hmc_grid.sh batch sleepcnn 2
for run in results/hmc/batch/sleepcnn/fold*_seed*; do
  [ -d "$run" ] || continue
  cal=$(ls "${run}"/*_cal.npz 2>/dev/null | head -1)
  test=$(ls "${run}"/*_test.npz 2>/dev/null | head -1)
  [ -n "$cal" ] && [ -n "$test" ] || continue
  [ -f "${run}/cp_diagnostic.json" ] && continue
  $PY -u -I scripts/run_cp_diagnostic.py --cal "$cal" --test "$test" \
    --out "${run}/cp_diagnostic.json" --n-repeats 20
done
if [ ! -f results/norm/adabn_hmc_sleepcnn.json ]; then
  $PY -u -I scripts/adapt_bn_stats.py --root results/hmc/batch \
    --data-dir data/processed/hmc --model sleepcnn --device cuda \
    --out results/norm/adabn_hmc_sleepcnn.json
fi
if [ ! -f results/hmc/batch/summary/probe_sleepcnn.json ]; then
  $PY -u -I scripts/probe_subject_identity.py --root results/hmc/batch \
    --data-dir data/processed/hmc --model sleepcnn --device cuda \
    --out results/hmc/batch/summary/probe_sleepcnn.json
fi
$PY -I scripts/aggregate_hmc.py --model sleepcnn \
  --adabn results/norm/adabn_hmc_sleepcnn.json \
  --out results/norm/arms_hmc_sleepcnn.json

echo "########## pipeline done"
