#!/usr/bin/env bash
# Resume the paper-2 compute after the two orphaned Python processes finish.
#
# The background shells that started these processes were reaped for low
# memory; the Python children survived and are still producing
# results/norm/adabn_sleepcnn.json (PID 32488) and the group_all/sleepcnn
# probe (PID 27544). They cannot be waited on through the job runner, so this
# script polls the process table instead, then runs everything that is still
# missing with one Python process at a time, which is what the machine can
# afford.
#
#   1. waiter: both PIDs gone, or 6 h elapsed
#   2. main/sleepcnn probe (its output was never written)
#   3. subject dependence for the two SleepCNN roots
#   4. aggregation, tables, figures
#   5. scripts/run_paper2_pipeline.sh (stages 1-7; completed stages are skipped)
set -u
PY=D:/Anaconda/envs/sleepcp/python.exe
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

alive() {  # alive PID
  tasklist //FI "PID eq $1" 2>/dev/null | grep -qi python
}

waited=0
while alive 32488 || alive 27544; do
  sleep 60
  waited=$((waited + 60))
  if [ "$waited" -ge 21600 ]; then
    echo "waiter: giving up after 6 h"
    break
  fi
  if [ $((waited % 1800)) -eq 0 ]; then
    echo "waiter: still running after $((waited / 60)) min"
  fi
done
echo "waiter: both processes gone after $((waited / 60)) min"
date

n_runs() {
  if [ -f "$1" ]; then
    $PY -I -c "import json,sys;print(len(json.load(open(sys.argv[1],encoding='utf-8')).get('runs',[])))" "$1"
  else
    echo 0
  fi
}

for root in results/main results/probe_norm/group_all; do
  tag=$(basename "$root")
  probe="${root}/summary/probe_sleepcnn.json"
  if [ "$(n_runs "$probe")" -lt 15 ]; then
    echo "=== probe sleepcnn ${tag}"
    $PY -u -I scripts/probe_subject_identity.py --root "$root" --model sleepcnn \
      --device cuda --out "$probe"
  fi
  dep="results/norm/subject_dependence/${tag}_sleepcnn.json"
  if [ ! -f "$dep" ]; then
    echo "=== subject dependence sleepcnn ${tag}"
    $PY -u -I scripts/subject_dependence.py --root "$root" --model sleepcnn \
      --device cuda --out "$dep"
  fi
done

echo "=== aggregate"
$PY -I scripts/aggregate_arms.py --out results/norm/arms.json
$PY -I scripts/make_tables2.py
$PY -I scripts/make_figures2.py

echo "=== pipeline (stages 1-7)"
bash scripts/run_paper2_pipeline.sh
echo "resume done"
