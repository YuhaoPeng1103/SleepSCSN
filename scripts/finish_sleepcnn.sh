#!/usr/bin/env bash
# Wait for the AdaBN evaluation of the second backbone, then finish the
# Sleep-EDF measurements that are still missing: subject probes for the
# SleepCNN checkpoint and for the group-normalised EEGNet (its probe file was
# written by an aborted run and holds fewer than fifteen runs), the
# subject-dependence summaries, the arm aggregation, the tables and the
# figures.
#
# Another job may already be producing the same probes. Each probe is
# therefore waited for rather than recomputed: the wait ends as soon as the
# file holds fifteen runs, and the probe is run here only if the wait times
# out. Restartable: nothing is recomputed when its output is already complete.
set -u
PY=D:/Anaconda/envs/sleepcp/python.exe
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

n_runs() {  # number of runs in a probe file, 0 if absent
  if [ -f "$1" ]; then
    $PY -I -c "import json,sys;print(len(json.load(open(sys.argv[1],encoding='utf-8')).get('runs',[])))" "$1"
  else
    echo 0
  fi
}

wait_for_probe() {  # wait_for_probe FILE TIMEOUT_SECONDS
  local f=$1 limit=$2 waited=0
  while [ "$(n_runs "$f")" -lt 15 ] && [ "$waited" -lt "$limit" ]; do
    sleep 30
    waited=$((waited + 30))
  done
}

while [ ! -f results/norm/adabn_sleepcnn.json ]; do sleep 20; done
echo "AdaBN(SleepCNN) ready"

for root in results/main results/probe_norm/group_all; do
  tag=$(basename "$root")
  for m in eegnet sleepcnn; do
    probe="${root}/summary/probe_${m}.json"
    if [ "$(n_runs "$probe")" -lt 15 ]; then
      echo "=== waiting for probe ${tag} ${m} (another job may be running it)"
      wait_for_probe "$probe" 5400
    fi
    if [ "$(n_runs "$probe")" -lt 15 ]; then
      echo "=== probe ${tag} ${m} (running here)"
      $PY -u -I scripts/probe_subject_identity.py --root "$root" --model "$m" \
        --device cuda --out "$probe"
    else
      echo "=== probe ${tag} ${m} complete ($(n_runs "$probe") runs)"
    fi
    dep="results/norm/subject_dependence/${tag}_${m}.json"
    if [ ! -f "$dep" ]; then
      echo "=== subject dependence ${tag} ${m}"
      $PY -u -I scripts/subject_dependence.py --root "$root" --model "$m" \
        --device cuda --out "$dep"
    fi
  done
done

echo "=== aggregate"
$PY -I scripts/aggregate_arms.py --out results/norm/arms.json
$PY -I scripts/make_tables2.py
$PY -I scripts/make_figures2.py
echo "finish_sleepcnn done"
