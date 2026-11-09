"""Collect every final result into one JSON and print compact tables.

The inputs are the per-run JSONs written by ``pilot_cssn4_fast.py`` (SCSN
matrix), ``tent_baseline.py`` (Tent) and ``adapt_bn_stats.py`` (layerwise
AdaBN), plus the cross-cohort JSON.  Means are over runs; the paired tests
stay in ``results/scsn_final/summary.json``.
"""

from __future__ import annotations

import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("results")
METRICS = ("accuracy", "kappa", "mf1", "ece", "coverage",
           "worst_subject_coverage", "coverage_sd", "set_size",
           "review_workload", "probe", "probe_within_mean")


def mean(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def tent_groups():
    groups = defaultdict(list)
    for path in sorted(glob.glob(str(ROOT / "tent" / "*.json"))):
        d = json.loads(Path(path).read_text())
        stem = Path(path).stem
        cohort = "hmc" if stem.startswith("hmc") else "edf"
        groups[(cohort, d.get("model", "eegnet"))].append(d)
    return groups


def adabn_groups():
    groups = {}
    for path in sorted(glob.glob(str(ROOT / "norm" / "adabn*.json"))):
        d = json.loads(Path(path).read_text())
        stem = Path(path).stem
        key = stem.replace("adabn_", "")
        groups[key] = d
    return groups


def main() -> None:
    summary = json.loads((ROOT / "scsn_final" / "summary.json").read_text())
    out = {"scsn": {}, "tent": {}, "adabn": {}}

    print("SCSN matrix")
    for key, group in summary["groups"].items():
        rows = group["summary"]
        out["scsn"][key] = {"n_runs": group["n_runs"], "model": group["model"],
                            "arms": rows}
        print(f"\n{key} (n={group['n_runs']})")
        for arm in ("bn", "marginal", "cond_full", "cond_full_lam0.5",
                    "cond_r4", "cond_r4_orth", "cond_hard"):
            if arm not in rows:
                continue
            r = rows[arm]
            print(f"  {arm:16s} " + " ".join(
                f"{m}={r[m]:.4f}" for m in
                ("accuracy", "worst_subject_coverage", "coverage_sd",
                 "probe", "probe_within_mean") if r.get(m) is not None))

    print("\nTent")
    for (cohort, model), runs in tent_groups().items():
        arms = sorted(set().union(*(r["results"].keys() for r in runs)))
        out["tent"][f"{cohort}_{model}"] = {"n_runs": len(runs), "arms": {}}
        print(f"\n{cohort} {model} (n={len(runs)})")
        for arm in arms:
            entries = [r["results"][arm] for r in runs]
            row = {m: mean(entries, m) for m in METRICS}
            out["tent"][f"{cohort}_{model}"]["arms"][arm] = row
            print(f"  {arm:10s} acc={row['accuracy']:.4f} "
                  f"worst={row['worst_subject_coverage']:.4f} "
                  f"sd={row['coverage_sd']:.4f} probe={row['probe']}")

    print("\nLayerwise AdaBN (existing protocol)")
    for key, d in adabn_groups().items():
        out["adabn"][key] = d["summary"]
        s = d["summary"]
        if s.get("accuracy") is None:
            continue
        print(f"  {key:18s} n={s.get('n_runs')} acc={s['accuracy']:.4f} "
              f"worst={s['worst_subject_coverage']:.4f} "
              f"sd={s['coverage_sd']:.4f} probe={s['probe_overall']:.4f}")

    cross = {}
    for path in sorted(glob.glob(str(ROOT / "scsn_cross" / "*.json"))):
        cross[Path(path).stem] = json.loads(Path(path).read_text())
    out["cross"] = cross

    (ROOT / "final_results.json").write_text(json.dumps(out, indent=1),
                                             encoding="utf-8")
    print("\nwrote results/final_results.json")


if __name__ == "__main__":
    main()
