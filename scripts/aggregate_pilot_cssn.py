"""Aggregate the CSSN pilot runs into paired comparisons.

Reads every ``results/pilot_cssn/<model>_fold*_seed*_pilot.json`` (or the
files named by ``--pattern``), keeps the arms present in all runs, and reports
means plus paired differences against ``bn`` and ``marginal``.

Example
-------
python scripts/aggregate_pilot_cssn.py --model eegnet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

METRICS = ("accuracy", "coverage", "worst_subject_coverage", "coverage_sd",
           "coverage_q10", "coverage_median", "coverage_below",
           "set_size", "probe", "probe_within_mean", "ece")
STAGES = ("W", "N1", "N2", "N3", "REM")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="eegnet")
    parser.add_argument("--dir", type=Path, default=Path("results/pilot_cssn"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    files = sorted(args.dir.glob(f"{args.model}_fold*_seed*.json"))
    runs = []
    for path in files:
        d = json.loads(path.read_text(encoding="utf-8"))
        d["_path"] = str(path)
        runs.append(d)
    if not runs:
        raise SystemExit(f"no runs found in {args.dir}")

    common = set(runs[0]["results"])
    for r in runs[1:]:
        common &= set(r["results"])
    common = [a for a in runs[0]["results"] if a in common]
    print(f"{len(runs)} runs; arms: {common}")

    def value(run, arm, metric):
        entry = run["results"][arm]
        if metric == "probe":
            return entry.get("probe")
        if metric == "probe_within_mean":
            vals = [v for v in (entry.get("probe_within") or {}).values()
                    if v is not None]
            return float(np.mean(vals)) if vals else None
        if metric in ("coverage_q10", "coverage_median", "coverage_below"):
            covs = np.asarray(list((entry.get("per_subject_coverage") or {}).values()),
                              dtype=float)
            if covs.size == 0:
                return None
            if metric == "coverage_q10":
                return float(np.percentile(covs, 10))
            if metric == "coverage_median":
                return float(np.median(covs))
            return float((covs < 0.9).mean())
        return entry.get(metric)

    def values(arm, metric):
        v = [value(r, arm, metric) for r in runs]
        return np.asarray([x for x in v if x is not None], dtype=float)

    def stage_value(run, arm, stage):
        return (run["results"][arm].get("probe_within") or {}).get(stage)

    summary = {}
    for arm in common:
        summary[arm] = {}
        for m in METRICS:
            v = values(arm, m)
            summary[arm][m] = float(np.mean(v)) if v.size else None
        for stage in STAGES:
            v = [stage_value(r, arm, stage) for r in runs]
            v = np.asarray([x for x in v if x is not None], dtype=float)
            summary[arm][f"within_{stage}"] = float(np.mean(v)) if v.size else None
    print("\nmeans:")
    header = "arm".ljust(22) + "".join(m[:12].rjust(14) for m in METRICS)
    print(header)
    for arm in common:
        print(arm.ljust(22) + "".join(
            (f"{summary[arm][m]:.4f}".rjust(14) if summary[arm][m] is not None else "nan".rjust(14))
            for m in METRICS))

    def paired(a, b, metric):
        va = np.asarray([value(r, a, metric) for r in runs], dtype=float)
        vb = np.asarray([value(r, b, metric) for r in runs], dtype=float)
        keep = np.isfinite(va) & np.isfinite(vb)
        d = va[keep] - vb[keep]
        n = d.size
        if n < 2:
            return None
        mean = float(d.mean())
        sd = float(d.std(ddof=1))
        ci = 1.96 * sd / np.sqrt(n)
        dz = mean / sd if sd > 0 else float("inf")
        try:
            p = float(wilcoxon(d).pvalue)
        except ValueError:
            p = float("nan")
        return {"mean_diff": mean, "ci95": [mean - ci, mean + ci],
                "cohen_dz": dz, "wilcoxon_p": p, "n": int(n)}

    pairs = {}
    for a in common:
        for b in ("bn", "marginal"):
            if a == b:
                continue
            key = f"{a}-vs-{b}"
            pairs[key] = {m: paired(a, b, m) for m in METRICS}
    print("\npaired differences (positive = first arm larger):")
    for key, d in pairs.items():
        bits = []
        for m in ("accuracy", "worst_subject_coverage", "coverage_sd", "probe",
                  "probe_within_mean"):
            s = d.get(m)
            if s:
                bits.append(f"{m}: {s['mean_diff']:+.4f} (p={s['wilcoxon_p']:.3f})")
        print(key + "  " + " | ".join(bits))

    def paired_stage(a, b, stage):
        va, vb = [], []
        for r in runs:
            x, y = stage_value(r, a, stage), stage_value(r, b, stage)
            if x is not None and y is not None:
                va.append(x)
                vb.append(y)
        d = np.asarray(va) - np.asarray(vb)
        if d.size < 2:
            return None
        try:
            p = float(wilcoxon(d).pvalue)
        except ValueError:
            p = float("nan")
        return float(d.mean()), p

    print("\nwithin-stage subject probe (mean over runs):")
    show = [a for a in ("bn", "marginal", "cond_full", "cond_full_lam0.5",
                        "cond_hard") if a in common]
    print("stage".ljust(6) + "".join(a.rjust(20) for a in show)
          + "   cond_full-marginal")
    for stage in STAGES:
        line = stage.ljust(6)
        for a in show:
            v = summary[a][f"within_{stage}"]
            line += (f"{v:.3f}".rjust(20) if v is not None else "nan".rjust(20))
        ps = paired_stage("cond_full", "marginal", stage) \
            if "cond_full" in common and "marginal" in common else None
        line += (f"   {ps[0]:+.3f} (p={ps[1]:.3f})" if ps else "")
        print(line)

    out = args.out or args.dir / f"{args.model}_pilot_summary.json"
    out.write_text(json.dumps({"n_runs": len(runs), "arms": common,
                               "summary": summary, "pairs": pairs}, indent=1),
                   encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
