"""Aggregate the final SCSN matrix into means, paired tests and sign counts.

Files are read from ``results/scsn_final/<tag>_<model>_fold<f>_seed<s>.json``.
Every group (cohort and normalisation) is aggregated separately, and every
comparison is paired over runs that share the checkpoint, the split, the test
epochs and the probe subsample.

Example
-------
python scripts/aggregate_scsn_final.py
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import t as t_dist, wilcoxon

METRICS = ("accuracy", "kappa", "mf1", "ece", "coverage", "coverage_sd",
           "worst_subject_coverage", "coverage_q10", "coverage_median",
           "coverage_below", "set_size", "review_workload", "probe",
           "probe_within_mean")
STAGES = ("W", "N1", "N2", "N3", "REM")
PATTERN = re.compile(r"^(?P<tag>.+)_(?P<model>eegnet|sleepcnn)_fold(?P<fold>\d+)_seed(?P<seed>\d+)\.json$")


def load_groups(directory: Path) -> dict:
    groups = defaultdict(list)
    for path in sorted(directory.glob("*.json")):
        match = PATTERN.match(path.name)
        if not match:
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        d["_fold"] = int(match.group("fold"))
        d["_seed"] = int(match.group("seed"))
        d["_tag"] = match.group("tag")
        d["_model"] = match.group("model")
        groups[(d["_tag"], d["_model"])].append(d)
    return groups


def value(run: dict, arm: str, metric: str):
    entry = run["results"].get(arm)
    if entry is None:
        return None
    if metric == "probe_within_mean":
        vals = [v for v in (entry.get("probe_within") or {}).values() if v is not None]
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


def stage_value(run: dict, arm: str, stage: str):
    return (run["results"].get(arm, {}).get("probe_within") or {}).get(stage)


def paired(runs, a, b, metric):
    va, vb = [], []
    for run in runs:
        x, y = value(run, a, metric), value(run, b, metric)
        if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
            va.append(x)
            vb.append(y)
    d = np.asarray(va, dtype=float) - np.asarray(vb, dtype=float)
    n = d.size
    if n < 2:
        return None
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    half = float(t_dist.ppf(0.975, n - 1) * sd / np.sqrt(n))
    dz = mean / sd if sd > 0 else float("inf")
    try:
        p = float(wilcoxon(d).pvalue)
    except ValueError:
        p = float("nan")
    return {"mean_diff": mean, "ci95": [mean - half, mean + half],
            "cohen_dz": dz, "wilcoxon_p": p, "n": int(n)}


def sign_count(runs, a, b, metric):
    """Fold-level sign count: average within fold, then count folds."""
    per_fold = defaultdict(list)
    for run in runs:
        x, y = value(run, a, metric), value(run, b, metric)
        if x is not None and y is not None:
            per_fold[run["_fold"]].append(x - y)
    signs = [np.mean(v) > 0 for v in per_fold.values()]
    return int(sum(signs)), len(signs)


def summarise(runs):
    common = set(runs[0]["results"])
    for run in runs[1:]:
        common &= set(run["results"])
    arms = [a for a in runs[0]["results"] if a in common]
    summary = {}
    for arm in arms:
        summary[arm] = {}
        for metric in METRICS:
            vals = [value(r, arm, metric) for r in runs]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            summary[arm][metric] = float(np.mean(vals)) if vals else None
        for stage in STAGES:
            vals = [stage_value(r, arm, stage) for r in runs]
            vals = [v for v in vals if v is not None and np.isfinite(v)]
            summary[arm][f"within_{stage}"] = float(np.mean(vals)) if vals else None

    pairs = {}
    contrasts = [(arm, "bn") for arm in arms if arm != "bn"]
    contrasts += [(arm, "marginal") for arm in arms
                  if arm not in ("bn", "marginal")]
    seen = set()
    for a, b in contrasts:
        if (a, b) in seen or (b, a) in seen or a == b:
            continue
        seen.add((a, b))
        key = f"{a}-vs-{b}"
        pairs[key] = {m: paired(runs, a, b, m) for m in
                      ("accuracy", "kappa", "ece", "coverage", "worst_subject_coverage",
                       "coverage_sd", "coverage_q10", "set_size", "review_workload",
                       "probe", "probe_within_mean")}
        pairs[key]["_signs"] = {m: sign_count(runs, a, b, m) for m in
                                ("accuracy", "worst_subject_coverage", "coverage_sd",
                                 "probe_within_mean")}
    return arms, summary, pairs


def main() -> None:
    directory = Path("results/scsn_final")
    groups = load_groups(directory)
    out = {"groups": {}}
    for (tag, model), runs in groups.items():
        arms, summary, pairs = summarise(runs)
        out["groups"][f"{tag}_{model}"] = {
            "tag": tag, "model": model, "n_runs": len(runs),
            "arms": arms, "summary": summary, "pairs": pairs}
        print(f"\n### {tag} ({model}), {len(runs)} runs")
        print("arm".ljust(24) + "".join(m[:10].rjust(11) for m in
              ("accuracy", "kappa", "mf1", "ece", "coverage", "worst_subject_coverage",
               "coverage_sd", "set_size", "probe", "probe_within_mean")))
        for arm in arms:
            row = summary[arm]
            print(arm.ljust(24) + "".join(
                (f"{row[m]:.4f}".rjust(11) if row[m] is not None else "nan".rjust(11))
                for m in ("accuracy", "kappa", "mf1", "ece", "coverage",
                          "worst_subject_coverage", "coverage_sd", "set_size",
                          "probe", "probe_within_mean")))
        for key in ("marginal-vs-bn", "cond_full-vs-bn", "cond_full-vs-marginal",
                    "cond_full_lam0.5-vs-marginal", "cond_r4-vs-marginal"):
            if key not in pairs:
                continue
            bits = []
            for m in ("accuracy", "worst_subject_coverage", "coverage_sd",
                      "probe", "probe_within_mean"):
                s = pairs[key].get(m)
                if s:
                    signs = pairs[key]["_signs"].get(m)
                    tail = (f", {signs[0]}/{signs[1]}" if signs else "")
                    bits.append(f"{m}: {s['mean_diff']:+.4f} "
                                f"(p={s['wilcoxon_p']:.3f}{tail})")
            print(key + "  " + " | ".join(bits))

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("\nwrote results/scsn_final/summary.json")


if __name__ == "__main__":
    main()
