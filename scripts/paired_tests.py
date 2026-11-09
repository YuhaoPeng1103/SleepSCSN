"""Paired per-patient comparisons between calibration rules.

The paper's headline claims are about the distribution of per-patient
coverage, and the methods are evaluated on the same subjects, so the
comparison is paired by patient. This script merges per-subject coverage
within each seed's five folds (every subject appears exactly once per seed),
averages across seeds, and runs a two-sided Wilcoxon signed-rank test on the
resulting pairs, plus a sign count.

Usage:
    python scripts/paired_tests.py --root results/main --out results/main/summary/paired.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def seed_of(path: Path) -> str:
    for part in path.parts:
        if part.startswith("fold") and "_seed" in part:
            return part.split("_seed")[1]
    return "0"


def collect(root: Path) -> dict[str, dict[str, list[float]]]:
    """method -> subject -> list of per-seed coverage values."""
    per_method: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    def merge(entries: list[dict]) -> dict[str, float]:
        # A subject appears once per backbone within a seed, so average the
        # two backbone values rather than letting the later file overwrite
        # the earlier one. Within one backbone the subject appears once (its
        # test fold); across folds the entries are disjoint, so the mean over
        # the collected values is the same average the headline tables use.
        acc: dict[str, list[float]] = defaultdict(list)
        for entry in entries:
            for subject, value in entry.get("per_subject_coverage", {}).items():
                acc[subject].append(float(value))
        return {s: float(np.mean(v)) for s, v in acc.items()}

    # group raw entries by (method, seed) so a subject appears once per seed
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for path in sorted(root.rglob("cp_diagnostic.json")):
        data = load(path)
        for name in ("global", "class_conditional"):
            entry = data.get("methods", {}).get(name)
            if entry:
                grouped[(name, seed_of(path))].append(entry)
    for path in sorted(root.rglob("subject_aci.json")):
        data = load(path)
        for name in ("global_aps", "global_deficit", "subject_aci_aps",
                     "subject_aci_deficit"):
            entry = data.get("methods", {}).get(name)
            if entry:
                grouped[(name, seed_of(path))].append(entry)

    for (method, _seed), entries in grouped.items():
        for subject, value in merge(entries).items():
            per_method[method][subject].append(value)

    # average over seeds -> one value per subject
    return {
        method: {s: float(np.mean(v)) for s, v in subjects.items()}
        for method, subjects in per_method.items()
    }


def compare(a: dict[str, float], b: dict[str, float], nominal: float | None = None) -> dict:
    subjects = sorted(set(a) & set(b))
    x = np.asarray([a[s] for s in subjects], dtype=float)
    y = np.asarray([b[s] for s in subjects], dtype=float)
    if nominal is not None:
        # Compare the per-patient distance from nominal, not the mean: the
        # claim is that adaptation shrinks the spread, and the mean is
        # deliberately left unchanged.
        x, y = np.abs(x - nominal), np.abs(y - nominal)
    diff = y - x
    if subjects and np.allclose(diff, 0):
        stat, p = float("nan"), 1.0
    elif subjects:
        stat, p = wilcoxon(x, y)
        stat, p = float(stat), float(p)
    else:
        stat, p = float("nan"), float("nan")
    return {
        "n_subjects": len(subjects),
        "mean_a": float(x.mean()) if subjects else None,
        "mean_b": float(y.mean()) if subjects else None,
        "n_b_greater": int((diff > 0).sum()),
        "n_b_lower": int((diff < 0).sum()),
        "median_diff": float(np.median(diff)) if subjects else None,
        "wilcoxon_stat": stat,
        "p_value": p,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/main"))
    parser.add_argument("--out", type=Path, default=Path("results/main/summary/paired.json"))
    args = parser.parse_args()

    cov = collect(args.root)
    # (a, b, nominal): when nominal is given, the test is on the absolute
    # deviation from nominal rather than on coverage itself.
    comparisons = {
        "aci_aps_vs_global_aps": ("global", "subject_aci_aps", None),
        "aci_deficit_vs_global_deficit": ("global_deficit", "subject_aci_deficit", None),
        "aci_aps_vs_mondrian": ("class_conditional", "subject_aci_aps", None),
        "mondrian_vs_global_aps": ("global", "class_conditional", None),
        "absdev_aci_aps_vs_global_aps": ("global", "subject_aci_aps", 0.9),
        "absdev_aci_deficit_vs_global_deficit": ("global_deficit", "subject_aci_deficit", 0.9),
        "absdev_aci_aps_vs_mondrian": ("class_conditional", "subject_aci_aps", 0.9),
        "absdev_mondrian_vs_global_aps": ("global", "class_conditional", 0.9),
    }
    out = {}
    for label, (a, b, nominal) in comparisons.items():
        if a in cov and b in cov:
            out[label] = {"a": a, "b": b, **compare(cov[a], cov[b], nominal)}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"{'comparison':<34}{'n':>4}{'mean a':>9}{'mean b':>9}{'up/down':>10}{'p':>10}")
    for label, r in out.items():
        print(
            f"{label:<34}{r['n_subjects']:>4}{r['mean_a']:>9.3f}{r['mean_b']:>9.3f}"
            f"{str(r['n_b_greater']) + '/' + str(r['n_b_lower']):>10}{r['p_value']:>10.4g}"
        )


if __name__ == "__main__":
    main()
