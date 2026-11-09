"""Sensitivity of the pooled comparison to the nominal level.

Every experiment in the paper is run at ``alpha = 0.1``. This script reuses the
fitted models --- the stored per-epoch probabilities --- to recompute the pooled
APS and pooled smoothed-deficit results at ``alpha`` in {0.05, 0.1, 0.2}, so
that the claim "the ordering of the scores does not depend on the level" can
be checked rather than assumed. Nothing is retrained and no test label is used
for calibration.

Usage::

    python scripts/alpha_sweep.py --root results/main \
        --out results/main/summary/alpha_sweep.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sleepcp.conformal import aps_score_matrix, split_conformal_quantile
from sleepcp.temporal import initial_distribution, smooth_grouped, transition_matrix


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def per_subject(sets: np.ndarray, labels: np.ndarray, subjects: np.ndarray,
                alpha: float) -> dict:
    by_subject = {}
    for s in np.unique(subjects):
        mask = subjects == s
        by_subject[str(s)] = float(sets[mask, labels[mask]].mean())
    return {
        "coverage": float(sets[np.arange(labels.size), labels].mean()),
        "set_size": float(sets.sum(axis=1).mean()),
        # Emitted so that ``main`` can merge one value per subject across runs
        # exactly as ``scripts/aggregate.py`` does; the summary fields below are
        # computed per run and are not what the paper reports.
        "per_subject_coverage": by_subject,
    }


def merge_runs(values: list[dict[str, float]]) -> dict[str, float]:
    """One value per subject, averaged over every run that contains it."""
    total: dict[str, float] = {}
    count: dict[str, int] = {}
    for rec in values:
        for subj, value in rec.items():
            total[subj] = total.get(subj, 0.0) + float(value)
            count[subj] = count.get(subj, 0) + 1
    return {subj: total[subj] / count[subj] for subj in total}


def run_one(cal_path: Path, test_path: Path, alphas: list[float]) -> dict:
    cal = load_npz(cal_path)
    test = load_npz(test_path)
    cal_labels = cal["labels"].astype(int)
    test_labels = test["labels"].astype(int)
    test_subjects = test["subjects"].astype(str)

    aps_cal = aps_score_matrix(cal["probs"].astype(float))
    aps_test = aps_score_matrix(test["probs"].astype(float))

    trans = transition_matrix(cal_labels, smoothing=1.0)
    init = initial_distribution(cal_labels)
    cal_sm = smooth_grouped(cal["probs"].astype(float),
                            cal["recordings"].astype(str), trans, init)
    test_sm = smooth_grouped(test["probs"].astype(float),
                             test["recordings"].astype(str), trans, init)

    def deficit(probs: np.ndarray) -> np.ndarray:
        return probs.max(axis=1, keepdims=True) - probs

    out = {}
    for alpha in alphas:
        for name, cal_scores, test_scores in (
            ("aps", aps_cal, aps_test),
            ("deficit", deficit(cal_sm), deficit(test_sm)),
        ):
            q = split_conformal_quantile(
                cal_scores[np.arange(cal_labels.size), cal_labels], alpha
            )
            rec = per_subject(test_scores <= q, test_labels, test_subjects, alpha)
            out[f"{name}_a{alpha:g}"] = rec
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["eegnet", "sleepcnn"])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.2])
    args = parser.parse_args()

    per_run = {}
    for model in args.models:
        for run_dir in sorted((args.root / model).glob("fold*_seed*")):
            cal = next(run_dir.glob("*_cal.npz"), None)
            test = next(run_dir.glob("*_test.npz"), None)
            if cal is None or test is None:
                continue
            print(f"== {model}/{run_dir.name} ==", flush=True)
            per_run[f"{model}/{run_dir.name}"] = run_one(cal, test, args.alphas)

    table = {}
    for key in sorted(next(iter(per_run.values()), {})):
        vals = [r[key] for r in per_run.values()]
        alpha = float(key.rsplit("_a", 1)[1])
        merged = merge_runs([v["per_subject_coverage"] for v in vals])
        subjects = np.asarray(list(merged.values()), dtype=float)
        table[key] = {}
        for field in ("coverage", "set_size"):
            arr = np.asarray([v[field] for v in vals], dtype=float)
            table[key][field] = {
                "mean": float(arr.mean()),
                "sd": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            }
        # Worst-patient, dispersion and violation rate use the merged vector
        # (each subject once) so they match ``scripts/aggregate.py``.
        table[key]["worst_subject_coverage"] = {
            "mean": float(subjects.min()), "sd": 0.0,
        }
        table[key]["std_subject_coverage"] = {
            "mean": float(subjects.std(ddof=1)) if subjects.size > 1 else 0.0,
            "sd": 0.0,
        }
        table[key]["subject_violation_rate"] = {
            "mean": float((subjects < 1.0 - alpha).mean()), "sd": 0.0,
        }
        table[key]["n_subjects"] = int(subjects.size)

    out = {"alphas": args.alphas, "n_runs": len(per_run), "table": table}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(f"\n{'alpha':>6} {'score':>8} {'coverage':>9} {'worst':>7} {'size':>6}")
    for alpha in args.alphas:
        for name in ("aps", "deficit"):
            r = table.get(f"{name}_a{alpha:g}")
            if not r:
                continue
            print(f"{alpha:>6.2f} {name:>8} {r['coverage']['mean']:>9.3f} "
                  f"{r['worst_subject_coverage']['mean']:>7.3f} "
                  f"{r['set_size']['mean']:>6.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
