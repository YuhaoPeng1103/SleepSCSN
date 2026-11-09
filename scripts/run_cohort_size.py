"""Coverage as a function of how many calibration subjects are available.

The pooled split-conformal guarantee depends on the calibration scores being
exchangeable with the test scores, not on the calibration cohort being large.
Increasing the number of calibration subjects reduces the *sampling* noise of
the threshold, but it cannot remove the between-subject heterogeneity that
makes a pooled threshold too tight for some patients and too loose for
others.  This script separates the two: it subsamples k calibration subjects,
recomputes the thresholds, and reports the marginal and worst-subject
coverage over the fixed test cohort.

Usage:
    python scripts/run_cohort_size.py --cal ..._cal.npz --test ..._test.npz \
        --out results/main/eegnet/fold0_seed0/cohort_size.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sleepcp.conformal import aps_score_matrix, split_conformal_quantile
from sleepcp.metrics import coverage
from sleepcp.temporal import forward_backward, initial_distribution, transition_matrix


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def groups(recordings: np.ndarray) -> list[np.ndarray]:
    out, start = [], 0
    while start < recordings.size:
        end = start + 1
        while end < recordings.size and recordings[end] == recordings[start]:
            end += 1
        out.append(np.arange(start, end))
        start = end
    return out


def smooth(probs: np.ndarray, recs: np.ndarray, trans: np.ndarray, init: np.ndarray) -> np.ndarray:
    out = np.empty_like(probs)
    for idx in groups(recs):
        out[idx] = forward_backward(probs[idx], trans, init)
    return out


def evaluate(sets: np.ndarray, labels: np.ndarray, subjects: np.ndarray, alpha: float) -> dict:
    per = {str(s): coverage(sets[subjects == s], labels[subjects == s]) for s in np.unique(subjects)}
    values = np.asarray(list(per.values()), dtype=float)
    return {
        "coverage": float(coverage(sets, labels)),
        "worst": float(values.min()),
        "p10": float(np.quantile(values, 0.1)),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "violation_rate": float((values < 1.0 - alpha).mean()),
        "per_subject": per,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16])
    args = parser.parse_args()

    cal_raw, test_raw = load_npz(args.cal), load_npz(args.test)
    cal = {k: cal_raw[k] for k in ("probs", "labels", "subjects", "recordings")}
    test = {k: test_raw[k] for k in ("probs", "labels", "subjects", "recordings")}
    cal["labels"] = cal["labels"].astype(int)
    test["labels"] = test["labels"].astype(int)
    cal["subjects"] = cal["subjects"].astype(str)
    test["subjects"] = test["subjects"].astype(str)

    trans = transition_matrix(cal["labels"])
    init = initial_distribution(cal["labels"])
    cal_smooth = smooth(cal["probs"].astype(float), cal["recordings"].astype(str), trans, init)
    test_smooth = smooth(test["probs"].astype(float), test["recordings"].astype(str), trans, init)

    rows = np.arange(cal["labels"].size)
    scores = {
        "aps": (
            aps_score_matrix(cal["probs"])[rows, cal["labels"]],
            aps_score_matrix(test["probs"]),
        ),
        "deficit": (
            (cal_smooth.max(axis=1) - cal_smooth[rows, cal["labels"]]),
            test_smooth.max(axis=1, keepdims=True) - test_smooth,
        ),
    }

    subjects = sorted(set(cal["subjects"].tolist()))
    ks = [k for k in args.k if k <= len(subjects)]
    if not ks:
        raise SystemExit(f"no k <= {len(subjects)} calibration subjects")

    rng = np.random.default_rng(args.seed)
    results = {"alpha": args.alpha, "n_cal_subjects": len(subjects),
               "n_test_subjects": int(len(np.unique(test["subjects"]))), "curves": {}}

    for name, (cal_true, test_matrix) in scores.items():
        curve = {}
        for k in ks:
            runs = []
            for _ in range(args.repeats):
                pick = rng.choice(len(subjects), size=k, replace=False)
                keep = np.isin(cal["subjects"], [subjects[i] for i in pick])
                q = split_conformal_quantile(cal_true[keep], args.alpha)
                sets = test_matrix <= q
                runs.append(evaluate(sets, test["labels"], test["subjects"], args.alpha))
            curve[str(k)] = {
                key: {
                    "mean": float(np.mean([r[key] for r in runs])),
                    "sd": float(np.std([r[key] for r in runs], ddof=1)) if len(runs) > 1 else 0.0,
                }
                for key in ("coverage", "worst", "p10", "std", "violation_rate")
            }
            print(
                f"{name} k={k:<3} coverage={curve[str(k)]['coverage']['mean']:.3f} "
                f"worst={curve[str(k)]['worst']['mean']:.3f} "
                f"p10={curve[str(k)]['p10']['mean']:.3f}"
            )
        results["curves"][name] = curve

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
