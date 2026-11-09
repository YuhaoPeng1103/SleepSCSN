"""Diagnose conformal coverage under random-epoch vs subject-wise calibration.

This script is the first go/no-go experiment. It loads the calibration and test
probabilities produced by ``scripts/train_eegnet.py`` and compares:

* random-epoch calibration (optimistic, leaks subject identity);
* global split CP (calibration and test subjects disjoint);
* class-conditional (Mondrian) CP;
* subject-neighborhood CP with shrinkage;
* a per-subject oracle (not deployable; upper bound).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sleepcp.conformal import aps_score_matrix, shrink_quantile, split_conformal_quantile
from sleepcp.metrics import (
    coverage,
    empty_rate,
    per_class_coverage,
    review_workload,
    set_size,
    singleton_rate,
)


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def subject_features(probs: np.ndarray) -> np.ndarray:
    """Unlabeled subject descriptor used for neighborhood matching."""
    mean_probs = probs.mean(axis=0)
    mean_max = probs.max(axis=1).mean()
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1).mean()
    return np.concatenate([mean_probs, [mean_max, entropy]])


def summarize(sets: np.ndarray, labels: np.ndarray, subjects: np.ndarray) -> dict:
    per_subject = {}
    for s in np.unique(subjects):
        mask = subjects == s
        per_subject[str(s)] = coverage(sets[mask], labels[mask])
    values = np.asarray(list(per_subject.values()), dtype=float)
    return {
        "coverage": coverage(sets, labels),
        "per_class_coverage": per_class_coverage(sets, labels).tolist(),
        "set_size": set_size(sets),
        "singleton_rate": singleton_rate(sets),
        "empty_rate": empty_rate(sets),
        "review_workload": review_workload(sets),
        "per_subject_coverage": per_subject,
        "mean_subject_coverage": float(values.mean()),
        "worst_subject_coverage": float(values.min()),
        "p10_subject_coverage": float(np.quantile(values, 0.1)),
    }


def sets_from_threshold(scores: np.ndarray, threshold: float | np.ndarray) -> np.ndarray:
    return np.asarray(scores) <= np.asarray(threshold)


def global_cp(cal: dict, test: dict, alpha: float) -> tuple[np.ndarray, float]:
    q = split_conformal_quantile(cal["true_scores"], alpha)
    return sets_from_threshold(test["score_matrix"], q), q


def class_conditional_cp(cal: dict, test: dict, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    n_classes = cal["score_matrix"].shape[1]
    q = np.array(
        [
            split_conformal_quantile(cal["true_scores"][cal["labels"] == k], alpha)
            for k in range(n_classes)
        ]
    )
    return sets_from_threshold(test["score_matrix"], q[None, :]), q


def neighborhood_cp(
    cal: dict,
    test: dict,
    alpha: float,
    k: int,
    shrinkage: float,
    global_q: float,
) -> tuple[np.ndarray, np.ndarray]:
    cal_subjects = np.unique(cal["subjects"])
    cal_features = np.stack(
        [subject_features(cal["probs"][cal["subjects"] == s]) for s in cal_subjects]
    )
    thresholds = np.empty(test["labels"].size, dtype=float)
    for s in np.unique(test["subjects"]):
        mask = test["subjects"] == s
        feat = subject_features(test["probs"][mask])
        dist = np.linalg.norm(cal_features - feat[None, :], axis=1)
        neighbors = cal_subjects[np.argsort(dist)[: min(k, cal_subjects.size)]]
        neighbor_mask = np.isin(cal["subjects"], neighbors)
        local_q = split_conformal_quantile(cal["true_scores"][neighbor_mask], alpha)
        thresholds[mask] = shrink_quantile(local_q, global_q, shrinkage)
    return sets_from_threshold(test["score_matrix"], thresholds[:, None]), thresholds


def oracle_subject_cp(test: dict, alpha: float) -> np.ndarray:
    thresholds = np.empty(test["labels"].size, dtype=float)
    for s in np.unique(test["subjects"]):
        mask = test["subjects"] == s
        thresholds[mask] = split_conformal_quantile(test["true_scores"][mask], alpha)
    return sets_from_threshold(test["score_matrix"], thresholds[:, None])


def random_epoch_cp(test: dict, alpha: float, n_repeats: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = test["labels"].size
    coverages, sizes = [], []
    for _ in range(n_repeats):
        order = rng.permutation(n)
        half = n // 2
        cal_idx, test_idx = order[:half], order[half:]
        q = split_conformal_quantile(test["true_scores"][cal_idx], alpha)
        sets = test["score_matrix"][test_idx] <= q
        coverages.append(coverage(sets, test["labels"][test_idx]))
        sizes.append(set_size(sets))
    return {
        "coverage_mean": float(np.mean(coverages)),
        "coverage_std": float(np.std(coverages)),
        "set_size_mean": float(np.mean(sizes)),
        "n_repeats": n_repeats,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--n-repeats", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, nargs="+", default=[3, 5, 10])
    parser.add_argument("--shrinkage", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    args = parser.parse_args()

    cal_raw = load_npz(args.cal)
    test_raw = load_npz(args.test)
    cal = {
        "probs": cal_raw["probs"],
        "labels": cal_raw["labels"].astype(int),
        "subjects": cal_raw["subjects"].astype(str),
    }
    test = {
        "probs": test_raw["probs"],
        "labels": test_raw["labels"].astype(int),
        "subjects": test_raw["subjects"].astype(str),
    }
    cal["score_matrix"] = aps_score_matrix(cal["probs"])
    test["score_matrix"] = aps_score_matrix(test["probs"])
    cal["true_scores"] = cal["score_matrix"][np.arange(cal["labels"].size), cal["labels"]]
    test["true_scores"] = test["score_matrix"][np.arange(test["labels"].size), test["labels"]]

    results = {
        "alpha": args.alpha,
        "n_cal": int(cal["labels"].size),
        "n_test": int(test["labels"].size),
        "cal_subjects": sorted(set(cal["subjects"].tolist())),
        "test_subjects": sorted(set(test["subjects"].tolist())),
        "methods": {},
    }

    results["methods"]["random_epoch"] = random_epoch_cp(
        test, args.alpha, args.n_repeats, args.seed
    )

    sets, q = global_cp(cal, test, args.alpha)
    results["methods"]["global"] = {"q": q, **summarize(sets, test["labels"], test["subjects"])}

    sets, q = class_conditional_cp(cal, test, args.alpha)
    results["methods"]["class_conditional"] = {
        "q": q.tolist(),
        **summarize(sets, test["labels"], test["subjects"]),
    }

    global_q = results["methods"]["global"]["q"]
    for k in args.k:
        for lam in args.shrinkage:
            name = f"neighborhood_k{k}_shrink{lam:g}"
            sets, q = neighborhood_cp(cal, test, args.alpha, k, lam, global_q)
            results["methods"][name] = {
                "q_mean": float(np.mean(q)),
                "q_min": float(np.min(q)),
                "q_max": float(np.max(q)),
                **summarize(sets, test["labels"], test["subjects"]),
            }

    sets = oracle_subject_cp(test, args.alpha)
    results["methods"]["oracle_subject"] = summarize(sets, test["labels"], test["subjects"])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
