"""Per-subject adaptive conformal inference (ACI) for sleep staging.

Split conformal gives a marginal guarantee: 90% of *all* epochs are covered.
Adaptive conformal inference (Gibbs & Candes) instead updates the miscoverage
target after every observation,

    alpha_{t+1} = alpha_t + gamma_t (alpha - 1{Y_t not in C_t}),

which drives the long-run miscoverage of a *stream* to alpha even when
exchangeability fails. Standard ACI runs one stream over all subjects, so the
guarantee is again pooled: a hard patient can be under-covered while an easy
one is over-covered. Here every subject is its own stream, with its own
alpha_i,t, so the long-run rate is controlled per patient.

Feedback assumption: ACI needs the true label after each prediction. In sleep
staging this is realistic -- a technologist reviews the automatic hypnogram and
the correction is available immediately -- but it is still an assumption, and
we say so.

The threshold is read from the sorted calibration scores at level 1 - alpha_t,
so no extra model fitting is needed per subject.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sleepcp.conformal import aps_score_matrix, split_conformal_quantile
from sleepcp.metrics import (
    coverage,
    empty_rate,
    per_class_coverage,
    review_workload,
    set_size,
    singleton_rate,
    temporal_consistency_grouped,
)
from sleepcp.temporal import forward_backward, initial_distribution, transition_matrix


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def recording_groups(recordings: np.ndarray) -> list[np.ndarray]:
    groups, start, n = [], 0, recordings.size
    while start < n:
        end = start + 1
        while end < n and recordings[end] == recordings[start]:
            end += 1
        groups.append(np.arange(start, end))
        start = end
    return groups


def smooth(probs: np.ndarray, groups: list[np.ndarray], trans: np.ndarray, init: np.ndarray) -> np.ndarray:
    out = np.empty_like(probs)
    for idx in groups:
        out[idx] = forward_backward(probs[idx], trans, init)
    return out


def threshold_at(sorted_scores: np.ndarray, alpha: float) -> float:
    """Split-conformal threshold for a (possibly non-nominal) level."""
    n = sorted_scores.size
    if n == 0:
        return float("inf")
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    if level >= 1.0:
        return float(sorted_scores[-1])
    if level <= 0.0:
        return float("-inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    k = min(max(k, 0), n - 1)
    return float(sorted_scores[k])


def step_size(t: int, gamma0: float, schedule: str, poly_power: float) -> float:
    """Step at epoch ``t`` (0-indexed).

    ``sqrt`` is gamma0 / sqrt(t + 2): sum_t gamma_t diverges but sum_t gamma_t^2
    does too, so it does not meet the square-summability condition under which
    the long-run ACI bound is proved. ``poly`` with power p in (0.5, 1] meets
    it, and p = 0.6 is the value reported in the robustness check.
    """
    if schedule == "poly":
        return gamma0 / (t + 2.0) ** poly_power
    return gamma0 / np.sqrt(t + 2.0)


def run_aci(
    cal_scores_sorted: np.ndarray,
    test_scores: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    gamma0: float,
    schedule: str = "sqrt",
    poly_power: float = 0.6,
) -> np.ndarray:
    """Boolean sets for one stream, updated after every observed label."""
    n = labels.size
    sets = np.zeros(test_scores.shape, dtype=bool)
    alpha_t = alpha
    for t in range(n):
        q = threshold_at(cal_scores_sorted, float(alpha_t))
        sets[t] = test_scores[t] <= q
        miss = 0.0 if sets[t, labels[t]] else 1.0
        step = step_size(t, gamma0, schedule, poly_power)
        alpha_t = float(np.clip(alpha_t + step * (alpha - miss), 0.001, 0.999))
    return sets


def summarize(
    sets: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    recordings: np.ndarray,
    alpha: float,
) -> dict:
    per_subject = {
        str(s): coverage(sets[subjects == s], labels[subjects == s]) for s in np.unique(subjects)
    }
    values = np.asarray(list(per_subject.values()), dtype=float)
    return {
        "coverage": coverage(sets, labels),
        "per_class_coverage": per_class_coverage(sets, labels).tolist(),
        "set_size": set_size(sets),
        "singleton_rate": singleton_rate(sets),
        "empty_rate": empty_rate(sets),
        "review_workload": review_workload(sets),
        "temporal_consistency": temporal_consistency_grouped(sets, recordings),
        "n_epochs": int(labels.size),
        "per_subject_coverage": per_subject,
        "mean_subject_coverage": float(values.mean()),
        "worst_subject_coverage": float(values.min()),
        "p10_subject_coverage": float(np.quantile(values, 0.1)),
        "std_subject_coverage": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "subject_violation_rate": float((values < 1.0 - alpha).mean()),
    }


def aci_all_subjects(
    cal_scores_sorted: np.ndarray,
    score_matrix: np.ndarray,
    data: dict,
    alpha: float,
    gamma0: float,
    min_per_subject: int,
    schedule: str = "sqrt",
    poly_power: float = 0.6,
) -> np.ndarray:
    """Run one ACI stream per subject; every stream starts at ``alpha``."""
    sets = np.zeros(score_matrix.shape, dtype=bool)
    for s in np.unique(data["subjects"]):
        idx = np.flatnonzero(data["subjects"] == s)
        if idx.size < min_per_subject:
            # Too short to adapt; the empty set counts as a miss, which is the
            # conservative choice and is never silent in the metrics.
            continue
        sets[idx] = run_aci(
            cal_scores_sorted, score_matrix[idx], data["labels"][idx],
            alpha, gamma0, schedule, poly_power,
        )
    return sets


def choose_gamma_loso(
    sc: dict,
    cal: dict,
    cal_subjects: list[str],
    alpha: float,
    grid: list[float],
    min_per_subject: int,
    criterion: str = "shortfall",
    schedule: str = "sqrt",
    poly_power: float = 0.6,
) -> dict:
    """Choose the ACI step size on calibration subjects, never on the test set.

    Each calibration subject is held out in turn: its epochs form the stream,
    the remaining calibration subjects supply the score pool for the
    threshold. The step size that best controls per-subject miscoverage on
    those held-out streams is then applied unchanged to the test subjects.

    Selection criterion (fixed before seeing any result): minimise the mean
    one-sided coverage shortfall ``max(0, (1 - alpha) - cov_i)``, i.e. how far
    the worst-off patients fall below nominal, averaging only over the
    subjects that are actually below nominal; ties go to the smaller mean set
    size. Mean shortfall is preferred over worst-case because a single very
    short stream can dominate a worst-case statistic. ``criterion="worst"``
    selects on the minimum held-out coverage instead, for a deployment that
    cares about the single worst patient; the default is what the paper
    reports.
    """
    # One held-out stream per eligible calibration subject, with the pool built
    # from the other calibration subjects. The pools do not depend on gamma0,
    # so they are built once.
    streams = []
    for s in cal_subjects:
        idx = np.flatnonzero(cal["subjects"] == s)
        if idx.size < min_per_subject:
            continue
        pool = np.sort(sc["cal_true"][np.flatnonzero(cal["subjects"] != s)])
        if pool.size == 0:
            continue
        streams.append((s, pool, sc["cal_matrix"][idx], cal["labels"][idx]))
    detail: dict = {}
    best = None
    for gamma0 in grid:
        shortfalls, sizes, coverages = [], [], []
        for _s, pool, matrix, labels_s in streams:
            sets = run_aci(pool, matrix, labels_s, alpha, gamma0, schedule, poly_power)
            cov = coverage(sets, labels_s)
            coverages.append(float(cov))
            shortfalls.append(max(0.0, (1.0 - alpha) - float(cov)))
            sizes.append(float(sets.sum(axis=1).mean()))
        if not coverages:
            continue
        mean_shortfall = float(np.mean(shortfalls))
        mean_size = float(np.mean(sizes))
        detail[f"{gamma0:g}"] = {
            "mean_shortfall": mean_shortfall,
            "mean_set_size": mean_size,
            "mean_coverage": float(np.mean(coverages)),
            "worst_coverage": float(np.min(coverages)),
            "n_subjects": len(coverages),
        }
        if criterion == "worst":
            key = (-float(np.min(coverages)), mean_size)
        else:
            key = (mean_shortfall, mean_size)
        if best is None or key < best[0]:
            best = (key, float(gamma0))
    if best is None:
        return {"selected": float(grid[0]), "detail": detail}
    return {"selected": best[1], "detail": detail}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument(
        "--gamma0",
        type=float,
        nargs="+",
        default=[0.005, 0.01, 0.02, 0.05, 0.1],
        help="ACI step scale; the update uses gamma0 / sqrt(t + 2).",
    )
    parser.add_argument("--min-per-subject", type=int, default=100)
    parser.add_argument(
        "--schedule",
        choices=["sqrt", "poly"],
        default="sqrt",
        help="Step schedule: gamma0/sqrt(t+2) (default) or "
             "gamma0/(t+2)**p with p=--poly-power. The default does not meet "
             "the square-summability condition; 'poly' with 0.5 < p <= 1 does.",
    )
    parser.add_argument("--poly-power", type=float, default=0.6)
    parser.add_argument(
        "--criterion",
        choices=["shortfall", "worst"],
        default="shortfall",
        help="LOSO selection objective: mean one-sided shortfall (default, "
             "the paper's protocol) or the minimum held-out coverage.",
    )
    args = parser.parse_args()

    cal_raw, test_raw = load_npz(args.cal), load_npz(args.test)
    cal = {
        "probs": cal_raw["probs"].astype(float),
        "labels": cal_raw["labels"].astype(int),
        "subjects": cal_raw["subjects"].astype(str),
        "recordings": cal_raw["recordings"].astype(str),
    }
    test = {
        "probs": test_raw["probs"].astype(float),
        "labels": test_raw["labels"].astype(int),
        "subjects": test_raw["subjects"].astype(str),
        "recordings": test_raw["recordings"].astype(str),
    }

    trans = transition_matrix(cal["labels"])
    init = initial_distribution(cal["labels"])
    cal_groups = recording_groups(cal["recordings"])
    test_groups = recording_groups(test["recordings"])
    cal_smooth = smooth(cal["probs"], cal_groups, trans, init)
    test_smooth = smooth(test["probs"], test_groups, trans, init)

    cal_rows = np.arange(cal["labels"].size)
    cal_aps = aps_score_matrix(cal["probs"])
    cal_deficit_matrix = cal_smooth.max(axis=1, keepdims=True) - cal_smooth
    test_deficit_matrix = test_smooth.max(axis=1, keepdims=True) - test_smooth
    scores = {
        "aps": {
            "cal_true": cal_aps[cal_rows, cal["labels"]],
            "cal_matrix": cal_aps,
            "test_matrix": aps_score_matrix(test["probs"]),
        },
        "deficit": {
            "cal_true": cal_deficit_matrix[cal_rows, cal["labels"]],
            "cal_matrix": cal_deficit_matrix,
            "test_matrix": test_deficit_matrix,
        },
    }

    results = {
        "alpha": args.alpha,
        "schedule": args.schedule,
        "poly_power": args.poly_power if args.schedule == "poly" else None,
        "transitions": trans.tolist(),
        "methods": {},
    }

    # Static global CP on the same score, for reference.
    for name, sc in scores.items():
        q = split_conformal_quantile(sc["cal_true"], args.alpha)
        sets = sc["test_matrix"] <= q
        results["methods"][f"global_{name}"] = summarize(
            sets, test["labels"], test["subjects"], test["recordings"], args.alpha
        )

    cal_subjects = sorted(set(cal["subjects"]))
    for name, sc in scores.items():
        order = np.sort(sc["cal_true"])
        gam = choose_gamma_loso(
            sc, cal, cal_subjects, args.alpha, args.gamma0,
            args.min_per_subject, args.criterion, args.schedule, args.poly_power,
        )
        results["selected_gamma0_" + name] = gam["selected"]
        results["gamma_loso_" + name] = gam["detail"]

        for gamma0 in args.gamma0:
            sets = aci_all_subjects(
                order, sc["test_matrix"], test, args.alpha, gamma0,
                args.min_per_subject, args.schedule, args.poly_power,
            )
            results["methods"][f"subject_aci_{name}_g{gamma0:g}"] = summarize(
                sets, test["labels"], test["subjects"], test["recordings"], args.alpha
            )

        # The deployable entry: gamma0 chosen without touching the test subjects.
        sets = aci_all_subjects(
            order, sc["test_matrix"], test, args.alpha, gam["selected"],
            args.min_per_subject, args.schedule, args.poly_power,
        )
        results["methods"][f"subject_aci_{name}"] = summarize(
            sets, test["labels"], test["subjects"], test["recordings"], args.alpha
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"{'method':<28}{'cov':>7}{'size':>7}{'sing':>7}{'cons':>7}{'worst':>7}{'p10':>7}{'viol':>7}")
    for key, m in results["methods"].items():
        print(
            f"{key:<28}{m['coverage']:>7.3f}{m['set_size']:>7.2f}{m['singleton_rate']:>7.3f}"
            f"{m['temporal_consistency']:>7.3f}{m['worst_subject_coverage']:>7.3f}"
            f"{m['p10_subject_coverage']:>7.3f}{m['subject_violation_rate']:>7.3f}"
        )


if __name__ == "__main__":
    main()
