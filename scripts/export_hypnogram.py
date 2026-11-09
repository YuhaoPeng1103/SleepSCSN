"""Export per-epoch prediction sets for one subject, for the hypnogram figure.

The figure needs the actual set at every epoch, not the aggregate statistics
that ``run_sacp.py`` and ``run_subject_aci.py`` write.  This script recomputes
the four constructions the paper compares, all on the same test subject and
the same calibration data:

``global_aps``
    APS on the raw softmax, pooled threshold.
``smooth_aps``
    APS on the forward--backward posterior, pooled threshold.
``deficit``
    Posterior deficit ``max_k gamma - gamma(y)``, pooled threshold.
``aci``
    Posterior deficit with one ACI stream for this subject, starting from the
    pooled threshold.  ``gamma0`` is read from the ``subject_aci.json`` of the
    same run (its LOSO-selected value) unless ``--gamma0`` is given.

Usage:
    python scripts/export_hypnogram.py \
        --cal ..._cal.npz --test ..._test.npz --aci-json .../subject_aci.json \
        --subject SC414 --out paper/figures/hypnogram_SC414.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sleepcp.conformal import aps_score_matrix, split_conformal_quantile
from sleepcp.temporal import (
    forward_backward,
    initial_distribution,
    transition_matrix,
)


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


def threshold_at(sorted_scores: np.ndarray, alpha: float) -> float:
    n = sorted_scores.size
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    k = min(max(k, 0), n - 1)
    return float(sorted_scores[k])


def aci_sets(
    cal_scores_sorted: np.ndarray,
    score_matrix: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    gamma0: float,
) -> np.ndarray:
    sets = np.zeros(score_matrix.shape, dtype=bool)
    alpha_t = alpha
    for t in range(labels.size):
        q = threshold_at(cal_scores_sorted, float(alpha_t))
        sets[t] = score_matrix[t] <= q
        miss = 0.0 if sets[t, labels[t]] else 1.0
        alpha_t = float(np.clip(alpha_t + gamma0 / np.sqrt(t + 2.0) * (alpha - miss), 0.001, 0.999))
    return sets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--aci-json", required=True, type=Path)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--gamma0", type=float, default=None)
    parser.add_argument("--start", type=int, default=None, help="first epoch of the window")
    parser.add_argument("--window", type=int, default=180)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    cal_raw, test_raw = load_npz(args.cal), load_npz(args.test)
    cal_probs = cal_raw["probs"].astype(float)
    test_probs = test_raw["probs"].astype(float)
    cal_labels = cal_raw["labels"].astype(int)
    test_labels = test_raw["labels"].astype(int)
    test_subjects = test_raw["subjects"].astype(str)
    cal_recs = cal_raw["recordings"].astype(str)
    test_recs = test_raw["recordings"].astype(str)

    alpha = 0.1
    if args.aci_json.exists():
        with open(args.aci_json, encoding="utf-8") as f:
            aci = json.load(f)
        alpha = float(aci.get("alpha", alpha))
        # The LOSO-selected value is the deployable one; the JSON key is
        # "selected_gamma0_<score>".
        gamma = args.gamma0 if args.gamma0 is not None else aci.get("selected_gamma0_deficit")
    else:
        gamma = args.gamma0
    if gamma is None:
        raise SystemExit("no gamma0: pass --gamma0 or a subject_aci.json with a selected value")

    trans = transition_matrix(cal_labels)
    init = initial_distribution(cal_labels)
    cal_smooth = smooth(cal_probs, cal_recs, trans, init)
    test_smooth = smooth(test_probs, test_recs, trans, init)

    cal_aps = aps_score_matrix(cal_probs)
    test_aps = aps_score_matrix(test_probs)
    cal_aps_s = aps_score_matrix(cal_smooth)
    test_aps_s = aps_score_matrix(test_smooth)
    cal_def = cal_smooth.max(axis=1, keepdims=True) - cal_smooth
    test_def = test_smooth.max(axis=1, keepdims=True) - test_smooth

    rows = np.arange(cal_labels.size)
    q_aps = split_conformal_quantile(cal_aps[rows, cal_labels], alpha)
    q_aps_s = split_conformal_quantile(cal_aps_s[rows, cal_labels], alpha)
    cal_true_def = cal_def[rows, cal_labels]
    q_def = split_conformal_quantile(cal_true_def, alpha)

    idx = np.flatnonzero(test_subjects == args.subject)
    if idx.size == 0:
        raise SystemExit(f"subject {args.subject} not present in the test split")

    # The stream must run over the subject's own epochs in order, which is what
    # --subject selects. A subject with two recordings still forms one stream.
    aci = aci_sets(np.sort(cal_true_def), test_def[idx], test_labels[idx], alpha, float(gamma))

    sets = {
        "global_aps": test_aps[idx] <= q_aps,
        "smooth_aps": test_aps_s[idx] <= q_aps_s,
        "deficit": test_def[idx] <= q_def,
        "aci": aci,
    }
    subject_labels = test_labels[idx]
    subject_recs = test_recs[idx]

    if args.start is None:
        # Choose the window that contains the most epochs at which the pooled
        # APS set misses the truth, so the figure shows the failure it is
        # about rather than a quiet stretch of N2.
        miss = ~sets["global_aps"][np.arange(subject_labels.size), subject_labels]
        if miss.any():
            best, best_count = 0, -1
            for s in range(0, max(1, subject_labels.size - args.window), 5):
                count = int(miss[s : s + args.window].sum())
                if count > best_count:
                    best, best_count = s, count
            start = best
        else:
            start = 0
    else:
        start = args.start
    lo = max(0, min(start, max(0, subject_labels.size - 1)))
    hi = min(subject_labels.size, lo + args.window)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        labels=subject_labels[lo:hi],
        recordings=subject_recs[lo:hi],
        start=np.array(lo),
        windowed=np.array(True),
        alpha=np.array(alpha),
        gamma0=np.array(float(gamma)),
        q_global=np.array(q_aps),
        q_smooth_aps=np.array(q_aps_s),
        q_deficit=np.array(q_def),
        **{k: v[lo:hi] for k, v in sets.items()},
    )
    for k, v in sets.items():
        cov = v[np.arange(subject_labels.size), subject_labels].mean()
        print(f"{args.subject} {k:<12} size={v.sum(axis=1).mean():.2f}  coverage={cov:.3f}")
    print(f"window [{lo}, {hi}) of {subject_labels.size} epochs -> {args.out}")


if __name__ == "__main__":
    main()
