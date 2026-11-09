"""Core conformal utilities for SleepCP.

The module implements:

* split conformal quantiles with the finite-sample correction;
* APS nonconformity scores for multi-class sleep staging;
* subject-aware quantile shrinkage;
* a dynamic-programming decoder for temporally consistent prediction sets.

All functions are NumPy-only so they can be unit-tested without PyTorch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def split_conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Return the split-conformal threshold for a target miscoverage ``alpha``.

    The threshold is the ``ceil((n + 1) * (1 - alpha)) / n`` empirical
    quantile. If the required quantile exceeds one, there are too few
    calibration points and ``inf`` is returned (a vacuous predictor).
    """
    scores = np.asarray(scores, dtype=float).ravel()
    n = scores.size
    if n == 0:
        return float("inf")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    if level > 1.0:
        return float("inf")
    return float(np.quantile(scores, level, method="higher"))


def threshold_at(sorted_scores: np.ndarray, alpha: float) -> float:
    """Split-conformal threshold from pre-sorted calibration scores.

    Identical to :func:`split_conformal_quantile` but avoids re-sorting when
    the threshold has to be recomputed many times, as in adaptive conformal
    inference.
    """
    n = sorted_scores.size
    if n == 0:
        return float("inf")
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    if level > 1.0:
        return float("inf")
    k = int(np.ceil(level * n)) - 1
    return float(sorted_scores[min(max(k, 0), n - 1)])


def run_aci(
    cal_scores_sorted: np.ndarray,
    test_scores: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    gamma0: float,
) -> np.ndarray:
    """Boolean prediction sets for one adaptive-conformal-inference stream.

    ``alpha_t`` is updated after every observed label by
    ``alpha_{t+1} = alpha_t + gamma0 / sqrt(t + 2) (alpha - err_t)``. The
    schedule satisfies no long-run guarantee in itself; it is the empirical
    construction studied in the paper, not the Bastani et al. design.
    """
    n = labels.size
    sets = np.zeros(test_scores.shape, dtype=bool)
    alpha_t = float(alpha)
    for t in range(n):
        q = threshold_at(cal_scores_sorted, alpha_t)
        sets[t] = test_scores[t] <= q
        miss = 0.0 if sets[t, labels[t]] else 1.0
        alpha_t = float(
            np.clip(alpha_t + gamma0 / np.sqrt(t + 2.0) * (alpha - miss), 0.001, 0.999)
        )
    return sets


def aps_score(probs: np.ndarray, label: int) -> float:
    """Adaptive prediction set (APS) score for one example.

    ``probs`` is a probability vector; lower scores are better. The score is
    the cumulative probability mass of all classes at least as likely as the
    true label.
    """
    probs = np.asarray(probs, dtype=float).ravel()
    if probs.size == 0:
        raise ValueError("probs must be non-empty")
    if not 0 <= label < probs.size:
        raise ValueError("label out of range")
    threshold = probs[label]
    return float(probs[probs >= threshold].sum())


def aps_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Vectorized APS scores for a batch of probability vectors."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int).ravel()
    if probs.ndim != 2 or probs.shape[0] != labels.size:
        raise ValueError("probs must be (n, k) and labels must be (n,)")
    out = np.empty(labels.size, dtype=float)
    for i, (row, y) in enumerate(zip(probs, labels)):
        out[i] = aps_score(row, int(y))
    return out


def aps_score_matrix(probs: np.ndarray) -> np.ndarray:
    """APS score of every candidate label for every example.

    Returns an ``(n, k)`` matrix where ``out[i, y]`` is the APS score of
    label ``y`` for example ``i``.
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2:
        raise ValueError("probs must be (n, k)")
    n, k = probs.shape
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumulative = np.cumsum(sorted_probs, axis=1)
    scores = np.empty((n, k), dtype=float)
    scores[np.arange(n)[:, None], order] = cumulative
    return scores


def raps_score_matrix(
    probs: np.ndarray,
    k_reg: int = 1,
    lam: float = 0.01,
) -> np.ndarray:
    """Regularised APS scores for every candidate label.

    Same as :func:`aps_score_matrix` but with a penalty ``lam`` for labels
    ranked below ``k_reg``, which discourages the long tail of low-probability
    labels from entering the set.
    """
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2:
        raise ValueError("probs must be (n, k)")
    n, k = probs.shape
    order = np.argsort(-probs, axis=1)
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumulative = np.cumsum(sorted_probs, axis=1)
    ranks = np.arange(1, k + 1, dtype=float)[None, :]
    sorted_scores = cumulative + lam * np.maximum(0.0, ranks - k_reg)
    scores = np.empty((n, k), dtype=float)
    scores[np.arange(n)[:, None], order] = sorted_scores
    return scores


def tune_raps(
    probs: np.ndarray,
    labels: np.ndarray,
    alpha: float,
    k_reg_grid: tuple[int, ...] = (1, 2, 3),
    lam_grid: tuple[float, ...] = (0.001, 0.01, 0.05, 0.1, 0.2),
    val_frac: float = 0.3,
    seed: int = 0,
) -> tuple[int, float]:
    """Choose RAPS hyperparameters on a held-out part of the calibration set.

    The calibration data are split in two; the first part sets the threshold
    and the second measures set size subject to coverage being at least
    ``1 - alpha``. The smallest average set wins.
    """
    labels = np.asarray(labels, dtype=int)
    n = labels.size
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_val = max(int(round(val_frac * n)), 1)
    val_idx, fit_idx = order[:n_val], order[n_val:]

    best = (1, 0.01)
    best_size = np.inf
    for k_reg in k_reg_grid:
        for lam in lam_grid:
            scores = raps_score_matrix(probs, k_reg=k_reg, lam=lam)
            true = scores[np.arange(n), labels]
            q = split_conformal_quantile(true[fit_idx], alpha)
            if not np.isfinite(q):
                continue
            sets = scores[val_idx] <= q
            cov = sets[np.arange(val_idx.size), labels[val_idx]].mean()
            if cov < 1.0 - alpha - 1e-6:
                continue
            size = float(sets.sum(axis=1).mean())
            if size < best_size:
                best_size, best = size, (k_reg, lam)
    return best


def shrink_quantile(
    local_quantile: float,
    global_quantile: float,
    shrinkage: float,
) -> float:
    """Convex combination of a local and a global conformal quantile.

    ``shrinkage=0`` uses the global quantile; ``shrinkage=1`` uses the local
    quantile. Values in between trade bias for variance.
    """
    if not 0.0 <= shrinkage <= 1.0:
        raise ValueError("shrinkage must be in [0, 1]")
    if not np.isfinite(local_quantile) or not np.isfinite(global_quantile):
        return float("inf")
    return float((1.0 - shrinkage) * global_quantile + shrinkage * local_quantile)


@dataclass(frozen=True)
class TemporalSetResult:
    """Prediction sets and the decoded path for one sequence block."""

    sets: np.ndarray  # (T, K) boolean
    path: np.ndarray  # (T,) int
    min_scores: np.ndarray  # (T, K) minimum total score through each state


def temporal_dp_sets(
    scores: np.ndarray,
    transitions: np.ndarray,
    threshold: float,
    beta: float = 1.0,
) -> TemporalSetResult:
    """Temporally consistent prediction sets via dynamic programming.

    Parameters
    ----------
    scores:
        ``(T, K)`` array of per-epoch, per-class nonconformity scores. Lower is
        better.
    transitions:
        ``(K, K)`` non-negative transition costs. ``transitions[a, b]`` is the
        cost of moving from class ``a`` to class ``b``.
    threshold:
        Calibrated upper bound on the *mean* sequence score.
    beta:
        Scale of the transition cost.

    Returns
    -------
    TemporalSetResult
        ``sets[t, y]`` is True when there exists a sequence through ``y`` at
        time ``t`` whose mean score is at most ``threshold``.

    Notes
    -----
    The sequence score is

        S(y_1:T) = mean_t scores[t, y_t]
                   + beta * mean_{t=2:T} transitions[y_{t-1}, y_t].

    The forward-backward DP computes the minimum score of any path through
    each ``(t, y)``; the set contains the labels whose minimum is within the
    threshold. This is the projection of the sequence-level conformal set.
    """
    scores = np.asarray(scores, dtype=float)
    transitions = np.asarray(transitions, dtype=float)
    if scores.ndim != 2:
        raise ValueError("scores must be (T, K)")
    T, K = scores.shape
    if transitions.shape != (K, K):
        raise ValueError("transitions must be (K, K)")
    if T == 0:
        return TemporalSetResult(
            sets=np.zeros((0, K), dtype=bool),
            path=np.zeros(0, dtype=int),
            min_scores=np.zeros((0, K), dtype=float),
        )

    # The block score averages point scores over T and transition costs over
    # T - 1. Rescale beta so that dividing the DP total by T gives exactly the
    # same block score as ``block_mean_score``.
    beta_eff = beta * (T / (T - 1)) if T > 1 else 0.0

    # Forward pass: V[t, y] = min cost of a path ending at y at time t.
    forward = np.empty((T, K), dtype=float)
    backptr = np.zeros((T, K), dtype=int)
    forward[0] = scores[0]
    for t in range(1, T):
        # candidate[a, b] = cost of ending at a at t-1 and moving to b at t
        candidate = forward[t - 1][:, None] + beta_eff * transitions
        backptr[t] = np.argmin(candidate, axis=0)
        forward[t] = scores[t] + candidate[backptr[t], np.arange(K)]

    # Backward pass: B[t, y] = min cost of the suffix after t, given state y
    # at t, excluding scores[t, y] itself. For t = T - 1 the suffix is the
    # final epoch and the transition into it.
    backward = np.zeros((T, K), dtype=float)
    if T > 1:
        candidate = beta_eff * transitions + scores[T - 1][None, :]
        backward[T - 2] = np.min(candidate, axis=1)
        for t in range(T - 3, -1, -1):
            candidate = (
                beta_eff * transitions
                + scores[t + 1][None, :]
                + backward[t + 1][None, :]
            )
            backward[t] = np.min(candidate, axis=1)

    # Minimum total score of a full path through (t, y). The score at (t, y)
    # appears in both forward and backward, so subtract it once.
    min_total = forward + backward
    mean_score = min_total / float(T)
    sets = mean_score <= threshold

    # Decode the best path by backtracking from the best final state.
    path = np.empty(T, dtype=int)
    path[-1] = int(np.argmin(forward[-1]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = int(backptr[t, path[t]])

    return TemporalSetResult(sets=sets, path=path, min_scores=mean_score)


def temporal_dp_margin_sets(
    scores: np.ndarray,
    transitions: np.ndarray,
    delta: float,
    beta: float = 1.0,
    center: str = "global",
) -> TemporalSetResult:
    """Viterbi-centered conformal sets with a calibrated margin.

    The DP first finds the minimum block score ``M_t(y)`` of any path through
    ``(t, y)``. The margin of ``y`` at ``t`` is how much a path through ``y``
    costs over a reference, and the set at ``t`` holds every label whose margin
    is at most ``delta``.

    ``center`` selects the reference. ``"global"`` subtracts the minimum over
    the whole block, ``M* = min_{t,y} M_t(y)``; ``"position"`` subtracts
    ``min_k M_t(k)``, comparing labels at the same epoch under the same
    surrounding context. Since ``M* <= min_k M_t(k)``, position-centred
    margins are pointwise no larger. The two coincide exactly whenever the
    block-optimal path is constant, because such a path visits every epoch;
    on sleep blocks this is the common case, and the two constructions
    therefore give the same sets. The option is kept so the claim can be
    checked rather than assumed.
    """
    result = temporal_dp_sets(scores, transitions, threshold=float("inf"), beta=beta)
    if center == "global":
        margins = result.min_scores - np.min(result.min_scores)
    elif center == "position":
        margins = result.min_scores - result.min_scores.min(axis=1, keepdims=True)
    else:
        raise ValueError(f"unknown center: {center!r}")
    return TemporalSetResult(
        sets=margins <= delta,
        path=result.path,
        min_scores=margins,
    )


def block_mean_score(
    scores: np.ndarray,
    labels: np.ndarray,
    transitions: np.ndarray,
    beta: float = 1.0,
) -> float:
    """Mean sequence score of a ground-truth label sequence."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int).ravel()
    transitions = np.asarray(transitions, dtype=float)
    if scores.ndim != 2 or scores.shape[0] != labels.size:
        raise ValueError("scores must be (T, K) and labels must be (T,)")
    T = labels.size
    if T == 0:
        return 0.0
    point = float(scores[np.arange(T), labels].mean())
    if T == 1:
        return point
    trans = float(transitions[labels[:-1], labels[1:]].mean())
    return point + beta * trans
