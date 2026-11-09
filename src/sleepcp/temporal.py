"""Temporal smoothing utilities for sleep staging.

Sleep stages form a slowly varying sequence, so neighbouring epochs carry
information about the current one.  This module implements the forward-backward
posterior and Viterbi decoding for a first-order hidden Markov model whose
emission probabilities are the neural network's per-epoch softmax outputs and
whose transition matrix is estimated from the training labels.

The smoothed posterior is what SeqCP uses as its scoring distribution: APS
scores computed on ``gamma`` are both better behaved (fewer near-one scores)
and temporally coherent, which yields smaller and more stable prediction sets
than scoring the raw softmax.
"""

from __future__ import annotations

import numpy as np


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def transition_matrix(
    labels: np.ndarray,
    n_classes: int = 5,
    smoothing: float = 1.0,
) -> np.ndarray:
    """Row-stochastic transition matrix estimated from a label sequence."""
    labels = np.asarray(labels, dtype=int).ravel()
    counts = np.full((n_classes, n_classes), float(smoothing))
    if labels.size > 1:
        valid = (labels[:-1] >= 0) & (labels[1:] >= 0)
        np.add.at(counts, (labels[:-1][valid], labels[1:][valid]), 1.0)
    return counts / counts.sum(axis=1, keepdims=True)


def initial_distribution(labels: np.ndarray, n_classes: int = 5) -> np.ndarray:
    counts = np.bincount(np.asarray(labels, dtype=int).ravel(), minlength=n_classes).astype(float)
    counts = np.maximum(counts, 1e-3)
    return counts / counts.sum()


def forward_backward(
    probs: np.ndarray,
    transitions: np.ndarray,
    init: np.ndarray | None = None,
) -> np.ndarray:
    """Smoothed posteriors ``gamma[t, k] = P(Y_t = k | X_1:T)``.

    Parameters
    ----------
    probs:
        ``(T, K)`` emission probabilities, rows sum to one.
    transitions:
        ``(K, K)`` row-stochastic transition matrix.
    init:
        Optional ``(K,)`` initial state distribution.
    """
    probs = np.asarray(probs, dtype=float)
    transitions = np.asarray(transitions, dtype=float)
    if probs.ndim != 2:
        raise ValueError("probs must be (T, K)")
    T, K = probs.shape
    if transitions.shape != (K, K):
        raise ValueError("transitions must be (K, K)")
    if T == 0:
        return np.zeros((0, K), dtype=float)

    logp = np.log(np.clip(probs, 1e-12, 1.0))
    logA = np.log(np.clip(transitions, 1e-12, 1.0))
    if init is None:
        logpi = np.full(K, -np.log(K))
    else:
        logpi = np.log(np.clip(np.asarray(init, dtype=float), 1e-12, 1.0))

    alpha = np.empty((T, K), dtype=float)
    alpha[0] = logpi + logp[0]
    for t in range(1, T):
        alpha[t] = logp[t] + _logsumexp(alpha[t - 1][:, None] + logA, axis=0)

    beta = np.zeros((T, K), dtype=float)
    for t in range(T - 2, -1, -1):
        beta[t] = _logsumexp(logA + (logp[t + 1] + beta[t + 1])[None, :], axis=1)

    gamma = alpha + beta
    gamma -= np.max(gamma, axis=1, keepdims=True)
    gamma = np.exp(gamma)
    gamma /= gamma.sum(axis=1, keepdims=True)
    return gamma


def viterbi(
    probs: np.ndarray,
    transitions: np.ndarray,
    init: np.ndarray | None = None,
) -> np.ndarray:
    """Most likely state sequence under the same HMM."""
    probs = np.asarray(probs, dtype=float)
    transitions = np.asarray(transitions, dtype=float)
    T, K = probs.shape
    if T == 0:
        return np.zeros(0, dtype=int)
    logp = np.log(np.clip(probs, 1e-12, 1.0))
    logA = np.log(np.clip(transitions, 1e-12, 1.0))
    logpi = np.full(K, -np.log(K)) if init is None else np.log(np.clip(init, 1e-12, 1.0))

    delta = np.empty((T, K), dtype=float)
    back = np.zeros((T, K), dtype=int)
    delta[0] = logpi + logp[0]
    for t in range(1, T):
        candidate = delta[t - 1][:, None] + logA
        back[t] = np.argmax(candidate, axis=0)
        delta[t] = logp[t] + candidate[back[t], np.arange(K)]
    path = np.empty(T, dtype=int)
    path[-1] = int(np.argmax(delta[-1]))
    for t in range(T - 1, 0, -1):
        path[t - 1] = int(back[t, path[t]])
    return path


def smooth_grouped(
    probs: np.ndarray,
    groups: np.ndarray,
    transitions: np.ndarray,
    init: np.ndarray | None = None,
) -> np.ndarray:
    """Apply :func:`forward_backward` independently within each group.

    ``groups`` is an integer or string array; contiguous runs with the same
    value are treated as one sequence (one recording).
    """
    probs = np.asarray(probs, dtype=float)
    groups = np.asarray(groups)
    out = np.empty_like(probs)
    start = 0
    n = groups.size
    while start < n:
        end = start + 1
        while end < n and groups[end] == groups[start]:
            end += 1
        out[start:end] = forward_backward(probs[start:end], transitions, init)
        start = end
    return out
