"""Tests for the HMM smoothing utilities."""

from __future__ import annotations

import itertools

import numpy as np

from sleepcp.temporal import forward_backward, smooth_grouped, transition_matrix, viterbi


def _brute_force_posterior(probs, transitions, init):
    T, K = probs.shape
    logp = np.log(probs)
    logA = np.log(transitions)
    logpi = np.log(init)
    weights = []
    paths = []
    for path in itertools.product(range(K), repeat=T):
        path = np.asarray(path)
        lp = logpi[path[0]] + logp[0, path[0]]
        for t in range(1, T):
            lp += logA[path[t - 1], path[t]] + logp[t, path[t]]
        weights.append(lp)
        paths.append(path)
    weights = np.asarray(weights)
    weights -= weights.max()
    w = np.exp(weights)
    w /= w.sum()
    gamma = np.zeros((T, K))
    for path, p in zip(paths, w):
        gamma[np.arange(T), path] += p
    return gamma


def test_forward_backward_matches_enumeration():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(3), size=4)
    transitions = rng.dirichlet(np.ones(3) * 2.0, size=3)
    init = np.array([0.5, 0.3, 0.2])
    gamma = forward_backward(probs, transitions, init)
    expected = _brute_force_posterior(probs, transitions, init)
    assert np.allclose(gamma, expected, atol=1e-10)
    assert np.allclose(gamma.sum(axis=1), 1.0)


def test_forward_backward_sharpens_with_strong_transitions():
    probs = np.array([[0.4, 0.6], [0.45, 0.55], [0.45, 0.55]])
    transitions = np.array([[0.99, 0.01], [0.01, 0.99]])
    init = np.array([0.5, 0.5])
    gamma = forward_backward(probs, transitions, init)
    # The middle epoch borrows evidence from its neighbours.
    assert gamma[1, 1] > probs[1, 1]


def test_viterbi_matches_enumeration():
    rng = np.random.default_rng(1)
    probs = rng.dirichlet(np.ones(3), size=5)
    transitions = rng.dirichlet(np.ones(3) * 2.0, size=3)
    init = np.array([0.4, 0.4, 0.2])
    path = viterbi(probs, transitions, init)
    T, K = probs.shape
    best, best_lp = None, -np.inf
    logp, logA, logpi = np.log(probs), np.log(transitions), np.log(init)
    for candidate in itertools.product(range(K), repeat=T):
        candidate = np.asarray(candidate)
        lp = logpi[candidate[0]] + logp[0, candidate[0]]
        for t in range(1, T):
            lp += logA[candidate[t - 1], candidate[t]] + logp[t, candidate[t]]
        if lp > best_lp:
            best, best_lp = candidate, lp
    assert np.array_equal(path, best)


def test_transition_matrix_rows_sum_to_one():
    labels = np.array([0, 0, 1, 1, 2, 2, 2, 1, 0])
    A = transition_matrix(labels, n_classes=3, smoothing=1.0)
    assert A.shape == (3, 3)
    assert np.allclose(A.sum(axis=1), 1.0)
    assert A[0, 0] > A[0, 2]


def test_smooth_grouped_respects_boundaries():
    rng = np.random.default_rng(2)
    probs = rng.dirichlet(np.ones(3), size=6)
    transitions = np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
    groups = np.array([0, 0, 0, 1, 1, 1])
    out = smooth_grouped(probs, groups, transitions)
    # Each group is smoothed separately, matching a direct call.
    assert np.allclose(out[:3], forward_backward(probs[:3], transitions))
    assert np.allclose(out[3:], forward_backward(probs[3:], transitions))
