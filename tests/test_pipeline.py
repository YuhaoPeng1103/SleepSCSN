"""End-to-end checks of the conformal procedures on synthetic data.

These are slower than the unit tests but they test the property that matters:
the calibrated procedure actually delivers its nominal coverage on held-out
data drawn from the calibration distribution.
"""

from __future__ import annotations

import numpy as np
import pytest

from sleepcp.conformal import (
    aps_score_matrix,
    block_mean_score,
    split_conformal_quantile,
    temporal_dp_margin_sets,
    temporal_dp_sets,
)
from sleepcp.metrics import cohen_kappa_score, f1_score


def _toy_probs(rng: np.random.Generator, n: int, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Probabilities and labels from a Dirichlet model with real signal."""
    alpha = np.ones(k) * 0.6
    probs = rng.dirichlet(alpha, size=n)
    # Sharpen towards a random "true" class for most epochs.
    labels = rng.integers(0, k, size=n)
    probs[np.arange(n), labels] += rng.uniform(0.5, 2.0, size=n)
    probs /= probs.sum(axis=1, keepdims=True)
    return probs, labels


def test_split_conformal_coverage_on_held_out() -> None:
    rng = np.random.default_rng(0)
    probs, labels = _toy_probs(rng, 4000)
    scores = aps_score_matrix(probs)
    true = scores[np.arange(labels.size), labels]
    q = split_conformal_quantile(true[:2000], 0.1)
    covered = true[2000:] <= q
    assert covered.mean() == pytest.approx(0.9, abs=0.02)


def test_margin_sets_deliver_marginal_coverage() -> None:
    """SeqCP-Margin must hit nominal coverage at the epoch level."""
    rng = np.random.default_rng(1)
    n_blocks, block = 800, 20
    probs, labels = _toy_probs(rng, n_blocks * block)
    deficit = 1.0 - probs
    transitions = 1.0 - np.full((5, 5), 0.2)

    def margin_block(b: int) -> np.ndarray:
        sl = slice(b * block, (b + 1) * block)
        result = temporal_dp_margin_sets(deficit[sl], transitions, delta=float("inf"), beta=0.0)
        return result.min_scores - np.min(result.min_scores)

    cal_idx = range(0, 400)
    cal_true = np.array(
        [margin_block(b)[np.arange(block), labels[b * block : (b + 1) * block]] for b in cal_idx]
    ).ravel()
    delta = split_conformal_quantile(cal_true, 0.1)

    covered = []
    for b in range(400, n_blocks):
        sl = slice(b * block, (b + 1) * block)
        result = temporal_dp_margin_sets(deficit[sl], transitions, delta=delta, beta=0.0)
        covered.append(result.sets[np.arange(block), labels[sl]])
    rate = np.concatenate(covered).mean()
    # Validity, not exact calibration: the DP margin is a discrete statistic
    # with a large atom at zero, so split CP over-covers rather than landing
    # exactly on 1 - alpha. Every reported coverage is likewise a lower bound.
    assert rate >= 0.9 - 1e-9
    assert rate < 1.0


def test_block_projection_covers_every_epoch_of_the_true_sequence() -> None:
    """Theorem 1 in miniature: if the true block is in the block set, every
    epoch is in its projected set, and the projection is never smaller than the
    per-epoch sets built at the same level."""
    rng = np.random.default_rng(2)
    n_blocks, block = 400, 20
    probs, labels = _toy_probs(rng, n_blocks * block)
    scores = aps_score_matrix(probs)
    transitions = 1.0 - np.full((5, 5), 0.2)

    cal_scores = [
        block_mean_score(scores[b * block : (b + 1) * block],
                         labels[b * block : (b + 1) * block], transitions, beta=1.0)
        for b in range(200)
    ]
    q = split_conformal_quantile(np.asarray(cal_scores), 0.1)
    true_scores = scores[np.arange(labels.size), labels]
    q_epoch = split_conformal_quantile(true_scores[: 200 * block], 0.1)

    sizes, epoch_sizes, feasible_blocks = [], [], 0
    for b in range(200, n_blocks):
        sl = slice(b * block, (b + 1) * block)
        block_sets = temporal_dp_sets(scores[sl], transitions, threshold=q, beta=1.0).sets
        covered = block_sets[np.arange(block), labels[sl]]
        in_set = block_mean_score(scores[sl], labels[sl], transitions, beta=1.0) <= q
        if in_set:
            feasible_blocks += 1
            assert covered.all(), "a feasible true block must cover every epoch"
        sizes.append(block_sets.mean())
        epoch_sizes.append((scores[sl] <= q_epoch).mean())
    assert feasible_blocks > 0
    assert np.mean(sizes) >= np.mean(epoch_sizes)


def test_margin_at_beta_zero_is_the_per_epoch_deficit() -> None:
    """With no transition cost the DP minimum factorises, so SeqCP-Margin
    must reduce exactly to per-epoch conformal prediction on the deficit
    score ``max_k gamma - gamma(y)``. This is the property the paper uses to
    attribute any gain at beta > 0 to the dynamic program itself."""
    rng = np.random.default_rng(3)
    T, K = 20, 5
    gamma = rng.dirichlet(np.ones(K) * 0.3, size=T)
    scores = 1.0 - gamma
    transitions = 1.0 - np.full((K, K), 0.2)  # irrelevant at beta = 0

    margins = temporal_dp_margin_sets(scores, transitions, delta=float("inf"), beta=0.0)
    deficit = scores - scores.min(axis=1, keepdims=True)
    # temporal_dp_sets reports the block mean, i.e. the total divided by T.
    assert np.allclose(margins.min_scores, deficit / T, atol=1e-12)


def test_position_centred_margins_never_exceed_global() -> None:
    rng = np.random.default_rng(4)
    T, K = 30, 5
    scores = 1.0 - rng.dirichlet(np.ones(K) * 0.5, size=T)
    A = np.full((K, K), 0.01)
    np.fill_diagonal(A, 0.96)
    transitions = 1.0 - A
    for beta in (0.5, 1.0, 2.0):
        g = temporal_dp_margin_sets(scores, transitions, float("inf"), beta=beta, center="global")
        p = temporal_dp_margin_sets(scores, transitions, float("inf"), beta=beta, center="position")
        assert np.all(p.min_scores <= g.min_scores + 1e-12)


def test_kappa_and_macro_f1_are_correct() -> None:
    y = np.array([0, 1, 2, 2, 3, 4, 0, 1])
    p = np.array([0, 1, 2, 3, 3, 4, 0, 0])
    # Accuracy 6/8, chance agreement 13/64, so kappa = 17/24 - 13/64 normalised.
    assert cohen_kappa_score(y, p) == pytest.approx(0.6862745098039216, abs=1e-9)
    scores = f1_score(y, p, average=None, labels=np.arange(5))
    assert scores.shape == (5,)
    assert f1_score(y, p, average="macro") == pytest.approx(scores.mean())
