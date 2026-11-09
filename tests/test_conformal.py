import numpy as np

from sleepcp.conformal import (
    aps_score,
    aps_score_matrix,
    aps_scores,
    block_mean_score,
    shrink_quantile,
    split_conformal_quantile,
    temporal_dp_sets,
)


def test_split_conformal_quantile_is_conservative():
    scores = np.arange(1.0, 101.0)
    q = split_conformal_quantile(scores, alpha=0.1)
    assert q >= float(np.quantile(scores, 0.9))


def test_split_conformal_quantile_vacuous_when_too_few_points():
    q = split_conformal_quantile(np.arange(5.0), alpha=0.05)
    assert np.isinf(q)


def test_aps_score_counts_mass_at_least_as_likely():
    probs = np.array([0.2, 0.5, 0.3])
    assert np.isclose(aps_score(probs, 1), 0.5)
    assert np.isclose(aps_score(probs, 0), 1.0)
    assert np.isclose(aps_score(probs, 2), 0.8)


def test_aps_scores_vectorized_matches_loop():
    rng = np.random.default_rng(0)
    probs = rng.dirichlet(np.ones(5), size=20)
    labels = rng.integers(0, 5, size=20)
    expected = np.array([aps_score(p, y) for p, y in zip(probs, labels)])
    assert np.allclose(aps_scores(probs, labels), expected)


def test_aps_score_matrix_matches_per_label_scores():
    rng = np.random.default_rng(3)
    probs = rng.dirichlet(np.ones(5), size=20)
    matrix = aps_score_matrix(probs)
    expected = np.array([[aps_score(p, y) for y in range(5)] for p in probs])
    assert np.allclose(matrix, expected)


def test_shrink_quantile_interpolates():
    assert np.isclose(shrink_quantile(1.0, 3.0, 0.5), 2.0)
    assert np.isclose(shrink_quantile(1.0, 3.0, 0.0), 3.0)
    assert np.isclose(shrink_quantile(1.0, 3.0, 1.0), 1.0)


def test_temporal_dp_returns_full_sets_for_large_threshold():
    scores = np.array([[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    transitions = np.zeros((2, 2))
    result = temporal_dp_sets(scores, transitions, threshold=10.0)
    assert result.sets.all()
    assert result.path.shape == (3,)


def test_temporal_dp_returns_empty_sets_for_small_threshold():
    scores = np.array([[0.0, 1.0], [1.0, 0.0]])
    transitions = np.zeros((2, 2))
    result = temporal_dp_sets(scores, transitions, threshold=-1.0)
    assert not result.sets.any()


def test_temporal_dp_high_transition_cost_gives_constant_path():
    scores = np.array([[0.0, 10.0], [10.0, 0.0], [0.0, 10.0]])
    transitions = np.array([[0.0, 1.0], [1.0, 0.0]])
    result = temporal_dp_sets(scores, transitions, threshold=100.0, beta=100.0)
    assert np.unique(result.path).size == 1


def test_temporal_dp_projection_contains_true_path_when_score_passes():
    rng = np.random.default_rng(1)
    T, K = 12, 5
    scores = rng.random((T, K))
    transitions = rng.random((K, K))
    labels = rng.integers(0, K, size=T)
    score = block_mean_score(scores, labels, transitions, beta=0.5)
    result = temporal_dp_sets(scores, transitions, threshold=score, beta=0.5)
    # If the true sequence is within the threshold, every true label must be
    # in the projected set at its epoch.
    assert result.sets[np.arange(T), labels].all()


def test_temporal_dp_matches_brute_force_min_scores():
    import itertools

    rng = np.random.default_rng(2)
    T, K = 4, 3
    scores = rng.random((T, K))
    transitions = rng.random((K, K))
    beta = 0.7
    result = temporal_dp_sets(scores, transitions, threshold=1.0, beta=beta)

    best = np.full((T, K), np.inf)
    for path in itertools.product(range(K), repeat=T):
        path = np.array(path)
        value = block_mean_score(scores, path, transitions, beta=beta)
        for t, y in enumerate(path):
            best[t, y] = min(best[t, y], value)
    assert np.allclose(result.min_scores, best)
