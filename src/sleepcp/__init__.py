"""SleepCP: conformal prediction utilities for sleep staging."""

from .conformal import (
    TemporalSetResult,
    aps_score,
    aps_score_matrix,
    aps_scores,
    block_mean_score,
    raps_score_matrix,
    shrink_quantile,
    split_conformal_quantile,
    temporal_dp_margin_sets,
    temporal_dp_sets,
    tune_raps,
)
from .temporal import (
    forward_backward,
    initial_distribution,
    smooth_grouped,
    transition_matrix,
    viterbi,
)

__all__ = [
    "TemporalSetResult",
    "aps_score",
    "aps_score_matrix",
    "aps_scores",
    "block_mean_score",
    "raps_score_matrix",
    "shrink_quantile",
    "split_conformal_quantile",
    "temporal_dp_margin_sets",
    "temporal_dp_sets",
    "forward_backward",
    "initial_distribution",
    "smooth_grouped",
    "transition_matrix",
    "viterbi",
]
