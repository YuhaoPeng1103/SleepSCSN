"""Evaluation metrics for SleepCP."""

from __future__ import annotations

import numpy as np


def cohen_kappa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cohen's kappa, computed from the confusion matrix.

    Implemented here rather than imported from scikit-learn so that the whole
    evaluation stack stays NumPy-only and can be tested with just NumPy.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    labels = np.union1d(y_true, y_pred)
    k = labels.size
    index = {int(v): i for i, v in enumerate(labels)}
    conf = np.zeros((k, k), dtype=float)
    np.add.at(conf, ([index[int(v)] for v in y_true], [index[int(v)] for v in y_pred]), 1.0)
    n = conf.sum()
    if n == 0:
        return float("nan")
    observed = np.trace(conf) / n
    expected = float((conf.sum(axis=0) * conf.sum(axis=1)).sum()) / (n * n)
    if abs(1.0 - expected) < 1e-12:
        return 1.0 if abs(observed - 1.0) < 1e-12 else 0.0
    return float((observed - expected) / (1.0 - expected))


def f1_score(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "macro", labels: np.ndarray | None = None
) -> float | np.ndarray:
    """Per-class or averaged F1, matching scikit-learn's default conventions.

    ``average=None`` returns the per-class array; ``average='macro'`` returns
    the unweighted mean, with absent classes scoring 0 (as in scikit-learn's
    default ``zero_division=0``).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if labels is None:
        labels = np.union1d(y_true, y_pred)
    labels = np.asarray(labels, dtype=int)
    scores = np.zeros(labels.size, dtype=float)
    for i, c in enumerate(labels):
        tp = float(((y_true == c) & (y_pred == c)).sum())
        fp = float(((y_true != c) & (y_pred == c)).sum())
        fn = float(((y_true == c) & (y_pred != c)).sum())
        denom = 2.0 * tp + fp + fn
        scores[i] = 2.0 * tp / denom if denom > 0 else 0.0
    if average is None:
        return scores
    if average == "macro":
        return float(scores.mean())
    raise ValueError(f"unsupported average: {average}")


def coverage(sets: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of examples whose true label is in the prediction set."""
    sets = np.asarray(sets, dtype=bool)
    labels = np.asarray(labels, dtype=int)
    return float(sets[np.arange(labels.size), labels].mean())


def per_class_coverage(sets: np.ndarray, labels: np.ndarray, n_classes: int = 5) -> np.ndarray:
    sets = np.asarray(sets, dtype=bool)
    labels = np.asarray(labels, dtype=int)
    out = np.full(n_classes, np.nan, dtype=float)
    for k in range(n_classes):
        mask = labels == k
        if mask.any():
            out[k] = sets[mask, k].mean()
    return out


def set_size(sets: np.ndarray) -> float:
    return float(np.asarray(sets, dtype=bool).sum(axis=1).mean())


def singleton_rate(sets: np.ndarray) -> float:
    return float((np.asarray(sets, dtype=bool).sum(axis=1) == 1).mean())


def empty_rate(sets: np.ndarray) -> float:
    return float((np.asarray(sets, dtype=bool).sum(axis=1) == 0).mean())


def review_workload(sets: np.ndarray) -> float:
    """Fraction of epochs that would need human review (set size > 1)."""
    return float((np.asarray(sets, dtype=bool).sum(axis=1) > 1).mean())


def temporal_consistency(sets: np.ndarray) -> float:
    """Mean Jaccard overlap between adjacent prediction sets.

    ``sets`` must be ordered by time. Higher is more consistent.
    """
    sets = np.asarray(sets, dtype=bool)
    if sets.shape[0] < 2:
        return float("nan")
    a = sets[:-1]
    b = sets[1:]
    inter = (a & b).sum(axis=1)
    union = (a | b).sum(axis=1)
    return float((inter / np.maximum(union, 1)).mean())


def temporal_consistency_grouped(sets: np.ndarray, groups: np.ndarray) -> float:
    """Adjacent-set Jaccard restricted to pairs inside the same group.

    ``groups`` marks the recording (or block) each epoch belongs to. Pairs that
    straddle a group boundary are excluded, so concatenating independent
    recordings does not fabricate inconsistency.
    """
    sets = np.asarray(sets, dtype=bool)
    groups = np.asarray(groups)
    if sets.shape[0] < 2:
        return float("nan")
    same = groups[:-1] == groups[1:]
    if not same.any():
        return float("nan")
    a = sets[:-1][same]
    b = sets[1:][same]
    inter = (a & b).sum(axis=1)
    union = (a | b).sum(axis=1)
    return float((inter / np.maximum(union, 1)).mean())


def sequence_coverage(sets: np.ndarray, labels: np.ndarray, boundaries: np.ndarray) -> float:
    """Fraction of sequences for which every epoch's label is in its set.

    ``boundaries`` is a boolean array marking the first epoch of each
    sequence; the last sequence runs to the end.
    """
    sets = np.asarray(sets, dtype=bool)
    labels = np.asarray(labels, dtype=int)
    boundaries = np.asarray(boundaries, dtype=bool)
    starts = np.flatnonzero(boundaries)
    if starts.size == 0 or starts[0] != 0:
        starts = np.concatenate([[0], starts])
    starts = np.unique(starts)
    hits = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < starts.size else labels.size
        hits.append(bool(sets[np.arange(start, end), labels[start:end]].all()))
    return float(np.mean(hits))


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Top-label expected calibration error."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (conf >= lo) & (conf < hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def classification_metrics(logits_or_probs: np.ndarray, labels: np.ndarray) -> dict:
    probs = np.asarray(logits_or_probs, dtype=float)
    if probs.ndim == 2 and not np.allclose(probs.sum(axis=1), 1.0, atol=1e-3):
        # Treat as logits.
        z = probs - probs.max(axis=1, keepdims=True)
        probs = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    pred = probs.argmax(axis=1)
    return {
        "accuracy": float((pred == labels).mean()),
        "kappa": float(cohen_kappa_score(labels, pred)),
        "mf1": float(f1_score(labels, pred, average="macro")),
        "per_class_f1": f1_score(labels, pred, average=None, labels=np.arange(probs.shape[1])).tolist(),
        "ece": expected_calibration_error(probs, labels),
    }
