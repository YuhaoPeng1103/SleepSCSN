"""Dataset loading and subject-wise splitting for SleepCP."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

STAGE_NAMES = ("W", "N1", "N2", "N3", "REM")
NUM_STAGES = len(STAGE_NAMES)


@dataclass(frozen=True)
class Recording:
    """One preprocessed sleep recording."""

    x: np.ndarray  # (n_epochs, n_samples)
    y: np.ndarray  # (n_epochs,) in [0, 4], -1 means unlabeled
    subject: str
    recording: str
    sfreq: float
    epoch_sec: float
    channel: str

    def labeled(self) -> "Recording":
        mask = self.y >= 0
        return Recording(
            x=self.x[mask],
            y=self.y[mask],
            subject=self.subject,
            recording=self.recording,
            sfreq=self.sfreq,
            epoch_sec=self.epoch_sec,
            channel=self.channel,
        )


def load_recording(path: str | Path) -> Recording:
    """Load a preprocessed ``.npz`` recording."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        return Recording(
            x=data["x"].astype(np.float32),
            y=data["y"].astype(np.int64),
            subject=str(data["subject"]),
            recording=str(data["recording"]),
            sfreq=float(data["sfreq"]),
            epoch_sec=float(data["epoch_sec"]),
            channel=str(data["channel"]),
        )


def load_dataset(data_dir: str | Path) -> list[Recording]:
    """Load every ``.npz`` recording in a directory, sorted by name."""
    paths = sorted(Path(data_dir).glob("*.npz"))
    if not paths:
        raise FileNotFoundError(f"no .npz recordings found in {data_dir}")
    return [load_recording(p) for p in paths]


def concatenate(recordings: Iterable[Recording]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate recordings into arrays plus subject and recording ids."""
    xs, ys, subjects, rec_names = [], [], [], []
    for rec in recordings:
        xs.append(rec.x)
        ys.append(rec.y)
        subjects.extend([rec.subject] * len(rec.y))
        rec_names.extend([rec.recording] * len(rec.y))
    return (
        np.concatenate(xs, axis=0),
        np.concatenate(ys, axis=0),
        np.asarray(subjects),
        np.asarray(rec_names),
    )


def unique_subjects(recordings: Iterable[Recording]) -> list[str]:
    return sorted({rec.subject for rec in recordings})


def subject_wise_split(
    subjects: Iterable[str],
    n_folds: int = 5,
    seed: int = 0,
    cal_frac: float = 0.2,
    test_frac: float = 0.2,
) -> list[tuple[list[str], list[str], list[str]]]:
    """Return disjoint train/calibration/test subject splits.

    The subject list is shuffled once and split into ``n_folds`` blocks. Fold
    ``f`` tests on block ``f`` and calibrates on ``cal_frac`` of the remaining
    subjects, so the test sets partition the cohort: every subject is tested
    exactly once. Calibration subjects are rotated across folds, so a subject
    serves as calibration data in some folds and as training data in others.
    Within a fold the three sets are disjoint and their union is the full
    subject list.
    """
    subjects = np.asarray(sorted(set(subjects)))
    n = len(subjects)
    if n < 3:
        raise ValueError("need at least three subjects")
    if cal_frac <= 0 or test_frac <= 0 or cal_frac + test_frac >= 1:
        raise ValueError("cal_frac and test_frac must be positive and sum to < 1")

    rng = np.random.default_rng(seed)
    order = rng.permutation(n)

    # Each fold's test set is a block of the shuffled subject list, so every
    # subject is tested exactly once across folds. Calibration subjects are
    # drawn from the *other* folds and rotated across folds so that a subject
    # is not always excluded from training.
    test_folds = np.array_split(order, n_folds)
    n_cal = max(1, int(round(cal_frac * n)))
    if n_cal >= n - max(len(f) for f in test_folds):
        raise ValueError("not enough non-test subjects for calibration")

    splits = []
    for fi, test_idx in enumerate(test_folds):
        test_set = set(int(i) for i in test_idx)
        pool = [int(i) for i in order if int(i) not in test_set]
        shift = (fi * n_cal) % len(pool)
        pool = pool[shift:] + pool[:shift]
        cal_idx = pool[:n_cal]
        train_idx = pool[n_cal:]
        if not train_idx:
            raise ValueError("fold too small for train/cal/test split")
        splits.append(
            (
                sorted(str(subjects[i]) for i in train_idx),
                sorted(str(subjects[i]) for i in cal_idx),
                sorted(str(subjects[i]) for i in test_idx),
            )
        )
    return splits


def random_epoch_split(
    subjects: np.ndarray,
    seed: int = 0,
    cal_frac: float = 0.2,
    test_frac: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Epoch-level random split (the deliberately optimistic diagnostic).

    Returns boolean masks ``(train, cal, test)``. This split leaks subject
    identity between calibration and test and is used only to show why
    epoch-wise conformal calibration can look valid while failing on new
    subjects.
    """
    subjects = np.asarray(subjects)
    n = subjects.size
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_test = int(round(test_frac * n))
    n_cal = int(round(cal_frac * n))
    test_idx = order[:n_test]
    cal_idx = order[n_test : n_test + n_cal]
    train_idx = order[n_test + n_cal :]
    train = np.zeros(n, dtype=bool)
    cal = np.zeros(n, dtype=bool)
    test = np.zeros(n, dtype=bool)
    train[train_idx] = True
    cal[cal_idx] = True
    test[test_idx] = True
    return train, cal, test


def select_subjects(subjects: np.ndarray, keep: Iterable[str]) -> np.ndarray:
    """Boolean mask of the epochs whose subject is in ``keep``."""
    keep = set(keep)
    return np.asarray([s in keep for s in subjects], dtype=bool)
