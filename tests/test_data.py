"""Tests for the subject-wise splitting logic."""

from __future__ import annotations

import numpy as np

from sleepcp.data import subject_wise_split


def test_splits_are_disjoint_and_cover_cohort():
    subjects = [f"S{i:02d}" for i in range(20)]
    splits = subject_wise_split(subjects, n_folds=5, seed=0, cal_frac=0.2)
    assert len(splits) == 5
    tested = []
    for train, cal, test in splits:
        assert not set(train) & set(cal)
        assert not set(train) & set(test)
        assert not set(cal) & set(test)
        assert len(set(train) | set(cal) | set(test)) == len(subjects)
        tested.extend(test)
    assert sorted(tested) == sorted(subjects)


def test_each_fold_has_all_three_roles():
    subjects = [f"S{i:02d}" for i in range(20)]
    for train, cal, test in subject_wise_split(subjects, n_folds=5, seed=3):
        assert len(test) == 4
        assert len(cal) == 4
        assert len(train) == 12


def test_split_is_deterministic():
    subjects = [f"S{i:02d}" for i in range(20)]
    a = subject_wise_split(subjects, n_folds=5, seed=7)
    b = subject_wise_split(subjects, n_folds=5, seed=7)
    assert a == b
