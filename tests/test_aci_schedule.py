"""Step schedules for per-patient ACI.

The default schedule gamma0/sqrt(t+2) has sum gamma = infinity but
sum gamma^2 = infinity as well, so it sits outside the square-summability
condition under which the long-run ACI bound is proved. The power schedule
gamma0/(t+2)^p meets it for 0.5 < p <= 1. These tests pin that distinction so
the robustness check cannot silently be run with the wrong exponent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "run_subject_aci", Path(__file__).resolve().parents[1] / "scripts" / "run_subject_aci.py"
)
assert _SPEC is not None and _SPEC.loader is not None
aci = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(aci)


def _partial_sums(power: float, n: int) -> tuple[float, float]:
    t = np.arange(n, dtype=float)
    g = 1.0 / (t + 2.0) ** power
    return float(g.sum()), float((g**2).sum())


def test_sqrt_schedule_violates_square_summability() -> None:
    # p = 0.5: the sum of squares grows like log n (and sum gamma like sqrt n),
    # so the tail mass added between n = 10k and 40k stays well above zero.
    s1, s2 = _partial_sums(0.5, 10_000)
    s1_big, s2_big = _partial_sums(0.5, 40_000)
    assert s1_big > 1.5 * s1  # diverges
    assert s2_big - s2 > 1.0  # does not converge


def test_poly_schedule_meets_the_condition() -> None:
    # p = 0.6: the sum still diverges while the sum of squares converges.
    s1, s2 = _partial_sums(0.6, 10_000)
    s1_big, s2_big = _partial_sums(0.6, 40_000)
    assert s1_big > 1.5 * s1  # still diverges
    assert s2_big - s2 < 0.3  # tail mass has converged


def test_step_size_dispatch() -> None:
    assert aci.step_size(0, 0.1, "sqrt", 0.6) == pytest.approx(0.1 / np.sqrt(2.0))
    assert aci.step_size(0, 0.1, "poly", 0.6) == pytest.approx(0.1 / 2.0**0.6)
    # the default argument must stay the sqrt schedule
    assert aci.step_size(3, 0.05, "sqrt", 0.6) == pytest.approx(0.05 / np.sqrt(5.0))


def test_run_aci_covers_when_threshold_is_not_binding() -> None:
    """A calibration pool above every test score makes every set cover."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, size=50)
    scores = np.full((50, 5), 10.0)
    cal = np.full(100, 20.0)  # quantile above every candidate score
    sets = aci.run_aci(cal, scores, labels, alpha=0.1, gamma0=0.01)
    assert sets.all()
    # and with a pool below every score, nothing covers
    sets = aci.run_aci(np.zeros(100), scores, labels, alpha=0.1, gamma0=0.01)
    assert not sets.any()
