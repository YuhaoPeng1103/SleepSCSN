"""Checks on the aggregation layer, which the paper's numbers pass through.

The failure these guard against is silent: if an aggregation pass reads its
own previous output back in, the formatted strings (``"0.899 \\pm 0.010"``)
either crash the next pass or, worse, are skipped by the ``isfinite`` filter
and quietly drop runs from the summary.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "aggregate", Path(__file__).resolve().parents[1] / "scripts" / "aggregate.py"
)
assert _SPEC is not None and _SPEC.loader is not None
aggregate = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(aggregate)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")


def test_discover_skips_requested_output_dir(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _touch(root / "a" / "x.json")
    _touch(root / "b" / "x.json")
    out = root / "summary"
    _touch(out / "x.json")

    found = aggregate.discover(root, "x.json", out)
    assert [p.relative_to(root).as_posix() for p in found] == ["a/x.json", "b/x.json"]


def test_discover_skips_every_excluded_dir(tmp_path: Path) -> None:
    root = tmp_path / "results"
    _touch(root / "a" / "x.json")
    _touch(root / "summary" / "x.json")
    _touch(root / "elsewhere" / "x.json")

    found = aggregate.discover(root, "x.json", [root / "summary", root / "elsewhere"])
    assert [p.relative_to(root).as_posix() for p in found] == ["a/x.json"]


def test_merge_subject_coverage_averages_over_runs() -> None:
    """Each subject must contribute one value, averaged over its runs.

    A plain ``dict.update`` would keep whichever run was read last, so the
    worst-patient and dispersion statistics would describe one seed while
    coverage and set size describe all of them.
    """
    entries = [
        {"per_subject_coverage": {"SC400": 0.9, "SC401": 0.7}},
        {"per_subject_coverage": {"SC400": 0.9, "SC401": 0.5}},
    ]
    merged = aggregate.merge_subject_coverage(entries)
    assert merged["SC400"] == pytest.approx(0.9)
    assert merged["SC401"] == pytest.approx(0.6)  # not 0.5 (last wins)


def test_selection_joint_counts_zero_through_m80(tmp_path: Path) -> None:
    """Joint count must be per run-variant, not recoverable from marginals."""
    root = tmp_path / "main"
    budgets = {0: 0.0, 10: 0.0, 20: 0.0, 40: 0.0, 80: 0.0}
    # Run 1: always zero up to m=80 -> counted.
    # Run 2: zero except at m=80 -> not counted, although four budgets are zero.
    for run, m80 in (("fold0_seed0", 0.0), ("fold1_seed0", 0.5)):
        curve = {str(m): {"lambda": v, "coverage": 0.9} for m, v in budgets.items()}
        curve["80"] = {"lambda": m80, "coverage": 0.9}
        path = root / "eegnet" / run / "sacp.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"budget_curve": curve, "global": {"coverage": 0.9}}),
            encoding="utf-8",
        )

    out = aggregate.aggregate_sacp(root, 0.1, root / "summary")
    joint = out["selection_joint"]["aps"]
    assert joint["n_runs"] == 2
    assert joint["n_zero_through_80"] == 1
    # Marginal rates are what they are meant to be and cannot answer the
    # joint question: frac_zero at m=80 is 0.5, at m=40 it is 1.0.
    assert out["selection"]["aps"]["80"]["frac_zero"] == pytest.approx(0.5)
    assert out["selection"]["aps"]["40"]["frac_zero"] == pytest.approx(1.0)
