"""Time the forward--backward smoothing on a fitted run.

Section 5.2 quotes the inference cost of the smoothed posterior as a few tens
of microseconds per epoch. Timing depends on the machine and its load, so the
number should be read as an order of magnitude; this script reproduces it for
the machine at hand.

Usage:
    python scripts/benchmark_smoothing.py \
        --npz results/main/eegnet/fold0_seed0/eegnet_fold0_seed0_test.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from sleepcp.temporal import (
    initial_distribution,
    smooth_grouped,
    transition_matrix,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True,
                        help="test probabilities written by a training run")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    with np.load(args.npz, allow_pickle=False) as data:
        probs = data["probs"].astype(float)
        labels = data["labels"].astype(int)
        groups = data["recordings"]

    # The transitions are estimated from these labels only to give the
    # recursion realistic work; no output of this script enters any result.
    valid = labels >= 0
    transitions = transition_matrix(labels[valid])
    init = initial_distribution(labels[valid])

    smooth_grouped(probs, groups, transitions, init)  # warm-up
    times = []
    for _ in range(max(1, args.repeats)):
        t0 = time.perf_counter()
        smooth_grouped(probs, groups, transitions, init)
        times.append(time.perf_counter() - t0)
    best = min(times)

    per_epoch = best / len(probs)
    print(f"{len(probs)} epochs in {best * 1e3:.1f} ms "
          f"(best of {len(times)})")
    print(f"{per_epoch * 1e6:.1f} us/epoch on one CPU core")
    print(f"{per_epoch * 960 * 1e3:.0f} ms for a 960-epoch recording")


if __name__ == "__main__":
    main()
