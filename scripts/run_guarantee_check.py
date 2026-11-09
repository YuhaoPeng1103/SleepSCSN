"""Empirical check of the block-level guarantee (Theorem 1).

Theorem 1 says that if blocks are exchangeable and the block score of the
true sequence is below the calibrated threshold, then *every* epoch in the
block is covered. It also says the probability of that event is at least
1 - alpha - 1/(n_B + 1).

This script pools the epochs of the calibration and test outputs, forms blocks
of a fixed length, and repeatedly splits the blocks at random into a
calibration and a test half. Random block assignment is the setting in which
the exchangeability assumption actually holds; the subject-wise split used
elsewhere in the paper deliberately violates it, and the gap between the two
is the empirical cost of subject shift.

Reported for each repeat:

* ``block_score_coverage``: fraction of test blocks whose true-sequence score
  is at or below the calibrated threshold. Should be >= 1 - alpha - 1/(n+1).
* ``simultaneous_coverage``: fraction of test blocks in which every epoch's
  true label is in its predicted set. Should equal block_score_coverage and
  in particular be >= 1 - alpha - 1/(n+1).
* ``marginal_coverage`` and ``set_size``: per-epoch behaviour of the same
  predictor, for comparison with per-epoch CP.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sleepcp.conformal import (
    aps_score_matrix,
    block_mean_score,
    split_conformal_quantile,
    temporal_dp_margin_sets,
    temporal_dp_sets,
)
from sleepcp.metrics import coverage, set_size
from sleepcp.temporal import initial_distribution, smooth_grouped, transition_matrix


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def contiguous_groups(recordings: np.ndarray) -> list[np.ndarray]:
    groups, start, n = [], 0, recordings.size
    while start < n:
        end = start + 1
        while end < n and recordings[end] == recordings[start]:
            end += 1
        groups.append(np.arange(start, end))
        start = end
    return groups


def blocks_from_groups(groups: list[np.ndarray], L: int) -> list[np.ndarray]:
    out = []
    for idx in groups:
        for s in range(0, idx.size - L + 1, L):
            out.append(idx[s : s + L])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--block-length", type=int, nargs="+", default=[5, 10, 20, 30])
    parser.add_argument("--beta", type=float, nargs="+", default=[0.0, 0.5, 1.0])
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cal_raw, test_raw = load_npz(args.cal), load_npz(args.test)
    probs = np.concatenate([cal_raw["probs"], test_raw["probs"]], axis=0).astype(float)
    labels = np.concatenate([cal_raw["labels"], test_raw["labels"]]).astype(int)
    recordings = np.concatenate(
        [cal_raw["recordings"].astype(str), test_raw["recordings"].astype(str)]
    )

    trans = transition_matrix(labels)
    init = initial_distribution(labels)
    smoothed = smooth_grouped(probs, recordings, trans, init)
    scores = aps_score_matrix(smoothed)
    trans_cost = 1.0 - trans

    groups = contiguous_groups(recordings)
    rng = np.random.default_rng(args.seed)
    results = {"alpha": args.alpha, "repeats": args.repeats, "block_lengths": {}}

    for L in args.block_length:
        blocks = blocks_from_groups(groups, L)
        n_blocks = len(blocks)
        rows = []
        for _ in range(args.repeats):
            perm = rng.permutation(n_blocks)
            half = n_blocks // 2
            cal_blocks = [blocks[i] for i in perm[:half]]
            test_blocks = [blocks[i] for i in perm[half:]]
            cal_labels = [labels[b] for b in cal_blocks]
            test_labels = [labels[b] for b in test_blocks]
            for beta in args.beta:
                block_scores = np.array(
                    [
                        block_mean_score(scores[b], y, trans_cost, beta=beta)
                        for b, y in zip(cal_blocks, cal_labels)
                    ]
                )
                q = split_conformal_quantile(block_scores, args.alpha)
                true_block_scores = np.array(
                    [
                        block_mean_score(scores[b], y, trans_cost, beta=beta)
                        for b, y in zip(test_blocks, test_labels)
                    ]
                )
                block_cov = float((true_block_scores <= q).mean())

                sim_cov, marg_bits, marg_labels = [], [], []
                for b, y in zip(test_blocks, test_labels):
                    dp = temporal_dp_sets(scores[b], trans_cost, threshold=q, beta=beta)
                    sim_cov.append(bool(dp.sets[np.arange(y.size), y].all()))
                    marg_bits.append(dp.sets)
                    marg_labels.append(y)
                sets = np.concatenate(marg_bits)
                ys = np.concatenate(marg_labels)
                rows.append(
                    {
                        "beta": beta,
                        "block_score_coverage": block_cov,
                        "simultaneous_coverage": float(np.mean(sim_cov)),
                        "marginal_coverage": coverage(sets, ys),
                        "set_size": set_size(sets),
                    }
                )

        by_beta = {}
        for beta in args.beta:
            sel = [r for r in rows if r["beta"] == beta]
            by_beta[str(beta)] = {
                key: {
                    "mean": float(np.mean([r[key] for r in sel])),
                    "std": float(np.std([r[key] for r in sel], ddof=1)),
                    "min": float(np.min([r[key] for r in sel])),
                }
                for key in ("block_score_coverage", "simultaneous_coverage", "marginal_coverage", "set_size")
            }
        results["block_lengths"][str(L)] = {"n_blocks": n_blocks, "by_beta": by_beta}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"alpha={args.alpha} target>={1 - args.alpha:.2f}")
    print(f"{'L':>4}{'beta':>6}{'block':>8}{'simul':>8}{'marg':>8}{'size':>7}")
    for L, entry in results["block_lengths"].items():
        for beta, stats in entry["by_beta"].items():
            print(
                f"{L:>4}{beta:>6}{stats['block_score_coverage']['mean']:>8.3f}"
                f"{stats['simultaneous_coverage']['mean']:>8.3f}"
                f"{stats['marginal_coverage']['mean']:>8.3f}"
                f"{stats['set_size']['mean']:>7.2f}"
            )


if __name__ == "__main__":
    main()
