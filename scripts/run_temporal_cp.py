"""Evaluate temporal conformal prediction (SeqCP) on calibration/test outputs.

Methods compared, all calibrated on the calibration subjects only:

``per_epoch_global``
    Standard split CP with APS scores on the raw softmax.

``seqcp_epoch``
    Split CP with APS scores on the forward-backward smoothed posterior. The
    transition matrix comes from the training labels (here: calibration
    labels), never from test labels.

``per_epoch_deficit_mondrian``
    The posterior-deficit score with one split-conformal threshold per stage
    (Mondrian). Stage coverage is near nominal by construction at the cost of
    a larger set than the pooled deficit threshold.

``seqcp_block_L*_beta*``
    Block-level scoring: S(y_1:T) = mean_t APS_t(y_t) + beta * mean_t
    transition_cost(y_{t-1}, y_t), calibrated on complete blocks. Test sets
    are the exact DP projection of the block-level conformal set, which gives
    simultaneous per-epoch coverage over the block.

``seqcp_margin_L*_beta*``
    Viterbi-centred margin sets. The DP margin M_t(y) - M* is calibrated on
    the true labels of calibration blocks, using the probability-deficit score
    1 - gamma so the margin is bounded.

For a given block length L all methods are evaluated on exactly the same
epochs (the ones covered by a complete length-L block), so the comparison is
apples-to-apples.
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
from sleepcp.metrics import (
    coverage,
    empty_rate,
    per_class_coverage,
    review_workload,
    set_size,
    singleton_rate,
    temporal_consistency_grouped,
)
from sleepcp.temporal import (
    forward_backward,
    initial_distribution,
    transition_matrix,
)


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def recording_groups(recordings: np.ndarray) -> list[np.ndarray]:
    """Contiguous index runs, one per recording."""
    groups = []
    start = 0
    n = recordings.size
    while start < n:
        end = start + 1
        while end < n and recordings[end] == recordings[start]:
            end += 1
        groups.append(np.arange(start, end))
        start = end
    return groups


def smooth_recordings(
    probs: np.ndarray,
    groups: list[np.ndarray],
    trans: np.ndarray,
    init: np.ndarray,
) -> np.ndarray:
    """Forward-backward posterior per recording, aligned to the input rows."""
    out = np.empty_like(probs)
    for idx in groups:
        out[idx] = forward_backward(probs[idx], trans, init)
    return out


def make_blocks(groups: list[np.ndarray], block_length: int) -> list[np.ndarray]:
    blocks = []
    for idx in groups:
        for start in range(0, idx.size - block_length + 1, block_length):
            blocks.append(idx[start : start + block_length])
    return blocks


def evaluate(
    sets: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    recordings: np.ndarray,
    n_blocks: int = 0,
) -> dict:
    per_subject = {}
    for s in np.unique(subjects):
        mask = subjects == s
        per_subject[str(s)] = coverage(sets[mask], labels[mask])
    values = np.asarray(list(per_subject.values()), dtype=float)
    return {
        "coverage": coverage(sets, labels),
        "per_class_coverage": per_class_coverage(sets, labels).tolist(),
        "set_size": set_size(sets),
        "singleton_rate": singleton_rate(sets),
        "empty_rate": empty_rate(sets),
        "review_workload": review_workload(sets),
        "temporal_consistency": temporal_consistency_grouped(sets, recordings),
        "n_epochs": int(labels.size),
        "n_blocks": int(n_blocks),
        "per_subject_coverage": per_subject,
        "mean_subject_coverage": float(values.mean()),
        "worst_subject_coverage": float(values.min()),
        "p10_subject_coverage": float(np.quantile(values, 0.1)),
    }


def block_sequence_coverage(bits: list[np.ndarray], labels_per_block: list[np.ndarray]) -> float:
    return float(np.mean([bool(b[np.arange(b.shape[0]), y].all()) for b, y in zip(bits, labels_per_block)]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--block-length", type=int, nargs="+", default=[10, 20, 30])
    parser.add_argument("--beta", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    parser.add_argument("--transition-source", default="cal", choices=["cal", "uniform"])
    parser.add_argument(
        "--transition-cost",
        default="one-minus",
        choices=["one-minus", "neglog"],
        help="Cost scale. 'one-minus' uses 1 - P(a->b) in [0, 1] so it is "
        "comparable to APS scores; 'neglog' uses -log P(a->b).",
    )
    parser.add_argument("--smoothing", type=float, default=1.0)
    parser.add_argument(
        "--margin-center",
        default="global",
        choices=["global", "position"],
        help="Reference the DP margin is measured against: the block minimum "
        "or the per-epoch minimum. They coincide when the block-optimal path "
        "is constant, which is the usual case on sleep blocks.",
    )
    args = parser.parse_args()

    cal_raw = load_npz(args.cal)
    test_raw = load_npz(args.test)
    cal = {
        "probs": cal_raw["probs"].astype(float),
        "labels": cal_raw["labels"].astype(int),
        "subjects": cal_raw["subjects"].astype(str),
        "recordings": cal_raw["recordings"].astype(str),
    }
    test = {
        "probs": test_raw["probs"].astype(float),
        "labels": test_raw["labels"].astype(int),
        "subjects": test_raw["subjects"].astype(str),
        "recordings": test_raw["recordings"].astype(str),
    }

    if args.transition_source == "uniform":
        trans = np.ones((5, 5)) / 5.0
    else:
        trans = transition_matrix(cal["labels"], smoothing=args.smoothing)
    init = initial_distribution(cal["labels"])

    cal_groups = recording_groups(cal["recordings"])
    test_groups = recording_groups(test["recordings"])

    cal_smooth = smooth_recordings(cal["probs"], cal_groups, trans, init)
    test_smooth = smooth_recordings(test["probs"], test_groups, trans, init)

    cal_aps_raw = aps_score_matrix(cal["probs"])
    test_aps_raw = aps_score_matrix(test["probs"])
    cal_aps_smooth = aps_score_matrix(cal_smooth)
    test_aps_smooth = aps_score_matrix(test_smooth)
    # Probability deficit: bounded, well behaved for margin calibration.
    cal_deficit = 1.0 - cal_smooth
    test_deficit = 1.0 - test_smooth

    # DP transition costs on a scale comparable to APS scores.
    if args.transition_cost == "neglog":
        trans_cost = -np.log(np.clip(trans, 1e-12, 1.0))
    else:
        trans_cost = 1.0 - trans

    cal_true_raw = cal_aps_raw[np.arange(cal["labels"].size), cal["labels"]]
    cal_true_smooth = cal_aps_smooth[np.arange(cal["labels"].size), cal["labels"]]

    q_raw = split_conformal_quantile(cal_true_raw, args.alpha)
    q_smooth = split_conformal_quantile(cal_true_smooth, args.alpha)

    # Posterior-deficit score max_k gamma - gamma(y). This is exactly what the
    # DP margin reduces to at beta = 0 (with no transition cost the minimum
    # over paths factorises), so it is the honest per-epoch comparator for
    # SeqCP-Margin: any gain at beta > 0 is attributable to the DP itself.
    cal_rows = np.arange(cal["labels"].size)
    cal_true_deficit = cal_smooth.max(axis=1) - cal_smooth[cal_rows, cal["labels"]]
    test_deficit_score = test_smooth.max(axis=1, keepdims=True) - test_smooth
    q_deficit = split_conformal_quantile(cal_true_deficit, args.alpha)

    results = {
        "alpha": args.alpha,
        "transition_source": args.transition_source,
        "margin_center": args.margin_center,
        "transitions": trans.tolist(),
        "q_per_epoch_raw": q_raw,
        "q_per_epoch_smoothed": q_smooth,
        "q_per_epoch_deficit": q_deficit,
        "methods": {},
    }

    # Full-recording per-epoch references (no block restriction).
    sets = test_aps_raw <= q_raw
    results["methods"]["per_epoch_global__all_epochs"] = evaluate(
        sets, test["labels"], test["subjects"], test["recordings"]
    )
    sets = test_aps_smooth <= q_smooth
    results["methods"]["seqcp_epoch__all_epochs"] = evaluate(
        sets, test["labels"], test["subjects"], test["recordings"]
    )
    sets = test_deficit_score <= q_deficit
    results["methods"]["per_epoch_deficit__all_epochs"] = evaluate(
        sets, test["labels"], test["subjects"], test["recordings"]
    )

    # Stage-conditional (Mondrian) variant of the same deficit score. It
    # restores stage-level coverage at the price of a larger set, and it is
    # the construction the per-stage table recommends when an omitted stage
    # is costly.
    n_classes = cal["probs"].shape[1]
    q_deficit_mondrian = np.array(
        [
            split_conformal_quantile(
                cal_true_deficit[cal["labels"] == k], args.alpha
            )
            for k in range(n_classes)
        ]
    )
    results["q_deficit_mondrian"] = q_deficit_mondrian.tolist()
    sets = test_deficit_score <= q_deficit_mondrian
    results["methods"]["per_epoch_deficit_mondrian__all_epochs"] = evaluate(
        sets, test["labels"], test["subjects"], test["recordings"]
    )

    for L in args.block_length:
        cal_blocks = make_blocks(cal_groups, L)
        test_blocks = make_blocks(test_groups, L)
        if not cal_blocks or not test_blocks:
            continue

        sel = np.concatenate(test_blocks)
        labels_sel = test["labels"][sel]
        subjects_sel = test["subjects"][sel]
        rec_sel = test["recordings"][sel]
        cal_sel = np.concatenate(cal_blocks)

        # Same-epoch baselines for this block length. Their thresholds are
        # calibrated on the same calibration epochs the block methods use
        # (the epochs inside calibration blocks), so the comparison isolates
        # the temporal construction rather than the calibration sample.
        cal_true_raw_L = cal_aps_raw[cal_sel][np.arange(cal_sel.size), cal["labels"][cal_sel]]
        cal_true_smooth_L = cal_aps_smooth[cal_sel][np.arange(cal_sel.size), cal["labels"][cal_sel]]
        cal_true_deficit_L = cal_true_deficit[cal_sel]
        q_raw_L = split_conformal_quantile(cal_true_raw_L, args.alpha)
        q_smooth_L = split_conformal_quantile(cal_true_smooth_L, args.alpha)
        q_deficit_L = split_conformal_quantile(cal_true_deficit_L, args.alpha)

        sets_raw = test_aps_raw[sel] <= q_raw_L
        results["methods"][f"per_epoch_global_L{L}"] = evaluate(
            sets_raw, labels_sel, subjects_sel, rec_sel, n_blocks=len(test_blocks)
        )
        sets_smooth = test_aps_smooth[sel] <= q_smooth_L
        results["methods"][f"seqcp_epoch_L{L}"] = evaluate(
            sets_smooth, labels_sel, subjects_sel, rec_sel, n_blocks=len(test_blocks)
        )
        sets_deficit = test_deficit_score[sel] <= q_deficit_L
        results["methods"][f"per_epoch_deficit_L{L}"] = evaluate(
            sets_deficit, labels_sel, subjects_sel, rec_sel, n_blocks=len(test_blocks)
        )

        cal_labels_per_block = [cal["labels"][b] for b in cal_blocks]
        test_labels_per_block = [test["labels"][b] for b in test_blocks]

        for beta in args.beta:
            # ---- block-level conformal projection -------------------------
            cal_block_scores = np.array(
                [
                    block_mean_score(cal_aps_smooth[b], y, trans_cost, beta=beta)
                    for b, y in zip(cal_blocks, cal_labels_per_block)
                ]
            )
            q = split_conformal_quantile(cal_block_scores, args.alpha)
            name = f"seqcp_block_L{L}_beta{beta:g}"
            if not np.isfinite(q):
                results["methods"][name] = {"q": q, "error": "vacuous"}
            else:
                bits = [
                    temporal_dp_sets(test_aps_smooth[b], trans_cost, threshold=q, beta=beta).sets
                    for b in test_blocks
                ]
                sets = np.concatenate(bits, axis=0)
                metrics = evaluate(sets, labels_sel, subjects_sel, rec_sel, len(test_blocks))
                metrics["sequence_coverage"] = block_sequence_coverage(bits, test_labels_per_block)
                metrics["q"] = q
                results["methods"][name] = metrics

            # ---- Viterbi-centred margin sets ------------------------------
            cal_margins = []
            for b, y in zip(cal_blocks, cal_labels_per_block):
                m = temporal_dp_margin_sets(
                    cal_deficit[b], trans_cost, delta=float("inf"), beta=beta,
                    center=args.margin_center,
                )
                cal_margins.append(m.min_scores[np.arange(y.size), y])
            delta = split_conformal_quantile(np.concatenate(cal_margins), args.alpha)
            name = f"seqcp_margin_L{L}_beta{beta:g}"
            if not np.isfinite(delta):
                results["methods"][name] = {"delta": delta, "error": "vacuous"}
            else:
                bits = [
                    temporal_dp_margin_sets(
                        test_deficit[b], trans_cost, delta=delta, beta=beta,
                        center=args.margin_center,
                    ).sets
                    for b in test_blocks
                ]
                sets = np.concatenate(bits, axis=0)
                metrics = evaluate(sets, labels_sel, subjects_sel, rec_sel, len(test_blocks))
                metrics["sequence_coverage"] = block_sequence_coverage(bits, test_labels_per_block)
                metrics["delta"] = delta
                results["methods"][name] = metrics

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Compact console summary.
    print(f"alpha={args.alpha} q_raw={q_raw:.4f} q_smoothed={q_smooth:.4f}")
    header = f"{'method':<34}{'cov':>7}{'size':>7}{'sing':>7}{'cons':>7}{'worst':>8}"
    print(header)
    print("-" * len(header))
    for name, m in results["methods"].items():
        if "error" in m:
            print(f"{name:<34}{m['error']:>7}")
            continue
        print(
            f"{name:<34}{m['coverage']:>7.3f}{m['set_size']:>7.2f}"
            f"{m['singleton_rate']:>7.3f}{m['temporal_consistency']:>7.3f}"
            f"{m['worst_subject_coverage']:>8.3f}"
        )


if __name__ == "__main__":
    main()
