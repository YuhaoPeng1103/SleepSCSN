"""Generate synthetic calibration/test probability files.

Purpose: exercise the whole conformal pipeline end to end without a trained
network. The files use exactly the schema written by ``train_eegnet.py``
(``probs``, ``labels``, ``subjects``, ``recordings``), so every analysis script
can be tested on them unchanged.

The generator is deliberately realistic in the two respects that matter for
this paper: stage transitions follow a Markov chain (so temporal smoothing has
something to exploit) and subjects differ in how well calibrated their
predictions are (so per-subject coverage has a spread to study).

Usage:
    python scripts/make_synthetic_data.py --out-dir results/synthetic --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]

# Rough stage-transition structure of a night: each row sums to one.
TRANS = np.array(
    [
        [0.94, 0.03, 0.02, 0.00, 0.01],  # W
        [0.10, 0.70, 0.17, 0.01, 0.02],  # N1
        [0.02, 0.05, 0.88, 0.03, 0.02],  # N2
        [0.01, 0.02, 0.10, 0.86, 0.01],  # N3
        [0.08, 0.10, 0.12, 0.00, 0.70],  # REM
    ]
)

INIT = np.array([0.9, 0.05, 0.04, 0.01, 0.0])


def sample_sequence(rng: np.random.Generator, n: int) -> np.ndarray:
    labels = np.empty(n, dtype=int)
    labels[0] = rng.choice(5, p=INIT)
    for t in range(1, n):
        labels[t] = rng.choice(5, p=TRANS[labels[t - 1]])
    return labels


def make_probs(
    rng: np.random.Generator, labels: np.ndarray, skill: float, n_classes: int = 5
) -> np.ndarray:
    """Softmax probabilities whose accuracy increases with ``skill``."""
    n = labels.size
    logits = rng.normal(0.0, 1.0, size=(n, n_classes))
    # Add a signal to the true class; a larger skill gives sharper, more
    # accurate predictions.
    logits[np.arange(n), labels] += 0.9 + 2.0 * skill
    # Per-epoch difficulty: some epochs are noisier than others.
    difficulty = rng.normal(0.0, 0.6, size=(n, 1))
    logits = logits + difficulty * rng.normal(0.0, 1.0, size=(n, n_classes))
    logits -= logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    return probs / probs.sum(axis=1, keepdims=True)


def build(
    seed: int,
    n_subjects: int,
    epochs_per_subject: tuple[int, int],
    cal_frac: float,
) -> tuple[dict, dict]:
    rng = np.random.default_rng(seed)
    subjects = [f"SC{400 + i:02d}" for i in range(n_subjects)]
    # Subject-level skill: a wide spread so worst-subject coverage differs
    # markedly from the marginal coverage, which is the effect under study.
    skills = {s: float(np.clip(rng.beta(5.0, 2.0), 0.05, 0.99)) for s in subjects}

    cal = {"probs": [], "labels": [], "subjects": [], "recordings": []}
    test = {"probs": [], "labels": [], "subjects": [], "recordings": []}
    for s in subjects:
        n = int(rng.integers(epochs_per_subject[0], epochs_per_subject[1]))
        labels = sample_sequence(rng, n)
        probs = make_probs(rng, labels, skills[s])
        rec = f"{s}1E0"
        n_cal = int(round(cal_frac * n))
        for bucket, sl, count in (
            (cal, slice(0, n_cal), n_cal),
            (test, slice(n_cal, n), n - n_cal),
        ):
            bucket["probs"].append(probs[sl])
            bucket["labels"].append(labels[sl])
            bucket["subjects"].append(np.repeat(s, count))
            bucket["recordings"].append(np.repeat(rec, count))

    def pack(bucket: dict) -> dict:
        return {
            "probs": np.concatenate(bucket["probs"]).astype(np.float32),
            "labels": np.concatenate(bucket["labels"]).astype(np.int64),
            "subjects": np.concatenate(bucket["subjects"]).astype(str),
            "recordings": np.concatenate(bucket["recordings"]).astype(str),
        }

    return pack(cal), pack(test)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("results/synthetic"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--subjects", type=int, default=20)
    parser.add_argument("--min-epochs", type=int, default=700)
    parser.add_argument("--max-epochs", type=int, default=1100)
    parser.add_argument("--cal-frac", type=float, default=0.35)
    args = parser.parse_args()

    cal, test = build(args.seed, args.subjects, (args.min_epochs, args.max_epochs), args.cal_frac)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "synthetic_cal.npz", **cal)
    np.savez_compressed(args.out_dir / "synthetic_test.npz", **test)
    acc = float((test["probs"].argmax(axis=1) == test["labels"]).mean())
    print(f"cal epochs {cal['labels'].size}  test epochs {test['labels'].size}  test acc {acc:.3f}")


if __name__ == "__main__":
    main()
