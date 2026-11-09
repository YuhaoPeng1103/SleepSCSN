"""Measure how much of a stager's prediction is about the patient, not the stage.

Three statistics are computed for every trained run, all from the test epochs
of the run's held-out subjects:

``centroid_ratio``
    For the penultimate representation, the distance of each subject's mean
    feature vector from the cohort mean, divided by the mean within-subject
    distance from that subject's own mean. A value near one means the subject
    occupies as much of the representation as the epoch-to-epoch variation
    within the subject; a value near zero means the representation is
    subject-invariant.

``between_within_ratio``
    Within each stage, predictions are summarised per subject by the mean
    softmax vector of that subject's epochs in the stage. The statistic is the
    mean pairwise total-variation distance between subject means divided by
    the mean distance of an epoch from its own subject mean. It answers the
    question "holding the stage fixed, how much of the predicted distribution
    is still explained by who the patient is".

``swap_shift``
    A causal check on the representation: every epoch's feature vector has its
    own subject's centroid subtracted and another subject's added, the linear
    head is reapplied, and the mean total-variation distance between the two
    predicted distributions is reported, together with the distance normalised
    by the length of the centroid displacement.

The script is label-using by construction (it needs the stage to hold it
fixed); it is a diagnostic, not a deployable score.

Example
-------
python scripts/subject_dependence.py --root results/probe_norm/group_all \
    --model eegnet --out results/subject_dependence/group_eegnet.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from probe_subject_identity import features_for  # noqa: E402

from sleepcp.data import load_dataset  # noqa: E402

STAGES = ("W", "N1", "N2", "N3", "REM")


def tv(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Total-variation distance between rows of two probability matrices."""
    return 0.5 * np.abs(p - q).sum(axis=1)


def head_for(model_name: str, checkpoint: Path, device: str = "cpu"):
    """Return a function mapping features to class probabilities."""
    from sleepcp.models import build_model

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    norm = "batch" if any(k.endswith("num_batches_tracked") for k in state) else "group"
    model = build_model(model_name, norm=norm).to(device)
    model.load_state_dict(state)
    model.eval()

    # EEGNet keeps the linear layer as ``fc``; SleepCNN ends its ``head``
    # (dropout -> linear) with the same layer.
    classifier = model.fc if hasattr(model, "fc") else model.head[-1]

    def probs(features: np.ndarray) -> np.ndarray:
        out = []
        with torch.no_grad():
            for start in range(0, features.shape[0], 2048):
                z = torch.as_tensor(features[start : start + 2048], dtype=torch.float32).to(device)
                logits = classifier(z)
                out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out, axis=0)

    return probs


def analyse(features: np.ndarray, prob_fn, labels: np.ndarray,
            subjects: np.ndarray) -> dict:
    probs = prob_fn(features)
    subs = np.unique(subjects)
    centroids = np.stack([features[subjects == s].mean(axis=0) for s in subs])
    cohort = features.mean(axis=0)
    between = np.linalg.norm(centroids - cohort, axis=1).mean()
    within = np.mean([np.linalg.norm(features[subjects == s] - centroids[i], axis=1).mean()
                      for i, s in enumerate(subs)])

    # Between-subject vs within-subject spread of the predicted distribution,
    # holding the stage fixed.
    per_stage = {}
    for k, name in enumerate(STAGES):
        mask = labels == k
        if mask.sum() < 200:
            continue
        means, spread = [], []
        for s in subs:
            sel = mask & (subjects == s)
            if sel.sum() < 20:
                continue
            m = probs[sel].mean(axis=0)
            means.append(m)
            spread.append(tv(probs[sel], np.repeat(m[None, :], sel.sum(), axis=0)).mean())
        if len(means) < 2:
            continue
        pairs = [tv(np.asarray(a)[None, :], np.asarray(b)[None, :])[0]
                 for a, b in itertools.combinations(means, 2)]
        per_stage[name] = {
            "between": float(np.mean(pairs)),
            "within": float(np.mean(spread)),
            "ratio": float(np.mean(pairs) / max(np.mean(spread), 1e-8)),
            "n_pairs": len(pairs),
        }

    # Causal swap: replace each epoch's subject centroid with the next
    # subject's centroid in the representation.
    index = {s: i for i, s in enumerate(subs)}
    order = {s: subs[(index[s] + 1) % len(subs)] for s in subs}
    swapped = features.copy()
    shifts = []
    for s in subs:
        sel = subjects == s
        delta = centroids[index[order[s]]] - centroids[index[s]]
        swapped[sel] += delta[None, :]
        shifts.append(np.linalg.norm(delta))
    p_swapped = prob_fn(swapped)
    shift = tv(probs, p_swapped)
    flips = float((probs.argmax(axis=1) != p_swapped.argmax(axis=1)).mean())

    return {
        "subjects": int(len(subs)),
        "centroid_norm": float(between),
        "within_norm": float(within),
        "centroid_ratio": float(between / max(within, 1e-8)),
        "per_stage": per_stage,
        "between_within_ratio": float(np.mean([v["ratio"] for v in per_stage.values()]))
        if per_stage else float("nan"),
        "swap_shift": float(shift.mean()),
        "swap_shift_per_unit": float(shift.mean() / max(np.mean(shifts), 1e-8)),
        "swap_flip_rate": flips,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/probe_norm/group_all"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/processed/sleep-edf-20"))
    parser.add_argument("--model", default="eegnet")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    recordings = {rec.recording: rec for rec in load_dataset(args.data_dir)}
    runs = []
    for test_path in sorted(args.root.glob(f"{args.model}/fold*_seed*/*_test.npz")):
        checkpoint = Path(str(test_path)[: -len("_test.npz")] + ".pt")
        if not checkpoint.exists():
            continue
        with np.load(test_path, allow_pickle=False) as data:
            subjects = np.asarray(data["subjects"])
            labels = np.asarray(data["labels"])
            names = np.asarray(data["recordings"])
        order = sorted(set(names.tolist()))
        x = np.concatenate([recordings[name].x for name in order])
        y = np.concatenate([recordings[name].y for name in order])
        keep = y >= 0
        x, y = x[keep], y[keep]
        features = features_for(args.model, checkpoint, x, args.device).astype(np.float32)
        prob_fn = head_for(args.model, checkpoint, args.device)
        entry = {"run": str(test_path.parent), **analyse(features, prob_fn, y, subjects)}
        runs.append(entry)
        print(f"{test_path.parent}: centroid_ratio={entry['centroid_ratio']:.3f} "
              f"between_within={entry['between_within_ratio']:.3f} "
              f"swap_tv={entry['swap_shift']:.4f} flips={entry['swap_flip_rate']:.3f}",
              flush=True)

    def mean(key: str) -> float:
        return float(np.mean([r[key] for r in runs]))

    summary = {
        "model": args.model,
        "n_runs": len(runs),
        "centroid_ratio": mean("centroid_ratio"),
        "between_within_ratio": mean("between_within_ratio"),
        "swap_shift": mean("swap_shift"),
        "swap_shift_per_unit": mean("swap_shift_per_unit"),
        "swap_flip_rate": mean("swap_flip_rate"),
        "per_stage": {
            name: float(np.mean([r["per_stage"][name]["ratio"] for r in runs
                                 if name in r["per_stage"]]))
            for name in STAGES
            if any(name in r["per_stage"] for r in runs)
        },
    }
    out = args.out or args.root / "summary" / f"subject_dependence_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "runs": runs}, indent=1),
                   encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
