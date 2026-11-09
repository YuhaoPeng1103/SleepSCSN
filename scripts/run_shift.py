"""Coverage under input-level distribution shift.

Conformal coverage is a statement about exchangeability. A wearable EEG
recorder violates that assumption in mundane ways: noise, electrode impedance
drift, sampling-clock offsets. This script applies controlled corruptions to
the *test* signals, keeps the calibration set clean, and measures how far the
nominal 90% coverage falls.

For each corruption the script also reports two adapted variants: a
*recalibrated* one, where the same corruption is applied to the calibration
signals before the threshold is computed, and a *per-patient ACI* one, where
the threshold is adapted online on the corrupted stream with the clean
calibration set as its starting point. The first isolates how much of the loss
is a miscalibration of a clean-data threshold; the second is the method the
paper actually deploys, and it needs no corrupted calibration data at all.

Usage:
    python scripts/run_shift.py \
        --data-dir data/processed/sleep-edf-20 \
        --run-json results/main/eegnet/fold0_seed0/eegnet_fold0_seed0.json \
        --checkpoint results/main/eegnet/fold0_seed0/eegnet_fold0_seed0.pt \
        --cal results/main/eegnet/fold0_seed0/eegnet_fold0_seed0_cal.npz \
        --out results/main/eegnet/fold0_seed0/shift.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sleepcp.conformal import (
    aps_score_matrix,
    run_aci,
    split_conformal_quantile,
    threshold_at,
)
from sleepcp.data import concatenate, load_dataset, select_subjects
from sleepcp.metrics import coverage, set_size
from sleepcp.models import build_model
from sleepcp.training import make_loader, predict


# --------------------------------------------------------------------------
# corruptions
# --------------------------------------------------------------------------
def add_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    power = np.maximum((x**2).mean(axis=1, keepdims=True), 1e-8)
    sigma = np.sqrt(power / (10.0 ** (snr_db / 10.0)))
    return x + rng.normal(0.0, 1.0, size=x.shape) * sigma


def scale(x: np.ndarray, factor: float) -> np.ndarray:
    return x * factor


def add_drift(x: np.ndarray, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    """Slow baseline wander, as from a loosening electrode."""
    n, length = x.shape
    freq = rng.uniform(0.02, 0.2, size=(n, 1))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(n, 1))
    t = np.arange(length, dtype=float)[None, :] / length
    wave = np.sin(2.0 * np.pi * freq * t * length / 100.0 + phase)
    wave -= wave.mean(axis=1, keepdims=True)
    wave /= np.maximum(wave.std(axis=1, keepdims=True), 1e-8)
    return x + amplitude * wave


def time_shift(x: np.ndarray, shift_sec: float) -> np.ndarray:
    """Circular shift, as from a sampling-clock offset."""
    k = int(round(shift_sec * 100.0)) % x.shape[1]
    return np.roll(x, k, axis=1) if k else x


def dropout(x: np.ndarray, rate: float, rng: np.random.Generator) -> np.ndarray:
    """Zero out a fraction of epochs entirely."""
    out = x.copy()
    mask = rng.random(x.shape[0]) < rate
    out[mask] = 0.0
    return out


CORRUPTIONS = {
    "noise": ("snr_db", [30.0, 20.0, 10.0, 5.0, 0.0]),
    "scale": ("factor", [0.5, 0.75, 1.25, 1.5]),
    "drift": ("amplitude", [0.25, 0.5, 1.0, 2.0]),
    "shift": ("shift_sec", [0.5, 1.0, 2.0, 5.0]),
    "dropout": ("rate", [0.02, 0.05, 0.1, 0.2]),
}


def apply_corruption(name: str, value: float, x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if name == "noise":
        return add_noise(x, value, rng)
    if name == "scale":
        return scale(x, value)
    if name == "drift":
        return add_drift(x, value, rng)
    if name == "shift":
        return time_shift(x, value)
    if name == "dropout":
        return dropout(x, value, rng)
    raise ValueError(name)


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
def predict_probs(model, x: np.ndarray, device: torch.device, batch_size: int = 128) -> np.ndarray:
    loader = make_loader(x, np.zeros(x.shape[0], dtype=int), batch_size=batch_size, shuffle=False)
    probs, _ = predict(model, loader, device)
    return probs


def summarize(probs: np.ndarray, labels: np.ndarray, subjects: np.ndarray, alpha: float, q: float) -> dict:
    scores = aps_score_matrix(probs)
    out = summarize_sets(scores <= q, labels, subjects, alpha)
    # Accuracy separates "the sets stopped covering" from "the model stopped
    # being right": recalibration can restore coverage to a model whose
    # accuracy has collapsed, and the two must not be conflated.
    out["accuracy"] = float((probs.argmax(axis=1) == labels).mean())
    return out


def summarize_sets(sets: np.ndarray, labels: np.ndarray, subjects: np.ndarray, alpha: float) -> dict:
    per_subject = {
        str(s): coverage(sets[subjects == s], labels[subjects == s]) for s in np.unique(subjects)
    }
    values = np.asarray(list(per_subject.values()), dtype=float)
    return {
        "coverage": coverage(sets, labels),
        "set_size": set_size(sets),
        "worst_subject_coverage": float(values.min()),
        "p10_subject_coverage": float(np.quantile(values, 0.1)),
        "n_subjects_below_nominal": int((values < 1.0 - alpha).sum()),
        "per_subject_coverage": per_subject,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--run-json", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cal", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma0", type=float, default=0.05, help="ACI step scale")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--corruptions", nargs="+", default=list(CORRUPTIONS), choices=list(CORRUPTIONS)
    )
    args = parser.parse_args()

    with open(args.run_json, encoding="utf-8") as f:
        run = json.load(f)
    with np.load(args.cal, allow_pickle=False) as data:
        cal_probs = data["probs"].astype(float)
        cal_labels = data["labels"].astype(int)
    cal_true = aps_score_matrix(cal_probs)[np.arange(cal_labels.size), cal_labels]
    q_clean = split_conformal_quantile(cal_true, args.alpha)
    cal_sorted = np.sort(cal_true)

    recordings = [rec.labeled() for rec in load_dataset(args.data_dir)]
    x, y, subjects, _ = concatenate(recordings)
    test_mask = select_subjects(subjects, keep=set(run["test_subjects"]))
    x_test, y_test, s_test = x[test_mask], y[test_mask], subjects[test_mask]
    cal_mask = select_subjects(subjects, keep=set(run["cal_subjects"]))
    x_cal, y_cal = x[cal_mask], y[cal_mask]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(run["model"], n_classes=5)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.to(device).eval()

    rng = np.random.default_rng(args.seed)
    clean = summarize(predict_probs(model, x_test, device), y_test, s_test, args.alpha, q_clean)
    results = {
        "alpha": args.alpha,
        "q_clean": q_clean,
        "n_test_epochs": int(y_test.size),
        "clean": clean,
        "corruptions": {},
    }

    for name in args.corruptions:
        _, values = CORRUPTIONS[name]
        block = {}
        for value in values:
            x_cor = apply_corruption(name, value, x_test, rng)
            probs = predict_probs(model, x_cor, device)
            clean_stats = summarize(probs, y_test, s_test, args.alpha, q_clean)

            # Recalibration: apply the same corruption to the calibration
            # signals and rebuild the threshold from them.
            x_cal_cor = apply_corruption(name, value, x_cal, rng)
            cal_probs_cor = predict_probs(model, x_cal_cor, device)
            cal_scores_cor = aps_score_matrix(cal_probs_cor)[np.arange(y_cal.size), y_cal]
            q_cor = split_conformal_quantile(cal_scores_cor, args.alpha)
            recal = summarize(probs, y_test, s_test, args.alpha, q_cor)

            # Per-patient ACI on the corrupted stream, started from the clean
            # threshold. No corrupted calibration data is used.
            scores = aps_score_matrix(probs)
            sets = np.zeros(scores.shape, dtype=bool)
            for s in np.unique(s_test):
                idx = np.flatnonzero(s_test == s)
                sets[idx] = run_aci(cal_sorted, scores[idx], y_test[idx], args.alpha, args.gamma0)
            aci = summarize_sets(sets, y_test, s_test, args.alpha)

            block[str(value)] = {"clean_q": clean_stats, "recalibrated": recal, "aci": aci}
            print(
                f"{name}={value:<5} clean={clean_stats['coverage']:.3f} "
                f"recal={recal['coverage']:.3f} aci={aci['coverage']:.3f} "
                f"| aci worst={aci['worst_subject_coverage']:.3f}"
            )
        results["corruptions"][name] = block

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"clean coverage={clean['coverage']:.3f} set={clean['set_size']:.2f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
