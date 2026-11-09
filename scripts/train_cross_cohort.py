"""Train on one Sleep-EDF cohort and export probabilities on another.

The cassette cohort (SC4xx) and the telemetry cohort (ST7xx) were recorded
with different equipment on different patient groups.  This script trains a
backbone on the cassette subjects (holding a few of them out for conformal
calibration) and writes calibration/test ``.npz`` files in exactly the format
``train_eegnet.py`` produces, so the whole downstream analysis
(``run_sacp.py``, ``run_temporal_cp.py``, ``run_subject_aci.py``) applies
unchanged.

The split is the fold-0 calibration block of the main protocol, with the
cassette test block folded into training: every cassette subject is either
training or calibration data, and no telemetry subject is ever seen.

Usage:
    python scripts/train_cross_cohort.py \
        --train-dir data/processed/sleep-edf-20 \
        --test-dir data/processed/sleep-telemetry \
        --out-dir results/cross/sc2st/eegnet/seed0 \
        --model eegnet --seed 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from sleepcp.data import (
    concatenate,
    load_dataset,
    select_subjects,
    subject_wise_split,
    unique_subjects,
)
from sleepcp.metrics import classification_metrics
from sleepcp.models import build_model, count_parameters
from sleepcp.training import class_weights, make_loader, predict, set_seed, train_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", required=True, type=Path)
    parser.add_argument("--test-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--model", default="eegnet")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-cal", type=int, default=4, help="subjects held out for calibration")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-class-weights", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    src = [rec.labeled() for rec in load_dataset(args.train_dir)]
    x, y, subjects, rec_names = concatenate(src)

    # Reuse the main protocol's fold-0 split to choose the calibration
    # subjects, then give every other source subject to training.
    _, cal_subjects, _ = subject_wise_split(
        unique_subjects(src), n_folds=5, seed=42, cal_frac=args.n_cal / len(unique_subjects(src))
    )[0]
    train_pool = [s for s in unique_subjects(src) if s not in set(cal_subjects)]

    rng = np.random.default_rng(args.seed)
    order = list(rng.permutation(train_pool))
    n_val = max(1, int(round(args.val_frac * len(order))))
    val_subjects, fit_subjects = order[:n_val], order[n_val:]

    def mask_for(names):
        return select_subjects(subjects, keep=set(names))

    train_mask, val_mask = mask_for(fit_subjects), mask_for(val_subjects)
    cal_mask = mask_for(cal_subjects)

    dst = [rec.labeled() for rec in load_dataset(args.test_dir)]
    x_te, y_te, s_te, r_te = concatenate(dst)

    print(
        f"train={len(fit_subjects)} subj/{train_mask.sum()} ep, "
        f"val={len(val_subjects)}/{val_mask.sum()} ep, "
        f"cal={len(cal_subjects)}/{cal_mask.sum()} ep, "
        f"test={len(set(s_te))} subj/{y_te.size} ep"
    )

    device = torch.device(args.device)
    model = build_model(args.model, n_classes=5)
    print(f"model={args.model} params={count_parameters(model):,} device={device}")

    train_loader = make_loader(x[train_mask], y[train_mask], batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(x[val_mask], y[val_mask], batch_size=args.batch_size, shuffle=False)
    cal_loader = make_loader(x[cal_mask], y[cal_mask], batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(x_te, y_te, batch_size=args.batch_size, shuffle=False)

    weights = None if args.no_class_weights else class_weights(y[train_mask])
    model, history = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        class_weights=weights,
    )

    cal_probs, cal_labels = predict(model, cal_loader, device)
    test_probs, test_labels = predict(model, test_loader, device)

    cal_metrics = classification_metrics(cal_probs, cal_labels)
    test_metrics = classification_metrics(test_probs, test_labels)
    print("cal metrics:", json.dumps(cal_metrics, indent=2))
    print("test metrics:", json.dumps(test_metrics, indent=2))

    stem = f"{args.model}_cross_seed{args.seed}"
    torch.save(model.state_dict(), args.out_dir / f"{stem}.pt")
    np.savez_compressed(
        args.out_dir / f"{stem}_cal.npz",
        probs=cal_probs,
        labels=cal_labels,
        subjects=subjects[cal_mask],
        recordings=rec_names[cal_mask],
    )
    np.savez_compressed(
        args.out_dir / f"{stem}_test.npz",
        probs=test_probs,
        labels=test_labels,
        subjects=s_te,
        recordings=r_te,
    )
    result = {
        "model": args.model,
        "fold": 0,
        "seed": args.seed,
        "device": str(device),
        "cross_cohort": True,
        "train_dir": str(args.train_dir),
        "test_dir": str(args.test_dir),
        "class_weights": None if weights is None else weights.tolist(),
        "train_subjects": sorted(fit_subjects),
        "val_subjects": sorted(val_subjects),
        "cal_subjects": sorted(cal_subjects),
        "test_subjects": sorted(set(s_te.tolist())),
        "cal_metrics": cal_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "n_test": int(y_te.size),
    }
    with open(args.out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out_dir / stem}.json")


if __name__ == "__main__":
    main()
