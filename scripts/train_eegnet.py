"""Train EEGNet on preprocessed Sleep-EDF recordings and save calibration/test probabilities."""

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
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--model", default="eegnet")
    parser.add_argument("--norm", default="batch", choices=["batch", "instance", "group"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-subjects", type=int, default=0, help="debug: keep only N subjects")
    parser.add_argument("--no-class-weights", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    recordings = [rec.labeled() for rec in load_dataset(args.data_dir)]
    if args.limit_subjects:
        keep = sorted(unique_subjects(recordings))[: args.limit_subjects]
        recordings = [rec for rec in recordings if rec.subject in set(keep)]
    x, y, subjects, rec_names = concatenate(recordings)
    print(f"loaded {x.shape[0]} labeled epochs from {len(recordings)} recordings, {len(set(subjects))} subjects")

    splits = subject_wise_split(
        unique_subjects(recordings),
        n_folds=args.n_folds,
        seed=42,
        cal_frac=0.2,
        test_frac=0.2,
    )
    if args.fold >= len(splits):
        raise ValueError(f"fold {args.fold} out of range (n_folds={len(splits)})")
    train_subjects, cal_subjects, test_subjects = splits[args.fold]
    print(f"fold {args.fold}: train={train_subjects}")
    print(f"fold {args.fold}: cal={cal_subjects}")
    print(f"fold {args.fold}: test={test_subjects}")

    # Hold out a subset of training subjects for early stopping.
    rng = np.random.default_rng(args.seed + args.fold)
    train_subjects = list(rng.permutation(train_subjects))
    n_val = max(1, int(round(args.val_frac * len(train_subjects))))
    val_subjects = train_subjects[:n_val]
    fit_subjects = train_subjects[n_val:]
    if not fit_subjects:
        fit_subjects, val_subjects = train_subjects, train_subjects

    def mask_for(names):
        return select_subjects(subjects, keep=set(names))

    train_mask = mask_for(fit_subjects)
    val_mask = mask_for(val_subjects)
    cal_mask = mask_for(cal_subjects)
    test_mask = mask_for(test_subjects)

    train_loader = make_loader(x[train_mask], y[train_mask], batch_size=args.batch_size, shuffle=True)
    val_loader = make_loader(x[val_mask], y[val_mask], batch_size=args.batch_size, shuffle=False)
    cal_loader = make_loader(x[cal_mask], y[cal_mask], batch_size=args.batch_size, shuffle=False)
    test_loader = make_loader(x[test_mask], y[test_mask], batch_size=args.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = build_model(args.model, n_classes=5, norm=args.norm)
    print(f"model={args.model} norm={args.norm} params={count_parameters(model):,} device={device}")
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

    stem = f"{args.model}_fold{args.fold}_seed{args.seed}"
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
        subjects=subjects[test_mask],
        recordings=rec_names[test_mask],
    )
    result = {
        "model": args.model,
        "fold": args.fold,
        "seed": args.seed,
        "device": str(device),
        "n_train": int(train_mask.sum()),
        "n_val": int(val_mask.sum()),
        "n_cal": int(cal_mask.sum()),
        "n_test": int(test_mask.sum()),
        "train_subjects": fit_subjects,
        "val_subjects": val_subjects,
        "cal_subjects": cal_subjects,
        "test_subjects": test_subjects,
        "cal_metrics": cal_metrics,
        "test_metrics": test_metrics,
        "history": history,
        "class_weights": None if weights is None else weights.tolist(),
    }
    with open(args.out_dir / f"{stem}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"saved {stem} to {args.out_dir}")


if __name__ == "__main__":
    main()
