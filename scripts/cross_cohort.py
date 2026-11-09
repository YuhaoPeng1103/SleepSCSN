"""Evaluate a Sleep-EDF checkpoint on an unseen cohort, with and without AdaBN.

A network trained on one cohort is applied, unchanged, to recordings of a
second cohort acquired with a different montage and scored under a different
manual. Nothing is retrained. Two inference modes are compared:

``cohort``
    the running statistics learned on the source cohort are used as they are.

``recording``
    the running statistics are re-estimated on the recording being scored
    (AdaBN); no label of the target cohort is read.

The point of the comparison is that the two modes share every weight, so any
difference between them is attributable to the statistics.

Example
-------
python scripts/cross_cohort.py --model eegnet --norm batch \
    --source results/main/eegnet/fold0_seed0 --target data/processed/hmc \
    --out results/norm/cross_cohort/eegnet_batch.json --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adapt_bn_stats import bn_layers  # noqa: E402

from sleepcp.data import load_dataset  # noqa: E402
from sleepcp.metrics import cohen_kappa_score, expected_calibration_error, f1_score  # noqa: E402
from sleepcp.models import build_model  # noqa: E402


def predict(model, x: np.ndarray, device: str, batch_size: int = 512) -> np.ndarray:
    out = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch = torch.as_tensor(x[start : start + batch_size])[:, None, :].to(device)
            out.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(out, axis=0)


def adapt(model, x: np.ndarray, device: str) -> None:
    layers = bn_layers(model)
    model.eval()
    for layer in layers:
        layer.reset_running_stats()
        layer.momentum = None
        layer.train()
    with torch.no_grad():
        for start in range(0, x.shape[0], 256):
            batch = torch.as_tensor(x[start : start + 256])[:, None, :].to(device)
            model(batch)
    for layer in layers:
        layer.eval()


def scores(y: np.ndarray, prob: np.ndarray, groups: np.ndarray) -> dict:
    pred = prob.argmax(axis=1)
    per_rec = [float((pred[groups == g] == y[groups == g]).mean())
               for g in np.unique(groups)]
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "kappa": float(cohen_kappa_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "ece": float(expected_calibration_error(prob, y)),
        "recording_accuracy_mean": float(np.mean(per_rec)),
        "recording_accuracy_worst": float(np.min(per_rec)),
        "recording_accuracy_sd": float(np.std(per_rec, ddof=1)),
        "per_recording_accuracy": per_rec,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="eegnet")
    parser.add_argument("--norm", default="batch", choices=["batch", "group", "instance"])
    parser.add_argument("--source", type=Path, required=True,
                        help="run directory holding the checkpoint")
    parser.add_argument("--target", type=Path, default=Path("data/processed/hmc"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = next(args.source.glob("*.pt"))
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_model(args.model, norm=args.norm).to(args.device)
    model.load_state_dict(state)
    model.eval()

    recordings = [rec for rec in load_dataset(args.target)]
    x = np.concatenate([rec.x for rec in recordings])
    y = np.concatenate([rec.y for rec in recordings])
    groups = np.repeat(np.arange(len(recordings)),
                       [len(rec.y) for rec in recordings])
    keep = y >= 0
    x, y, groups = x[keep], y[keep], groups[keep]

    result = {"checkpoint": str(checkpoint), "target": str(args.target)}
    result["cohort"] = scores(y, predict(model, x, args.device), groups)
    if args.norm == "batch":
        prob = []
        for i, rec in enumerate(recordings):
            keep_r = rec.y >= 0
            adapt(model, rec.x[keep_r], args.device)
            prob.append(predict(model, rec.x[keep_r], args.device))
        result["recording"] = scores(y, np.concatenate(prob), groups)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for mode in ("cohort", "recording"):
        if mode in result:
            s = result[mode]
            print(f"{mode:10s} acc={s['accuracy']:.4f} kappa={s['kappa']:.4f} "
                  f"mf1={s['macro_f1']:.4f} ece={s['ece']:.4f} "
                  f"rec_sd={s['recording_accuracy_sd']:.4f}", flush=True)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
