"""Tent-style test-time adaptation baseline for one run.

Tent (Wang et al., 2021) adapts the affine parameters of every batch
normalisation layer by minimising the entropy of the model's predictions on
the unlabelled test recording, one gradient step per batch. This script
applies it per recording on a frozen checkpoint, then evaluates the same
metrics as the SCSN pilot (accuracy, kappa, macro-F1, ECE, per-subject
conformal coverage, subject probe).

The arm ``bn`` is the unadapted checkpoint. ``tent1`` and ``tent5`` are the
predictions after one and five entropy steps respectively. Adaptation uses
only the unlabelled epochs of the recording being scored; calibration
recordings are adapted in exactly the same way.

Example
-------
python scripts/tent_baseline.py --model eegnet --fold 0 --seed 0 \
    --out results/tent/eegnet_fold0_seed0.json
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sleepcp.conformal import aps_score_matrix, split_conformal_quantile  # noqa: E402
from sleepcp.data import load_dataset  # noqa: E402
from sleepcp.metrics import (cohen_kappa_score, coverage,  # noqa: E402
                             expected_calibration_error, f1_score,
                             review_workload, set_size)
from sleepcp.models import build_model  # noqa: E402

STAGES = ("W", "N1", "N2", "N3", "REM")
PROBE_MAX = 0


def final_linear(model: torch.nn.Module, model_name: str) -> torch.nn.Linear:
    return model.fc if model_name == "eegnet" else model.head[3]


def penultimate(model: torch.nn.Module, model_name: str, x: np.ndarray,
                device: str = "cpu", batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    feats, logits = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch = torch.as_tensor(x[start : start + batch_size])[:, None, :].to(device)
            if model_name == "eegnet":
                z = model.bn1(model.conv1(batch[:, :, None, :]))
                z = model.bn2(model.conv2(z))
                z = torch.nn.functional.elu(z)
                z = model.drop1(model.pool1(z))
                z = model.conv4(model.conv3(z))
                z = model.bn3(z)
                z = torch.nn.functional.elu(z)
                z = model.drop2(model.pool2(z))
            else:
                z = model.features(batch)
                z = model.head[0](z)
                z = model.head[1](z)
                z = model.head[2](z)
            z = torch.flatten(z, 1)
            feats.append(z.cpu().numpy())
            logits.append(final_linear(model, model_name)(z).cpu().numpy())
    return np.concatenate(feats), np.concatenate(logits)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def bn_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    return [m for m in model.modules()
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]


def probe(features: np.ndarray, labels: np.ndarray) -> float | None:
    labels = np.unique(np.asarray(labels), return_inverse=True)[1]
    if len(np.unique(labels)) < 2 or np.bincount(labels).min() < 5:
        return None
    if PROBE_MAX and features.shape[0] > PROBE_MAX:
        rng = np.random.default_rng(0)
        keep = rng.choice(features.shape[0], size=PROBE_MAX, replace=False)
        features, labels = features[keep], labels[keep]
    clf = LogisticRegression(max_iter=300, tol=1e-3)
    return float(cross_val_score(clf, features, labels, cv=3).mean())


def tent_adapt(model: torch.nn.Module, initial: dict, x: np.ndarray, device: str,
               steps: int, lr: float, batch_size: int = 128,
               opt: str = "adam") -> None:
    """Adapt ``model`` in place on ``x`` with Tent's entropy objective."""
    model.load_state_dict(initial)
    layers = bn_layers(model)
    for layer in layers:
        layer.reset_running_stats()
        layer.momentum = None
        layer.train()
    params = [p for layer in layers for p in (layer.weight, layer.bias)
              if p is not None]
    if opt == "sgd":
        # Tent's published optimiser: SGD with momentum, lr 2.5e-4.
        optimizer = torch.optim.SGD(params, lr=2.5e-4, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(params, lr=lr)
    x_t = torch.as_tensor(x)[:, None, :].to(device)
    generator = torch.Generator().manual_seed(0)
    for _ in range(steps):
        order = torch.randperm(x_t.shape[0], generator=generator).to(device)
        for start in range(0, x_t.shape[0], batch_size):
            batch = x_t[order[start : start + batch_size]]
            logits = model(batch)
            logp = torch.log_softmax(logits, dim=1)
            loss = -(logp.exp() * logp).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def metrics_for(z_cal: np.ndarray, z_test: np.ndarray, y_cal: np.ndarray,
                y_test: np.ndarray, s_cal: np.ndarray, s_test: np.ndarray,
                W: np.ndarray, b: np.ndarray, alpha: float) -> dict:
    cal_p = softmax(z_cal @ W.T + b)
    test_p = softmax(z_test @ W.T + b)
    q = split_conformal_quantile(
        aps_score_matrix(cal_p)[np.arange(y_cal.size), y_cal], alpha)
    sets = aps_score_matrix(test_p) <= q
    per_subject = {str(sub): float(coverage(sets[s_test == sub], y_test[s_test == sub]))
                   for sub in np.unique(s_test)}
    cov_arr = np.asarray(list(per_subject.values()), dtype=float)
    pred = test_p.argmax(axis=1)
    return {
        "accuracy": float((pred == y_test).mean()),
        "kappa": float(cohen_kappa_score(y_test, pred)),
        "mf1": float(f1_score(y_test, pred, average="macro")),
        "ece": float(expected_calibration_error(test_p, y_test)),
        "coverage": float(coverage(sets, y_test)),
        "set_size": float(set_size(sets)),
        "review_workload": float(review_workload(sets)),
        "worst_subject_coverage": float(cov_arr.min()),
        "coverage_sd": float(cov_arr.std(ddof=1)),
        "coverage_q10": float(np.percentile(cov_arr, 10)),
        "coverage_median": float(np.median(cov_arr)),
        "per_subject_coverage": per_subject,
        "per_stage_recall": {STAGES[k]: float((pred[y_test == k] == k).mean())
                             for k in range(5) if (y_test == k).sum() > 0},
        "probe": probe(z_test, s_test),
        "probe_within": {STAGES[k]: probe(z_test[y_test == k], s_test[y_test == k])
                         for k in range(5) if (y_test == k).sum() >= 50},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="eegnet")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", type=Path, default=Path("results/main"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/sleep-edf-20"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--probe-max", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    global PROBE_MAX
    PROBE_MAX = args.probe_max

    run_dir = args.root / args.model / f"fold{args.fold}_seed{args.seed}"
    stem = f"{args.model}_fold{args.fold}_seed{args.seed}"
    run_json = json.loads((run_dir / f"{stem}.json").read_text(encoding="utf-8"))
    split = {
        "cal": sorted(set(run_json["cal_subjects"])),
        "test": sorted(set(run_json["test_subjects"])),
    }
    state = torch.load(run_dir / f"{stem}.pt", map_location="cpu", weights_only=True)
    model = build_model(args.model).to(args.device)
    model.load_state_dict(state)
    model.eval()
    initial = copy.deepcopy(model.state_dict())
    W = final_linear(model, args.model).weight.detach().cpu().numpy()
    b = final_linear(model, args.model).bias.detach().cpu().numpy()

    recordings = {rec.recording: rec for rec in load_dataset(args.data_dir)}
    chosen = {tag: sorted({r.recording for r in recordings.values()
                           if r.subject in set(subjects)})
              for tag, subjects in split.items()}

    arms = {"bn": {}, "tent_sgd": {}, "tent1": {}, "tent5": {}}
    for tag, names in chosen.items():
        for name in names:
            rec = recordings[name]
            keep = rec.y >= 0
            x, y = rec.x[keep], rec.y[keep]
            # Unadapted model first.
            z, _ = penultimate(model, args.model, x, args.device)
            arms["bn"].setdefault(tag, []).append((z, y, rec.subject, rec.recording))
            # Published Tent configuration: SGD, one pass over the recording.
            tent_adapt(model, initial, x, args.device, 1, args.lr, opt="sgd")
            zg, _ = penultimate(model, args.model, x, args.device)
            arms["tent_sgd"].setdefault(tag, []).append((zg, y, rec.subject, rec.recording))
            # Adam variants: predictions after one step and after all steps.
            tent_adapt(model, initial, x, args.device, 1, args.lr)
            z1, _ = penultimate(model, args.model, x, args.device)
            arms["tent1"].setdefault(tag, []).append((z1, y, rec.subject, rec.recording))
            if args.steps > 1:
                tent_adapt(model, initial, x, args.device, args.steps, args.lr)
                zs, _ = penultimate(model, args.model, x, args.device)
            else:
                zs = z1
            arms["tent5"].setdefault(tag, []).append((zs, y, rec.subject, rec.recording))
    model.load_state_dict(initial)

    results = {}
    for arm, data in arms.items():
        z_cal = np.concatenate([row[0] for row in data["cal"]])
        y_cal = np.concatenate([row[1] for row in data["cal"]])
        s_cal = np.concatenate([[row[2]] * len(row[1]) for row in data["cal"]])
        z_test = np.concatenate([row[0] for row in data["test"]])
        y_test = np.concatenate([row[1] for row in data["test"]])
        s_test = np.concatenate([[row[2]] * len(row[1]) for row in data["test"]])
        results[arm] = metrics_for(z_cal, z_test, y_cal, y_test, s_cal, s_test,
                                   W, b, args.alpha)
        entry = results[arm]
        print(f"{arm:8s} acc={entry['accuracy']:.4f} ece={entry['ece']:.4f} "
              f"worst={entry['worst_subject_coverage']:.4f} "
              f"sd={entry['coverage_sd']:.4f} probe={entry['probe']}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": args.model, "fold": args.fold, "seed": args.seed,
        "steps": args.steps, "lr": args.lr, "results": results,
    }, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
