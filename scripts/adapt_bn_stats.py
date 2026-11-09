"""Recompute batch-norm statistics on each new recording (AdaBN).

Batch normalisation applies running statistics estimated on the training
cohort. Those statistics are what makes a subject's activations depend on how
far the subject sits from the cohort, so replacing them with statistics
estimated on the recording being scored removes that dependence without
changing any weight. The script measures what this does to accuracy,
prediction-set coverage and the subject-identifiability of the features.

For every run of a batch-normalised checkpoint it

1. resets the running statistics of each normalisation layer and estimates
   them from the epochs of the recording itself (no labels are used),
2. predicts every epoch of the recording with the adapted statistics,
3. calibrates on the adapted calibration recordings and reports per-subject
   APS coverage, set size and review workload, and
4. probes the adapted features for subject identity with the same logistic
   regression used elsewhere.

Group-normalised checkpoints have no running statistics; they are skipped.

Example
-------
python scripts/adapt_bn_stats.py --root results/main --model eegnet \
    --out results/norm/adabn_eegnet.json --device cuda
"""

from __future__ import annotations

import argparse
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
from sleepcp.metrics import (cohen_kappa_score, coverage, expected_calibration_error,
                             f1_score, set_size)  # noqa: E402
from sleepcp.models import build_model  # noqa: E402

STAGES = ("W", "N1", "N2", "N3", "REM")


def bn_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    return [m for m in model.modules()
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]


def adapt(model: torch.nn.Module, x: np.ndarray, device: str) -> None:
    """Replace running statistics with statistics of ``x``."""
    layers = bn_layers(model)
    model.eval()
    for layer in layers:
        layer.reset_running_stats()
        layer.momentum = None          # cumulative average over all batches
        layer.train()
    with torch.no_grad():
        for start in range(0, x.shape[0], 256):
            batch = torch.as_tensor(x[start : start + 256])[:, None, :].to(device)
            model(batch)
    for layer in layers:
        layer.eval()


def features_and_probs(model: torch.nn.Module, x: np.ndarray,
                       device: str) -> tuple[np.ndarray, np.ndarray]:
    feats, probs = [], []
    with torch.no_grad():
        for start in range(0, x.shape[0], 512):
            batch = torch.as_tensor(x[start : start + 512])[:, None, :].to(device)
            if hasattr(model, "features"):
                z = model.features(batch)
            else:
                z = model.bn1(model.conv1(batch[:, :, None, :]))
                z = model.bn2(model.conv2(z))
                z = torch.nn.functional.elu(z)
                z = model.drop1(model.pool1(z))
                z = model.conv4(model.conv3(z))
                z = model.bn3(z)
                z = torch.nn.functional.elu(z)
                z = model.drop2(model.pool2(z))
            feats.append(torch.flatten(z, 1).cpu().numpy())
            probs.append(torch.softmax(model(batch), dim=1).cpu().numpy())
    return np.concatenate(feats), np.concatenate(probs)


def probe(features: np.ndarray, labels: np.ndarray) -> float | None:
    labels = np.unique(np.asarray(labels), return_inverse=True)[1]
    if len(np.unique(labels)) < 2 or np.bincount(labels).min() < 5:
        return None
    return float(cross_val_score(LogisticRegression(max_iter=3000), features, labels,
                                 cv=5).mean())


def predict_adapted(model, recordings, names, device):
    feats, probs = [], []
    for name in names:
        rec = recordings[name]
        x = rec.x
        adapt(model, x, device)
        f, p = features_and_probs(model, x, device)
        keep = rec.y >= 0
        feats.append(f[keep])
        probs.append(p[keep])
    return np.concatenate(feats), np.concatenate(probs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/main"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/sleep-edf-20"))
    parser.add_argument("--model", default="eegnet")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    recordings = {rec.recording: rec for rec in load_dataset(args.data_dir)}
    rows = []
    for test_path in sorted(args.root.glob(f"{args.model}/fold*_seed*/*_test.npz")):
        stem = str(test_path)[: -len("_test.npz")]
        checkpoint = Path(stem + ".pt")
        cal_path = Path(stem + "_cal.npz")
        if not checkpoint.exists() or not cal_path.exists():
            continue
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not any(k.endswith("num_batches_tracked") for k in state):
            print(f"skip {test_path.parent}: not batch-normalised")
            continue
        model = build_model(args.model, norm="batch").to(args.device)
        model.load_state_dict(state)
        model.eval()

        with np.load(test_path, allow_pickle=False) as data:
            test_subjects, test_labels = data["subjects"], data["labels"]
            test_names = sorted(set(data["recordings"].tolist()))
        with np.load(cal_path, allow_pickle=False) as data:
            cal_subjects, cal_labels = data["subjects"], data["labels"]
            cal_names = sorted(set(data["recordings"].tolist()))

        cal_f, cal_p = predict_adapted(model, recordings, cal_names, args.device)
        test_f, test_p = predict_adapted(model, recordings, test_names, args.device)

        q = split_conformal_quantile(aps_score_matrix(cal_p)[np.arange(len(cal_labels)),
                                                            cal_labels], args.alpha)
        sets = aps_score_matrix(test_p) <= q
        pred = test_p.argmax(axis=1)
        per_subject = {}
        for subject in np.unique(test_subjects):
            mask = test_subjects == subject
            per_subject[str(subject)] = float(coverage(sets[mask], test_labels[mask]))

        entry = {
            "run": str(test_path.parent),
            "accuracy": float((pred == test_labels).mean()),
            "kappa": float(cohen_kappa_score(test_labels, pred)),
            "mf1": float(f1_score(test_labels, pred, average="macro")),
            "ece": float(expected_calibration_error(test_p, test_labels)),
            "coverage": float(coverage(sets, test_labels)),
            "set_size": float(set_size(sets)),
            "review_workload": float((sets.sum(axis=1) > 1).mean()),
            "worst_subject_coverage": float(min(per_subject.values())),
            "coverage_sd": float(np.std(list(per_subject.values()), ddof=1)),
            "per_subject_coverage": per_subject,
            "probe_overall": probe(test_f, test_subjects),
            "probe_within": {},
        }
        for k, name in enumerate(STAGES):
            mask = test_labels == k
            if mask.sum() >= 50:
                entry["probe_within"][name] = probe(test_f[mask], test_subjects[mask])
        rows.append(entry)
        print(f"{test_path.parent}: acc={entry['accuracy']:.3f} "
              f"cov={entry['coverage']:.3f} set={entry['set_size']:.2f} "
              f"worst={entry['worst_subject_coverage']:.3f} "
              f"probe={entry['probe_overall']:.3f}", flush=True)

    summary = {
        "model": args.model,
        "n_runs": len(rows),
        "accuracy": float(np.mean([r["accuracy"] for r in rows])),
        "coverage": float(np.mean([r["coverage"] for r in rows])),
        "set_size": float(np.mean([r["set_size"] for r in rows])),
        "review_workload": float(np.mean([r["review_workload"] for r in rows])),
        "worst_subject_coverage": float(np.mean([r["worst_subject_coverage"] for r in rows])),
        "coverage_sd": float(np.mean([r["coverage_sd"] for r in rows])),
        "probe_overall": float(np.nanmean([r["probe_overall"] for r in rows])),
    }
    out = args.out or args.root / "summary" / f"adabn_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "runs": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
