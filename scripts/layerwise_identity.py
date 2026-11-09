"""Localise where a stager's representation becomes patient-specific.

For a trained checkpoint the activations of each normalisation block are
summarised per epoch by global average pooling and then, per recording, by the
mean of those summaries. Two probes are fitted at every depth:

``epoch_probe``
    multinomial logistic regression from the pooled epoch activation to the
    patient (five-fold cross-validated).

``recording_probe``
    leave-one-out logistic regression from the *mean* activation of a
    recording to the patient. One sample per recording, so a score well above
    chance means the per-recording average of the activations, a single vector
    with no time resolution left, already identifies the patient.

The between/within ratio of the recording means is reported as well: how far
the recordings sit from each other relative to the spread of their own epochs.

Example
-------
python scripts/layerwise_identity.py --root results/main --model eegnet \
    --out results/norm/layerwise/main_eegnet.json --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score, cross_val_predict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sleepcp.data import load_dataset  # noqa: E402
from sleepcp.models import build_model  # noqa: E402


def blocks(model_name: str, model: torch.nn.Module):
    """Return the activation-producing stages of a backbone, in order."""
    if model_name == "eegnet":
        def stage1(x):
            return model.bn1(model.conv1(x))

        def stage2(x):
            return model.bn2(model.conv2(stage1(x)))

        def stage3(x):
            z = torch.nn.functional.elu(stage2(x))
            z = model.drop1(model.pool1(z))
            return model.conv4(model.conv3(z))

        def stage4(x):
            return model.bn3(stage3(x))

        return [("conv1", stage1), ("conv2", stage2),
                ("conv3", stage3), ("penultimate", stage4)]
    layers = list(model.features.children())
    stages, index = [], []
    for i, layer in enumerate(layers):
        index.append(i)
        # SleepCNN places the non-linearity inside each block, so a top-level
        # ReLU test finds nothing; take any child that contains a convolution.
        if isinstance(layer, torch.nn.Sequential) and any(
            isinstance(module, torch.nn.Conv1d) for module in layer.modules()
        ):
            stages.append((f"block{len(stages) + 1}",
                           _prefix(model.features, index)))
    if not stages:
        raise ValueError(f"no convolutional blocks found for {model_name}")
    return stages


def _prefix(sequential: torch.nn.Sequential, index: list[int]):
    kept = [sequential[i] for i in index]

    def run(x):
        for layer in kept:
            x = layer(x)
        return x

    return run


def pool(activation: torch.Tensor) -> torch.Tensor:
    """Average over every axis except the batch."""
    z = activation
    while z.ndim > 2:
        z = z.mean(dim=-1)
    return z


def epoch_features(model_name: str, checkpoint: Path, x: np.ndarray,
                   device: str) -> dict[str, np.ndarray]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    norm = "batch" if any(k.endswith("num_batches_tracked") for k in state) else "group"
    model = build_model(model_name, norm=norm).to(device)
    model.load_state_dict(state)
    model.eval()
    stages = blocks(model_name, model)
    out = {name: [] for name, _ in stages}
    with torch.no_grad():
        for start in range(0, x.shape[0], 256):
            chunk = torch.as_tensor(x[start : start + 256]).to(device)
            if model_name == "eegnet":
                batch = chunk[:, None, None, :]
            else:
                batch = chunk[:, None, :]
            for name, fn in stages:
                out[name].append(pool(fn(batch)).cpu().numpy())
    feats = {name: np.concatenate(v, axis=0).astype(np.float32) for name, v in out.items()}
    return feats, model


def epoch_probe(features: np.ndarray, labels: np.ndarray) -> float | None:
    labels = np.unique(labels, return_inverse=True)[1]
    if len(np.unique(labels)) < 2 or np.bincount(labels).min() < 5:
        return None
    return float(cross_val_score(LogisticRegression(max_iter=3000), features, labels,
                                 cv=5).mean())


def head_accuracy(model, features: np.ndarray, labels: np.ndarray) -> float:
    with torch.no_grad():
        z = torch.as_tensor(features, dtype=torch.float32).to(next(model.parameters()).device)
        logits = model.fc(z) if hasattr(model, "fc") else model.head[-1](z)
        pred = logits.argmax(dim=1).cpu().numpy()
    return float((pred == labels).mean())


def stage_probe(features: np.ndarray, subjects: np.ndarray,
                labels: np.ndarray) -> float | None:
    """Mean within-stage subject probe, so architecture cannot drive it."""
    scores = []
    for k in np.unique(labels):
        mask = labels == k
        if mask.sum() < 50:
            continue
        score = epoch_probe(features[mask], subjects[mask])
        if score is not None:
            scores.append(score)
    return float(np.mean(scores)) if scores else None


def variance_share(features: np.ndarray, groups: np.ndarray) -> float:
    """Fraction of feature variance that is constant within a recording."""
    resid = features.copy()
    for g in np.unique(groups):
        mask = groups == g
        resid[mask] -= features[mask].mean(axis=0)
    total = features.var(axis=0).mean()
    return float(1.0 - resid.var(axis=0).mean() / max(total, 1e-12))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/main"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/processed/sleep-edf-20"))
    parser.add_argument("--model", default="eegnet")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--limit-runs", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    recordings = {rec.recording: rec for rec in load_dataset(args.data_dir)}
    runs = []
    paths = sorted(args.root.glob(f"{args.model}/fold*_seed*/*_test.npz"))
    if args.limit_runs:
        paths = paths[: args.limit_runs]
    for test_path in paths:
        checkpoint = Path(str(test_path)[: -len("_test.npz")] + ".pt")
        if not checkpoint.exists():
            continue
        with np.load(test_path, allow_pickle=False) as data:
            names = sorted(set(data["recordings"].tolist()))
            subjects = np.asarray(data["subjects"])
        x = np.concatenate([recordings[name].x for name in names])
        y = np.concatenate([recordings[name].y for name in names])
        keep = y >= 0
        x, y = x[keep], y[keep]
        # ``names`` is sorted the same way the test arrays were built, so the
        # per-epoch recording index is recovered from the epoch counts.
        counts = [int((recordings[name].y >= 0).sum()) for name in names]
        rec_index = np.repeat(np.arange(len(names)), counts)
        rec_subject = np.asarray([recordings[name].subject for name in names])

        feats, model = epoch_features(args.model, checkpoint, x, args.device)
        entry = {"run": str(test_path.parent), "layers": {}}
        for name, z in feats.items():
            means = np.stack([z[rec_index == i].mean(axis=0) for i in range(len(names))])
            resid = z.copy()
            for i in range(len(names)):
                resid[rec_index == i] -= means[i]
            cohort = means.mean(axis=0)
            between = np.linalg.norm(means - cohort, axis=1).mean()
            within = np.mean([np.linalg.norm(z[rec_index == i] - means[i], axis=1).mean()
                              for i in range(len(names))])
            entry["layers"][name] = {
                "dim": int(z.shape[1]),
                "epoch_probe": epoch_probe(z, subjects),
                "centered_probe": epoch_probe(resid, subjects),
                "stage_probe": stage_probe(z, subjects, y),
                "centered_stage_probe": stage_probe(resid, subjects, y),
                "variance_share": variance_share(z, rec_index),
                "centroid_ratio": float(between / max(within, 1e-8)),
            }
            print(f"{test_path.parent.name} {name:12s} d={z.shape[1]:4d} "
                  f"probe={entry['layers'][name]['epoch_probe']:.3f} "
                  f"centered={entry['layers'][name]['centered_probe']:.3f} "
                  f"vshare={entry['layers'][name]['variance_share']:.3f} "
                  f"ratio={entry['layers'][name]['centroid_ratio']:.3f}", flush=True)
        last = list(feats)[-1]
        z = feats[last]
        entry["acc"] = head_accuracy(model, z, y)
        entry["acc_centered"] = head_accuracy(
            model, z - np.repeat(
                np.stack([z[rec_index == i].mean(axis=0) for i in range(len(names))]),
                counts, axis=0), y)
        print(f"{test_path.parent.name} head acc={entry['acc']:.3f} "
              f"centered={entry['acc_centered']:.3f}", flush=True)
        runs.append(entry)

    summary = {}
    for name in runs[0]["layers"]:
        summary[name] = {
            key: float(np.mean([r["layers"][name][key] for r in runs]))
            for key in ("epoch_probe", "centered_probe", "stage_probe",
                        "centered_stage_probe", "variance_share", "centroid_ratio")
        }
        summary[name]["dim"] = runs[0]["layers"][name]["dim"]
    summary["acc"] = float(np.mean([r["acc"] for r in runs]))
    summary["acc_centered"] = float(np.mean([r["acc_centered"] for r in runs]))
    summary["n_runs"] = len(runs)
    summary["chance"] = float(1.0 / len(np.unique(subjects)))
    out = args.out or args.root / "summary" / f"layerwise_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "runs": runs}, indent=1),
                   encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
