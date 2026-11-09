"""Probe how much identity a trained representation carries about the patient.

For each run directory, the penultimate features of the test epochs are
extracted and a multinomial logistic regression predicts the subject from
them, with five-fold cross-validation. The probe is repeated within each
stage, so that the score is not driven by the different stage mixes of
different patients: a high within-stage score means the representation
encodes *how this patient's EEG looks*, not merely *what sleep they had*.

Example
-------
python scripts/probe_subject_identity.py --root results/main \
    --data-dir data/processed/sleep-edf-20 --out results/main/summary/probe.json
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

from sleepcp.data import load_dataset  # noqa: E402
from sleepcp.models import build_model  # noqa: E402

STAGES = ("W", "N1", "N2", "N3", "REM")


def features_for(model_name: str, checkpoint: Path, x: np.ndarray,
                 device: str = "cpu") -> np.ndarray:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # ``num_batches_tracked`` exists only for batchnorm, so its absence marks a
    # group- or instance-normalised checkpoint.
    norm = "batch" if any(k.endswith("num_batches_tracked") for k in state) else "group"
    model = build_model(model_name, norm=norm).to(device)
    model.load_state_dict(state)
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, x.shape[0], 256):
            batch = torch.as_tensor(x[start : start + 256])[:, None, :].to(device)
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
            out.append(torch.flatten(z, 1).cpu().numpy())
    return np.concatenate(out, axis=0)


def probe(features: np.ndarray, labels: np.ndarray) -> float | None:
    labels = np.unique(np.asarray(labels), return_inverse=True)[1]
    if len(np.unique(labels)) < 2:
        return None
    if np.bincount(labels).min() < 5:
        return None
    return float(
        cross_val_score(
            LogisticRegression(max_iter=3000), features, labels, cv=5
        ).mean()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/main"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/sleep-edf-20"))
    parser.add_argument("--model", default="eegnet")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    recordings = {rec.recording: rec for rec in load_dataset(args.data_dir)}
    results = []
    for test_path in sorted(args.root.glob(f"{args.model}/fold*_seed*/*_test.npz")):
        stem = str(test_path)[: -len("_test.npz")]
        checkpoint = Path(stem + ".pt")
        if not checkpoint.exists():
            continue
        with np.load(test_path, allow_pickle=False) as data:
            subjects = data["subjects"]
            labels = data["labels"]
            names = data["recordings"]
        order = sorted(set(names.tolist()))
        x = np.concatenate([recordings[name].x for name in order])
        y = np.concatenate([recordings[name].y for name in order])
        keep = y >= 0
        x = x[keep]
        features = features_for(args.model, checkpoint, x, args.device)

        entry = {
            "run": str(test_path.parent),
            "subjects": int(len(np.unique(subjects))),
            "overall": probe(features, subjects),
            "chance": float(1.0 / len(np.unique(subjects))),
            "within_stage": {},
        }
        for k, name in enumerate(STAGES):
            mask = labels == k
            if mask.sum() < 50:
                continue
            entry["within_stage"][name] = probe(features[mask], subjects[mask])
        results.append(entry)
        print(
            f"{test_path.parent}: overall={entry['overall']:.3f} "
            f"chance={entry['chance']:.2f} within="
            + ", ".join(f"{k}:{v:.2f}" for k, v in entry["within_stage"].items()
                        if v is not None),
            flush=True,
        )

    summary = {
        "n_runs": len(results),
        "overall": float(np.mean([r["overall"] for r in results if r["overall"]])),
        "chance": float(np.mean([r["chance"] for r in results])),
        "within_stage": {
            name: float(np.mean([r["within_stage"][name] for r in results
                                 if r["within_stage"].get(name) is not None]))
            for name in STAGES
            if any(r["within_stage"].get(name) is not None for r in results)
        },
    }
    out = args.out or args.root / "summary" / "probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "runs": results}, indent=1),
                   encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
