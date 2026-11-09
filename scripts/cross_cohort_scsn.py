"""Apply SCSN to a checkpoint transferred across cohorts.

A model trained on Sleep-EDF (fold 0, seed 0) is evaluated on every epoch of
HMC. Nothing is retrained and no HMC label is read by the corrections. Four
inference modes are compared:

``cohort``
    source-cohort normalisation statistics, no correction.
``adabn``
    post-hoc marginal correction: subtract the recording's global mean shift.
``scsn``
    stage-conditional correction with soft assignments (lambda = 1).
``scsn_lam0.5``
    the same correction at half strength.

Stage means and the cohort mean are estimated on the source training subjects.
An optional subject probe measures how much identity survives each correction
in the transferred representation.

Example
-------
python scripts/cross_cohort_scsn.py --model eegnet \
    --source results/main/eegnet/fold0_seed0 \
    --out results/scsn_cross/eegnet.json
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
from sleepcp.metrics import cohen_kappa_score, expected_calibration_error, f1_score  # noqa: E402
from sleepcp.models import build_model  # noqa: E402

STAGES = ("W", "N1", "N2", "N3", "REM")


def final_linear(model: torch.nn.Module, model_name: str) -> torch.nn.Linear:
    return model.fc if model_name == "eegnet" else model.head[3]


def penultimate(model: torch.nn.Module, model_name: str, x: np.ndarray,
                device: str = "cpu", batch_size: int = 512) -> np.ndarray:
    feats = []
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
            feats.append(torch.flatten(z, 1).cpu().numpy())
    return np.concatenate(feats)


def adapt_bn(model: torch.nn.Module, initial: dict, x: np.ndarray,
             device: str) -> None:
    """Re-estimate every batch-normalisation statistic on ``x`` (AdaBN)."""
    model.load_state_dict(initial)
    layers = [m for m in model.modules()
              if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))]
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


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def estimate_stage_means(z: np.ndarray, y: np.ndarray) -> dict:
    return {k: z[y == k].mean(axis=0) for k in range(5) if (y == k).sum() > 0}


def recording_shift(z: np.ndarray, probs: np.ndarray, mu_k: dict,
                    shrink: float = 10.0) -> dict:
    out = {}
    for k, m in mu_k.items():
        w = probs[:, k]
        total = float(w.sum())
        if total <= 1e-9:
            continue
        out[k] = (w @ z + shrink * m) / (total + shrink) - m
    return out


def correct(z: np.ndarray, probs: np.ndarray, mu_k: dict, mu_global: np.ndarray,
            lam: float) -> np.ndarray:
    shift = recording_shift(z, probs, mu_k)
    cond = np.zeros_like(z)
    for k, d in shift.items():
        cond += probs[:, [k]] * d[None, :]
    delta = z.mean(axis=0) - mu_global
    return z - delta - lam * (cond - delta)


def probe(features: np.ndarray, labels: np.ndarray, cap: int, cv: int) -> float | None:
    labels = np.unique(np.asarray(labels), return_inverse=True)[1]
    if len(np.unique(labels)) < 2:
        return None
    if cap and features.shape[0] > cap:
        rng = np.random.default_rng(0)
        keep = rng.choice(features.shape[0], size=cap, replace=False)
        features, labels = features[keep], labels[keep]
    clf = LogisticRegression(max_iter=300, tol=1e-3)
    return float(cross_val_score(clf, features, labels, cv=cv).mean())


def scores(z: np.ndarray, y: np.ndarray, W: np.ndarray, b: np.ndarray,
           recordings: np.ndarray) -> dict:
    p = softmax(z @ W.T + b)
    pred = p.argmax(axis=1)
    per_rec = [float((pred[recordings == r] == y[recordings == r]).mean())
               for r in np.unique(recordings)]
    return {
        "n": int(len(y)),
        "accuracy": float((pred == y).mean()),
        "kappa": float(cohen_kappa_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "ece": float(expected_calibration_error(p, y)),
        "recording_accuracy_mean": float(np.mean(per_rec)),
        "recording_accuracy_worst": float(np.min(per_rec)),
        "recording_accuracy_sd": float(np.std(per_rec, ddof=1)),
        "per_recording_accuracy": per_rec,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="eegnet")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-data", type=Path,
                        default=Path("data/processed/sleep-edf-20"))
    parser.add_argument("--target", type=Path, default=Path("data/processed/hmc"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--shrink", type=float, default=10.0)
    parser.add_argument("--probe-cap", type=int, default=5000)
    parser.add_argument("--probe-cv", type=int, default=3)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    checkpoint = next(args.source.glob("*.pt"))
    run_json = json.loads(
        args.source.joinpath(f"{checkpoint.stem}.json").read_text(encoding="utf-8"))
    train_subjects = set(run_json["train_subjects"]) | set(run_json["val_subjects"])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    norm = "batch" if any(k.endswith("num_batches_tracked") for k in state) else "group"
    model = build_model(args.model, norm=norm).to(args.device)
    model.load_state_dict(state)
    model.eval()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    W = final_linear(model, args.model).weight.detach().cpu().numpy()
    b = final_linear(model, args.model).bias.detach().cpu().numpy()

    def features(x: np.ndarray, adapted: bool) -> np.ndarray:
        if adapted:
            adapt_bn(model, initial, x, args.device)
        else:
            model.load_state_dict(initial)
        return penultimate(model, args.model, x, args.device)

    # Stage means on the source training subjects, in both the cohort space and
    # the per-recording (AdaBN) space.
    source = [rec for rec in load_dataset(args.source_data)
              if rec.subject in train_subjects]
    src = {"cohort": ([], []), "adapted": ([], [])}
    for rec in source:
        keep = rec.y >= 0
        for mode in src:
            src[mode][0].append(features(rec.x[keep], mode == "adapted"))
            src[mode][1].append(rec.y[keep])
    model.load_state_dict(initial)
    templates = {}
    for mode, (zs, ys) in src.items():
        z_src, y_src = np.concatenate(zs), np.concatenate(ys)
        templates[mode] = (estimate_stage_means(z_src, y_src), z_src.mean(axis=0))
        print(f"source {mode}: {z_src.shape}, stage counts "
              f"{ {STAGES[k]: int((y_src == k).sum()) for k in range(5)} }", flush=True)

    # Target, in both spaces.
    target = [rec for rec in load_dataset(args.target)]
    tgt = {"cohort": ([], [], []), "adapted": ([], [], [])}
    for i, rec in enumerate(target):
        keep = rec.y >= 0
        for mode in tgt:
            tgt[mode][0].append(features(rec.x[keep], mode == "adapted"))
            tgt[mode][1].append(rec.y[keep])
            tgt[mode][2].append(np.full(int(keep.sum()), i))
    model.load_state_dict(initial)

    result = {"checkpoint": str(checkpoint), "target": str(args.target)}
    mu_k, mu_global = templates["cohort"]
    z, y, rec_ids = (np.concatenate(v) for v in tgt["cohort"])
    p = softmax(z @ W.T + b)
    corrected = {
        "cohort": z,
        "adabn": z - (z.mean(axis=0) - mu_global)[None, :],
        "scsn": correct(z, p, mu_k, mu_global, 1.0),
        "scsn_lam0.5": correct(z, p, mu_k, mu_global, 0.5),
    }
    # Oracle diagnostic: correct with the true stage labels. This is an upper
    # bound and uses information the method never has; it isolates the effect
    # of pseudo-label error.
    shift = {k: z[y == k].mean(axis=0) - mu_k[k] for k in mu_k}
    oracle = z.copy()
    for k, d in shift.items():
        oracle[y == k] -= d[None, :]
    corrected["scsn_oracle"] = oracle

    # Layerwise AdaBN followed by SCSN in the adapted feature space.
    mu_k_a, mu_global_a = templates["adapted"]
    z_a, y_a, rec_ids_a = (np.concatenate(v) for v in tgt["adapted"])
    p_a = softmax(z_a @ W.T + b)
    corrected["adabn_layerwise"] = z_a
    corrected["ada_scsn"] = correct(z_a, p_a, mu_k_a, mu_global_a, 1.0)
    corrected["ada_scsn_lam0.5"] = correct(z_a, p_a, mu_k_a, mu_global_a, 0.5)

    for arm, z_arm in corrected.items():
        y_use, rec_use = (y_a, rec_ids_a) if arm.startswith("ada_") else (y, rec_ids)
        entry = scores(z_arm, y_use, W, b, rec_use)
        entry["probe"] = probe(z_arm, rec_use, args.probe_cap, args.probe_cv)
        result[arm] = entry
        print(f"{arm:18s} acc={entry['accuracy']:.4f} kappa={entry['kappa']:.4f} "
              f"ece={entry['ece']:.4f} worst={entry['recording_accuracy_worst']:.4f} "
              f"sd={entry['recording_accuracy_sd']:.4f} probe={entry['probe']}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
