"""Layerwise AdaBN, and SCSN stacked on it, for one run.

Every batch-normalisation layer is re-estimated on the unlabelled epochs of
the recording being scored (AdaBN). The penultimate features of the adapted
model are then either left alone (``adabn_layerwise``) or corrected with SCSN
in the adapted feature space (``ada_scsn``). Stage prototypes for the latter
are estimated on the source training subjects, each adapted to its own
recording in the same way.

The metrics, the probe protocol and the conformal evaluation are identical to
``pilot_cssn4_fast.py`` so that the arms can be paired by fold and seed.

Example
-------
python scripts/adabn_final.py --model eegnet --fold 0 --seed 0 \
    --out results/adabn_final/edf_bn_eegnet_fold0_seed0.json
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
from sleepcp.data import load_recording  # noqa: E402
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


def adapt_bn(model: torch.nn.Module, initial: dict, x: np.ndarray, device: str) -> None:
    model.load_state_dict(initial)
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


def probe(features: np.ndarray, labels: np.ndarray) -> float | None:
    labels = np.unique(np.asarray(labels), return_inverse=True)[1]
    if len(np.unique(labels)) < 2 or np.bincount(labels).min() < 5:
        return None
    if PROBE_MAX and features.shape[0] > PROBE_MAX:
        rng = np.random.default_rng(0)
        keep = rng.choice(features.shape[0], size=PROBE_MAX, replace=False)
        features, labels = features[keep], labels[keep]
    return float(cross_val_score(LogisticRegression(max_iter=300, tol=1e-3),
                                 features, labels, cv=3).mean())


def estimate_stage_means(z: np.ndarray, y: np.ndarray) -> dict:
    return {k: z[y == k].mean(axis=0) for k in range(5) if (y == k).sum() > 0}


def correct(z: np.ndarray, probs: np.ndarray, mu_k: dict, mu_global: np.ndarray,
            lam: float, shrink: float = 10.0) -> np.ndarray:
    cond = np.zeros_like(z)
    for k, m in mu_k.items():
        w = probs[:, k]
        total = float(w.sum())
        if total <= 1e-9:
            continue
        delta = (w @ z + shrink * m) / (total + shrink) - m
        cond += probs[:, [k]] * delta[None, :]
    delta = z.mean(axis=0) - mu_global
    return z - delta - lam * (cond - delta)


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
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--probe-max", type=int, default=5000)
    parser.add_argument("--skip-stacked", action="store_true",
                        help="skip the source-subject pass and the AdaBN+SCSN arms")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    global PROBE_MAX
    PROBE_MAX = args.probe_max

    run_dir = args.root / args.model / f"fold{args.fold}_seed{args.seed}"
    stem = f"{args.model}_fold{args.fold}_seed{args.seed}"
    run_json = json.loads((run_dir / f"{stem}.json").read_text(encoding="utf-8"))
    train_subjects = sorted(set(run_json["train_subjects"]) | set(run_json["val_subjects"]))
    cal_subjects = sorted(set(run_json["cal_subjects"]))
    test_subjects = sorted(set(run_json["test_subjects"]))

    state = torch.load(run_dir / f"{stem}.pt", map_location="cpu", weights_only=True)
    model = build_model(args.model).to(args.device)
    model.load_state_dict(state)
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    W = final_linear(model, args.model).weight.detach().cpu().numpy()
    b = final_linear(model, args.model).bias.detach().cpu().numpy()

    # The flat HMC files are named after the subject, so the source-subject
    # pass can be skipped without loading the whole cohort.  Sleep-EDF files
    # are named after the recording, not the subject, and are loaded in full.
    paths = sorted(Path(args.data_dir).glob("*.npz"))
    wanted = set(cal_subjects) | set(test_subjects)
    if args.skip_stacked and any(p.stem in wanted for p in paths):
        paths = [p for p in paths if p.stem in wanted]
    recordings = {rec.recording: rec
                  for rec in (load_recording(p) for p in paths)}

    def names_for(subjects):
        return sorted({r.recording for r in recordings.values()
                       if r.subject in set(subjects)})

    def extract(names):
        zs, ys, ss, rs = [], [], [], []
        for name in names:
            rec = recordings[name]
            keep = rec.y >= 0
            adapt_bn(model, initial, rec.x[keep], args.device)
            z, _ = penultimate(model, args.model, rec.x[keep], args.device)
            zs.append(z)
            ys.append(rec.y[keep])
            ss.extend([rec.subject] * int(keep.sum()))
            rs.extend([rec.recording] * int(keep.sum()))
        return (np.concatenate(zs), np.concatenate(ys),
                np.asarray(ss), np.asarray(rs))

    z_cal, y_cal, s_cal, _ = extract(names_for(cal_subjects))
    z_test, y_test, s_test, _ = extract(names_for(test_subjects))

    results = {
        "adabn_layerwise": metrics_for(z_cal, z_test, y_cal, y_test,
                                       s_cal, s_test, W, b, args.alpha),
    }
    if not args.skip_stacked:
        z_tr, y_tr, s_tr, _ = extract(names_for(train_subjects))
        mu_k = estimate_stage_means(z_tr, y_tr)
        mu_global = z_tr.mean(axis=0)
        p_cal = softmax(z_cal @ W.T + b)
        p_test = softmax(z_test @ W.T + b)
        for lam in (1.0, 0.5):
            arm = "ada_scsn" if lam == 1.0 else f"ada_scsn_lam{lam}"
            results[arm] = metrics_for(
                correct(z_cal, p_cal, mu_k, mu_global, lam),
                correct(z_test, p_test, mu_k, mu_global, lam),
                y_cal, y_test, s_cal, s_test, W, b, args.alpha)
    model.load_state_dict(initial)

    for arm, entry in results.items():
        print(f"{arm:18s} acc={entry['accuracy']:.4f} ece={entry['ece']:.4f} "
              f"worst={entry['worst_subject_coverage']:.4f} "
              f"sd={entry['coverage_sd']:.4f} probe={entry['probe']} "
              f"within={entry['probe_within']}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "model": args.model, "fold": args.fold, "seed": args.seed,
        "results": results,
    }, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
