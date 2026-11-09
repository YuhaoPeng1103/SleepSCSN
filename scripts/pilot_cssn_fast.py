"""Pilot for stage-conditional subject-subspace normalisation (CSSN).

This is a *post-hoc* pilot on existing batch-normalised checkpoints. It does
not retrain anything. For a given run it

1. extracts penultimate features for the training, calibration and test
   recordings of one fold;
2. estimates, per sleep stage, the between-subject covariance of the
   per-subject class-conditional feature means, and keeps its top-r
   eigenvectors as the stage-conditional subject subspace;
3. evaluates several corrections of the recording's feature mean:
   - ``bn``: no correction;
   - ``marginal``: subtract the recording mean shift (post-hoc AdaBN);
   - ``cond_full``: subtract the soft-stage-weighted class-conditional shift;
   - ``cond_r{r}``: subtract only its projection on the stage-conditional
     subject subspace;
   - ``cond_r{r}_orth``: same, with the subspace orthogonalised against the
     classifier's row space (predictions should be unchanged exactly).

Everything is paired: all arms share the same features, so differences are
attributable to the correction.

Example
-------
python scripts/pilot_cssn.py --model eegnet --fold 0 --seed 0 \
    --out results/pilot_cssn/eegnet_fold0_seed0.json
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
from sleepcp.metrics import (cohen_kappa_score, coverage,  # noqa: E402
                             expected_calibration_error, f1_score,
                             review_workload, set_size)
from sleepcp.models import build_model  # noqa: E402

STAGES = ("W", "N1", "N2", "N3", "REM")
TRAIN_REC = "train"
CAL_REC = "cal"
TEST_REC = "test"


def final_linear(model: torch.nn.Module, model_name: str) -> torch.nn.Linear:
    if model_name == "eegnet":
        return model.fc
    return model.head[3]


def penultimate(model: torch.nn.Module, model_name: str, x: np.ndarray,
                device: str = "cpu", batch_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Return penultimate features and logits for ``x`` of shape (n, T)."""
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
                z = model.head[0](z)  # adaptive average pool
                z = model.head[1](z)  # flatten
                z = model.head[2](z)  # dropout (identity in eval)
            z = torch.flatten(z, 1)
            feats.append(z.cpu().numpy())
            logits.append(final_linear(model, model_name)(z).cpu().numpy())
    return np.concatenate(feats), np.concatenate(logits)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def hard_probs(probs: np.ndarray) -> np.ndarray:
    """One-hot pseudo-labels from the model's own argmax prediction."""
    out = np.zeros_like(probs)
    out[np.arange(probs.shape[0]), probs.argmax(axis=1)] = 1.0
    return out


def extract_split(recordings, names, model, model_name, device):
    zs, ys, ss, rs = [], [], [], []
    for name in names:
        rec = recordings[name]
        keep = rec.y >= 0
        z, _ = penultimate(model, model_name, rec.x[keep], device)
        zs.append(z)
        ys.append(rec.y[keep])
        ss.extend([rec.subject] * int(keep.sum()))
        rs.extend([rec.recording] * int(keep.sum()))
    return (np.concatenate(zs), np.concatenate(ys), np.asarray(ss), np.asarray(rs))


def estimate_subspaces(z: np.ndarray, y: np.ndarray, subjects: np.ndarray,
                       rank: int, min_epochs: int = 20):
    """Per-stage subject means, between-subject covariance and top-r bases."""
    subject_ids = sorted(set(subjects.tolist()))
    mu_k, U_k, U_orth_k, counts = {}, {}, {}, {}
    for k in range(5):
        mask_k = y == k
        if mask_k.sum() == 0:
            continue
        mu_k[k] = z[mask_k].mean(axis=0)
        means = []
        for s in subject_ids:
            m = mask_k & (subjects == s)
            if m.sum() >= min_epochs:
                means.append(z[m].mean(axis=0))
        counts[k] = len(means)
        if len(means) < 3:
            continue
        means = np.asarray(means)
        # Between-subject covariance of class-conditional means. Each subject
        # contributes one point, so a subject with many epochs does not
        # dominate the subspace.
        centred = means - means.mean(axis=0, keepdims=True)
        cov = centred.T @ centred / max(len(means) - 1, 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        r = min(rank, len(means) - 1, z.shape[1])
        U_k[k] = eigvecs[:, order[:r]]
    return mu_k, U_k, counts


def orthogonalise(U: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Remove from the columns of U their projection on the row space of W.

    Returns an orthonormal basis for the surviving directions. The number of
    columns can shrink when the subject subspace lies almost entirely in the
    classifier's row space, which is itself a useful diagnostic.
    """
    Q, _ = np.linalg.qr(W.T)  # (d, rank)
    Uo = U - Q @ (Q.T @ U)
    Uu, s, _ = np.linalg.svd(Uo, full_matrices=False)
    return Uu[:, s > 1e-8]


def estimate_recording_shift(z: np.ndarray, probs: np.ndarray, mu_k: dict,
                             shrink: float = 1.0) -> dict:
    """Soft-stage-weighted class-conditional mean shift for one recording.

    ``shrink`` is the pseudo-count with which each stage mean is pulled toward
    the training cohort's class mean. It matters for rare stages (N1) where a
    recording has little mass.
    """
    out = {}
    for k, m in mu_k.items():
        w = probs[:, k]
        total = float(w.sum())
        if total <= 1e-9:
            continue
        out[k] = (w @ z + shrink * m) / (total + shrink) - m
    return out


def apply_arm(z: np.ndarray, probs: np.ndarray, shift: dict, arm: str,
              U: dict | None = None, lam: float = 1.0) -> np.ndarray:
    """Return the corrected features for one arm."""
    if arm == "bn":
        return z
    if arm == "marginal":
        # ``shift`` carries the global mean under key -1 when available.
        delta = shift.get(-1, np.zeros(z.shape[1]))
        return z - lam * delta
    correction = np.zeros_like(z)
    for k, d in shift.items():
        if k < 0:
            continue
        d_use = d
        if U is not None and k in U:
            d_use = U[k] @ (U[k].T @ d)
        correction += probs[:, [k]] * d_use[None, :]
    return z - lam * correction


PROBE_MAX = 0


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


def metrics_for(z: np.ndarray, W: np.ndarray, b: np.ndarray, y: np.ndarray,
                subjects: np.ndarray) -> dict:
    logits = z @ W.T + b
    p = softmax(logits)
    pred = p.argmax(axis=1)
    return {
        "accuracy": float((pred == y).mean()),
        "kappa": float(cohen_kappa_score(y, pred)),
        "mf1": float(f1_score(y, pred, average="macro")),
        "ece": float(expected_calibration_error(p, y)),
        "probe": probe(z, subjects),
        "probe_within": {
            STAGES[k]: probe(z[y == k], subjects[y == k])
            for k in range(5) if (y == k).sum() >= 50
        },
    }


def select_lambda(z_cal, c_cal, y_cal, s_cal, r_cal, W, b, grid, alpha):
    """Choose the correction strength on held-out calibration recordings.

    The calibration recordings are split into two halves; for each candidate
    ``lam`` the conformal quantile is computed on one half and per-subject
    coverage on the other, in both directions. The selected value minimises
    the average coverage dispersion subject to the average coverage staying
    within one point of the nominal level; if no candidate meets the
    constraint, the candidate with the highest average coverage is used.
    """
    recs = sorted(set(r_cal.tolist()))
    halves = (recs[0::2], recs[1::2])
    rows = []
    for lam in grid:
        p = softmax((z_cal - lam * c_cal) @ W.T + b)
        scores = aps_score_matrix(p)
        stats = []
        for score_half, eval_half in ((halves[0], halves[1]), (halves[1], halves[0])):
            sm = np.isin(r_cal, score_half)
            em = np.isin(r_cal, eval_half)
            q = split_conformal_quantile(
                scores[sm][np.arange(sm.sum()), y_cal[sm]], alpha)
            sets = scores[em] <= q
            covs = [float(coverage(sets[s_cal[em] == sub], y_cal[em][s_cal[em] == sub]))
                    for sub in np.unique(s_cal[em])]
            stats.append((float(np.mean(covs)), float(np.std(covs, ddof=1)),
                          float(np.min(covs))))
        rows.append({"lambda": float(lam),
                     "mean": float(np.mean([s[0] for s in stats])),
                     "sd": float(np.mean([s[1] for s in stats])),
                     "worst": float(np.min([s[2] for s in stats]))})
    ok = [row for row in rows if row["mean"] >= 1.0 - alpha - 0.01]
    best = min(ok, key=lambda row: (row["sd"], -row["mean"])) if ok else \
        max(rows, key=lambda row: row["mean"])
    return best["lambda"], rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="eegnet")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--root", type=Path, default=Path("results/main"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/sleep-edf-20"))
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--min-epochs", type=int, default=20)
    parser.add_argument("--shrink", type=float, default=10.0)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--probe-max", type=int, default=0,
                        help="subsample at most this many epochs for each probe "
                             "(0 = use all; useful on large cohorts)")
    parser.add_argument("--arms", default=None,
                        help="comma-separated arm names to evaluate (default: all)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    global PROBE_MAX
    PROBE_MAX = args.probe_max

    run_dir = args.root / args.model / f"fold{args.fold}_seed{args.seed}"
    stem = f"{args.model}_fold{args.fold}_seed{args.seed}"
    run_json = json.loads((run_dir / f"{stem}.json").read_text(encoding="utf-8"))
    train_subjects = sorted(set(run_json["train_subjects"]) | set(run_json["val_subjects"]))
    cal_subjects = sorted(set(run_json["cal_subjects"]))
    test_subjects = sorted(set(run_json["test_subjects"]))

    recordings = {rec.recording: rec for rec in load_dataset(args.data_dir)}
    state = torch.load(run_dir / f"{stem}.pt", map_location="cpu", weights_only=True)
    norm = "batch" if any(k.endswith("num_batches_tracked") for k in state) else "group"
    model = build_model(args.model, norm=norm).to(args.device)
    model.load_state_dict(state)
    model.eval()
    W = final_linear(model, args.model).weight.detach().cpu().numpy()
    b = final_linear(model, args.model).bias.detach().cpu().numpy()

    cache = args.cache or Path("results/pilot_cssn") / f"{stem}_features.npz"
    if cache.exists():
        with np.load(cache) as d:
            data = {k: d[k] for k in d.files}
        print(f"loaded cache {cache}")
    else:
        paths = list(recordings.values())
        def names_for(keep):
            return sorted({r.recording for r in paths if r.subject in set(keep)})
        data = {}
        for tag, keep in ((TRAIN_REC, train_subjects), (CAL_REC, cal_subjects),
                          (TEST_REC, test_subjects)):
            z, y, s, r = extract_split(recordings, names_for(keep), model,
                                       args.model, args.device)
            data[f"z_{tag}"], data[f"y_{tag}"] = z, y
            data[f"s_{tag}"], data[f"r_{tag}"] = s, r
            print(f"{tag}: {z.shape} from {len(names_for(keep))} recordings", flush=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **data)
        print(f"wrote cache {cache}")

    z_tr, y_tr, s_tr = data["z_train"], data["y_train"], data["s_train"]
    mu_k, U_r, counts = estimate_subspaces(z_tr, y_tr, s_tr, args.rank, args.min_epochs)
    mu_global = z_tr.mean(axis=0)
    U_orth = {k: orthogonalise(U, W) for k, U in U_r.items()}
    print("per-stage subjects used:", counts)

    # Build arms once. ``rank`` in the name is the requested rank; some stages
    # may contribute fewer components.
    arms = ["bn", "marginal", "cond_full", "cond_hard",
            "cond_full_auto", "cond_hard_auto"]
    for r in (1, 2, 4, 8):
        arms += [f"cond_r{r}", f"cond_r{r}_orth"]
    arms += ["cond_full_lam0.25", "cond_full_lam0.5", "cond_full_lam0.75",
             "cond_hard_lam0.5", "cond_r4_lam0.5"]
    if args.arms:
        wanted = [a.strip() for a in args.arms.split(",") if a.strip()]
        unknown = [a for a in wanted if a not in arms]
        if unknown:
            raise ValueError(f"unknown arms: {unknown}")
        arms = wanted

    def mean_shift(z, probs):
        shift = estimate_recording_shift(z, probs, mu_k, args.shrink)
        shift[-1] = z.mean(axis=0) - mu_global
        return shift

    # Precompute per-recording shifts from the *uncorrected* probabilities.
    rec_shifts = {}
    for tag in (CAL_REC, TEST_REC):
        z, y, s, r = data[f"z_{tag}"], data[f"y_{tag}"], data[f"s_{tag}"], data[f"r_{tag}"]
        p = softmax(z @ W.T + b)
        for name in sorted(set(r.tolist())):
            m = r == name
            rec_shifts[(tag, name)] = mean_shift(z[m], p[m])

    results = {}
    cal_labels = data["y_cal"]
    cal_subjects_arr = data["s_cal"]
    cal_recordings_arr = data["r_cal"]
    test_labels = data["y_test"]
    test_subjects_arr = data["s_test"]

    for arm in arms:
        lam = 1.0
        base = arm
        if "_lam" in arm:
            base, lam_s = arm.split("_lam", 1)
            lam = float(lam_s)
        if base in ("cond_full", "cond_hard", "cond_full_auto", "cond_hard_auto"):
            U_use = None
        elif base == "cond_r1":
            U_use = {k: v[:, :1] for k, v in U_r.items()}
        elif base == "cond_r2":
            U_use = {k: v[:, :2] for k, v in U_r.items()}
        elif base == "cond_r4":
            U_use = {k: v[:, :4] for k, v in U_r.items()}
        elif base == "cond_r8":
            U_use = {k: v[:, :8] for k, v in U_r.items()}
        elif base.endswith("_orth"):
            U_use = U_orth
            if base.startswith("cond_r1"):
                U_use = {k: v[:, :1] for k, v in U_orth.items()}
            elif base.startswith("cond_r2"):
                U_use = {k: v[:, :2] for k, v in U_orth.items()}
            elif base.startswith("cond_r4"):
                U_use = {k: v[:, :4] for k, v in U_orth.items()}
            elif base.startswith("cond_r8"):
                U_use = {k: v[:, :8] for k, v in U_orth.items()}
        else:
            U_use = None

        def corrected(tag):
            z = data[f"z_{tag}"]
            r = data[f"r_{tag}"]
            out = np.empty_like(z)
            for name in sorted(set(r.tolist())):
                m = r == name
                p = softmax(z[m] @ W.T + b)
                if base == "cond_hard":
                    p = hard_probs(p)
                    shift = mean_shift(z[m], p)
                else:
                    shift = rec_shifts[(tag, name)]
                out[m] = apply_arm(z[m], p, shift, base, U_use, lam)
            return out

        def correction_vectors(tag):
            """Per-epoch conditional correction before any shrinkage."""
            z = data[f"z_{tag}"]
            r = data[f"r_{tag}"]
            out = np.zeros_like(z)
            for name in sorted(set(r.tolist())):
                m = r == name
                p = softmax(z[m] @ W.T + b)
                if base == "cond_hard_auto":
                    p = hard_probs(p)
                    shift = mean_shift(z[m], p)
                else:
                    shift = rec_shifts[(tag, name)]
                for k, d in shift.items():
                    if k < 0:
                        continue
                    out[m] += p[:, [k]] * d[None, :]
            return out

        def evaluate(z_cal_c, z_test_c, lam_used, extra=None):
            cal_p = softmax(z_cal_c @ W.T + b)
            test_p = softmax(z_test_c @ W.T + b)
            q = split_conformal_quantile(
                aps_score_matrix(cal_p)[np.arange(cal_labels.size), cal_labels],
                args.alpha)
            sets = aps_score_matrix(test_p) <= q
            per_subject = {}
            for subject in np.unique(test_subjects_arr):
                m = test_subjects_arr == subject
                per_subject[str(subject)] = float(coverage(sets[m], test_labels[m]))
            entry = {
                "arm": arm,
                "lambda": lam_used,
                "accuracy": float((test_p.argmax(axis=1) == test_labels).mean()),
                "kappa": float(cohen_kappa_score(test_labels, test_p.argmax(axis=1))),
                "mf1": float(f1_score(test_labels, test_p.argmax(axis=1), average="macro")),
                "ece": float(expected_calibration_error(test_p, test_labels)),
                "coverage": float(coverage(sets, test_labels)),
                "set_size": float(set_size(sets)),
                "review_workload": float(review_workload(sets)),
                "worst_subject_coverage": float(min(per_subject.values())),
                "coverage_sd": float(np.std(list(per_subject.values()), ddof=1)),
                "per_subject_coverage": per_subject,
                "probe": probe(z_test_c, test_subjects_arr),
                "probe_within": {
                    STAGES[k]: probe(z_test_c[test_labels == k],
                                     test_subjects_arr[test_labels == k])
                    for k in range(5) if (test_labels == k).sum() >= 50
                },
            }
            if extra:
                entry.update(extra)
            results[arm] = entry
            print(f"{arm:24s} acc={entry['accuracy']:.4f} cov={entry['coverage']:.4f} "
                  f"worst={entry['worst_subject_coverage']:.4f} sd={entry['coverage_sd']:.4f} "
                  f"set={entry['set_size']:.3f} probe={entry['probe']} "
                  f"within={entry['probe_within']}", flush=True)

        if base in ("cond_full_auto", "cond_hard_auto"):
            c_cal = correction_vectors(CAL_REC)
            c_test = correction_vectors(TEST_REC)
            grid = (0.0, 0.25, 0.5, 0.75, 1.0)
            chosen, rows = select_lambda(
                data["z_cal"], c_cal, cal_labels, cal_subjects_arr,
                cal_recordings_arr, W, b, grid, args.alpha)
            print(f"{arm}: selected lambda={chosen}; {rows}", flush=True)
            evaluate(data["z_cal"] - chosen * c_cal,
                     data["z_test"] - chosen * c_test, chosen,
                     {"selected_lambda": chosen, "selection": rows})
            continue

        z_cal = corrected(CAL_REC)
        z_test = corrected(TEST_REC)
        evaluate(z_cal, z_test, lam)

    out = args.out or Path("results/pilot_cssn") / f"{stem}_pilot.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "fold": args.fold, "seed": args.seed,
        "train_subjects": train_subjects, "cal_subjects": cal_subjects,
        "test_subjects": test_subjects, "stage_subject_counts": counts,
        "rank": args.rank, "shrink": args.shrink, "results": results,
    }, indent=1), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
