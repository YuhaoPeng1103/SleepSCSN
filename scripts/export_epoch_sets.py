"""Export epoch-level prediction sets for the hypnogram figure.

``run_sacp.py`` reports aggregate statistics; the hypnogram figure needs the
set of every individual epoch for one subject.  This script recomputes the
global and subject-adaptive sets exactly as ``run_sacp.py`` does, using the
warm-up budget and the shrinkage weight selected in its JSON, and writes a
compact ``.npz`` for the chosen subject.

Usage:
    python scripts/export_epoch_sets.py \
        --cal .../cal.npz --test .../test.npz \
        --sacp-json .../sacp.json --subject SC411 --budget 20 \
        --out .../epoch_sets_SC411.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sleepcp.conformal import (
    aps_score_matrix,
    raps_score_matrix,
    shrink_quantile,
    split_conformal_quantile,
)
from sleepcp.temporal import initial_distribution, smooth_grouped, transition_matrix


def load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cal", required=True, type=Path)
    parser.add_argument("--test", required=True, type=Path)
    parser.add_argument("--sacp-json", required=True, type=Path)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--score", default="aps", choices=["aps", "raps"])
    parser.add_argument("--smooth", action="store_true")
    parser.add_argument("--start", type=int, default=0, help="first epoch of the window")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    with open(args.sacp_json, encoding="utf-8") as f:
        sacp = json.load(f)
    entry = sacp["budget_curve"][str(args.budget)]
    shrinkage = float(entry["lambda"])

    cal_raw = load_npz(args.cal)
    test_raw = load_npz(args.test)
    cal_probs = cal_raw["probs"].astype(float)
    test_probs = test_raw["probs"].astype(float)
    cal_labels = cal_raw["labels"].astype(int)
    test_labels = test_raw["labels"].astype(int)
    test_subjects = test_raw["subjects"].astype(str)

    if args.smooth:
        trans = transition_matrix(cal_labels)
        init = initial_distribution(cal_labels)
        cal_probs = smooth_grouped(cal_probs, cal_raw["recordings"].astype(str), trans, init)
        test_probs = smooth_grouped(test_probs, test_raw["recordings"].astype(str), trans, init)

    if args.score == "raps":
        info = sacp.get("raps") or {}
        cal_scores = raps_score_matrix(cal_probs, k_reg=info.get("k_reg", 1), lam=info.get("lam", 0.01))
        test_scores = raps_score_matrix(test_probs, k_reg=info.get("k_reg", 1), lam=info.get("lam", 0.01))
    else:
        cal_scores = aps_score_matrix(cal_probs)
        test_scores = aps_score_matrix(test_probs)

    cal_true = cal_scores[np.arange(cal_labels.size), cal_labels]
    global_q = split_conformal_quantile(cal_true, sacp["alpha"])

    idx = np.flatnonzero(test_subjects == args.subject)
    if idx.size == 0:
        raise SystemExit(f"subject {args.subject} not present in the test split")
    warm = idx[: min(args.budget, idx.size)]
    local_q = (
        split_conformal_quantile(test_scores[warm, test_labels[warm]], sacp["alpha"])
        if args.budget > 0
        else float("inf")
    )
    q = shrink_quantile(local_q, global_q, shrinkage) if np.isfinite(local_q) else global_q

    global_sets = test_scores[idx] <= global_q
    sacp_sets = test_scores[idx] <= q
    recordings = test_raw["recordings"].astype(str)[idx]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        labels=test_labels[idx],
        global_sets=global_sets,
        sacp_sets=sacp_sets,
        recordings=recordings,
        start=np.array(args.start),
        q_global=np.array(global_q),
        q_sacp=np.array(q),
        shrinkage=np.array(shrinkage),
    )
    print(
        f"{args.subject}: {idx.size} epochs  q_global={global_q:.4f}  q_sacp={q:.4f} "
        f"(lambda={shrinkage:.2f})  global={global_sets.sum(axis=1).mean():.2f} "
        f"sacp={sacp_sets.sum(axis=1).mean():.2f} labels/epoch"
    )


if __name__ == "__main__":
    main()
