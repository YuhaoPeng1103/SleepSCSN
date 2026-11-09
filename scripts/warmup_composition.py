"""Stage composition of a warm-up, and of the rest of the night.

Section 5.3 of the paper explains the failure of warm-up calibration with the
fact that a warm-up is a poor sample of the night: the first ``m`` epochs of
a subject's night are mostly wake, while the rest of the night is mostly N2
and REM. This script computes that comparison from the released test
probability files, so the percentages quoted in the text are reproducible.

The warm-up is the first ``m`` epochs of a subject, exactly as
``run_sacp.py`` defines it (``--warmup-select first``), with the subject's
recordings concatenated in the order the test array stores them. Seed 0 is
sufficient: the split and the labels do not depend on the training seed, so
the script reads one backbone at seed 0 and each subject once.

Usage:
    python scripts/warmup_composition.py --root results/main \
        --out results/main/summary/warmup_composition.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

STAGES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/main"))
    parser.add_argument("--backbone", default="eegnet")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--m", type=int, default=160)
    parser.add_argument(
        "--out", type=Path,
        default=Path("results/main/summary/warmup_composition.json"),
    )
    args = parser.parse_args()

    files = sorted(
        (args.root / args.backbone).glob(f"fold*_seed{args.seed}/*_test.npz")
    )
    if not files:
        raise SystemExit(f"no test arrays under {args.root}/{args.backbone}")

    warm_counts = {k: 0 for k in STAGES}
    rest_counts = {k: 0 for k in STAGES}
    per_subject: dict[str, dict] = {}
    for path in files:
        data = np.load(path, allow_pickle=False)
        labels = data["labels"].astype(int)
        subjects = data["subjects"].astype(str)
        for subject in np.unique(subjects):
            y = labels[subjects == subject]
            if y.size == 0:
                continue
            warm, rest = y[: args.m], y[args.m:]
            entry = per_subject.setdefault(
                subject, {"warm_epochs": 0, "rest_epochs": 0, "warm": {}, "rest": {}}
            )
            entry["warm_epochs"] += int(warm.size)
            entry["rest_epochs"] += int(rest.size)
            for k in STAGES:
                warm_counts[k] += int((warm == k).sum())
                rest_counts[k] += int((rest == k).sum())
                entry["warm"][STAGES[k]] = entry["warm"].get(STAGES[k], 0) + int(
                    (warm == k).sum()
                )
                entry["rest"][STAGES[k]] = entry["rest"].get(STAGES[k], 0) + int(
                    (rest == k).sum()
                )

    def fractions(counts: dict[int, int]) -> dict[str, float]:
        total = sum(counts.values())
        return {STAGES[k]: (v / total if total else 0.0) for k, v in counts.items()}

    warm_frac, rest_frac = fractions(warm_counts), fractions(rest_counts)
    result = {
        "m": args.m,
        "warm_epochs": int(sum(warm_counts.values())),
        "rest_epochs": int(sum(rest_counts.values())),
        "warm_minutes": args.m / 2.0,
        "warm_fractions": warm_frac,
        "rest_fractions": rest_frac,
        "delta": {k: warm_frac[k] - rest_frac[k] for k in warm_frac},
        "n_subjects": len(per_subject),
        "per_subject": per_subject,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(
        f"m={args.m} ({args.m / 2:.0f} min): "
        f"wake {warm_frac['W']:.3f} against {rest_frac['W']:.3f}; "
        f"REM {warm_frac['REM']:.3f} against {rest_frac['REM']:.3f}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
