"""Control experiment for AdaBN: matched versus mismatched statistics.

If the drop in the subject probe under AdaBN were an artefact of the network
seeing unusual statistics rather than of the statistics being *the right
ones*, then statistics estimated on a different recording should do the same
thing. This script measures the other direction: every test recording is
evaluated with the statistics of another recording of the same test fold
(cyclic shift), so the magnitude and the structure of the statistics are
unchanged but their correspondence to the recording is destroyed.

Three numbers per run are produced:

``cohort``      the frozen training-cohort statistics (no adaptation),
``matched``     statistics of the recording itself (AdaBN),
``mismatched``  statistics of the next recording of the fold.

The claim predicts ``cohort`` ~ ``mismatched`` >> ``matched`` for the subject
probe, and no accuracy difference between the three beyond noise.

Example
-------
python scripts/ablate_matched_stats.py --root results/main --model eegnet \
    --device cuda --out results/norm/adabn_control_eegnet.json
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

from adapt_bn_stats import adapt, features_and_probs, probe  # noqa: E402

from sleepcp.data import load_dataset  # noqa: E402
from sleepcp.models import build_model  # noqa: E402


def evaluate(model, recordings, names: list[str], stats_from: list[str],
             device: str) -> tuple[np.ndarray, np.ndarray]:
    """Features and probabilities of ``names``, using ``stats_from`` for BN."""
    feats, probs = [], []
    for name, source in zip(names, stats_from):
        if source is None:
            model.eval()
        else:
            adapt(model, recordings[source].x, device)
        f, p = features_and_probs(model, recordings[name].x, device)
        keep = recordings[name].y >= 0
        feats.append(f[keep])
        probs.append(p[keep])
    return np.concatenate(feats), np.concatenate(probs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("results/main"))
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/processed/sleep-edf-20"))
    parser.add_argument("--model", default="eegnet")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    recordings = {rec.recording: rec for rec in load_dataset(args.data_dir)}
    rows = []
    for test_path in sorted(args.root.glob(f"{args.model}/fold*_seed*/*_test.npz")):
        checkpoint = Path(str(test_path)[: -len("_test.npz")] + ".pt")
        if not checkpoint.exists():
            continue
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not any(k.endswith("num_batches_tracked") for k in state):
            continue
        model = build_model(args.model, norm="batch").to(args.device)
        model.load_state_dict(state)
        model.eval()

        with np.load(test_path, allow_pickle=False) as data:
            subjects = np.asarray(data["subjects"])
            names = sorted(set(data["recordings"].tolist()))
        if len(names) < 2:
            continue
        shifted = names[1:] + names[:1]

        entry = {"run": str(test_path.parent), "recordings": names}
        for mode, source in (("cohort", [None] * len(names)),
                             ("matched", names),
                             ("mismatched", shifted)):
            f, p = evaluate(model, recordings, names, source, args.device)
            entry[f"probe_{mode}"] = probe(f, subjects)
            entry[f"accuracy_{mode}"] = float((p.argmax(axis=1) ==
                                               np.concatenate(
                                                   [recordings[n].y[recordings[n].y >= 0]
                                                    for n in names])).mean())
        rows.append(entry)
        print(f"{test_path.parent}: cohort={entry['probe_cohort']:.3f} "
              f"matched={entry['probe_matched']:.3f} "
              f"mismatched={entry['probe_mismatched']:.3f} "
              f"acc={entry['accuracy_cohort']:.3f}/{entry['accuracy_matched']:.3f}/"
              f"{entry['accuracy_mismatched']:.3f}", flush=True)

    summary = {key: float(np.nanmean([r[key] for r in rows]))
               for key in rows[0] if key.startswith(("probe_", "accuracy_"))}
    summary["n_runs"] = len(rows)
    summary["model"] = args.model
    out = args.out or args.root / "summary" / f"adabn_control_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "runs": rows}, indent=1),
                   encoding="utf-8")
    print(json.dumps(summary, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
