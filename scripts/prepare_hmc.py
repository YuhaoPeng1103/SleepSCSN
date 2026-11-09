"""Preprocess the HMC sleep-staging cohort into per-recording ``.npz`` files.

The HMC Sleep Staging Database (PhysioNet, CC-BY 4.0) contains 151 whole-night
polysomnographies recorded at a clinical sleep centre, scored for the five AASM
stages. Each recording is shipped as ``SNXXX.edf`` (signals) and
``SNXXX_sleepscoring.txt`` (one row per annotation). Records were clipped to
the scoring interval, so unlike Sleep-EDF no sleep-period crop is needed.

Example
-------
python scripts/prepare_hmc.py \
    --data-dir data/hmc/recordings \
    --out-dir data/processed/hmc \
    --channel C4-M1 \
    --resample 100 \
    --workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

# ``python -I`` (used when reading untrusted recordings) drops the script
# directory from sys.path, so add it back before importing the sibling module.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Reuse the Sleep-EDF helpers: the epoch labelling, filtering, normalisation
# and windowing rules are identical, only the file format differs.
from prepare_sleep_edf import (  # noqa: E402
    labels_for_epochs,
    map_stage,
    normalize_windows,
    read_signal,
)


def read_hypnogram_csv(path: Path) -> list[tuple[float, float, int]]:
    """Read ``(onset, duration, label)`` from the HMC scoring CSV.

    Columns are ``Date, Time, Recording onset, Duration, Annotation, Linked
    channel``. Zero-duration rows are markers (lights off/on) and map to
    label ``-1``; they are kept so the onset ordering is preserved but are
    ignored when assigning epoch labels.
    """
    events: list[tuple[float, float, int]] = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        # The header is written as ``Date, Time, Recording onset, ...``; keep
        # the spaces out of the field names before looking them up.
        if reader.fieldnames:
            reader.fieldnames = [name.strip() for name in reader.fieldnames]
        for row in reader:
            annotation = (row.get("Annotation") or "").strip()
            if not annotation:
                continue
            try:
                onset = float(row["Recording onset"])
                duration = float(row["Duration"])
            except (KeyError, TypeError, ValueError):
                continue
            events.append((onset, duration, map_stage(annotation)))
    events.sort(key=lambda item: item[0])
    return events


def process_recording(
    edf_path: Path,
    out_dir: Path,
    channel: str,
    resample: float,
    bandpass: tuple[float, float],
    epoch_sec: float,
    normalize: str,
    min_overlap: float,
) -> dict:
    recording = edf_path.stem
    subject = recording
    csv_path = edf_path.with_name(f"{recording}_sleepscoring.txt")
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    events = read_hypnogram_csv(csv_path)
    data, sfreq, channel_name = read_signal(edf_path, channel, resample, bandpass)

    epoch_len = int(round(epoch_sec * sfreq))
    n_epochs = data.size // epoch_len
    if n_epochs == 0:
        raise ValueError(f"recording too short: {edf_path.name}")
    x = data[: n_epochs * epoch_len].reshape(n_epochs, epoch_len)
    y = labels_for_epochs(events, n_epochs, epoch_sec, min_overlap=min_overlap)
    x = normalize_windows(x, normalize)

    out_path = out_dir / f"{recording}.npz"
    np.savez_compressed(
        out_path,
        x=x.astype(np.float32),
        y=y,
        subject=subject,
        recording=recording,
        sfreq=sfreq,
        epoch_sec=epoch_sec,
        channel=channel_name,
        start_epoch=np.int64(0),
    )
    counts = {int(k): int((y == k).sum()) for k in range(5)}
    return {
        "recording": recording,
        "subject": subject,
        "n_epochs": int(n_epochs),
        "n_labeled": int((y >= 0).sum()),
        "counts": counts,
        "channel": channel_name,
        "sfreq": sfreq,
    }


def _worker(kwargs: dict) -> dict:
    return process_recording(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/hmc/recordings"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed/hmc"))
    parser.add_argument("--channel", default="C4-M1")
    parser.add_argument("--resample", type=float, default=100.0)
    parser.add_argument("--bandpass", type=float, nargs=2, default=(0.5, 30.0))
    parser.add_argument("--epoch-sec", type=float, default=30.0)
    parser.add_argument("--normalize", choices=["none", "recording", "epoch"],
                        default="recording")
    parser.add_argument("--min-overlap", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    edfs = sorted(args.data_dir.glob("SN*.edf"))
    edfs = [p for p in edfs if not p.stem.endswith("_sleepscoring")]
    if args.limit:
        edfs = edfs[: args.limit]
    if not edfs:
        raise SystemExit(f"no recordings found under {args.data_dir}")

    jobs = [
        dict(
            edf_path=path,
            out_dir=args.out_dir,
            channel=args.channel,
            resample=args.resample,
            bandpass=tuple(args.bandpass),
            epoch_sec=args.epoch_sec,
            normalize=args.normalize,
            min_overlap=args.min_overlap,
        )
        for path in edfs
    ]

    records = []
    with Pool(processes=args.workers) as pool:
        for i, record in enumerate(pool.imap_unordered(_worker, jobs), 1):
            records.append(record)
            print(
                f"[{i}/{len(jobs)}] {record['recording']}: "
                f"{record['n_labeled']} labeled epochs, "
                f"channel {record['channel']}",
                flush=True,
            )

    records.sort(key=lambda item: item["recording"])
    summary = {
        "recordings": records,
        "n_recordings": len(records),
        "n_epochs": int(sum(r["n_epochs"] for r in records)),
        "n_labeled": int(sum(r["n_labeled"] for r in records)),
        "counts": {
            str(k): int(sum(r["counts"].get(str(k), r["counts"].get(k, 0))
                            for r in records))
            for k in range(5)
        },
        "channel": args.channel,
        "resample": args.resample,
        "bandpass": list(args.bandpass),
        "normalize": args.normalize,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    total = summary["n_labeled"]
    print(f"wrote {len(records)} recordings, {total} labeled epochs to {args.out_dir}")


if __name__ == "__main__":
    main()
