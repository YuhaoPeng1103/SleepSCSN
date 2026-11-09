"""Preprocess Sleep-EDF PSG/hypnogram pairs into per-recording ``.npz`` files.

Example
-------
python scripts/prepare_sleep_edf.py \
    --data-dir data/sleep-edf/sleep-cassette \
    --out-dir data/processed/sleep-edf-20 \
    --subset sleep-edf-20 \
    --channel Fpz-Cz \
    --resample 100
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

STAGE_MAP = {
    "sleep stage w": 0,
    "sleep stage 1": 1,
    "sleep stage n1": 1,
    "sleep stage 2": 2,
    "sleep stage n2": 2,
    "sleep stage 3": 3,
    "sleep stage n3": 3,
    # R&K stage 4 is deep sleep: it merges into N3 under AASM, not into REM.
    "sleep stage 4": 3,
    "sleep stage n4": 3,
    "sleep stage r": 4,
    "sleep stage rem": 4,
    "movement time": -1,
    "sleep stage ?": -1,
    "sleep stage m": -1,
}

# Sleep-EDF-20 uses the SC4xx subjects with two-digit subject numbers 00-19.
SLEEP_EDF_20_SUBJECTS = {f"SC4{i:02d}" for i in range(20)}


def parse_recording_id(path: Path) -> tuple[str, str]:
    """Return ``(subject, recording)`` from a Sleep-EDF file stem.

    ``SC4001E0-PSG.edf`` -> subject ``SC400``, recording ``SC4001E0``.
    """
    stem = path.name.split("-")[0]
    match = re.match(r"^(SC|ST)(\d{4})([A-Z0-9]+)$", stem)
    if not match:
        raise ValueError(f"cannot parse Sleep-EDF recording id from {path.name}")
    prefix, number, suffix = match.groups()
    subject = f"{prefix}{number[:3]}"
    return subject, stem


def map_stage(description: str) -> int:
    return STAGE_MAP.get(description.strip().lower(), -1)


def _call(fn, *args, **kwargs):
    """Call an MNE function, dropping ``verbose`` where MNE removed it.

    MNE 1.13 removed the ``verbose`` keyword from ``read_annotations`` (and
    from several raw methods); older versions require it to stay quiet. Rather
    than pin a version, retry once without the keyword.
    """
    try:
        return fn(*args, **kwargs)
    except TypeError as exc:
        if "verbose" not in str(exc):
            raise
        kwargs.pop("verbose", None)
        return fn(*args, **kwargs)


def read_hypnogram(path: Path) -> list[tuple[float, float, int]]:
    """Read hypnogram annotations as ``(onset, duration, label)``."""
    import mne

    annotations = _call(mne.read_annotations, path, verbose="ERROR")
    events = []
    for onset, duration, description in zip(
        annotations.onset, annotations.duration, annotations.description
    ):
        label = map_stage(description)
        events.append((float(onset), float(duration), label))
    # Fill zero-duration annotations with the gap to the next event.
    events.sort(key=lambda e: e[0])
    filled = []
    for i, (onset, duration, label) in enumerate(events):
        if duration <= 0:
            next_onset = events[i + 1][0] if i + 1 < len(events) else onset + 30.0
            duration = max(next_onset - onset, 30.0)
        filled.append((onset, duration, label))
    return filled


def labels_for_epochs(
    events: list[tuple[float, float, int]],
    n_epochs: int,
    epoch_sec: float,
    min_overlap: float = 0.5,
) -> np.ndarray:
    labels = np.full(n_epochs, -1, dtype=np.int64)
    for i in range(n_epochs):
        start = i * epoch_sec
        end = start + epoch_sec
        best_overlap = 0.0
        best_label = -1
        for onset, duration, label in events:
            if label < 0:
                continue
            overlap = max(0.0, min(end, onset + duration) - max(start, onset))
            if overlap > best_overlap:
                best_overlap = overlap
                best_label = label
        if best_overlap >= min_overlap * epoch_sec:
            labels[i] = best_label
    return labels


def sleep_window(
    labels: np.ndarray, epoch_sec: float, wake_margin_min: float
) -> tuple[int, int]:
    """Contiguous epoch range spanning the scored night.

    Sleep-EDF recordings are ~24 h long but only the sleep period is scored:
    keeping every annotated epoch yields a cohort that is 68% wake and a class
    distribution that no published result matches. We keep the first to the
    last epoch scored as a sleep stage (N1-N3/REM) plus ``wake_margin_min``
    minutes of wake on either side, which reproduces the canonical
    ~42.3k-epoch Sleep-EDF-20 splits. A negative margin disables cropping.

    Returns inclusive ``(lo, hi)`` indices; unlabeled epochs inside the window
    stay in place so epoch alignment with the original recording is preserved.
    """
    if wake_margin_min < 0:
        return 0, labels.size - 1
    sleep = np.flatnonzero((labels >= 1) & (labels <= 4))
    if sleep.size == 0:
        return 0, labels.size - 1
    margin = int(round(wake_margin_min * 60.0 / epoch_sec))
    return max(0, int(sleep[0]) - margin), min(labels.size - 1, int(sleep[-1]) + margin)


def read_signal(path: Path, channel: str, resample: float, bandpass: tuple[float, float]) -> tuple[np.ndarray, float, str]:
    import mne

    raw = _call(mne.io.read_raw_edf, path, preload=True, verbose="ERROR")
    names = raw.ch_names
    matches = [name for name in names if channel.lower() in name.lower()]
    if not matches:
        raise ValueError(f"channel {channel!r} not found in {path.name}; available: {names}")
    raw.pick_channels([matches[0]])
    if resample and abs(raw.info["sfreq"] - resample) > 1e-6:
        _call(raw.resample, resample, verbose="ERROR")
    if bandpass is not None:
        _call(raw.filter, bandpass[0], bandpass[1], verbose="ERROR")
    data = raw.get_data()[0].astype(np.float64)
    return data, float(raw.info["sfreq"]), matches[0]


def normalize_windows(x: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return x
    if mode == "recording":
        return (x - x.mean()) / (x.std() + 1e-8)
    if mode == "epoch":
        mean = x.mean(axis=1, keepdims=True)
        std = x.std(axis=1, keepdims=True)
        return (x - mean) / (std + 1e-8)
    raise ValueError(f"unknown normalize mode: {mode}")


def process_pair(
    psg_path: Path,
    hyp_path: Path,
    out_dir: Path,
    channel: str,
    resample: float,
    bandpass: tuple[float, float],
    epoch_sec: float,
    normalize: str,
    min_overlap: float,
    wake_margin_min: float = 30.0,
) -> dict:
    subject, recording = parse_recording_id(psg_path)
    events = read_hypnogram(hyp_path)
    data, sfreq, channel_name = read_signal(psg_path, channel, resample, bandpass)

    epoch_len = int(round(epoch_sec * sfreq))
    n_epochs = data.size // epoch_len
    if n_epochs == 0:
        raise ValueError(f"recording too short: {psg_path.name}")
    x = data[: n_epochs * epoch_len].reshape(n_epochs, epoch_len)
    # Labels come first: normalization is per recording and must be computed on
    # the same samples that are kept, so the crop happens before normalization.
    y = labels_for_epochs(events, n_epochs, epoch_sec, min_overlap=min_overlap)
    lo, hi = sleep_window(y, epoch_sec, wake_margin_min)
    x = normalize_windows(x[lo : hi + 1], normalize)
    y = y[lo : hi + 1]

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
        start_epoch=np.int64(lo),
    )

    counts = {int(k): int((y == k).sum()) for k in range(5)}
    return {
        "recording": recording,
        "subject": subject,
        "psg": psg_path.name,
        "hypnogram": hyp_path.name,
        "n_epochs": int(y.size),
        "n_epochs_full": int(n_epochs),
        "start_epoch": int(lo),
        "n_labeled": int((y >= 0).sum()),
        "n_unlabeled": int((y < 0).sum()),
        "label_counts": counts,
        "channel": channel_name,
        "sfreq": sfreq,
    }


def find_pairs(data_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for psg in sorted(data_dir.glob("*-PSG.edf")):
        # The PSG and hypnogram suffixes differ (e.g. SC4001E0-PSG.edf and
        # SC4001EC-Hypnogram.edf), so match on the six-character recording
        # prefix instead of the full stem.
        prefix = psg.name[:6]
        candidates = sorted(data_dir.glob(f"{prefix}*-Hypnogram.edf"))
        if not candidates:
            print(f"[warn] no hypnogram for {psg.name}")
            continue
        pairs.append((psg, candidates[0]))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--subset", default="sleep-edf-20", choices=["sleep-edf-20", "sleep-edf-78", "all"])
    parser.add_argument("--channel", default="Fpz-Cz")
    parser.add_argument("--resample", type=float, default=100.0)
    parser.add_argument("--bandpass", type=float, nargs=2, default=(0.5, 30.0))
    parser.add_argument("--epoch-sec", type=float, default=30.0)
    parser.add_argument("--normalize", default="recording", choices=["recording", "epoch", "none"])
    parser.add_argument("--min-overlap", type=float, default=0.5)
    parser.add_argument(
        "--wake-margin-min",
        type=float,
        default=30.0,
        help="Minutes of wake kept on either side of the scored sleep period; "
        "negative keeps every annotated epoch (not recommended: ~68%% wake).",
    )
    parser.add_argument("--limit", type=int, default=0, help="process only the first N pairs (debug)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs = find_pairs(args.data_dir)
    if args.subset == "sleep-edf-20":
        pairs = [(p, h) for p, h in pairs if parse_recording_id(p)[0] in SLEEP_EDF_20_SUBJECTS]
    if args.limit:
        pairs = pairs[: args.limit]
    if not pairs:
        raise SystemExit(f"no PSG/hypnogram pairs found in {args.data_dir}")

    metadata = []
    for i, (psg, hyp) in enumerate(pairs, start=1):
        try:
            meta = process_pair(
                psg,
                hyp,
                args.out_dir,
                args.channel,
                args.resample,
                tuple(args.bandpass),
                args.epoch_sec,
                args.normalize,
                args.min_overlap,
                args.wake_margin_min,
            )
            metadata.append(meta)
            print(f"[{i}/{len(pairs)}] {meta['recording']}: {meta['n_labeled']} labeled epochs")
        except Exception as exc:
            print(f"[{i}/{len(pairs)}] FAILED {psg.name}: {exc}")

    summary = {
        "data_dir": str(args.data_dir),
        "out_dir": str(args.out_dir),
        "subset": args.subset,
        "channel_requested": args.channel,
        "resample": args.resample,
        "bandpass": list(args.bandpass),
        "epoch_sec": args.epoch_sec,
        "normalize": args.normalize,
        "wake_margin_min": args.wake_margin_min,
        "n_recordings": len(metadata),
        "n_subjects": len({m["subject"] for m in metadata}),
        "recordings": metadata,
    }
    with open(args.out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {len(metadata)} recordings to {args.out_dir}")
    print(f"subjects: {summary['n_subjects']}")


if __name__ == "__main__":
    main()
