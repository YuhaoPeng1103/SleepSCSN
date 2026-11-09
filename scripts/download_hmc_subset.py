"""Download a handful of HMC recordings without fetching the whole archive.

Useful for a pipeline smoke test before the 13.8 GB archive finishes, or for a
small-scale reproduction. Reuses the ranged-request fetcher from
``download_hmc.py`` so the transfer behaves the same way as the full download.

Example
-------
python scripts/download_hmc_subset.py \
    --out data/hmc/recordings --ids SN002 SN003 SN004 \
    --workers 4 --chunks-per-file 6
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from download_hmc import fetch_chunk  # noqa: E402

BASE = "https://physionet-open.s3.amazonaws.com/hmc-sleep-staging/1.1/recordings"


def download_file(url: str, out: Path, workers: int, chunks_per_file: int) -> None:
    if out.exists() and out.stat().st_size > 0:
        print(f"  keep {out.name} ({out.stat().st_size / 1e6:.1f} MB)")
        return
    response = requests.head(url, allow_redirects=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers["Content-Length"])
    chunk = (total + chunks_per_file - 1) // chunks_per_file
    tmp = out.with_suffix(out.suffix + ".tmp")
    part_dir = out.with_name(out.name + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for index in range(chunks_per_file):
        start = index * chunk
        end = min(start + chunk, total) - 1
        if start > end:
            break
        jobs.append((part_dir / f"part_{index:03d}", start, end))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(fetch_chunk, url, path, start, end)
            for path, start, end in jobs
        ]
        for future in as_completed(futures):
            future.result()
    with open(tmp, "wb") as handle:
        for path, _, _ in jobs:
            with open(path, "rb") as source:
                handle.write(source.read())
    if tmp.stat().st_size != total:
        raise RuntimeError(f"{out.name}: {tmp.stat().st_size} != {total}")
    tmp.replace(out)
    for path, _, _ in jobs:
        path.unlink()
    part_dir.rmdir()
    print(f"  wrote {out.name} ({total / 1e6:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("data/hmc/recordings"))
    parser.add_argument("--ids", nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunks-per-file", type=int, default=6)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for rec in args.ids:
        for name in (f"{rec}.edf", f"{rec}_sleepscoring.txt"):
            url = f"{BASE}/{name}"
            out = args.out / name
            print(f"{name}", flush=True)
            try:
                download_file(url, out, args.workers, args.chunks_per_file)
            except Exception as exc:  # noqa: BLE001
                print(f"  failed: {exc}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
