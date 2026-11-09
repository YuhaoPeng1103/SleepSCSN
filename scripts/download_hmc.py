"""Download the HMC sleep-staging archive with resumable ranged requests.

PhysioNet serves the archive from object storage, and a single connection is
throttled to a few tens of kilobytes per second. The script splits the file
into fixed-size chunks, fetches them over several connections, and keeps the
finished chunks on disk, so an interrupted download resumes instead of
starting over.

Example
-------
python scripts/download_hmc.py \
    --url https://physionet-open.s3.amazonaws.com/hmc-sleep-staging/hmc-sleep-staging-1.1.zip \
    --out data/hmc/hmc.zip --workers 24
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


def remote_size(url: str) -> int:
    response = requests.head(url, allow_redirects=True, timeout=60)
    response.raise_for_status()
    size = response.headers.get("Content-Length")
    if size is None:
        raise RuntimeError("server did not report Content-Length")
    return int(size)


def fetch_chunk(url: str, path: Path, start: int, end: int, retries: int = 6) -> int:
    """Fetch one inclusive byte range, retrying with backoff."""
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        return expected
    for attempt in range(retries):
        try:
            with requests.get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                stream=True,
                timeout=(30, 300),
            ) as response:
                response.raise_for_status()
                tmp = path.with_suffix(path.suffix + ".part")
                with open(tmp, "wb") as handle:
                    for block in response.iter_content(chunk_size=1 << 20):
                        handle.write(block)
                if tmp.stat().st_size != expected:
                    raise IOError(
                        f"chunk {path.name}: got {tmp.stat().st_size}, "
                        f"expected {expected}"
                    )
                os.replace(tmp, path)
                return expected
        except Exception as exc:  # noqa: BLE001 - retried below
            if attempt == retries - 1:
                raise
            delay = 2 ** attempt
            print(f"  retry {attempt + 1}/{retries} for {path.name}: {exc}",
                  file=sys.stderr, flush=True)
            time.sleep(delay)
    raise RuntimeError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default="https://physionet-open.s3.amazonaws.com/hmc-sleep-staging/"
        "hmc-sleep-staging-1.1.zip",
    )
    parser.add_argument("--out", type=Path, default=Path("data/hmc/hmc.zip"))
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--chunk-mb", type=int, default=8)
    parser.add_argument("--keep-chunks", action="store_true")
    args = parser.parse_args()

    total = remote_size(args.url)
    chunk = args.chunk_mb << 20
    n_chunks = (total + chunk - 1) // chunk
    chunk_dir = args.out.with_name(args.out.name + ".chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    print(f"total {total / 1e9:.2f} GB in {n_chunks} chunks of {args.chunk_mb} MB")

    jobs = []
    for index in range(n_chunks):
        start = index * chunk
        end = min(start + chunk, total) - 1
        jobs.append((index, chunk_dir / f"chunk_{index:05d}", start, end))

    done = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_chunk, args.url, path, start, end): path
            for _, path, start, end in jobs
        }
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 25 == 0 or done == n_chunks:
                elapsed = time.time() - started
                rate = done * chunk / max(elapsed, 1e-9)
                print(
                    f"  {done}/{n_chunks} chunks, {rate / 1e6:.2f} MB/s, "
                    f"{elapsed / 60:.1f} min elapsed",
                    flush=True,
                )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    with open(tmp, "wb") as handle:
        for _, path, _, _ in jobs:
            with open(path, "rb") as source:
                shutil.copyfileobj(source, handle, length=16 << 20)
    size = tmp.stat().st_size
    if size != total:
        raise RuntimeError(f"assembled {size} bytes, expected {total}")
    os.replace(tmp, args.out)
    print(f"wrote {args.out} ({size / 1e9:.2f} GB)")

    if not args.keep_chunks:
        shutil.rmtree(chunk_dir)


if __name__ == "__main__":
    main()
