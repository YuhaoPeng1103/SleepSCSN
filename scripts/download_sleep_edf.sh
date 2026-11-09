#!/usr/bin/env bash
# Download Sleep-EDF Expanded (sleep-cassette) from PhysioNet.
#
# Usage:
#   bash scripts/download_sleep_edf.sh 20   # 20 subjects, 39 recordings (~3 GB)
#   bash scripts/download_sleep_edf.sh 78   # all SC/ST subjects (~10 GB)
#
# Requires: curl. The data are open access (ODC-BY 1.0).
set -euo pipefail

SUBSET="${1:-20}"
BASE="https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
OUT="data/sleep-edf/sleep-cassette"
mkdir -p "$OUT"

LISTING="$(mktemp)"
trap 'rm -f "$LISTING"' EXIT
curl -sS --max-time 120 "$BASE/" -o "$LISTING"

# Extract EDF file names from the directory listing.
grep -oE '[A-Z]{2}[0-9A-Z]+-[A-Za-z]+\.edf' "$LISTING" | sort -u > "$OUT/filelist.txt"

if [[ "$SUBSET" == "20" ]]; then
  # SC400..SC419 (20 subjects, 40 recordings)
  grep -E '^SC4(0[0-9]|1[0-9])' "$OUT/filelist.txt" > "$OUT/selected.txt"
else
  cp "$OUT/filelist.txt" "$OUT/selected.txt"
fi

count=$(wc -l < "$OUT/selected.txt")
echo "selected $count files for subset=$SUBSET"

while IFS= read -r f; do
  if [[ -s "$OUT/$f" ]]; then
    continue
  fi
  echo "downloading $f"
  curl -L --retry 3 --retry-delay 2 -C - -o "$OUT/$f" "$BASE/$f"
done < "$OUT/selected.txt"

echo "done: $(ls "$OUT"/*.edf 2>/dev/null | wc -l) EDF files in $OUT"
