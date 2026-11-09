#!/usr/bin/env bash
# Parallel, resumable download of the Sleep-EDF file list.
#
# Usage:
#   bash scripts/download_parallel.sh [jobs]
#
# Reads data/sleep-edf/sleep-cassette/selected.txt (written by
# download_sleep_edf.sh) and fetches every missing file with N workers.
set -euo pipefail

JOBS="${1:-6}"
BASE="https://physionet.org/files/sleep-edfx/1.0.0/sleep-cassette"
OUT="data/sleep-edf/sleep-cassette"
LIST="${2:-$OUT/selected.txt}"

if [[ ! -s "$LIST" ]]; then
  echo "missing $LIST; run scripts/download_sleep_edf.sh 20 first"
  exit 1
fi

export BASE OUT
mkdir -p "$OUT"

fetch_one() {
  local f="$1"
  local target="$OUT/$f"
  if [[ -s "$target" ]] && ! [[ -f "$target.part" ]]; then
    echo "skip $f"
    return 0
  fi
  # Download into .part, then rename so an interrupted file is never mistaken
  # for a complete one.
  local part="$target.part"
  if [[ -s "$target" && ! -f "$part" ]]; then
    mv "$target" "$part"
  fi
  curl -sS -L --retry 5 --retry-delay 3 -C - -o "$part" "$BASE/$f" \
    && mv "$part" "$target" \
    && echo "done $f $(stat -c%s "$target" 2>/dev/null || echo '?')"
}
export -f fetch_one

cat "$LIST" | xargs -P "$JOBS" -I{} bash -c 'fetch_one "$@"' _ {}

echo "files present: $(ls "$OUT"/*.edf 2>/dev/null | wc -l)"
