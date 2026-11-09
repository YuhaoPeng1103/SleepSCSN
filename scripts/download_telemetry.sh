#!/usr/bin/env bash
# Download the Sleep-EDF Expanded *telemetry* cohort (ST700-ST721).
#
# The cassette cohort (SC4xx) and the telemetry cohort (ST7xx) were recorded
# with different equipment and protocols, so telemetry is the natural
# cross-cohort shift test for a method developed on the cassette data. It is
# also small: 22 subjects, 44 recordings.
#
# Usage:
#   bash scripts/download_telemetry.sh [jobs]
set -euo pipefail

JOBS="${1:-4}"
BASE="https://physionet.org/files/sleep-edfx/1.0.0/sleep-telemetry"
OUT="data/sleep-edf/sleep-telemetry"
mkdir -p "$OUT"

LISTING="$(mktemp)"
trap 'rm -f "$LISTING"' EXIT
curl -sS --max-time 120 "$BASE/" -o "$LISTING"
grep -oE 'ST[0-9A-Z]+-[A-Za-z]+\.edf' "$LISTING" | sort -u > "$OUT/filelist.txt"

count=$(wc -l < "$OUT/filelist.txt")
echo "telemetry listing: $count files"
if [[ "$count" -eq 0 ]]; then
  echo "could not parse the directory listing; aborting"
  exit 1
fi

export BASE OUT
fetch_one() {
  local f="$1"
  local target="$OUT/$f"
  if [[ -s "$target" && ! -f "$target.part" ]]; then
    echo "skip $f"
    return 0
  fi
  local part="$target.part"
  if [[ -s "$target" && ! -f "$part" ]]; then
    mv "$target" "$part"
  fi
  curl -sS -L --retry 5 --retry-delay 3 -C - -o "$part" "$BASE/$f" \
    && mv "$part" "$target" \
    && echo "done $f"
}
export -f fetch_one

cat "$OUT/filelist.txt" | xargs -P "$JOBS" -I{} bash -c 'fetch_one "$@"' _ {}
echo "edf files present: $(ls "$OUT"/*.edf 2>/dev/null | wc -l)"
