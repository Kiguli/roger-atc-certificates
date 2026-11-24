#!/usr/bin/env bash
# Full ATC certificate/ablation run (fresh draws, per-draw outcomes).
# Order: QuartzNet first (fast model), certificate splits before ablation
# splits, so certificates are available early for downstream analysis.
# Each combo checkpoints per-file and resumes, so this script is safe to
# re-run after an interruption.
set -e
cd "$(dirname "$0")"
unset SNRS ATC_MMAX ATC_TAG          # pin the default 8-SNR grid / full split
PY=../.venv/Scripts/python.exe

for combo in \
    "qn  atcosim certificate" \
    "qn  atco2   certificate" \
    "qn  atcosim ablation" \
    "qn  atco2   ablation" \
    "ds2 atcosim certificate" \
    "ds2 atco2   certificate" \
    "ds2 atcosim ablation" \
    "ds2 atco2   ablation" ; do
  echo "=== $combo start $(date) ==="
  $PY run_atc_split.py $combo 1000
done
echo "=== ALL DONE $(date) ==="
