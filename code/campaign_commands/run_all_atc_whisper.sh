#!/usr/bin/env bash
# Experiment 2: Whisper-base on both ATC corpora (fresh draws, per-draw
# outcomes). Run AFTER run_all_atc.sh so the two experiments never contend
# for the GPU. Certificate splits first, then ablation splits. Per-file
# checkpoint/resume, so safe to re-run after any interruption.
set -e
cd "$(dirname "$0")"
unset SNRS ATC_MMAX ATC_TAG          # pin the default 8-SNR grid / full split
PY=../.venv/Scripts/python.exe

for combo in \
    "wh atcosim certificate" \
    "wh atco2   certificate" \
    "wh atcosim ablation" \
    "wh atco2   ablation" ; do
  echo "=== $combo start $(date) ==="
  $PY run_atc_split.py $combo 1000
done
echo "=== WHISPER ALL DONE $(date) ==="
