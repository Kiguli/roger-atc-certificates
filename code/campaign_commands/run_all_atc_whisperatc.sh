#!/usr/bin/env bash
# Experiment 3: Whisper-ATC (jlvdoorn/whisper-large-v3-atco2-asr-atcosim,
# van Doorn's fine-tune on ATCO2-ASR + ATCOSIM) on both corpora -- the FULL
# identical pipeline (8 SNRs, N=1000, both splits). ~1.5B params: expect
# multiple days of GPU; per-file checkpoint/resume makes interruption safe.
# Run strictly after experiment 2 (the exp3 watcher handles sequencing).
set -e
cd "$(dirname "$0")"
unset SNRS ATC_MMAX ATC_TAG          # pin the default 8-SNR grid / full split
PY=../.venv/Scripts/python.exe

for combo in \
    "whatc atcosim certificate" \
    "whatc atco2   certificate" \
    "whatc atcosim ablation" \
    "whatc atco2   ablation" ; do
  echo "=== $combo start $(date) ==="
  $PY run_atc_split.py $combo 1000
done
echo "=== WHISPER-ATC ALL DONE $(date) ==="
