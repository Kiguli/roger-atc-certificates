#!/bin/bash
# Wait for the 4-country multi-supervisor (_multi.log) to finish, then certify the
# UWB-ATCC held-out (curated, disjoint-from-training, noisy 8->16 kHz Czech ATC).
# Runs after the others so only one Whisper-large is ever resident on the GPU.
cd "$REPO_BUILD/RadioStar/Stage-5-atc"
PY="python"
LOG="_uwb.log"
echo "[$(date)] waiting for multi-country run (ALL COUNTRIES DONE)..." > "$LOG"
for i in $(seq 1 600); do
  grep -q "ALL COUNTRIES DONE" _multi.log 2>/dev/null && break
  sleep 60
done
echo "[$(date)] === uwb : filter ===" >> "$LOG"
HELDOUT_DIR=heldout_uwb "$PY" filter_heldout.py 90 >> "$LOG" 2>&1
echo "[$(date)] === uwb : certify (N=1000, SNRs 60/40/20) ===" >> "$LOG"
HELDOUT_DIR=heldout_uwb SNRS=60,40,20 HELDOUT_BATCH=64 "$PY" \
    run_heldout.py heldout_uwb/good 1000 heldout_uwb >> "$LOG" 2>&1
echo "[$(date)] === uwb : DONE ($(($(wc -l < results_atc_whisper_atc_heldout_uwb.csv)-1)) cells) ===" >> "$LOG"
echo "[$(date)] UWB run DONE" >> "$LOG"
