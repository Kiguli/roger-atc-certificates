#!/bin/bash
# Serially certify the additional held-out country sets (Whisper-ATC, N=1000,
# SNRs 60/40/20), one at a time so only one Whisper-large is ever on the GPU.
cd "$REPO_BUILD/RadioStar/Stage-5-atc"
PY="python"
LOG="_multi.log"
echo "[$(date)] multi-country held-out run start" > "$LOG"
for cc in nl no se ca; do
  echo "[$(date)] === $cc : filter ===" >> "$LOG"
  HELDOUT_DIR=heldout_$cc "$PY" filter_heldout.py 90 >> "$LOG" 2>&1
  echo "[$(date)] === $cc : certify (N=1000) ===" >> "$LOG"
  HELDOUT_DIR=heldout_$cc SNRS=60,40,20 HELDOUT_BATCH=64 "$PY" \
      run_heldout.py heldout_$cc/good 1000 heldout_$cc >> "$LOG" 2>&1
  echo "[$(date)] === $cc : DONE ($(($(wc -l < results_atc_whisper_atc_heldout_$cc.csv)-1)) cells) ===" >> "$LOG"
done
echo "[$(date)] ALL COUNTRIES DONE" >> "$LOG"
