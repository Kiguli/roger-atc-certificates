#!/bin/bash
# Wait for the US held-out run (results_atc_whisper_atc_heldout.csv) to finish,
# then run the EU (Polish) filter + certification, so the two Whisper-ATC GPU
# jobs never run concurrently (VRAM).
cd "$REPO_BUILD/RadioStar/Stage-5-atc"
PY="python"
LOG="_eu_supervisor.log"
echo "[$(date)] waiting for US run to finish..." > "$LOG"
for i in $(seq 1 150); do
  grep -q "wrote" _heldout.log 2>/dev/null && break
  sleep 60
done
echo "[$(date)] US run finished (or timeout); EU filter (GPU)" >> "$LOG"
HELDOUT_DIR=heldout_eu "$PY" filter_heldout.py 90 >> "$LOG" 2>&1
echo "[$(date)] EU certification (N=1000, SNRs 60/40/20)" >> "$LOG"
SNRS=60,40,20 "$PY" run_heldout.py heldout_eu/good 1000 heldout_eu >> "$LOG" 2>&1
echo "[$(date)] EU run DONE" >> "$LOG"
