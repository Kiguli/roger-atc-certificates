#!/bin/bash
# Scale every held-out corpus to 250 certified clips (was 90; NL was 55).
# Serial so only one Whisper-large is ever resident on the GPU. run_heldout.py
# resumes from the existing CSVs, so only the NEW clips are certified.
# Segmentation is deterministic (verified byte-identical), so previously
# certified clips keep their filenames and rows.
cd "$REPO_BUILD/RadioStar/Stage-5-atc"
PY="python"
LOG="_250.log"
TARGET=250
echo "[$(date)] scale-to-$TARGET campaign start" > "$LOG"

# UWB-ATCC first: the headline matched comparison. Already utterance-segmented,
# so no VAD -- just fetch more from HuggingFace.
echo "[$(date)] === uwb : fetch ===" >> "$LOG"
UWB_LIMIT=420 "$PY" fetch_uwb.py >> "$LOG" 2>&1
echo "[$(date)] === uwb : filter ===" >> "$LOG"
HELDOUT_DIR=heldout_uwb "$PY" filter_heldout.py $TARGET >> "$LOG" 2>&1
echo "[$(date)] === uwb : certify ===" >> "$LOG"
HELDOUT_DIR=heldout_uwb SNRS=60,40,20 HELDOUT_BATCH=64 "$PY" \
    run_heldout.py heldout_uwb/good 1000 heldout_uwb >> "$LOG" 2>&1
echo "[$(date)] === uwb : DONE ($(($(wc -l < results_atc_whisper_atc_heldout_uwb.csv)-1)) cells) ===" >> "$LOG"

# Streamed feeds. "dir:tag:segtarget" -- NL needs more segments (lower pass rate).
for spec in heldout_eu:heldout_eu:400 heldout_no:heldout_no:400 \
            heldout_se:heldout_se:400 heldout_ca:heldout_ca:400 \
            heldout:heldout:400 heldout_nl:heldout_nl:700; do
  d="${spec%%:*}"; rest="${spec#*:}"; tag="${rest%%:*}"; segt="${rest##*:}"
  echo "[$(date)] === $d : segment ($segt) ===" >> "$LOG"
  HELDOUT_DIR=$d "$PY" segment_heldout.py $segt >> "$LOG" 2>&1
  echo "[$(date)] === $d : filter ($TARGET) ===" >> "$LOG"
  HELDOUT_DIR=$d "$PY" filter_heldout.py $TARGET >> "$LOG" 2>&1
  echo "[$(date)] === $d : certify ===" >> "$LOG"
  HELDOUT_DIR=$d SNRS=60,40,20 HELDOUT_BATCH=64 "$PY" \
      run_heldout.py $d/good 1000 $tag >> "$LOG" 2>&1
  echo "[$(date)] === $d : DONE ($(($(wc -l < results_atc_whisper_atc_$tag.csv)-1)) cells) ===" >> "$LOG"
done
echo "[$(date)] SCALE-250 ALL DONE" >> "$LOG"
