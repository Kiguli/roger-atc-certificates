@echo off
rem Whisper-base revision campaign: raw transcripts (digit fix)
cd /d %REPO_BUILD%\\RadioStar\Stage-5-atc
set PYTHONUNBUFFERED=1
python run_atc_v2.py wh atco2 1000 >> _v2_wh_atco2.log 2>&1
python run_atc_v2.py wh atcosim 1000 >> _v2_wh_atcosim.log 2>&1
echo CHAIN B DONE >> _chainB.done
