@echo off
rem Whisper-ATC: rerun the 2 ATCO2 messages whose clean decode contains numerals
cd /d %REPO_BUILD%\\RadioStar\Stage-5-atc
set PYTHONUNBUFFERED=1
set V2_FILES=131.wav,819.wav
set V2_TAG=_numerals
python run_atc_v2.py whatc atco2 1000 >> _v2_whatc_atco2_numerals.log 2>&1
echo DONE >> _chainN.done
