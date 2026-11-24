@echo off
rem Waits for the QuartzNet rerun (chain Q), then regenerates every revision output into rev1/
cd /d %REPO_BUILD%\\RadioStar\Stage-5-atc
:wait
if exist _chainQ.done goto go
timeout /t 120 /nobreak >nul
goto wait
:go
set PYTHONUNBUFFERED=1
python run_rev1_analysis.py --require-complete --original whisper_atc,deepspeech2 --patch-tag _numerals,_tf32fix >> _rev1_analysis.log 2>&1
echo CHAIN R DONE >> _chainR.done
