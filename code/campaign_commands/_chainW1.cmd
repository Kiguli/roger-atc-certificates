@echo off
rem Whisper-base revision rerun, shard 1 of 3
cd /d %REPO_BUILD%\\RadioStar\Stage-5-atc
set PYTHONUNBUFFERED=1
set V2_SHARD=1/3
python run_atc_v2.py wh atco2 1000 >> _v2_wh_atco2_shard1.log 2>&1
python run_atc_v2.py wh atcosim 1000 >> _v2_wh_atcosim_shard1.log 2>&1
echo DONE >> _chainW1.done
