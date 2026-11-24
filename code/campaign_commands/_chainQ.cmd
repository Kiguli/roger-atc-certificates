@echo off
rem Remaining non-Whisper-base jobs, sequential (runs alongside the Whisper-base shards)
cd /d %REPO_BUILD%\\RadioStar\Stage-5-atc
set PYTHONUNBUFFERED=1
set ZN_NO_TF32=1
python run_zeronoise.py ds2 atco2 1000 >> _zn_ds2_atco2_notf32.log 2>&1
python run_zeronoise.py ds2 atcosim 1000 >> _zn_ds2_atcosim_notf32.log 2>&1
set ZN_NO_TF32=
echo ZN_CTC >> _chainQ.progress
python run_zeronoise.py wh atco2 1000 >> _zn_wh_atco2.log 2>&1
python run_zeronoise.py wh atcosim 1000 >> _zn_wh_atcosim.log 2>&1
python run_zeronoise.py whatc atco2 1000 >> _zn_whatc_atco2.log 2>&1
python run_zeronoise.py whatc atcosim 1000 >> _zn_whatc_atcosim.log 2>&1
echo ZN_WHISPER >> _chainQ.progress
set V2_NO_TF32=1
python run_atc_v2.py qn atco2 1000 >> _v2_qn_atco2.log 2>&1
python run_atc_v2.py qn atcosim 1000 >> _v2_qn_atcosim.log 2>&1
set V2_NO_TF32=
echo CHAIN Q DONE >> _chainQ.done
