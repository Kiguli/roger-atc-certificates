@echo off
rem DeepSpeech2: recompute the single message with a TF32 zero-noise effect (ATCOSIM 710.wav), TF32 off
cd /d %REPO_BUILD%\\RadioStar\Stage-5-atc
set PYTHONUNBUFFERED=1
set V2_FILES=710.wav
set V2_TAG=_tf32fix
set V2_NO_TF32=1
python run_atc_v2.py ds2 atcosim 1000 >> _v2_ds2_atcosim_tf32fix.log 2>&1
echo DONE >> _chainP.done
