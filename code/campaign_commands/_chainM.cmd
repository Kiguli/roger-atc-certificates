@echo off
rem TMLR revision GPU queue (sequential: concurrent jobs starve the Whisper decode)
cd /d %REPO_BUILD%\\RadioStar\Stage-5-atc
set PYTHONUNBUFFERED=1
rem 1-2. zero-noise control, campaign configuration (TF32 on), CTC
python run_zeronoise.py qn atcosim 1000 >> _zn_qn_atcosim.log 2>&1
python run_zeronoise.py ds2 atco2 1000 >> _zn_ds2_atco2.log 2>&1
python run_zeronoise.py ds2 atcosim 1000 >> _zn_ds2_atcosim.log 2>&1
rem 3. zero-noise control, TF32 off, CTC
set ZN_NO_TF32=1
python run_zeronoise.py qn atcosim 1000 >> _zn_qn_atcosim_notf32.log 2>&1
python run_zeronoise.py ds2 atco2 1000 >> _zn_ds2_atco2_notf32.log 2>&1
python run_zeronoise.py ds2 atcosim 1000 >> _zn_ds2_atcosim_notf32.log 2>&1
set ZN_NO_TF32=
echo STEP3 >> _chainM.progress
rem 4. Whisper-base revision rerun (digit-preserving normalization, raw transcripts)
python run_atc_v2.py wh atco2 1000 >> _v2_wh_atco2.log 2>&1
python run_atc_v2.py wh atcosim 1000 >> _v2_wh_atcosim.log 2>&1
echo STEP4 >> _chainM.progress
rem 5. QuartzNet revision rerun, TF32 off
set V2_NO_TF32=1
python run_atc_v2.py qn atco2 1000 >> _v2_qn_atco2.log 2>&1
python run_atc_v2.py qn atcosim 1000 >> _v2_qn_atcosim.log 2>&1
set V2_NO_TF32=
echo STEP5 >> _chainM.progress
rem 6. zero-noise control, Whisper and Whisper-ATC (N0=264 for Whisper-ATC)
python run_zeronoise.py wh atco2 1000 >> _zn_wh_atco2.log 2>&1
python run_zeronoise.py wh atcosim 1000 >> _zn_wh_atcosim.log 2>&1
python run_zeronoise.py whatc atco2 1000 >> _zn_whatc_atco2.log 2>&1
python run_zeronoise.py whatc atcosim 1000 >> _zn_whatc_atcosim.log 2>&1
echo STEP6 >> _chainM.progress
rem 7. DeepSpeech2 revision rerun, TF32 off (drop if DS2 shows no TF32 artifact)
set V2_NO_TF32=1
python run_atc_v2.py ds2 atco2 1000 >> _v2_ds2_atco2.log 2>&1
python run_atc_v2.py ds2 atcosim 1000 >> _v2_ds2_atcosim.log 2>&1
set V2_NO_TF32=
echo CHAIN M DONE >> _chainM.done
