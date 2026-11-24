@echo off
rem Waits for chain A, then: CTC zero-noise with TF32 off, then TF32-off revision campaign
cd /d %REPO_BUILD%\\RadioStar\Stage-5-atc
:wait
if exist _chainA.done goto go
timeout /t 300 /nobreak >nul
goto wait
:go
set PYTHONUNBUFFERED=1
set ZN_NO_TF32=1
python run_zeronoise.py qn atcosim 1000 >> _zn_qn_atcosim_notf32.log 2>&1
python run_zeronoise.py ds2 atco2 1000 >> _zn_ds2_atco2_notf32.log 2>&1
python run_zeronoise.py ds2 atcosim 1000 >> _zn_ds2_atcosim_notf32.log 2>&1
set ZN_NO_TF32=
set V2_NO_TF32=1
python run_atc_v2.py qn atco2 1000 >> _v2_qn_atco2.log 2>&1
python run_atc_v2.py qn atcosim 1000 >> _v2_qn_atcosim.log 2>&1
python run_atc_v2.py ds2 atco2 1000 >> _v2_ds2_atco2.log 2>&1
python run_atc_v2.py ds2 atcosim 1000 >> _v2_ds2_atcosim.log 2>&1
echo CHAIN C2 DONE >> _chainC2.done
