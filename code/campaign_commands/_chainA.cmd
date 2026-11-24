@echo off
rem Zero-noise control (reviewer 3CgC), campaign configuration
cd /d %REPO_BUILD%\\RadioStar\Stage-5-atc
set PYTHONUNBUFFERED=1
python run_zeronoise.py qn atco2 1000 >> _zn_qn_atco2.log 2>&1
python run_zeronoise.py qn atcosim 1000 >> _zn_qn_atcosim.log 2>&1
python run_zeronoise.py ds2 atco2 1000 >> _zn_ds2_atco2.log 2>&1
python run_zeronoise.py ds2 atcosim 1000 >> _zn_ds2_atcosim.log 2>&1
python run_zeronoise.py wh atco2 1000 >> _zn_wh_atco2.log 2>&1
python run_zeronoise.py wh atcosim 1000 >> _zn_wh_atcosim.log 2>&1
python run_zeronoise.py whatc atco2 1000 >> _zn_whatc_atco2.log 2>&1
python run_zeronoise.py whatc atcosim 1000 >> _zn_whatc_atcosim.log 2>&1
echo CHAIN A DONE >> _chainA.done
