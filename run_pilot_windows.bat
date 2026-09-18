@echo off
setlocal
cd /d "%~dp0"
if not exist .venv py -3 -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install -q -r requirements.txt
python run_experiment.py
if errorlevel 1 pause & exit /b 1
python plot_results.py
pause
