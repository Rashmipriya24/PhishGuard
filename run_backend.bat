@echo off
echo Starting PhishGuard Backend...
cd /d "%~dp0backend"
python api\main.py
pause
