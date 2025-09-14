@echo off
cd /d %~dp0
REM load environment variables from .env (simple approach)
for /f "tokens=*" %%i in ('.env') do set %%i
python bot.py