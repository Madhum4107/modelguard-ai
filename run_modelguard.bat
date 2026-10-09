@echo off
setlocal
cd /d "%~dp0"
if not exist "%~dp0start_modelguard.ps1" (
  echo [ERROR] start_modelguard.ps1 was not found.
  echo Extract the ZIP completely before running this file.
  pause
  exit /b 1
)
where python >nul 2>&1
if errorlevel 1 (
  echo [ERROR] Python was not found in PATH.
  echo Install Python 3.11+ and enable "Add Python to PATH", then run again.
  pause
  exit /b 1
)
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_modelguard.ps1"
set "EXITCODE=%ERRORLEVEL%"
if not "%EXITCODE%"=="0" echo [ERROR] ModelGuard launcher exited with code %EXITCODE%.
pause
exit /b %EXITCODE%
