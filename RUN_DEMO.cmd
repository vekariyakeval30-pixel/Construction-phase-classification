@echo off
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" demo.py
) else if exist "..\..\work\ml-env\Scripts\python.exe" (
  "..\..\work\ml-env\Scripts\python.exe" demo.py
) else (
  python demo.py
)
echo.
pause
