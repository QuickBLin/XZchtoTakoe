@echo off
setlocal
cd /d "%~dp0"
if exist "%USERPROFILE%\anaconda3\python.exe" (
  "%USERPROFILE%\anaconda3\python.exe" -c "import sklearn, numpy, scipy, tkinter" >nul 2>nul
  if not errorlevel 1 goto anaconda
)
if not exist ".venv\Scripts\python.exe" (
  py -3.13 -m venv .venv
  if errorlevel 1 goto fail
)
".venv\Scripts\python.exe" -c "import sklearn, numpy, scipy, tkinter"
if errorlevel 1 (
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 goto fail
)
".venv\Scripts\python.exe" app.py
if errorlevel 1 goto fail
exit /b 0
:anaconda
"%USERPROFILE%\anaconda3\python.exe" app.py
if errorlevel 1 goto fail
exit /b 0
:fail
echo Startup failed. Open README.md for Python and VS Code setup.
pause
exit /b 1
