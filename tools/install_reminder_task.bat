@echo off
chcp 65001 >nul
setlocal

REM Install upload reminder to Windows Task Scheduler.
REM 6 daily golden-hour timepoints (Beijing UTC+8): 10:00 12:00 14:00 19:00 21:00 23:00
REM Double-click to run; on "[OK] 6 timepoints registered" close.
REM Use uninstall_reminder_task.bat to remove.

set "TASK_BASE=FitnessVideoPipeline_UploadReminder"
set "PROJECT_DIR=F:\wkspace\fitness-video-pipeline"
set "PYTHON_EXE=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "SCRIPT=%PROJECT_DIR%\tools\upload_reminder.py"

REM Check venv python
if not exist "%PYTHON_EXE%" (
    echo [ERR] Cannot find %PYTHON_EXE%
    echo       Run uv sync first to create .venv
    pause
    exit /b 1
)

REM Build the full TR string once (avoids block-scope variable expansion issues in for-loops)
set "TASK_TR=\"%PYTHON_EXE%\" \"%SCRIPT%\""

REM Delete any old tasks (ignore errors)
schtasks /delete /tn "%TASK_BASE%_10" /f >nul 2>&1
schtasks /delete /tn "%TASK_BASE%_12" /f >nul 2>&1
schtasks /delete /tn "%TASK_BASE%_14" /f >nul 2>&1
schtasks /delete /tn "%TASK_BASE%_19" /f >nul 2>&1
schtasks /delete /tn "%TASK_BASE%_21" /f >nul 2>&1
schtasks /delete /tn "%TASK_BASE%_23" /f >nul 2>&1

REM Register 6 daily tasks. Explicit per-hour calls (more reliable than for-loop with carets).
echo Registering 10:00 trigger...
schtasks /create /tn "%TASK_BASE%_10" /tr "%TASK_TR%" /sc daily /st 10:00:00 /rl highest /ru "%USERNAME%" /f
if errorlevel 1 goto err

echo Registering 12:00 trigger...
schtasks /create /tn "%TASK_BASE%_12" /tr "%TASK_TR%" /sc daily /st 12:00:00 /rl highest /ru "%USERNAME%" /f
if errorlevel 1 goto err

echo Registering 14:00 trigger...
schtasks /create /tn "%TASK_BASE%_14" /tr "%TASK_TR%" /sc daily /st 14:00:00 /rl highest /ru "%USERNAME%" /f
if errorlevel 1 goto err

echo Registering 19:00 trigger...
schtasks /create /tn "%TASK_BASE%_19" /tr "%TASK_TR%" /sc daily /st 19:00:00 /rl highest /ru "%USERNAME%" /f
if errorlevel 1 goto err

echo Registering 21:00 trigger...
schtasks /create /tn "%TASK_BASE%_21" /tr "%TASK_TR%" /sc daily /st 21:00:00 /rl highest /ru "%USERNAME%" /f
if errorlevel 1 goto err

echo Registering 23:00 trigger...
schtasks /create /tn "%TASK_BASE%_23" /tr "%TASK_TR%" /sc daily /st 23:00:00 /rl highest /ru "%USERNAME%" /f
if errorlevel 1 goto err

echo.
echo [OK] 6 daily timepoints registered.
echo   %TASK_BASE%_10  daily 10:00
echo   %TASK_BASE%_12  daily 12:00
echo   %TASK_BASE%_14  daily 14:00
echo   %TASK_BASE%_19  daily 19:00
echo   %TASK_BASE%_21  daily 21:00
echo   %TASK_BASE%_23  daily 23:00
echo.
echo To uninstall run: tools\uninstall_reminder_task.bat
echo.
pause
endlocal
exit /b 0

:err
echo.
echo [ERR] schtasks /create failed (see error above).
echo.
pause
endlocal
exit /b 1
