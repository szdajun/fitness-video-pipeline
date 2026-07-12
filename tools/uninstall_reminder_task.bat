@echo off
chcp 65001 >nul
setlocal

REM Uninstall upload reminder (Task Scheduler 6 timepoints).
REM Double-click to run; on "[OK] 6 removed" close.

set "TASK_BASE=FitnessVideoPipeline_UploadReminder"

set "H=10"
schtasks /delete /tn "%TASK_BASE%_10" /f >nul 2>&1
if errorlevel 1 goto :warn_10
echo [OK] Removed %TASK_BASE%_10
goto :next_12
:warn_10
echo [WARN] Delete %TASK_BASE%_10 failed (may not exist)
:next_12

set "H=12"
schtasks /delete /tn "%TASK_BASE%_12" /f >nul 2>&1
if errorlevel 1 goto :warn_12
echo [OK] Removed %TASK_BASE%_12
goto :next_14
:warn_12
echo [WARN] Delete %TASK_BASE%_12 failed (may not exist)
:next_14

set "H=14"
schtasks /delete /tn "%TASK_BASE%_14" /f >nul 2>&1
if errorlevel 1 goto :warn_14
echo [OK] Removed %TASK_BASE%_14
goto :next_19
:warn_14
echo [WARN] Delete %TASK_BASE%_14 failed (may not exist)
:next_19

set "H=19"
schtasks /delete /tn "%TASK_BASE%_19" /f >nul 2>&1
if errorlevel 1 goto :warn_19
echo [OK] Removed %TASK_BASE%_19
goto :next_21
:warn_19
echo [WARN] Delete %TASK_BASE%_19 failed (may not exist)
:next_21

set "H=21"
schtasks /delete /tn "%TASK_BASE%_21" /f >nul 2>&1
if errorlevel 1 goto :warn_21
echo [OK] Removed %TASK_BASE%_21
goto :next_23
:warn_21
echo [WARN] Delete %TASK_BASE%_21 failed (may not exist)
:next_23

set "H=23"
schtasks /delete /tn "%TASK_BASE%_23" /f >nul 2>&1
if errorlevel 1 goto :warn_23
echo [OK] Removed %TASK_BASE%_23
goto :done_uninstall
:warn_23
echo [WARN] Delete %TASK_BASE%_23 failed (may not exist)
:done_uninstall

echo.
pause
endlocal
exit /b 0
