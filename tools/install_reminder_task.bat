@echo off
REM 安装上传提醒到 Windows Task Scheduler.
REM 6 个黄金时段时点 (北京 UTC+8): 10:00 12:00 14:00 19:00 21:00 23:00
REM
REM 双击运行, 看到 "SUCCESS" 即可关闭. 卸载用 uninstall_reminder_task.bat.

setlocal
set "TASK_NAME=FitnessVideoPipeline_UploadReminder"
set "PROJECT_DIR=F:\wkspace\fitness-video-pipeline"
set "PYTHON_EXE=%PROJECT_DIR%\.venv\Scripts\python.exe"
set "SCRIPT=%PROJECT_DIR%\tools\upload_reminder.py"

REM 检查 venv python
if not exist "%PYTHON_EXE%" (
    echo [ERR] 找不到 %PYTHON_EXE%
    echo       请先跑 uv sync 创建 .venv
    pause
    exit /b 1
)

REM 删除旧的 (如果存在)
schtasks /delete /tn "%TASK_NAME%_10" /f >nul 2>&1
schtasks /delete /tn "%TASK_NAME%_12" /f >nul 2>&1
schtasks /delete /tn "%TASK_NAME%_14" /f >nul 2>&1
schtasks /delete /tn "%TASK_NAME%_19" /f >nul 2>&1
schtasks /delete /tn "%TASK_NAME%_21" /f >nul 2>&1
schtasks /delete /tn "%TASK_NAME%_23" /f >nul 2>&1

for %%H in (10 12 14 19 21 23) do (
    echo 注册 %%H:00 触发器...
    schtasks /create ^
        /tn "%TASK_NAME%_%%H" ^
        /tr "\"%PYTHON_EXE%\" \"%SCRIPT%\"" ^
        /sc daily ^
        /st %%H:00:00 ^
        /rl highest ^
        /ru "%USERNAME%" ^
        /f >nul
    if errorlevel 1 (
        echo [ERR] 创建 %%H:00 触发器失败
        pause
        exit /b 1
    )
)

echo.
echo [OK] 6 个时点已注册:
echo   %TASK_NAME%_10  每天 10:00
echo   %TASK_NAME%_12  每天 12:00
echo   %TASK_NAME%_14  每天 14:00
echo   %TASK_NAME%_19  每天 19:00
echo   %TASK_NAME%_21  每天 21:00
echo   %TASK_NAME%_23  每天 23:00
echo.
echo 卸载跑: tools\uninstall_reminder_task.bat
echo.
pause
endlocal
