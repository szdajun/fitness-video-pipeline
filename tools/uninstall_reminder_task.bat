@echo off
REM 卸载上传提醒 (Task Scheduler 6 个时点).
REM 双击运行, 看到 "[OK] 已卸载" 即可.

setlocal
set "TASK_NAME=FitnessVideoPipeline_UploadReminder"

for %%H in (10 12 14 19 21 23) do (
    schtasks /delete /tn "%TASK_NAME%_%%H" /f >nul 2>&1
    if errorlevel 1 (
        echo [WARN] 卸载 %%H:00 失败 (可能不存在)
    ) else (
        echo [OK] 已卸载 %%H:00 触发器
    )
)

echo.
pause
endlocal
