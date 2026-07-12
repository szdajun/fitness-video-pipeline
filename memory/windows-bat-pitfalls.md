---
name: windows-bat-pitfalls
description: "【2026-07-13 踩坑】Windows .bat 在 cmd 默认 GBK 环境下的 4 类典型坑: 中文乱码 / for-loop 块内变量扩展 / if-else 单行嵌套 / schtasks 需管理员. 写 .bat 必加 chcp 65001 + 拆 for-loop + 用 goto label 不用 subroutine."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 07fe6098-be61-4131-9c17-7c18206be2c0
---

# Windows .bat 踩坑教训 (2026-07-13, upload reminder .bat 修)

## 一句话

写 Windows .bat **必须** `chcp 65001 >nul` + 拆 `for-loop` 为显式命令 + 用 `goto label` 不用 `subroutine` (call :func). schtasks /create 需要管理员权限, 右键 → 以管理员身份运行.

## 4 类典型坑 (修法钉死)

### 坑 1: GBK 中文乱码
**症状**: .bat 文件 UTF-8 编码, cmd 默认 GBK, `echo [ERR] 找不到` 显示成 `'鏨找不到'`, `if errorlevel 1` 报 "命令语法不正确".
**修法**: 第一行加 `chcp 65001 >nul` (强制 UTF-8) + 把所有 echo 改成英文 (最稳, ANSI 兼容).
**示范** (`tools/install_reminder_task.bat` 顶):
```batch
@echo off
chcp 65001 >nul
setlocal
```

### 坑 2: for-loop 块内变量扩展失败
**症状**:
```batch
for %%H in (10 12 14 19 21 23) do (
    schtasks /create /tn "%TASK_BASE%_%%H" /tr "\"%PYTHON_EXE%\" \"%SCRIPT%\"" /sc daily /st %%H:00:00 ...
)
```
输出 `'注册 %%H:00'` + `'“AME00” ^'` 乱码 + "文件名语法不正确".

**根因**:
- cmd `for` 块内 `%VAR%` 默认**延迟展开**, 必须在 `setlocal enabledelayedexpansion` 下用 `!VAR!` 才能用
- 块内 `^` 续行 + `%VAR%` + 块语法组合触发 cmd 解析器多种 bug
- `%%H:00` 被当成变量名而不是字面 `:00`

**修法**: **拆 for-loop 为 6 个显式命令**. 例 (从 install .bat 抄):
```batch
echo Registering 10:00 trigger...
schtasks /create /tn "%TASK_BASE%_10" /tr "%TASK_TR%" /sc daily /st 10:00:00 ...
if errorlevel 1 goto :err
echo Registering 12:00 trigger...
schtasks /create /tn "%TASK_BASE%_12" /tr "%TASK_TR%" /sc daily /st 12:00:00 ...
...
```
**预解 TR 变量** (`set "TASK_TR=\"%PYTHON_EXE%\" \"%SCRIPT%\""`) 后所有命令直接用 `%TASK_TR%`, 不在块内展开.

### 坑 3: if-else 单行嵌套在重定向后报 "was unexpected at this time"
**症状**:
```batch
schtasks /delete /tn "%TASK_BASE%_10" /f >nul 2>&1
if errorlevel 1 (echo [WARN] Failed) else (echo [OK] Removed)
```
输出: `'带 was unexpected at this time'` 退出码 255.

**根因**: cmd 在 `>nul 2>&1` 重定向后, `if errorlevel N (cmd1) else (cmd2)` 单行嵌套解析失败 (cmd parser 对管道/重定向 + 单行 if-else 嵌套支持不完善).

**修法 A (用 goto label)** ✅ 推荐:
```batch
schtasks /delete /tn "%TASK_BASE%_10" /f >nul 2>&1
if errorlevel 1 goto :warn_10
echo [OK] Removed %TASK_BASE%_10
goto :next_12
:warn_10
echo [WARN] Delete %TASK_BASE%_10 failed
:next_12
```

**修法 B (call :subroutine)** — 看似优雅但**会双调用**, 输出 "[WARN]\n[OK]\n" 同一任务. **不推荐**.

**修法 C (多行 if + 嵌套)**:
```batch
if errorlevel 1 (
    echo [WARN] Failed
) else (
    echo [OK] Removed
)
```
多行括号版能跑, 但**仍然推荐 goto label** (更易读, 无嵌套深度限制).

### 坑 4: schtasks /create "Access is denied"
**症状**: 跑 install 报 `ERROR: Access is denied`, 退出码 1, 一个任务都没创建.
**根因**: Task Scheduler 创建任务需要管理员权限.
**修法**: **右键 install .bat → 以管理员身份运行**. uninstall 不需要.
**建议**: install .bat 顶部加 echo 提示:
```batch
echo IMPORTANT: Right-click and "Run as administrator"
pause
```

## 调试 .bat 4 步

1. `cmd //c "tools\\xxx.bat"` 从 bash 跑, stdout/stderr 都能看见
2. `echo.` 加在关键命令后看是否执行到
3. `pause` 加在末尾防止窗口闪关
4. 简单命令拆出来单跑, 排除变量/语法问题

## 历史

- 2026-07-13 upload reminder 修 .bat 4 类坑一次踩全, commit `0652b05` 修好
- 修了后 uninstall 干净 (6 个 [WARN] "may not exist)" exit 0)
- install 跑出 "Access is denied" → 用户需右键管理员
