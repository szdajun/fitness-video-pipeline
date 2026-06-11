"""
lib/observability.py - 进度显示 + 结构化日志

设计:
  - 进度: tqdm 风格, 类似下载文件
  - 日志: logs/YYYY-MM-DD.log 按天文件, 同时打印到 stderr
  - 装饰器 @stage_progress 给 step_* 函数自动加进度+日志+耗时

用法:
    from lib.observability import setup_logging, ProgressTracker, stage_progress

    setup_logging()  # 配 logs/2026-06-12.log

    # 方式 1: 用上下文管理器
    with ProgressTracker("跑 v2 pipeline", total=8) as p:
        for stage in stages:
            p.update(1, label=stage.name)
            stage.run()

    # 方式 2: 装饰器
    @stage_progress("Step 2a: 拼 16:9 final")
    def step_make_16x9(...):
        ...
"""
import functools
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"


def setup_logging(log_level: str = "INFO", retention_days: int = 30) -> Path:
    """配置全局 logging.
    - 文件: logs/YYYY-MM-DD.log (按天追加)
    - 控制台: stderr, 跟进度条配合
    - 自动清理 retention_days 之外的旧文件
    返回当天日志路径.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"{today}.log"

    fmt = "%(asctime)s [%(levelname)5s] %(name)s | %(message)s"
    date_fmt = "%H:%M:%S"

    # 清掉之前的 handler 防重复
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt, date_fmt))
    root.addHandler(fh)

    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(logging.Formatter("[%(levelname)5s] %(message)s"))
    sh.setLevel(logging.WARNING)  # 控制台只显示 WARN/ERROR, INFO 只进文件
    root.addHandler(sh)

    # 清理 30 天前的日志
    _cleanup_old_logs(retention_days)

    logging.info("=" * 60)
    logging.info(f"会话开始 - {datetime.now().isoformat(timespec='seconds')}")
    logging.info(f"日志文件: {log_file}")
    return log_file


def _cleanup_old_logs(retention_days: int):
    cutoff = time.time() - retention_days * 86400
    for f in LOG_DIR.glob("*.log"):
        try:
            if f.stat().st_mtime < cutoff:
                f.unlink()
        except OSError:
            pass


def get_logger(name: str = "pipeline") -> logging.Logger:
    return logging.getLogger(name)


# ============================================================
#  进度条 (tqdm 风格, 类似下载文件)
# ============================================================

class ProgressTracker:
    """单 stage 进度条上下文.
    用法:
        with ProgressTracker("跑 v2 pipeline", total=8) as p:
            p.update(1, label="pose_detect")
            ...
    无 tqdm 时降级到 print.
    """
    def __init__(self, desc: str, total: int = 100, unit: str = "step",
                 leave: bool = True, log: bool = True):
        self.desc = desc
        self.total = total
        self.unit = unit
        self.log = log
        self.start = None
        self.bar = None
        self.leave = leave
        self._logger = logging.getLogger("progress")

    def __enter__(self):
        self.start = time.time()
        if self.log:
            self._logger.info(f"[START] {self.desc} (total={self.total} {self.unit})")
        if HAS_TQDM:
            self.bar = tqdm(
                total=self.total, desc=self.desc, unit=self.unit,
                leave=self.leave, ncols=100,
                bar_format="{desc:30s} [{bar:30}] {n_fmt}/{total_fmt} {percentage:3.0f}% [{elapsed}<{remaining}] {postfix}",
            )
        else:
            print(f"[START] {self.desc}")
        return self

    def update(self, n: int = 1, label: Optional[str] = None):
        if self.bar:
            if label:
                self.bar.set_postfix_str(label)
            self.bar.update(n)
        else:
            done = (self.bar.n if self.bar else 0) + n
            pct = 100 * done / self.total if self.total else 0
            print(f"  [{self.desc}] {done}/{self.total} {pct:.0f}%  {label or ''}")
        if self.log and label:
            self._logger.debug(f"  [{self.desc}] +{n} → {label}")

    def __exit__(self, exc_type, exc_val, tb):
        elapsed = time.time() - self.start
        if self.bar:
            self.bar.close()
        ok = exc_type is None
        msg = f"[{'DONE' if ok else 'FAIL'}] {self.desc} ({elapsed:.1f}s)"
        if exc_val:
            msg += f" — {exc_type.__name__}: {exc_val}"
        if self.log:
            (self._logger.info if ok else self._logger.error)(msg)
        print(msg)
        return False  # 不吞异常


def stage_progress(desc: str):
    """装饰器: 给函数自动加进度 + 日志 + 耗时.
    用法:
        @stage_progress("Step 1: 跑 v2 pipeline")
        def step_run_pipeline(cfg):
            ...
    """
    def deco(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            log = logging.getLogger("stage")
            t0 = time.time()
            print()
            print("=" * 60)
            print(desc)
            print("=" * 60)
            log.info(f"[STAGE START] {desc}")
            try:
                result = func(*args, **kwargs)
                dt = time.time() - t0
                log.info(f"[STAGE DONE]  {desc} ({dt:.1f}s)")
                print(f"  ✓ {desc} 完成 ({dt:.1f}s)")
                return result
            except Exception as e:
                dt = time.time() - t0
                log.exception(f"[STAGE FAIL]  {desc} ({dt:.1f}s) — {type(e).__name__}: {e}")
                print(f"  ✗ {desc} 失败 ({dt:.1f}s): {e}")
                raise
        return wrapper
    return deco


# ============================================================
#  ffmpeg 调用包装 (实时解析帧进度)
# ============================================================

def run_ffmpeg_with_progress(cmd, total_frames: int, desc: str = "ffmpeg"):
    """跑 ffmpeg 命令并显示帧进度. cmd 是 ffmpeg 参数列表.
    自动追加 -progress pipe:1 解析输出.
    """
    import subprocess, re
    log = logging.getLogger("ffmpeg")
    log.info(f"[FFMPEG START] {desc}: total={total_frames}")
    log.debug(f"  cmd: {' '.join(cmd[:8])}...")

    # 在 -y 后插入 -progress pipe:1, 但要确保 cmd[0] 是 ffmpeg
    cmd = list(cmd)
    if "-progress" not in cmd:
        cmd.insert(1, "-progress")
        cmd.insert(2, "pipe:1")

    with ProgressTracker(desc, total=total_frames, unit="frame", leave=True) as p:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="ignore",
        )
        last = 0
        for line in proc.stdout:
            m = re.match(r"frame=(\d+)", line.strip())
            if m:
                cur = int(m.group(1))
                if cur > last:
                    p.update(min(cur - last, total_frames - p.bar.n if p.bar else cur - last))
                    last = cur
        rc = proc.wait()
        if rc != 0:
            err = proc.stderr.read()[-500:] if proc.stderr else ""
            log.error(f"[FFMPEG FAIL] {desc}: rc={rc} {err}")
            raise RuntimeError(f"{desc} 失败 (rc={rc})")
    return rc


if __name__ == "__main__":
    # 自检
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

    log_file = setup_logging("DEBUG")
    print(f"日志: {log_file}")

    log = get_logger("test")
    log.info("INFO 消息测试")
    log.warning("WARNING 测试")

    with ProgressTracker("假装跑 stage", total=20, unit="frame") as p:
        for i in range(20):
            time.sleep(0.05)
            p.update(1, label=f"frame {i+1}")

    @stage_progress("Step 演示")
    def demo():
        time.sleep(0.5)
        return "ok"

    print(f"结果: {demo()}")
    print(f"\n查看日志: {log_file}")
