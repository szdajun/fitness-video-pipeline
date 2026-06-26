"""tools/reset_gpu.py — 跑 pipeline 前重置 4070 GPU 状态

目的:
  - 释放 onnxruntime / ffmpeg 残留显存占用
  - 重置 GPU clocks 到默认 (释放电源管理降频)
  - 杀掉残留 GPU 进程
  - Python 侧 torch.cuda.empty_cache() 清碎片

用法:
    python tools/reset_gpu.py            # 实际重置
    python main.py process ... --reset-gpu  # pipeline 自动调
"""
import os
import subprocess
import sys


def _run(cmd, check=False):
    """Run shell command, return (returncode, stdout)"""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=15)
        return r.returncode, (r.stdout or "").strip()
    except Exception as e:
        return -1, str(e)


def reset_gpu(verbose=True):
    """重置 GPU 状态. 返回 True 成功, False 失败但继续 pipeline.

    注意:
      - 需要管理员权限才能 reset GPU clocks (PowerMizer 关闭)
      - 没权限时静默跳过 clocks/memory reset, 仍杀残留进程 + 清 torch 缓存
    """
    if verbose:
        print("[reset_gpu] 重置 RTX 4070 状态...")

    # 1. 列出当前 GPU 进程
    rc, out = _run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                    "--format=csv,noheader,nounits"])
    if rc == 0 and out:
        if verbose:
            print(f"[reset_gpu] 当前 GPU 进程:\n{out}")
        # 杀残留 GPU 进程 (排除自己)
        my_pid = os.getpid()
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 1 and parts[0].isdigit():
                pid = int(parts[0])
                if pid != my_pid:
                    try:
                        os.kill(pid, 9)  # SIGKILL 立即
                        if verbose:
                            print(f"[reset_gpu]   杀残留 PID {pid}")
                    except (OSError, ProcessLookupError):
                        pass
    elif verbose:
        print("[reset_gpu] 无残留 GPU 进程")

    # 2. 重置 GPU clocks (释放电源管理降频) — 通常需要管理员
    rc, _ = _run(["nvidia-smi", "-rgc"])
    if rc == 0 and verbose:
        print("[reset_gpu] ✓ GPU clocks 重置到默认")
    elif verbose:
        print("[reset_gpu] GPU clocks 重置需管理员, 跳过 (Win+R, 不影响显存)")

    rc, _ = _run(["nvidia-smi", "-rmc"])
    if rc == 0 and verbose:
        print("[reset_gpu] ✓ GPU memory clocks 重置到默认")

    # 3. Python 侧 torch 清碎片
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            if verbose:
                alloc = torch.cuda.memory_allocated() // 1024 // 1024
                reserv = torch.cuda.memory_reserved() // 1024 // 1024
                print(f"[reset_gpu] ✓ torch 缓存清空 (alloc={alloc}MB, reserved={reserv}MB)")
    except ImportError:
        if verbose:
            print("[reset_gpu] torch 未安装, 跳过 torch 缓存清理")
    except Exception as e:
        if verbose:
            print(f"[reset_gpu] torch 清理失败: {e}")

    # 4. 等 2s 让 GPU idle
    import time
    time.sleep(2)

    if verbose:
        rc, out = _run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                        "--format=csv,noheader"])
        print(f"[reset_gpu] 当前 GPU 状态: {out}")

    return True


if __name__ == "__main__":
    success = reset_gpu(verbose=True)
    sys.exit(0 if success else 1)
