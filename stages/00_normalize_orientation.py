"""阶段 00a: 源视频方向 normalize (EXIF/displaymatrix 旋转锁进像素)

目的 (2026-07-10 竖屏源端到端通路):
- 检测 EXIF 旋转 / displaymatrix / 像素方向
- 把任何角度的源转码锁进 1080×1920 yuv420p+30fps+aac 像素
- 输出到 `source_videos/_normalized/{stem}_normalized.mp4`
- 走 `ctx.input_path = out_path` 套路 (跟 00_pre_deblock.py:51 一致)
- 增量跳过: ctx.normalized_path 已存在且 valid → 直接复用, 仍把 ctx.input_path 指向它

复用:
- `lib/source_detection.py` 共用检测
- `stages/00_pre_deblock.py:51` ctx.input_path = out_path 套路
- `stages/07_export.py:235-243` ffprobe 模板 (测源尺寸兜底)
"""

import subprocess
from pathlib import Path

from lib.utils import path_exists
from lib.source_detection import (
    detect_source_orientation,
    apply_transpose_filter,
    _is_already_vertical,
)


class NormalizeOrientationStage:
    def run(self, ctx):
        # 1. 增量跳过: normalized_path 已存在 → ctx.input_path 指向它, 退出
        normalized_existing = ctx.get("normalized_path")
        if normalized_existing and path_exists(str(normalized_existing)):
            ctx.input_path = Path(str(normalized_existing))
            print(f"    方向修复已存在, 跳过 → {Path(str(normalized_existing)).name}")
            return

        src = Path(str(ctx.input_path))
        if not src.exists():
            return

        # 2. 检测方向
        info = detect_source_orientation(str(src))
        src_w = info["src_w"]
        src_h = info["src_h"]
        rotation = info["rotation"]
        is_vertical = info["is_vertical"]
        needs_normalize = info["needs_normalize"]

        # 3. 不需要修复 (横屏源 OR 像素已是 9:16 且 rotation=0)
        if not is_vertical or (is_vertical and not needs_normalize):
            ctx.set("normalized_path", str(src))
            ctx.set("source_orientation", info)
            if is_vertical:
                print(f"    竖屏源像素已锁定 ({src_w}x{src_h}), 无需方向修复")
            else:
                print(f"    横屏源 ({src_w}x{src_h}), 跳过方向修复")
            return

        # 4. 需要转码: 输出路径 = src 同目录的 _normalized/{stem}_normalized.mp4
        out_dir = src.parent / "_normalized"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{src.stem}_normalized.mp4"

        if path_exists(str(out_path)):
            print(f"    方向修复产物已存在, 复用 → {out_path.name}")
            ctx.set("normalized_path", str(out_path))
            ctx.set("source_orientation", info)
            ctx.input_path = out_path
            return

        # 5. ffmpeg 转码: 仅 scale (不动 rotate — ffmpeg 默认自动应用 EXIF rotate,
        # 抽帧就是已正确的方向; 我们 -metadata rotate=0 输出重置标记即可).
        # 2026-07-10 教训: 加 -noautorotate + 手动 transpose 会导致"双重旋转"=颠倒
        # (铁娘子3 实测). 正确=不加 noautorotate, 让 ffmpeg 自己处理 rotate,
        # 我们只做 scale 把 1080x1920 锁进像素, 元数据 rotate=0 清理.
        vf_parts = [p for p in ["scale=1080:1920:flags=lanczos", "setsar=1", "fps=30"] if p]
        vf = ",".join(vf_parts)

        print(f"    方向修复: {src_w}x{src_h} rotate={rotation} → 1080x1920 (scale only)")

        cmd = [
            "ffmpeg", "-y",
            "-i", str(src),
            "-map", "0:v:0", "-map", "0:a?",
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
            "-metadata:s:v:0", "rotate=0",
            "-movflags", "+faststart",
            str(out_path),
        ]

        r = subprocess.run(cmd, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=600)
        if r.returncode != 0:
            err_tail = (r.stderr or "")[-300:]
            print(f"    方向修复失败, 回退源视频: {err_tail}")
            ctx.set("normalized_path", str(src))
            ctx.set("source_orientation", info)
            return

        ctx.set("normalized_path", str(out_path))
        ctx.set("source_orientation", info)
        ctx.input_path = out_path
        print(f"    输出: {out_path.name} (1080x1920 yuv420p+aac)")