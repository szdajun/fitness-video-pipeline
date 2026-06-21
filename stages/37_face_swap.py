"""阶段37: 美颜换脸 — 用 GFPGAN 增强照替换视频人脸，提升面部质感

设计原则: 有照片就换、没照片就 skip — 不用写死教练名单。
按以下顺序查找教练的换脸照片(在 tools/ 目录下):
  1. tools/{coach}_gfpgan.png        中文名 + gfpgan 增强
  2. tools/{coach}_face.{png,jpg}    中文名
  3. tools/{coach}.{png,jpg,bmp}     直接中文名
  4. COACH_ALIAS_MAP[coach]_*        拼音/英文别名 (兼容旧命名)
找不到就 skip,程序不报错。要给新教练加换脸: tools/ 丢一张照片即可。
"""

import os, sys, subprocess, gc
from pathlib import Path
from typing import Optional
from lib.utils import path_exists

# 拼音/英文别名映射 (用于兼容旧文件名约定)
# 新教练不必加这里, 直接 tools/{coach}.png 也认
COACH_ALIAS_MAP = {
    "艳青": "yanqing",
    "丽丽": "lili",
    "建玲": "jianling",
    "小红豆": "xhd",
    "枫林红": "flh",
    "郭海军": "haijun",
}


def find_coach_face(coach_name: str, tools_dir: str) -> Optional[str]:
    """按多种命名约定查找教练的换脸照片. 返回绝对路径或 None.
    优先级 (按图像质量从高到低):
      1. {coach}_gfpgan.png / {alias}_gfpgan.png  — GFPGAN 增强 (最优)
      2. {coach}_face.png / {alias}_face.png      — 高清 PNG
      3. {coach}_face.jpg / {alias}_face.jpg      — 高清 JPG
      4. {coach}.png/jpg/bmp / {alias}.*          — 原始照
    """
    if not coach_name:
        return None
    alias = COACH_ALIAS_MAP.get(coach_name)
    names = [coach_name] + ([alias] if alias else [])
    # 按后缀分组遍历, 保证高质量后缀先匹配 (跨名字)
    for suffix in ["_face_gfpgan.png", "_gfpgan.png", "_face.png", "_face.jpg", ".png", ".jpg", ".bmp"]:
        for name in names:
            path = os.path.join(tools_dir, f"{name}{suffix}")
            if os.path.exists(path):
                return path
    return None


class FaceSwapStage:
    def run(self, ctx):
        # 引擎已根据 stages.face_swap 决定是否运行此阶段
        if ctx.get("face_swap_path") and path_exists(ctx.get("face_swap_path")):
            print("    已存在，跳过")
            return

        # 检测教练
        lead_name = ctx.get("lead_name")
        if not lead_name:
            try:
                from lib.coach_profiles import detect_coach_from_filename
                lead_name = detect_coach_from_filename(str(ctx.input_path))
            except Exception:
                pass
        if not lead_name:
            print(f"    跳过: 未识别教练")
            return

        # 自适应查找: 有照片就换, 没照片就 skip
        tools_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tools"))
        source_face = find_coach_face(lead_name, tools_dir)
        if not source_face:
            print(f"    跳过: 教练 '{lead_name}' 在 tools/ 下无照片 (可放 {lead_name}.png 启用换脸)")
            return

        # 换脸目标: 优先显式指定 → mascot 输出 → 任意上游视频
        # (mascot 未开启时也能换脸, 不再硬依赖吉祥物; 下游仍读 mascot_path 接力)
        _TARGET_KEYS = [
            "face_swap_target", "mascot_path", "watermark_path",
            "skin_smooth_path", "denoise_path", "skin_tone_filter_path",
            "color_path", "ken_burns_path", "warped_path", "face_path",
            "h2v_path", "stabilized_path", "pre_deblock_path",
        ]
        target = None
        for _k in _TARGET_KEYS:
            _p = ctx.get(_k)
            if _p and path_exists(_p):
                target = _p
                break
        if not target:
            print(f"    跳过: 无可用上游视频作为换脸目标")
            return

        stem = Path(target).stem
        out_path = ctx.output_dir / f"{stem}_faceswap.mp4"

        # 换脸参数. face_swap 在 preset 里通常是 bool (开关), 参数单独放顶层 face_swap 块.
        fs_cfg = ctx.config.get("face_swap")
        if not isinstance(fs_cfg, dict):
            fs_cfg = {}
        gfpgan_strength = fs_cfg.get("gfpgan_strength", 0.5)  # 换脸后 GFPGAN 美颜修复强度
        color_match_strength = fs_cfg.get("color_match_strength", 0.8)  # 肤色迁移回原场景 (消偏色/过白)
        max_frames = fs_cfg.get("max_frames", 0)

        sys.path.insert(0, tools_dir)
        try:
            from tools.face_swap import process_video, ensure_source_photo, FFMPEG
            # 源照质量门控: 不合格 (脸小/模糊/低置信) 自动 GFPGAN 增强 → 存 {coach}_gfpgan.png 长期复用
            try:
                source_face = ensure_source_photo(source_face, lead_name, out_dir=tools_dir)
            except Exception as _e:
                print(f"    源照增强跳过: {_e}")
            print(f"    换脸: {lead_name} ← {Path(source_face).name} (GFPGAN={gfpgan_strength}, 色温={color_match_strength})")
            process_video(source_face, str(target), str(out_path),
                         max_frames=max_frames, gfpgan_strength=gfpgan_strength,
                         color_match_strength=color_match_strength)

            # 兜底：如果工具内部混音失败，手动用源视频音频混合
            if not os.path.exists(str(out_path)):
                tmp_vid = os.path.join(os.path.dirname(str(out_path)), "_tmp_vid.mp4")
                if os.path.exists(tmp_vid):
                    src_video = str(ctx.input_path)  # 源视频有音频
                    import subprocess as sp
                    sp.run([FFMPEG, "-y", "-i", tmp_vid, "-i", src_video,
                            "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                            "-map", "0:v:0", "-map", "1:a:0", "-shortest",
                            str(out_path)], check=False, capture_output=True, timeout=60)
                    try: os.remove(tmp_vid)
                    except: pass

            if os.path.exists(str(out_path)):
                ctx.set("face_swap_path", str(out_path))
                ctx.set("mascot_path", str(out_path))  # 下游吃换脸结果
                print(f"    换脸完成: {out_path.name}")
            else:
                print(f"    换脸失败: 输出未生成")
                ctx.set("face_swap_path", None)
        except Exception as e:
            print(f"    换脸失败: {e}")
            ctx.set("face_swap_path", None)
        finally:
            sys.path.remove(tools_dir)
            gc.collect()
            try:
                import torch
                torch.cuda.empty_cache()
            except:
                pass
