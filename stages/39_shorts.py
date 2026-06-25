"""Stage 39: YouTube Shorts / 抖音竖版 — 智能选段 + 诗词片头 + CTA片尾

YouTube Shorts: 30s 精华, 英文标题 + 诗词 + 双语CTA
抖音竖版: 完整版 + 可选片头片尾

用法:
    from stages.39_shorts import make_shorts, make_douyin_vertical
"""

import json, os, subprocess, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.coach_profiles import (
    get_shorts_poem, get_shorts_en, DEFAULT_SHORTS_POEM, DEFAULT_SHORTS_EN,
    _SHORTS_CTA_EN, detect_coach_from_filename,
)

FFMPEG = "C:/Users/18091/ffmpeg/ffmpeg.exe"
if not os.path.exists(FFMPEG):
    import shutil as _su
    FFMPEG = _su.which("ffmpeg") or "ffmpeg"
FFPROBE = FFMPEG.replace("ffmpeg.exe", "ffprobe.exe")

FONT = "C\\:/Windows/Fonts/simhei.ttf"
FONT_BOLD = "C\\:/Windows/Fonts/msyhbd.ttf"


# ── 工具 ──────────────────────────────────────────────

def _get_duration(video_path: str) -> float:
    """获取视频时长(秒)"""
    r = subprocess.run([FFPROBE, "-v", "quiet", "-show_entries",
                       "format=duration", "-of", "csv=p=0", str(video_path)],
                      capture_output=True, text=True, encoding="utf-8")
    return float(r.stdout.strip())


# ── 智能选段 ──────────────────────────────────────────

def _find_best_segment(video_path, keypoints_file, duration=30):
    """从运动量中选出最佳 N 秒段"""
    import json as _json
    if not os.path.exists(keypoints_file):
        return max(0, _get_duration(video_path) * 0.15)

    with open(keypoints_file, encoding="utf-8") as f:
        kp_data = _json.load(f)
    kps = kp_data.get("keypoints", kp_data)

    fps = 30
    window = int(duration * fps)
    try:
        frames = sorted(int(k) for k in kps.keys() if k.isdigit())
    except (ValueError, TypeError):
        frames = []
    if not frames:
        return max(0, _get_duration(video_path) * 0.15)
    total_frames = frames[-1] + 1
    total_frames = min(total_frames, 9000)

    best_start = int(total_frames * 0.15)
    best_score = -1

    for start in range(0, total_frames - window, int(fps * 0.5)):
        end = start + window
        motion = 0
        for f_idx in range(start, end, 3):
            entry = kps.get(str(f_idx))
            if entry and isinstance(entry, list) and entry:
                person = entry[0] if isinstance(entry[0], list) else entry
                visible = sum(1 for kp in person if isinstance(kp, (list, tuple)) and len(kp) > 2 and kp[2] > 0.3)
                motion += visible
        score = motion
        if score > best_score:
            best_score = score
            best_start = start

    return max(best_start / fps, 0)


# ── ffmpeg drawtext 转义 ───────────────────────────────

def _escape_ffmpeg_text(text: str) -> str:
    """转义 drawtext 中的特殊字符"""
    return (text.replace(":", "\\:")
            .replace("'", "\\'")
            .replace("%", "\\\\%"))


# ── 片头叠加滤镜 (英文标题 + 诗歌) ─────────────────────

def _opening_overlay_filter(coach_name: str, duration: float, fps: float):
    """生成片头文字叠加的 ffmpeg drawtext 滤镜链。

    结构: 顶层英文大字 → 小字副标题 → 中部黄色诗句
    渐显 0.5s → 停留 2.5s → 渐隐 0.5s (总共约 3.5s 后消失)
    """
    en = get_shorts_en(coach_name) or DEFAULT_SHORTS_EN
    title = en.get("title", DEFAULT_SHORTS_EN["title"])
    subtitle = en.get("subtitle", DEFAULT_SHORTS_EN["subtitle"])
    poem = get_shorts_poem(coach_name) or DEFAULT_SHORTS_POEM

    # 渐显/渐隐 alpha 表达式 (帧级)
    # 0-0.5s 渐显, 0.5-3.0s 停留, 3.0-3.5s 渐隐
    total_fade = 3.5
    alpha_expr = (
        f"if(lt(t,0.5), t/0.5, "
        f"if(lt(t,{total_fade - 0.5}), 1, "
        f"if(lt(t,{total_fade}), ({total_fade}-t)/0.5, 0)))"
    )

    title_esc = _escape_ffmpeg_text(title)
    sub_esc = _escape_ffmpeg_text(subtitle)
    poem_esc = _escape_ffmpeg_text(poem)

    filters = [
        # 英文大字 - 顶部居中, 黄色粗体
        f"drawtext=fontfile='{FONT_BOLD}':text='{title_esc}':"
        f"fontcolor=yellow@$ALPHA$:fontsize=52:"
        f"x=(w-text_w)/2:y=h*0.06:"
        f"borderw=3:bordercolor=black@0.7",

        # 英文副标题 - 大字下方
        f"drawtext=fontfile='{FONT}':text='{sub_esc}':"
        f"fontcolor=white@$ALPHA$:fontsize=28:"
        f"x=(w-text_w)/2:y=h*0.14:"
        f"borderw=2:bordercolor=black@0.5",

        # 中文诗句 - 中部偏上
        f"drawtext=fontfile='{FONT}':text='{poem_esc}':"
        f"fontcolor=yellow@$ALPHA$:fontsize=36:"
        f"x=(w-text_w)/2:y=h*0.30:"
        f"line_spacing=8:"
        f"borderw=2:bordercolor=black@0.7",
    ]

    # 替换 $ALPHA$ 为实际表达式
    filters = [f.replace("$ALPHA$", alpha_expr) for f in filters]
    return ",".join(filters)


# ── 片尾 CTA 滤镜 ─────────────────────────────────────

def _ending_cta_filter(duration: float):
    """片尾 CTA 三层字幕:
    ① 黄色大字 点赞 LIKE & SUBSCRIBE
    ② 白色小字 完整版 Full Workout on Channel
    ③ 灰色小字+红线 新视频 New Videos Daily

    最后 4s 渐显
    """
    total = duration
    fade_in_start = total - 4.0

    alpha_expr = (
        f"if(lt(t,{fade_in_start}), 0, "
        f"if(lt(t,{fade_in_start + 1.0}), (t-{fade_in_start})/1.0, 1))"
    )

    cta_lines = _SHORTS_CTA_EN  # ["点赞 LIKE & SUBSCRIBE 关注", "完整版 Full Workout on Channel", "新视频 New Videos Daily"]

    filters = [
        # 红色分割线
        f"drawtext=fontfile='{FONT}':text='———————————':"
        f"fontcolor=red@$ALPHA$:fontsize=24:"
        f"x=(w-text_w)/2:y=h*0.78",

        # ① 黄色大字 CTA
        f"drawtext=fontfile='{FONT_BOLD}':text='{_escape_ffmpeg_text(cta_lines[0])}':"
        f"fontcolor=yellow@$ALPHA$:fontsize=34:"
        f"x=(w-text_w)/2:y=h*0.82:"
        f"borderw=2:bordercolor=black@0.7",

        # ② 白色小字
        f"drawtext=fontfile='{FONT}':text='{_escape_ffmpeg_text(cta_lines[1])}':"
        f"fontcolor=white@$ALPHA$:fontsize=24:"
        f"x=(w-text_w)/2:y=h*0.88",

        # ③ 灰色小字
        f"drawtext=fontfile='{FONT}':text='{_escape_ffmpeg_text(cta_lines[2])}':"
        f"fontcolor=gray@$ALPHA$:fontsize=20:"
        f"x=(w-text_w)/2:y=h*0.93",
    ]

    filters = [f.replace("$ALPHA$", alpha_expr) for f in filters]
    return ",".join(filters)


# ── 主入口 ────────────────────────────────────────────

def make_shorts(src_path, output_dir, keypoints_file, duration=30, audio_src=None):
    """生成 YouTube Shorts (30s, 9:16, 英文+诗词+CTA)

    Args:
        src_path: 处理后的横版视频 (1920x1080, 含所有效果)
        output_dir: 输出目录
        keypoints_file: YOLO 关键点 JSON
        duration: Shorts 时长 (默认 30s)
        audio_src: 音频源 (默认同 src_path)

    Returns:
        输出文件路径, 失败返回 None
    """
    src_path = str(src_path)
    output_dir = str(output_dir)
    src_stem = Path(src_path).stem

    # 1. 检测教练名 (用于英文标题/诗歌)
    coach_name = detect_coach_from_filename(src_path)

    # 2. 找最佳起始位置
    start_time = _find_best_segment(src_path, keypoints_file, duration)

    # 3. 获取视频时长, 钳制 start
    total_dur = _get_duration(src_path)
    start_time = max(0, min(start_time, total_dur - duration))

    # 4. 构建滤镜链
    # 4a. 9:16 裁切 (1920x1080 → 1080x1920)
    crop_w = 608   # 1080 * 9/16
    crop_h = 1080
    crop_x = (1920 - crop_w) // 2  # 默认居中, 有 keypoints 时可动态调整
    crop_filter = f"crop={crop_w}:{crop_h}:{crop_x}:0,scale=1080:1920:flags=lanczos"

    # 4b. 片头文字 (渐显渐隐)
    fps = 30
    opening = _opening_overlay_filter(coach_name or "", duration, fps)

    # 4c. 片尾 CTA
    cta = _ending_cta_filter(duration)

    vf = f"{crop_filter},{opening},{cta}"

    # 5. FFmpeg 编码
    out_path = os.path.join(output_dir, f"{src_stem}_shorts_v2.mp4")

    audio_input = audio_src or src_path
    cmd = [
        FFMPEG, "-y",
        "-ss", str(start_time), "-t", str(duration),
        "-i", src_path,
        "-ss", str(start_time), "-t", str(duration),
        "-i", audio_input,
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        out_path
    ]

    print(f"    [Shorts] start={start_time:.1f}s dur={duration}s coach={coach_name or 'default'}")
    r = subprocess.run(cmd, capture_output=True, text=True,
                      encoding="utf-8", errors="replace", timeout=300)

    if r.returncode != 0:
        err = r.stderr[-300:] if r.stderr else ""
        print(f"    Shorts 失败: {err}")
        return None

    if os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
        print(f"    Shorts: {Path(out_path).name} ({os.path.getsize(out_path)//1024//1024}MB)")
        return out_path
    return None


def make_douyin_vertical(src_path, output_dir, keypoints_file,
                          audio_src=None, intro_path=None, outro_path=None):
    """生成抖音竖版 (完整版, 9:16, 可选片头片尾)

    Args:
        src_path: 处理后的横版视频
        output_dir: 输出目录
        keypoints_file: 关键点文件
        audio_src: 音频源
        intro_path: 片头视频 (可选)
        outro_path: 片尾视频 (可选)

    Returns:
        输出路径或 None
    """
    src_path = str(src_path)
    output_dir = str(output_dir)
    src_stem = Path(src_path).stem

    # 9:16 裁切滤镜
    crop_w = 608
    crop_h = 1080
    crop_x = (1920 - crop_w) // 2
    crop_filter = f"crop={crop_w}:{crop_h}:{crop_x}:0,scale=1080:1920:flags=lanczos"

    out_path = os.path.join(output_dir, f"{src_stem}_douyin.mp4")

    # 如果有片头片尾, concat 三段
    has_intro = intro_path and os.path.exists(str(intro_path))
    has_outro = outro_path and os.path.exists(str(outro_path))

    if has_intro or has_outro:
        # 构建 concat 列表
        concat_file = os.path.join(output_dir, "_douyin_concat.txt")
        inputs = []
        filter_parts = []
        with open(concat_file, "w", encoding="utf-8") as cf:
            if has_intro:
                cf.write(f"file '{str(intro_path).replace(chr(92), '/')}'\n")
                inputs.extend(["-i", str(intro_path)])
                filter_parts.append(f"[{len(inputs)//2 - 1}:v]crop=608:1080:(1920-608)//2:0,scale=1080:1920:flags=lanczos,setsar=1[v{len(inputs)//2 - 1}]")
            # main
            cf.write(f"file '{src_path.replace(chr(92), '/')}'\n")
            inputs.extend(["-i", src_path])
            filter_parts.append(f"[{len(inputs)//2 - 1}:v]{crop_filter},setsar=1[v{len(inputs)//2 - 1}]")
            if has_outro:
                cf.write(f"file '{str(outro_path).replace(chr(92), '/')}'\n")
                inputs.extend(["-i", str(outro_path)])
                filter_parts.append(f"[{len(inputs)//2 - 1}:v]crop=608:1080:(1920-608)//2:0,scale=1080:1920:flags=lanczos,setsar=1[v{len(inputs)//2 - 1}]")

        # concat all video streams
        v_inputs = "".join(f"[v{i}]" for i in range(len(filter_parts)))
        vf = ";".join(filter_parts) + f";{v_inputs}concat=n={len(filter_parts)}:v=1:a=0[v]"

        audio_input = audio_src or src_path
        cmd = [
            FFMPEG, "-y",
            *inputs,
            "-i", audio_input,
            "-filter_complex", vf,
            "-map", "[v]", "-map", f"{len(filter_parts)}:a:0",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            out_path
        ]
    else:
        audio_input = audio_src or src_path
        cmd = [
            FFMPEG, "-y",
            "-i", src_path,
            "-i", audio_input,
            "-vf", crop_filter,
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k",
            "-map", "0:v:0", "-map", "1:a:0",
            "-shortest",
            out_path
        ]

    has_extra = " +片头尾" if (has_intro or has_outro) else ""
    print(f"    [抖音竖版] 9:16 完整版{has_extra}")
    r = subprocess.run(cmd, capture_output=True, text=True,
                      encoding="utf-8", errors="replace", timeout=600)

    if r.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 10000:
        print(f"    抖音竖版: {Path(out_path).name} ({os.path.getsize(out_path)//1024//1024}MB)")
        return out_path

    err = r.stderr[-200:] if r.stderr else ""
    print(f"    抖音竖版失败: {err}")
    return None
