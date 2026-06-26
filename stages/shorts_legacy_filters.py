"""stages/shorts_legacy_filters.py — 原 39_shorts.py 抽出的片头/片尾滤镜

抽离原因 (2026-06-27): 重构 ShortsStage 时, 让 make_vertical(profile) 复用这些滤镜,
不再重写 (你说"用 youtube 滤镜更成熟"). 原 make_shorts/make_douyin_vertical 已弃用,
但滤镜逻辑保留作为单一真相源.

包含:
  - _opening_overlay_filter: 英文标题 + 副标题 + 中文诗词 + 渐显渐隐
  - _ending_cta_filter: 红分割线 + 黄色 CTA + 白色副标 + 灰色水印
  - _escape_ffmpeg_text: 中文 / 特殊字符转义
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from lib.coach_profiles import (
    get_shorts_poem, get_shorts_en, DEFAULT_SHORTS_POEM, DEFAULT_SHORTS_EN,
)


# 字体路径 (沿用原版)
FONT = r"C:/Windows/Fonts/msyh.ttc"
FONT_BOLD = r"C:/Windows/Fonts/msyhbd.ttc"


def _escape_ffmpeg_text(text: str) -> str:
    """ffmpeg drawtext 文本转义"""
    return (text.replace("\\", "\\\\")
               .replace(":", "\\:")
               .replace("'", "\\'"))


def _opening_overlay_filter(coach_name: str, duration: float) -> str:
    """片头: 英文标题 + 副标题 + 中文诗词 (渐显渐隐 ~3.5s)"""
    en = get_shorts_en(coach_name) or DEFAULT_SHORTS_EN
    title = en.get("title", DEFAULT_SHORTS_EN["title"])
    subtitle = en.get("subtitle", DEFAULT_SHORTS_EN["subtitle"])
    poem = get_shorts_poem(coach_name) or DEFAULT_SHORTS_POEM

    total_fade = 3.5
    alpha_expr = (
        f"if(lt(t,0.5), t/0.5, "
        f"if(lt(t,{total_fade - 0.5}), 1, "
        f"if(lt(t,{total_fade}), ({total_fade}-t)/0.5, 0)))"
    )
    alpha_opt = f"alpha='{alpha_expr}'"

    title_esc = _escape_ffmpeg_text(title)
    sub_esc = _escape_ffmpeg_text(subtitle)
    poem_esc = _escape_ffmpeg_text(poem)

    filters = [
        f"drawtext=fontfile='{FONT_BOLD}':text='{title_esc}':"
        f"fontcolor=yellow:fontsize=56:{alpha_opt}:"
        f"x=(w-text_w)/2:y=h*0.02:"
        f"borderw=3:bordercolor=black",

        f"drawtext=fontfile='{FONT}':text='{sub_esc}':"
        f"fontcolor=white:fontsize=40:{alpha_opt}:"
        f"x=(w-text_w)/2:y=h*0.08:"
        f"borderw=2:bordercolor=black",
    ]

    poem_lines = poem.strip().split("\n")
    for i, line in enumerate(poem_lines):
        line_esc = _escape_ffmpeg_text(line.strip())
        line_y = 0.17 + i * 0.045
        filters.append(
            f"drawtext=fontfile='{FONT}':text='{line_esc}':"
            f"fontcolor=yellow:fontsize=44:{alpha_opt}:"
            f"x=(w-text_w)/2:y=h*{line_y:.3f}:"
            f"borderw=2:bordercolor=black"
        )

    return ",".join(filters)


def _ending_cta_filter(duration: float) -> str:
    """片尾 CTA: 渐入 (最后 4s) 红分割线 + 黄色大字 + 白色副标 + 灰色水印"""
    total = duration
    fade_in_start = total - 4.0

    alpha_expr = (
        f"if(lt(t,{fade_in_start}), 0, "
        f"if(lt(t,{fade_in_start + 1.0}), (t-{fade_in_start})/1.0, 1))"
    )
    alpha_opt = f"alpha='{alpha_expr}'"

    cta_lines = (
        "SUBSCRIBE for more fitness!",
        "Like & Share with friends",
        "@xiliuying_fit",
    )

    filters = [
        f"drawtext=fontfile='{FONT}':text='———————————':"
        f"fontcolor=red:fontsize=32:{alpha_opt}:"
        f"x=(w-text_w)/2:y=h*0.76",

        f"drawtext=fontfile='{FONT_BOLD}':text='{_escape_ffmpeg_text(cta_lines[0])}':"
        f"fontcolor=yellow:fontsize=50:{alpha_opt}:"
        f"x=(w-text_w)/2:y=h*0.81:"
        f"borderw=3:bordercolor=black",

        f"drawtext=fontfile='{FONT}':text='{_escape_ffmpeg_text(cta_lines[1])}':"
        f"fontcolor=white:fontsize=40:{alpha_opt}:"
        f"x=(w-text_w)/2:y=h*0.87",

        f"drawtext=fontfile='{FONT}':text='{_escape_ffmpeg_text(cta_lines[2])}':"
        f"fontcolor=gray:fontsize=26:{alpha_opt}:"
        f"x=(w-text_w)/2:y=h*0.93",
    ]
    return ",".join(filters)
