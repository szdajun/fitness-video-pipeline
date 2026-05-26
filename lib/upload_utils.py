"""YouTube 上传工具 — 标题/描述/标签模板"""

import sys
import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

# YouTube 上传模块路径
YT_UPLOAD_PATH = r"F:\wkspace\ComfyUI\custom_nodes"

# ====== 频道品牌信息 ======
CHANNEL_NAME = "胭脂虎健身团"
BRAND = "细柳营·胭脂虎"
LOCATION = "汉细柳营故地 · 时代广场"

# ====== 标签 ======
LONG_TAGS = [
    "细柳营胭脂虎", "有氧健身操", "燃脂暴汗", "居家健身", "打工族健身",
    "每日打卡", "零基础健身", "减肥瘦身", "全身燃脂", "有氧运动",
    "健身操", "暴汗燃脂", "瘦身塑形", "居家运动", "室内健身",
    "fitness", "aerobics", "workout", "fatburn", "homeworkout",
]

SHORTS_TAGS = LONG_TAGS + ["Shorts", "YouTubeShorts", "fitnessshorts"]


def _today_str() -> str:
    return date.today().strftime("%Y-%m-%d")


def build_title(coach: str, record_date: str = "", video_type: str = "long") -> str:
    """生成视频标题"""
    if video_type == "short":
        return f"细柳营{coach} | 暴汗燃脂30秒 #Shorts"
    base = f"细柳营·{coach} | 有氧健身操·燃脂暴汗"
    if record_date:
        base += f" | {record_date}"
    else:
        base += " | 汉细柳营故地时代广场打卡"
    return base


def build_description(coach: str, record_date: str = "",
                      location: str = LOCATION,
                      video_type: str = "long") -> str:
    """生成视频说明"""
    date_str = record_date or _today_str()
    common = (
        f"{CHANNEL_NAME} · {BRAND}\n"
        f"带操人：{coach}\n"
        f"地点：{location}\n"
        f"日期：{date_str}\n"
    )
    if video_type == "short":
        return common + (
            "\n每天暴汗打卡，焕发身体活力！\n"
            "零基础也能跳，完整版在频道\n\n"
            "#Shorts #细柳营 #胭脂虎 #有氧健身操 #燃脂暴汗\n"
            "#fitness #workout #aerobics\n"
        )
    return common + (
        "\n每天暴汗打卡，焕发身体活力！\n"
        "零基础也能跳，男女老少不限\n"
        "汉细柳营故地，传承秦人血脉\n\n"
        "【训练特点】\n"
        " 全身有氧燃脂，高效瘦身塑形\n"
        " 节拍卡点，跟着音乐律动燃脂\n"
        " 适合居家锻炼，无需器械\n\n"
        f"【{CHANNEL_NAME}】\n"
        "细柳营系列健身操，在历史文化故地\n"
        "用汗水书写当代人的健康生活\n\n"
        "每晚更新，记得点赞关注！\n"
        f"订阅频道：https://youtube.com/@{CHANNEL_NAME}\n"
    )


def upload_video(video_path: str, title: str, description: str,
                 tags: list, privacy: str = "public", channel: str = "fitness",
                 publish_at: str = None, thumbnail_path: str = None):
    """上传单个视频到 YouTube"""
    sys.path.insert(0, YT_UPLOAD_PATH)
    try:
        from youtube_upload import upload_video as _upload
        return _upload(video_path, title, description=description,
                       tags=tags, privacy=privacy, channel=channel,
                       publish_at=publish_at, thumbnail_path=thumbnail_path)
    except ImportError as e:
        logger.error("导入 youtube_upload 失败: %s", e)
        raise
    except Exception as e:
        logger.error("上传失败 [%s]: %s", video_path, e)
        raise


def upload_pair(coach: str, long_path: str, short_path: str,
                record_date: str = "", privacy: str = "public"):
    """上传长视频+短视频一对。record_date 为录制日期，省略则用当天。"""
    results = {}

    date_str = record_date or _today_str()

    if long_path and Path(long_path).exists():
        title = build_title(coach, date_str, "long")
        desc = build_description(coach, date_str, video_type="long")
        print(f"[上传] 长视频: {title}")
        vid = upload_video(long_path, title, desc, LONG_TAGS, privacy)
        results["long"] = vid
        print(f"  => https://youtube.com/watch?v={vid}")

    if short_path and Path(short_path).exists():
        title = build_title(coach, date_str, "short")
        desc = build_description(coach, date_str, video_type="short")
        print(f"[上传] 短视频: {title}")
        vid = upload_video(short_path, title, desc, SHORTS_TAGS, privacy)
        results["short"] = vid
        print(f"  => https://youtube.com/watch?v={vid}")

    return results
