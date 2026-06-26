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
    """生成视频标题 — 使用教练 nickname + focus 模板

    模板: 【{nickname}】{coach}{focus}操 | {focus}跟练 | 细柳营健身
    示例: 【老兵不老】郭海军力量燃脂操 | 刚劲塑形跟练 | 细柳营健身
    """
    if video_type == "short":
        return f"细柳营{coach} | 暴汗燃脂30秒 #Shorts"

    # 取教练画像 (nickname + focus)
    nickname = coach
    focus = "燃脂"
    focus_trail = "燃脂"
    try:
        from lib.coach_profiles import COACH_PROFILES
        profile = COACH_PROFILES.get(coach, {})
        if profile.get("nickname"):
            nickname = profile["nickname"]
        if profile.get("focus"):
            focus = profile["focus"]
            focus_trail = focus
    except Exception:
        pass

    return f"【{nickname}】{coach}{focus}操 | {focus_trail}跟练 | 细柳营健身"


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
                 publish_at: str = None, thumbnail_path: str = None,
                 coach: str = None, video_type: str = "long"):
    """上传单个视频到 YouTube, 自动写 manifest"""
    sys.path.insert(0, YT_UPLOAD_PATH)
    try:
        from youtube_upload import upload_video as _upload
        ytid = _upload(video_path, title, description=description,
                       tags=tags, privacy=privacy, channel=channel,
                       publish_at=publish_at, thumbnail_path=thumbnail_path)
        # 自动写 manifest (避免下次重复上传)
        try:
            _write_manifest(video_path, coach or "", video_type, ytid, title, privacy, publish_at)
        except Exception as e:
            logger.warning("写 manifest 失败 (不影响上传): %s", e)
        return ytid
    except ImportError as e:
        logger.error("导入 youtube_upload 失败: %s", e)
        raise
    except Exception as e:
        logger.error("上传失败 [%s]: %s", video_path, e)
        raise


def _write_manifest(file_path: str, coach: str, video_type: str,
                    ytid: str, title: str, privacy: str, publish_at=None):
    """追加上传记录到 records/upload_manifest.json"""
    import json
    from datetime import datetime
    manifest_path = Path(__file__).parent.parent / "records" / "upload_manifest.json"
    entries = []
    if manifest_path.exists():
        try:
            entries = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            entries = []
    entry = {
        "file": str(file_path),
        "type": video_type,
        "coach": coach,
        "ytid": ytid,
        "url": f"https://www.youtube.com/watch?v={ytid}",
        "title": title,
        "privacy": privacy,
        "uploaded_at": datetime.now().isoformat(timespec="seconds"),
    }
    if publish_at:
        entry["publish_at"] = publish_at
    entries.append(entry)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  [manifest] 已记录: {ytid}")


def upload_pair(coach: str, long_path: str, short_path: str,
                record_date: str = "", privacy: str = "public"):
    """上传长视频+短视频一对。record_date 为录制日期，省略则用当天。

    Args:
        long_path: YT 主视频路径. 必须是 *final_16x9_1920x1080.mp4 (含片头片尾).
                   **不要传 *_full_16x9.mp4** (那是去头去尾的副本, 不适合 YT).
        short_path: YT Shorts 路径. *_yt_shorts.mp4.
    """
    results = {}

    # 2026-06-27: 警告 long_path 不是 main 含头尾的 final_path
    if long_path:
        lp = Path(long_path)
        if lp.exists() and "_full_16x9" in lp.name and "_final" not in lp.name:
            print(f"  [警告] long_path 是 full_16x9 副本 (去头去尾), 应传 *final_16x9_1920x1080.mp4!")
        elif lp.exists() and "_full_16x9" in lp.name:
            print(f"  [警告] long_path 包含 _full_16x9, 确认是 *_final_16x9_full_16x9.mp4 (副本), 不建议上传!")

    # 2026-06-27: 从文件名自动检测 coach (manifest entry 需要非空 coach 字段)
    if not coach:
        try:
            from lib.coach_profiles import detect_coach_from_filename
            detected = detect_coach_from_filename(Path(long_path or short_path or "").stem)
            if detected:
                coach = detected
                print(f"  [auto-detect] coach = {coach} (从文件名)")
        except Exception:
            pass
    _ = short_path  # avoid unused warning

    date_str = record_date or _today_str()

    if long_path and Path(long_path).exists():
        title = build_title(coach, date_str, "long")
        desc = build_description(coach, date_str, video_type="long")
        print(f"[上传] 长视频: {title}")
        # 2026-06-27: 传 coach + video_type 让 manifest entry 有完整字段
        vid = upload_video(long_path, title, desc, LONG_TAGS, privacy,
                           coach=coach, video_type="long")
        results["long"] = vid
        print(f"  => https://youtube.com/watch?v={vid}")

    if short_path and Path(short_path).exists():
        title = build_title(coach, date_str, "short")
        desc = build_description(coach, date_str, video_type="short")
        print(f"[上传] 短视频: {title}")
        vid = upload_video(short_path, title, desc, SHORTS_TAGS, privacy,
                           coach=coach, video_type="short")
        results["short"] = vid
        print(f"  => https://youtube.com/watch?v={vid}")

    return results
