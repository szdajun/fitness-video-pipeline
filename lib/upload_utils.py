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


def build_title(coach: str, record_date: str = "", video_type: str = "long",
                duration_sec: int = 30) -> str:
    """生成视频标题 — 使用教练 nickname + focus 模板

    长视频模板 (钉死, CLAUDE.md 2026-06-27):
        【{nickname}】{coach}{focus}操 | {focus}跟练 | 细柳营健身
        例: 【老兵不老】郭海军力量燃脂操 | 刚劲塑形跟练 | 细柳营健身

    Shorts 模板 (钉死, CLAUDE.md 2026-06-27):
        【{nickname}】{coach}{N秒}{shorts_focus}操 | {shorts_challenge} | 细柳营健身 #Shorts
        例: 【长安腰女】丽丽30秒暴汗燃脂操 | 瘦腰瘦腿挑战 | 细柳营健身 #Shorts

    Args:
        duration_sec: Shorts 时长(秒), 写入标题. 默认 30.
    """
    # 取教练画像 (nickname + focus + shorts_focus + shorts_challenge)
    nickname = coach
    focus = "燃脂"
    focus_trail = "燃脂"
    shorts_focus = focus  # fallback for 缺 profile 的教练
    shorts_challenge = "燃脂挑战"  # fallback
    try:
        from lib.coach_profiles import COACH_PROFILES
        profile = COACH_PROFILES.get(coach, {})
        if profile.get("nickname"):
            nickname = profile["nickname"]
        if profile.get("focus"):
            focus = profile["focus"]
            focus_trail = focus
        # 2026-06-27: 优先用 shorts_focus + shorts_challenge, fallback focus
        if profile.get("shorts_focus"):
            shorts_focus = profile["shorts_focus"]
        else:
            shorts_focus = focus
        if profile.get("shorts_challenge"):
            shorts_challenge = profile["shorts_challenge"]
        else:
            shorts_challenge = f"{focus}挑战"
    except Exception:
        pass

    if video_type == "short":
        return f"【{nickname}】{coach}{duration_sec}秒{shorts_focus}操 | {shorts_challenge} | 细柳营健身 #Shorts"

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
        # 2026-06-29: 大文件(>200MB)上传走 200 OK wrapped 异常分支时, _upload 返回的
        # ytid 可能是 search 误拿的旧视频 ID (新视频未索引). verify 真实 videoId 再写 manifest.
        try:
            ytid = _verify_uploaded_ytid(ytid, title, channel=channel)
        except Exception as e:
            logger.warning("verify ytid 失败 (不影响上传): %s", e)
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


def _verify_uploaded_ytid(ytid: str, expected_title: str,
                          channel: str = "fitness", retries: int = 8,
                          wait: float = 15.0, max_age_seconds: int = 900) -> str:
    """核验上传返回的 ytid 是否真实 (根治 200 OK wrapped 误拿旧视频 + 自污染盲区).

    大文件(>200MB) resumable 上传时 youtube_upload.py 遇到 200 OK 异常会用
    search.list(order=date) 拿最近视频 ID, 但新视频刚传完未被索引 → 误拿频道里
    已索引的旧视频. 更糟: 它会把新视频的 title/desc/tags update 到误拿的旧 ID 上
    (自污染), 于是旧视频标题被改成 = expected, 单看标题匹配的 verify 会被骗过.
    (2026-06-29 建玲主视频 257MB 撞这坑: 误拿灼华 rbn30w3IF4Q 并覆盖其元数据)

    解法: 除标题外再核 publishedAt 必须 < max_age_seconds (刚上传). 旧视频即便标题
    被自污染改对了, publishedAt 仍远早于 now → 判定误拿, 转 search 找真实新视频.

    Returns: 真实 videoId (核验通过或修正后); 兜底返回原 ytid 并告警.
    """
    import time
    from datetime import datetime, timezone
    from youtube_upload import get_authenticated_service
    expected = (expected_title or "").strip()
    if not expected:
        return ytid
    yt = get_authenticated_service(channel=channel)

    def _is_fresh(pub_at) -> bool:
        """publishedAt 距 now 是否 < max_age_seconds (排除误拿的旧视频)."""
        try:
            dt = datetime.fromisoformat(str(pub_at).replace("Z", "+00:00"))
            return (datetime.now(timezone.utc) - dt).total_seconds() < max_age_seconds
        except Exception:
            return True  # 解析失败不拦 (保守认为新鲜, 不误杀)

    # 1. 核验返回 ytid 的标题 + 新鲜度 (双条件, 防自污染盲区)
    try:
        items = yt.videos().list(part="snippet", id=ytid).execute().get("items", [])
        if items:
            s = items[0]["snippet"]
            title_match = s["title"].strip() == expected
            fresh = _is_fresh(s.get("publishedAt"))
            if title_match and fresh:
                return ytid  # 标题匹配 + 刚上传 = 真实新视频
            if title_match and not fresh:
                print(f"  [verify] ytid={ytid} 标题匹配但 publishedAt={s.get('publishedAt')} "
                      f"过旧 (>={max_age_seconds}s), 判定误拿旧视频(自污染), 转 search")
    except Exception as e:
        logger.warning("verify videos.list 失败: %s", e)

    # 2. 误拿旧视频或标题不符 → search 找真实新视频 (标题+新鲜度双匹配, 索引延迟重试)
    print(f"  [verify] 搜索真实新视频 (标题+新鲜度双匹配, 最长等 {retries * wait:.0f}s)...")
    for i in range(retries):
        try:
            res = yt.search().list(part="snippet", forMine=True, type="video",
                                   order="date", maxResults=10).execute()
            for it in res.get("items", []):
                if (it["snippet"]["title"].strip() == expected
                        and _is_fresh(it["snippet"]["publishedAt"])):
                    real = it["id"]["videoId"]
                    print(f"  [verify] 真实 videoId: {ytid} -> {real} "
                          f"(第 {i + 1} 次 search 命中)")
                    return real
        except Exception as e:
            logger.warning("verify search 失败 (重试): %s", e)
        time.sleep(wait)
    print(f"  [verify][WARN] 无法确认真实 videoId (返回 {ytid}), 请手动核对 manifest!")
    return ytid  # 兜底


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
                record_date: str = "", privacy: str = "public",
                short_duration: int = 30):
    """上传长视频+短视频一对。record_date 为录制日期，省略则用当天。

    Args:
        long_path: YT 主视频路径. 必须是 *final_16x9_1920x1080.mp4 (含片头片尾).
                   **不要传 *_full_16x9.mp4** (那是去头去尾的副本, 不适合 YT).
        short_path: YT Shorts 路径. *_yt_shorts.mp4.
        short_duration: Shorts 时长(秒), 写入 YT Shorts 标题 (默认 30).
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
        title = build_title(coach, date_str, "short", duration_sec=short_duration)
        desc = build_description(coach, date_str, video_type="short")
        print(f"[上传] 短视频: {title}")
        vid = upload_video(short_path, title, desc, SHORTS_TAGS, privacy,
                           coach=coach, video_type="short")
        results["short"] = vid
        print(f"  => https://youtube.com/watch?v={vid}")

    return results
