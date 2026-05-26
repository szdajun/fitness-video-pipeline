"""Fitness Video Auto-Publisher — 零人工介入，全自动处理+上传。

调用现有管线 (main.py) 完成全部特效处理（降噪/节拍闪光/高光/能量条/
片头片尾/水印/吉祥物/电影滤镜/弹幕/爆燃大字），然后叠加 hook、生成
缩略图+Shorts、上传 YouTube。

用法:
    python auto_publish.py              # 一次性：处理所有新视频
    python auto_publish.py --watch      # 持续监控，来一个处理一个
    python auto_publish.py --loop 60    # 每 60 秒扫描一次
"""
import os, sys, time, json, random, subprocess, glob, logging, argparse
from pathlib import Path
from datetime import datetime
from lib.coach_profiles import get_coach, detect_coach_from_filename, get_shorts_en, DEFAULT_CHANNEL

# ── 配置 ──────────────────────────────────────────────
SOURCE_DIR     = r"C:\Users\18091\Desktop\短视频素材"
PROCESSED_DIR  = os.path.join(SOURCE_DIR, "_processed")
STATE_FILE     = os.path.join(os.path.dirname(__file__), "auto_publish_state.json")
LOG_FILE       = os.path.join(os.path.dirname(__file__), "auto_publish.log")
DAY_COUNTER    = os.path.join(os.path.dirname(__file__), "day_counter.json")
OUTPUT_BASE    = os.path.join(os.path.dirname(__file__), "output")
PIPELINE       = os.path.join(os.path.dirname(__file__), "main.py")
PIPELINE_CFG   = os.path.join(os.path.dirname(__file__), "config.yaml")
VENV_PY        = os.path.join(os.path.dirname(__file__), "venv", "Scripts", "python.exe")
FFMPEG         = r"C:\Users\18091\ffmpeg\ffmpeg.exe"

CHANNEL       = "fitness"
PRIVACY       = "private"
TAGS          = [
    "有氧健身操", "减肥操", "暴汗燃脂", "瘦全身", "瘦肚子",
    "瘦大腿", "瘦手臂", "瘦腿", "居家运动", "在家健身",
    "零基础健身", "新手小白", "女性健身", "燃脂操",
    "细柳营胭脂虎", "每日打卡", "30天挑战", "免费健身",
    "大体重减肥", "快速燃脂", "塑形", "马甲线",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"),
              logging.StreamHandler()],
)
log = logging.getLogger("auto_publish")


def ensure_disk_space(src_path):
    """预估管线所需磁盘空间，不足则清理或报警"""
    src_size = os.path.getsize(src_path) / (1024 ** 3)  # GB
    # HD源: 中间文件约 15-20x 源文件大小。低码率源约 8-12x。
    # 留 2x 安全余量
    stages_enabled = 0
    try:
        import yaml
        with open(PIPELINE_CFG, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        stages_enabled = sum(1 for v in cfg.get("stages", {}).values() if v is True)
    except Exception:
        stages_enabled = 14
    multiplier = 22 if stages_enabled > 12 else 12
    needed = src_size * multiplier
    needed = max(needed, 5)  # 最少预留5GB

    drive = os.path.splitdrive(os.path.dirname(__file__))[0] + "\\"
    import shutil as _shutil
    free = _shutil.disk_usage(drive).free / (1024 ** 3)

    if free < needed:
        # 尝试清理 F:\wkspace\fitness-video-pipeline\_temp
        temp_dir = os.path.join(os.path.dirname(__file__), "_temp")
        if os.path.exists(temp_dir):
            try:
                _shutil.rmtree(temp_dir, ignore_errors=True)
                free = _shutil.disk_usage(drive).free / (1024 ** 3)
                log.info(f"清理 _temp 后剩余: {free:.1f} GB")
            except Exception:
                pass

    if free < needed:
        log.warning(f"磁盘空间不足! 需要 ~{needed:.0f}GB, 剩余 {free:.1f}GB")
        return False

    log.info(f"磁盘检查: 需要 ~{needed:.0f}GB, 剩余 {free:.1f}GB, 源文件 {src_size:.1f}GB ✓")
    return True

# ── 状态 ───────────────────────────────────────────────
def _load_json(path, default=None):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return default or {}

def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ── 日计数器 ───────────────────────────────────────────
def get_next_day(coach):
    coils = _load_json(DAY_COUNTER, {})
    day = coils.get(coach, 1)
    coils[coach] = day + 1
    _save_json(DAY_COUNTER, coils)
    return day

# ── SEO 文案（模板，无 AI）───────────────────────────────
def generate_title(coach_name, nickname, day):
    if not coach_name:
        return f"有氧健身操 暴汗燃脂减肥操 瘦全身零基础在家跳"
    return random.choice([
        f"30分钟暴汗燃脂 瘦肚子瘦大腿 零基础在家跳就能瘦 | {nickname}带操 Day{day}",
        f"在家就能跳的减肥操 瘦全身高效燃脂 新手小白友好 | {nickname}领操 Day{day}",
        f"零基础有氧健身操 瘦腿瘦腰瘦肚子 居家运动燃脂 | {nickname}细柳营 Day{day}",
        f"30分钟暴汗有氧操 大体重友好全身燃脂 站着就能瘦 | {nickname}带操 Day{day}",
    ])

def generate_description(coach, nickname, day):
    desc_line = coach.get("judgment", f"{nickname}领操有氧健身")
    return f"""【{DEFAULT_CHANNEL}Day{day}】{nickname}有氧健身操 | 零基础暴汗燃脂 瘦全身减肥操 在家就能跳

{desc_line}

🔥 新手友好 男女老少都能跟
📍 汉细柳营故地 · 西安时代广场
🎵 天黑了下班了吃过了乡党们锻炼了
💪 996后的救赎 苦中作乐 逆风飞扬
🏃 躺不平病不起 来细柳营拿身体对抗生活

📌 每天免费更新 点关注不迷路
✍️ 今天你打卡了吗？评论区喊一声「练了」！

#{DEFAULT_CHANNEL} #{'#' + nickname if nickname else ''} #有氧健身操 #减肥操 #暴汗燃脂 #瘦全身 #零基础 #在家健身 #打工族健身 #每日打卡 #30天挑战"""

# ── 视频处理（调用现有管线）─────────────────────────────
def run_pipeline(src_path):
    """调用 main.py 完整管线：降噪→节拍闪光→高光→能量条→片头片尾→水印→
    吉祥物→电影滤镜→弹幕→爆燃大字→导出。返回输出文件路径。"""
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["OPENCV_VIDEOIO_PRIORITY_LIST"] = "MSMF"
    subprocess.run(
        [VENV_PY, PIPELINE, "process", src_path, "-c", PIPELINE_CFG],
        check=True, env=env,
    )
    # 找最新输出
    videos = glob.glob(os.path.join(OUTPUT_BASE, "**", "*_final_*.mp4"), recursive=True)
    if not videos:
        videos = glob.glob(os.path.join(OUTPUT_BASE, "**", "*.mp4"), recursive=True)
    if not videos:
        raise RuntimeError("管线未生成输出")
    return max(videos, key=os.path.getmtime)

def add_hook_overlay(video_path, coach_nickname, day):
    """PIL+FFmpeg 叠加 challenge hook 文字到前 4 秒"""
    import cv2, numpy as np, tempfile, shutil
    from PIL import Image, ImageDraw, ImageFont

    hook = f"天黑了 下班了 吃过了 乡党们 锻炼了! | 细柳营Day{day} | {coach_nickname}带操"
    for fp in ["C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/msyh.ttc",
               "C:/Windows/Fonts/simhei.ttf"]:
        if os.path.exists(fp):
            font = ImageFont.truetype(fp, 36)
            break
    else:
        log.warning("无中文字体，跳过 hook")
        return video_path

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(7))
    hook_n = int(fps * 4)

    tmp = Path(tempfile.mkdtemp(prefix="hook_"))
    try:
        for fi in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            if fi < hook_n:
                pi = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                d = ImageDraw.Draw(pi)
                bb = d.textbbox((0, 0), hook, font=font)
                tw, th = bb[2] - bb[0], bb[3] - bb[1]
                tx, ty = (w - tw) // 2, h - th - 80
                pad = 12
                ov = frame.copy()
                cv2.rectangle(ov, (tx - pad, ty - pad),
                             (tx + tw + pad, ty + th + pad), (0, 0, 0), -1)
                frame = cv2.addWeighted(frame, 0.6, ov, 0.4, 0)
                pf = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                d2 = ImageDraw.Draw(pf)
                d2.text((tx, ty), hook, font=font, fill=(255, 255, 255))
                frame = cv2.cvtColor(np.array(pf), cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(tmp / f"f_{fi:06d}.png"), frame)
        cap.release()

        out = video_path.replace(".mp4", "_hook.mp4")
        subprocess.run([
            FFMPEG, "-y", "-framerate", str(fps),
            "-i", str(tmp / "f_%06d.png"),
            "-i", video_path, "-map", "0:v", "-map", "1:a",
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-c:a", "copy", "-shortest", out,
        ], capture_output=True, check=True)
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

def find_best_thumbnail_frame(keypoints_file, total_frames, fps):
    """用关键点数据找最燃封面帧：运动强度 + 人数 + 构图 + 姿态展开度"""
    if not keypoints_file or not os.path.exists(keypoints_file):
        return int(total_frames * 0.33)

    try:
        with open(keypoints_file, encoding="utf-8") as f:
            kps_data = json.load(f)
    except Exception:
        return int(total_frames * 0.33)

    kps = kps_data.get("keypoints", kps_data)
    if not isinstance(kps, dict):
        return int(total_frames * 0.33)

    best_score = -1.0
    best_frame = int(total_frames * 0.33)
    sample_interval = max(1, int(fps * 0.5))  # 每0.5秒采样

    for fi in range(0, total_frames, sample_interval):
        entry = kps.get(str(fi))
        if not entry or not isinstance(entry, list) or not entry:
            continue

        # 检查第一个人的关键点格式
        person0 = entry[0]
        if not isinstance(person0, list) or not person0:
            continue
        if not isinstance(person0[0], (list, tuple)):
            continue

        # ---- 人数分 ----
        visible_persons = 0
        for person in entry:
            if isinstance(person, list) and len(person) >= 12:
                vis = sum(1 for kp in person if len(kp) >= 3 and kp[2] > 0.3)
                if vis >= 6:
                    visible_persons += 1
        person_score = min(visible_persons, 8) / 8.0

        # ---- 姿态展开度 (手臂张开 + 腿分开 = 视觉冲击力) ----
        coach = person0
        spread_score = 0.0
        if len(coach) >= 17:
            # 手腕间距 (9, 10) 相对于肩宽 (5, 6)
            if all(len(coach[i]) >= 3 and coach[i][2] > 0.3 for i in [5, 6, 9, 10]):
                shoulder_w = abs(coach[5][0] - coach[6][0])
                wrist_w = abs(coach[9][0] - coach[10][0])
                if shoulder_w > 0.01:
                    spread_score += min(wrist_w / shoulder_w, 3.0) / 3.0
            # 脚踝间距 (15, 16) 相对于髋宽 (11, 12)
            if all(len(coach[i]) >= 3 and coach[i][2] > 0.3 for i in [11, 12, 15, 16]):
                hip_w = abs(coach[11][0] - coach[12][0])
                ankle_w = abs(coach[15][0] - coach[16][0])
                if hip_w > 0.01:
                    spread_score += min(ankle_w / hip_w, 3.0) / 3.0
            spread_score = min(spread_score, 1.0)

        # ---- 构图分 (教练居中) ----
        center_score = 0.5
        if len(coach) >= 13:
            pts = [coach[i] for i in [5, 6, 11, 12] if len(coach[i]) >= 3 and coach[i][2] > 0.3]
            if pts:
                cx = sum(v[0] for v in pts) / len(pts)
                center_score = 1.0 - abs(cx - 0.5) * 2

        # ---- 综�合得分 ----
        score = spread_score * 4.0 + person_score * 2.5 + center_score * 1.5

        if score > best_score:
            best_score = score
            best_frame = fi

    return best_frame


def generate_thumbnail(video_path, coach_nickname, title, day=1,
                       keypoints_file=None, coach_name=None):
    """智能封面：最燃帧 + 双语大字标题"""
    import cv2, numpy as np
    from PIL import Image, ImageDraw, ImageFont

    thumb = video_path.replace(".mp4", "_thumb.jpg")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4))

    # 智能选帧 or fallback
    best_fi = find_best_thumbnail_frame(keypoints_file, total, fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, best_fi)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None

    pi = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pi)
    ref = min(w, h)

    # 获取教练英文 Shorts 标题
    en_data = {}
    if coach_name:
        try:
            en_data = get_shorts_en(coach_name)
        except Exception:
            pass
    en_title = en_data.get("title", "DAILY AEROBIC WORKOUT")
    # 提取教练英文名（去掉中文后缀，如 "3-Mommy Coach · 三宝妈" → "3-Mommy Coach"）
    coach_en_name = en_data.get("subtitle", "Outdoor Group Fitness")
    if "·" in coach_en_name:
        coach_en_name = coach_en_name.split("·")[0].strip()

    # 字体
    try:
        font_en = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", int(ref * 0.10))
        font_mid = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", int(ref * 0.06))
        font_sm = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", int(ref * 0.045))
        font_cta = ImageFont.truetype("C:/Windows/Fonts/msyhbd.ttc", int(ref * 0.065))
    except Exception:
        return None

    # 半透明底条
    bar_top_h = int(ref * 0.42)
    bar_bot_h = int(ref * 0.18)
    overlay = Image.new("RGBA", pi.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rectangle([(0, 0), (w, bar_top_h)], fill=(0, 0, 0, 155))
    od.rectangle([(0, h - bar_bot_h), (w, h)], fill=(0, 0, 0, 155))

    # 顶部文字: 三层结构
    cy = int(ref * 0.04)
    row_gap = int(ref * 0.12)
    row_gap_sm = int(ref * 0.09)

    # 第一行: 英文标题（黄色大字）
    bbox = draw.textbbox((0, 0), en_title, font=font_en)
    tx = (w - (bbox[2] - bbox[0])) // 2
    od.text((tx + 3, cy + 3), en_title, font=font_en, fill=(0, 0, 0, 120))
    od.text((tx, cy), en_title, font=font_en, fill=(255, 220, 50))

    # 第二行: Daily Outdoor Aerobics（白色中号）
    line2 = "Daily Outdoor Aerobics"
    cy2 = cy + row_gap
    bbox2 = draw.textbbox((0, 0), line2, font=font_mid)
    tx2 = (w - (bbox2[2] - bbox2[0])) // 2
    od.text((tx2 + 2, cy2 + 2), line2, font=font_mid, fill=(0, 0, 0, 120))
    od.text((tx2, cy2), line2, font=font_mid, fill=(255, 255, 255))

    # 第三行: 教练英文名（白色小号）
    cy3 = cy2 + row_gap_sm
    bbox3 = draw.textbbox((0, 0), coach_en_name, font=font_sm)
    tx3 = (w - (bbox3[2] - bbox3[0])) // 2
    od.text((tx3 + 2, cy3 + 2), coach_en_name, font=font_sm, fill=(0, 0, 0, 120))
    od.text((tx3, cy3), coach_en_name, font=font_sm, fill=(255, 255, 255))

    # 底部 CTA: LIKE 👍 SUBSCRIBE ❤️
    cta_text = "LIKE 👍   SUBSCRIBE ❤️"
    bb4 = draw.textbbox((0, 0), cta_text, font=font_cta)
    tx4 = (w - (bb4[2] - bb4[0])) // 2
    ty4 = h - int(ref * 0.12)
    # 红色底色条
    pad_x, pad_y = 30, 16
    od.rectangle(
        [(tx4 - pad_x, ty4 - pad_y),
         (tx4 + (bb4[2] - bb4[0]) + pad_x, ty4 + (bb4[3] - bb4[1]) + pad_y)],
        fill=(220, 30, 30, 200))
    od.text((tx4 + 2, ty4 + 2), cta_text, font=font_cta, fill=(0, 0, 0, 100))
    od.text((tx4, ty4), cta_text, font=font_cta, fill=(255, 255, 50))

    Image.alpha_composite(pi.convert("RGBA"), overlay).convert("RGB").save(thumb, "JPEG", quality=92)
    return thumb

def make_shorts_clip(video_path, duration=15):
    """竖屏 Shorts 中心裁剪"""
    out = video_path.replace(".mp4", "_shorts.mp4")
    dur = _probe_duration(video_path)
    start = max(dur * 0.3, 1)
    subprocess.run([
        FFMPEG, "-y", "-ss", str(start), "-t", str(duration),
        "-i", video_path,
        "-vf", "crop=ih*9/16:ih,scale=1080:1920",
        "-c:v", "libx264", "-c:a", "aac", out,
    ], capture_output=True, check=True)
    return out

def _probe_duration(video_path):
    r2 = subprocess.run(["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                        "-of", "csv=p=0", video_path], capture_output=True, text=True)
    return float(r2.stdout.strip())

# ── 上传 ───────────────────────────────────────────────
def upload(video_path, title, description, tags, thumbnail_path=None):
    sys.path.insert(0, r"F:\wkspace\ComfyUI\custom_nodes")
    from youtube_upload import upload_video
    return upload_video(video_path, title, description=description, tags=tags,
                        privacy=PRIVACY, channel=CHANNEL,
                        thumbnail_path=thumbnail_path)

# ── 单视频全流程 ──────────────────────────────────────
def process_one(video_path):
    fname = os.path.basename(video_path)
    log.info(f"处理: {fname}")

    coach_name = detect_coach_from_filename(fname)
    coach = get_coach(coach_name) if coach_name else None
    nickname = coach["nickname"] if coach else None
    coach_key = coach["name"] if coach else coach_name
    day = get_next_day(coach_key) if coach_key else 1
    title = generate_title(coach_key, nickname, day)
    log.info(f"  Day{day} | {nickname or '未知'} | {title[:50]}")

    # 1. 完整管线（片头片尾/水印/特效/出片）
    log.info("  [1/5] 管线处理...")
    if not ensure_disk_space(video_path):
        log.error("磁盘空间不足，跳过")
        return False
    final = run_pipeline(video_path)

    # 2. Hook 叠加
    if nickname:
        log.info("  [2/5] Hook...")
        try:
            final = add_hook_overlay(final, nickname, day)
        except Exception as e:
            log.warning(f"  Hook 失败: {e}")

    # 3. 缩略图
    log.info("  [3/5] 缩略图...")
    thumb = None
    try:
        # 查找关键点文件用于智能选帧
        kp_file = None
        for root, dirs, files in os.walk(OUTPUT_BASE):
            for f in files:
                if f.endswith("_keypoints.json") and not f.endswith("_cropped_keypoints.json"):
                    kp_file = os.path.join(root, f)
                    break
            if kp_file:
                break
        thumb = generate_thumbnail(final, nickname, title, day=day,
                                   keypoints_file=kp_file, coach_name=coach_name)
    except Exception as e:
        log.warning(f"  缩略图失败: {e}")

    # 4. 智能 Shorts（高能段 + 教练居中 + 放大 + 美颜）
    log.info("  [4/5] Smart Shorts...")
    shorts = None
    try:
        from stages.shorts_maker import make_smart_shorts
        kp_file = os.path.join(OUTPUT_BASE, os.path.basename(final).split("_")[0] + "_keypoints.json")
        # find keypoints file in output subdirs
        for root, dirs, files in os.walk(OUTPUT_BASE):
            for f in files:
                if f.endswith("_keypoints.json"):
                    kp_file = os.path.join(root, f)
                    break
        shorts = make_smart_shorts(final, os.path.dirname(final), kp_file,
                                   beat_frames=None, duration=15, beauty=True)
    except Exception as e:
        log.warning(f"  Smart Shorts 失败: {e}, 回退普通Shorts")
        try:
            shorts = make_shorts_clip(final)
        except Exception:
            pass

    # 5. 上传
    log.info("  [5/5] 上传 YouTube...")
    desc = generate_description(coach, nickname, day)
    try:
        ytid = upload(final, title, desc, TAGS, thumb)
        log.info(f"  主视频: https://youtube.com/watch?v={ytid}")
    except Exception as e:
        log.error(f"  上传失败: {e}")
        return False

    if shorts and os.path.exists(shorts):
        try:
            upload(shorts, f"Day{day} 15秒暴汗燃脂 {nickname}领操 #Shorts",
                   "15秒暴汗挑战 完整版在频道", TAGS)
            log.info(f"  Shorts 已上传")
        except Exception as e:
            log.warning(f"  Shorts 上传失败: {e}")

    return True

# ── 批量扫描 ───────────────────────────────────────────
def scan_and_process():
    state = _load_json(STATE_FILE, {"processed": {}, "failed": {}})
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    videos = []
    for ext in ["*.mp4", "*.MP4", "*.mov", "*.MOV"]:
        videos.extend(glob.glob(os.path.join(SOURCE_DIR, ext)))

    new_videos = []
    for v in sorted(set(videos)):
        fid = os.path.basename(v)
        if fid in state["processed"]:
            continue
        if fid in state["failed"] and state["failed"][fid].get("retries", 0) >= 3:
            continue
        new_videos.append(v)

    if not new_videos:
        return

    log.info(f"发现 {len(new_videos)} 个新视频")
    for v in new_videos:
        fid = os.path.basename(v)
        # 等待文件稳定
        try:
            s1 = os.path.getsize(v)
            time.sleep(2)
            s2 = os.path.getsize(v)
            if s1 != s2 or s1 == 0:
                log.info(f"跳过 (文件未稳定): {fid}")
                continue
        except OSError:
            continue

        try:
            if process_one(v):
                state["processed"][fid] = datetime.now().strftime("%Y-%m-%d %H:%M")
                try:
                    os.rename(v, os.path.join(PROCESSED_DIR, fid))
                except OSError:
                    pass
        except Exception as e:
            log.error(f"异常 {fid}: {e}")
            state["failed"].setdefault(fid, {"retries": 0})
            state["failed"][fid]["retries"] += 1
        _save_json(STATE_FILE, state)

def watch_loop(interval=30):
    log.info(f"守护模式 — 监控 {SOURCE_DIR}，间隔 {interval}s")
    while True:
        try:
            scan_and_process()
        except Exception as e:
            log.error(f"扫描异常: {e}")
        time.sleep(interval)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="健身视频全自动发布器")
    p.add_argument("--watch", action="store_true", help="持续监控")
    p.add_argument("--loop", type=int, metavar="SEC", default=0, help="定时扫描间隔(秒)")
    args = p.parse_args()
    if args.watch:
        watch_loop()
    elif args.loop:
        watch_loop(args.loop)
    else:
        scan_and_process()
