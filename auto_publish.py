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

COACH_MAP = {
    "艳青": "胭脂虎", "艳玲": "俏玲珑", "丽丽": "腰女", "建玲": "三宝妈",
    "小红豆": "红娘子", "郭海军": "老兵不老", "枫林红": "霸道总裁",
    "李刚": "托塔天王", "小飞侠": "节拍战神",
}
CHANNEL       = "fitness"
PRIVACY       = "private"
TAGS          = [
    "细柳营胭脂虎", "有氧健身操", "打工族健身", "暴汗燃脂",
    "每日打卡", "30天挑战", "零基础健身", "华人健身",
    "996后的救赎", "苦中作乐",
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

# ── 教练检测 ───────────────────────────────────────────
def detect_coach(filename):
    for key, nick in COACH_MAP.items():
        if key in filename:
            return key, nick
    return None, None

# ── 日计数器 ───────────────────────────────────────────
def get_next_day(coach):
    coils = _load_json(DAY_COUNTER, {})
    day = coils.get(coach, 1)
    coils[coach] = day + 1
    _save_json(DAY_COUNTER, coils)
    return day

# ── SEO 文案（模板，无 AI）───────────────────────────────
def generate_title(coach, nickname, day):
    if not coach:
        return f"胭脂虎健身团 有氧健身操 暴汗燃脂"
    return random.choice([
        f"细柳营Day{day} | {nickname}有氧操 | 996后的暴汗救赎",
        f"打工族每日功课 Day{day} | {nickname}带操 | 苦中作乐暴汗燃脂",
        f"细柳营风雨无阻Day{day} | {nickname}领操 | 逆风飞扬燃脂操",
        f"Day{day}零基础有氧操 | {nickname}细柳营 | 躺不平病不起就练",
    ])

def generate_description(nickname, day):
    return f"""细柳营·胭脂虎 | {nickname}有氧操 | Day{day}

汉细柳营故地 打工族每日功课 天黑下班吃饱开练
996后的救赎 苦中作乐 逆风飞扬 躺不平就动起来

跟练打卡挑战：每天一条，30天见效果！今天你打卡了吗？

#细柳营胭脂虎 #{'#' + nickname if nickname else ''} #有氧健身操 #打工族健身 #暴汗燃脂 #每日打卡 #30天挑战"""

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

def generate_thumbnail(video_path, coach_nickname, title):
    """提取 2s 帧 + 文字生成 YouTube 缩略图"""
    import cv2, numpy as np
    from PIL import Image, ImageDraw, ImageFont

    thumb = video_path.replace(".mp4", "_thumb.jpg")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(3)), int(cap.get(4))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * 2))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None

    for fp in ["C:/Windows/Fonts/msyhbd.ttc", "C:/Windows/Fonts/simhei.ttf"]:
        if os.path.exists(fp):
            flg = ImageFont.truetype(fp, int(h * 0.09))
            break
    else:
        return None

    stitle = title.split("|")[0].strip() if "|" in title else title[:20]
    lines = [f"{coach_nickname}带操", stitle, "暴汗燃脂·每日打卡"]
    pi = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pi)
    ls = int(h * 0.04)
    lh = [d.textbbox((0, 0), ln, font=flg)[3] - d.textbbox((0, 0), ln, font=flg)[1] for ln in lines]
    tt = sum(lh) + ls * (len(lines) - 1)

    ov = Image.new("RGBA", pi.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    bt = h - tt - int(h * 0.12)
    od.rectangle([(0, bt - 20), (w, h)], fill=(0, 0, 0, 160))
    y = bt
    for i, ln in enumerate(lines):
        color = (255, 220, 50) if i == 0 else (255, 255, 255)
        tw = d.textbbox((0, 0), ln, font=flg)[2] - d.textbbox((0, 0), ln, font=flg)[0]
        cx = (w - tw) // 2
        od.text((cx + 1, y + 1), ln, font=flg, fill=(0, 0, 0, 80))
        od.text((cx, y), ln, font=flg, fill=color)
        y += lh[i] + ls

    Image.alpha_composite(pi.convert("RGBA"), ov).convert("RGB").save(thumb, "JPEG", quality=92)
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

    coach, nickname = detect_coach(fname)
    day = get_next_day(coach) if coach else 1
    title = generate_title(coach, nickname, day)
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
        thumb = generate_thumbnail(final, nickname, title)
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
    desc = generate_description(nickname, day)
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
