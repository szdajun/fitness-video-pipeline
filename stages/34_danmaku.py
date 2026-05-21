"""阶段34: 弹幕效果

屏幕飘过健身激励文字，节拍时出现。
"""

import cv2, numpy as np, random, json
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import tempfile, subprocess, shutil, ctypes, os
from lib.utils import path_exists

PHRASES = [
    # 健康养生
    "出汗排毒!", "提高代谢!", "越跳越健康!", "气血通全身!",
    "每天30分远离医院!", "运动是最好的药!", "免疫力UP!",
    "肩颈不酸了!", "腰不疼了!", "活到99!", "心肺功能拉满!",
    # 身材美女
    "身材绝了!", "这腰我爱了!", "腿越来越细!", "蜜桃臀我来啦!",
    "马甲线指日可待!", "锁骨养鱼!", "直角肩!",
    "穿什么都好看!", "老公眼睛直了!", "姐妹身材太好了吧!",
    "A4腰不是梦!", "反手摸肚脐!", "漫画腿!", "天鹅颈!",
    "素颜也自信!", "皮肤越跳越好!", "比昨天瘦了!",
    "小蛮腰养成中!", "背影杀手!", "前凸后翘!", "线条美炸!",
    "翘臀yyds!", "瘦下来世界都温柔了~", "牛仔裤松了!",
    "闺蜜问我瘦了多少!", "被自己身材美醒!", "镜子里的谁啊这么好看!",
    "气质拿捏!", "女人味越来越浓!", "自信放光芒!",
    # 健身激励
    "加油!", "坚持!", "再来一组!", "燃烧吧!", "棒棒哒!",
    "你可以的!", "别放弃!", "冲!", "666!", "太强了!",
    "拼了!", "流汗就是燃脂!", "受不了也要撑住!", "今天也要卷!",
    # 互动趣味
    "后面的姐妹跟上!", "我妈问我为啥对屏幕傻笑~",
    "教练下次轻点!", "刚来，这是啥神仙直播间!",
    "推荐给闺蜜了!", "每天必打卡!", "已收藏!", "转发了!",
    "床说：你又要去跳了?", "脂肪在哭!", "卡路里杀手!",
    "笑死，动作跟不上!", "新手友好!", "跳完再来一杯奶茶(doge)",
    # 细柳营/逆风精神
    "天黑了下班了开练了!", "细柳营报到!", "汉军故地今练兵!",
    "996后来暴汗!", "苦中作乐!", "逆风飞扬!",
    "躺不平就动起来!", "病不起就练起来!", "乡党们冲!",
    "秦人血脉醒!", "细柳营打卡!", "练就完了!",
    "打工人的救赎!", "汗砸地声响!", "两千年前这里也练兵!",
    # 男教练专属
    "腰细了肾好了!", "老婆今晚有福了!", "男人不能说不行!",
    "越跳越猛!", "铁腰子练出来!", "这体力谁顶得住!",
    "大哥太猛了!", "真汉子!", "男人的快乐就这么简单!",
    "跳完回家老婆夸!", "精神小伙!", "老当益壮!",
    "腹肌出来了!", "人鱼线!", "美女盯着色眯眯!",
    "这身材谁不迷糊!", "老公腰力越来越猛!", "倒三角帅炸!",
    "不给隔壁老王机会!", "自己老婆自己宠!", "练好身体守住家!",
    "男人不练老王就练!", "这体格谁敢来挖墙脚!", "帅到老王自动退!",
    "场上多练10分钟 床上金枪永不倒!", "越练越硬气!",
    "美女与野兽共舞 激情和汗水飞溅!", "点赞加关注!",
    "拥抱青春气息!", "沉浸健康氛围!",
]


class DanmakuStage:
    def run(self, ctx):
        if ctx.get("danmaku_path") and path_exists(ctx.get("danmaku_path")):
            print("    已存在，跳过")
            return

        cfg = ctx.config.get("danmaku", {})
        if not cfg.get("enabled", False):
            return

        input_path = (ctx.get("pip_path") or
                     ctx.get("filmlook_path") or
                     ctx.get("speedramp_path") or
                     ctx.get("mascot_path") or
                     ctx.get("watermark_path") or
                     ctx.get("energybar_path") or
                     str(ctx.input_path))
        if not input_path or not path_exists(input_path):
            return

        video_info = ctx.get("video_info")
        fps = video_info["fps"]
        max_frames = video_info.get("process_frames", video_info["frames"])

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开: {input_path}")
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        beat_frames = set(ctx.get("beat_frames", []))

        # 中文字体
        font_paths = ["C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf"]
        font_size = cfg.get("font_size", int(orig_h * 0.05))
        pil_font = None
        for fp in font_paths:
            if os.path.exists(fp):
                pil_font = ImageFont.truetype(fp, font_size)
                break
        if not pil_font:
            pil_font = ImageFont.load_default()

        # 预生成弹幕队列: (文字, 出现帧, 起始y, 颜色)
        interval = cfg.get("interval", 30)  # 每隔N帧可能出现
        danmaku_list = []
        for f in range(0, max_frames, interval):
            if f in beat_frames or random.random() < 0.3:
                text = random.choice(PHRASES)
                y = random.randint(orig_h // 6, orig_h * 5 // 6)
                colors = [(255, 255, 100), (255, 150, 100), (100, 255, 200),
                          (255, 200, 255), (255, 255, 255)]
                color = random.choice(colors)
                danmaku_list.append((text, f, y, color))

        out_path = ctx.output_dir / f"{Path(input_path).stem}_danmaku.mp4"
        tmpdir = Path(tempfile.mkdtemp(prefix="dm_"))

        GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
        GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
        GetShortPathNameW.restype = ctypes.c_uint

        def to_short(p):
            buf_size = GetShortPathNameW(str(p), None, 0)
            if buf_size == 0:
                return str(p)
            buf = ctypes.create_unicode_buffer(buf_size)
            GetShortPathNameW(str(p), buf, buf_size)
            return buf.value

        tmpdir_short = to_short(str(tmpdir))

        print(f"    弹幕: {len(danmaku_list)}条, size={font_size}")

        cap = cv2.VideoCapture(input_path)
        frame_idx = 0

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # 查找当前帧活动的弹幕
            active = []
            for dm in danmaku_list:
                text, start_f, y, color = dm
                lifetime = orig_w / 3  # 3px/frame = ~3秒横跨
                age = frame_idx - start_f
                if 0 <= age < lifetime * 1.2:
                    # 从右到左移动
                    progress = age / lifetime
                    x = orig_w - int(progress * (orig_w + orig_w * 0.3)) + int(orig_w * 0.3)
                    alpha = 1.0
                    if progress < 0.1:
                        alpha = progress / 0.1
                    elif progress > 0.9:
                        alpha = (1.0 - progress) / 0.1
                    active.append((text, x, y, color, alpha))

            # 用 PIL 画弹幕
            if active:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay)

                for text, x, y, color, alpha in active:
                    bbox = pil_font.getbbox(text)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    # 阴影
                    draw.text((x+2, y+2), text, font=pil_font, fill=(0, 0, 0, int(100*alpha)))
                    # 正文
                    draw.text((x, y), text, font=pil_font, fill=(*color, int(255*alpha)))

                pil_img = Image.alpha_composite(pil_img.convert("RGBA"), overlay)
                frame = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

            cv2.imwrite(f"{tmpdir_short}/f_{frame_idx:06d}.png", frame)
            frame_idx += 1
            if frame_idx % 120 == 0:
                print(f"    进度: {frame_idx}/{max_frames}")

        cap.release()
        print(f"    写入完成: {frame_idx} 帧，调用 FFmpeg 编码...")

        ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
        cmd = [ffmpeg, "-y", "-framerate", str(fps),
               "-i", f"{tmpdir_short}/f_%06d.png",
               "-c:v", "libx264", "-preset", "fast", "-crf", "1",
               "-pix_fmt", "yuv420p", "-an", str(out_path)]
        r = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
        shutil.rmtree(tmpdir, ignore_errors=True)

        if r.returncode != 0:
            print(f"    FFmpeg 编码失败: {r.stderr[-200:]}")
            ctx.set("danmaku_path", input_path)
            return

        ctx.set("danmaku_path", str(out_path))
        print(f"    输出: {out_path.name}")
