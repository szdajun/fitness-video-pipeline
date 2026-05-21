"""阶段29: 吉祥物贴纸 (姿态驱动版 v2)

侧面视角卡通猫，头左尾右，四肢分明。
"""

import cv2, numpy as np, json, math
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import tempfile, subprocess, shutil, ctypes
from lib.utils import path_exists


def _angle(p1, p2):
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])


def _mid(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def draw_side_cat(size, keypoints, frame_idx, on_beat=False):
    """站立卡通猫：头在上，身体竖直，四肢向下，尾巴后翘"""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    def kp(i):
        return keypoints[i] if i < len(keypoints) else (0, 0, 0)

    def limb_ang(i1, i2):
        p1, p2 = kp(i1), kp(i2)
        if p1[2] > 0.3 and p2[2] > 0.3:
            return _angle((p1[0], p1[1]), (p2[0], p2[1]))
        return None

    # 角度
    larm_ang = limb_ang(5, 7)    # 左肩→左肘
    rarm_ang = limb_ang(6, 8)    # 右肩→右肘
    lleg_ang = limb_ang(11, 13)  # 左髋→左膝
    rleg_ang = limb_ang(12, 14)  # 右髋→右膝

    # 躯干倾斜（人肩→人髋）
    lsh, rsh = kp(5), kp(6)
    lhi, rhi = kp(11), kp(12)
    torso_lean = 0
    if all(p[2] > 0.3 for p in [lsh, rsh, lhi, rhi]):
        sc = _mid((lsh[0], lsh[1]), (rsh[0], rsh[1]))
        hc = _mid((lhi[0], lhi[1]), (rhi[0], rhi[1]))
        torso_lean = _angle(sc, hc) - math.pi / 2

    # 比例 — 身体竖直
    cx, cy = size // 2, size // 2
    body_width = size // 6
    body_height = size // 3
    head_r = size // 7
    limb_w = max(2, size // 20)
    arm_len = size // 4
    leg_len = size // 5

    # 身体倾斜和呼吸 — 留出头和耳朵的空间
    lean_x = int(math.sin(torso_lean) * body_height * 0.4)
    bob = int(on_beat) * size // 25
    body_cx = cx + lean_x
    body_top = cy - body_height + bob + head_r  # 下移留出头部空间

    # ===== 身体（竖直椭圆=站立）=====
    body_x0 = body_cx - body_width
    body_y0 = body_top
    body_x1 = body_cx + body_width
    body_y1 = body_top + body_height * 2
    draw.ellipse([body_x0, body_y0, body_x1, body_y1],
                 fill=(255, 170, 90, 255))

    # 肚皮（浅色）
    belly_y0 = body_y1 - body_height
    draw.ellipse([body_cx - body_width // 2, belly_y0,
                  body_cx + body_width // 2, body_y1],
                 fill=(255, 200, 140, 255))

    # ===== 头（身体上方）=====
    hx = body_cx
    hy = body_top - head_r
    # 头随躯干倾斜
    hx += lean_x
    draw.ellipse([hx - head_r, hy - head_r, hx + head_r, hy + head_r],
                 fill=(255, 170, 90, 255))

    # 耳朵
    draw.polygon([(hx - head_r + 4, hy - 2),
                  (hx - head_r//2, hy - head_r - 10),
                  (hx, hy - 2)],
                 fill=(255, 140, 70, 255))
    draw.polygon([(hx + head_r - 4, hy - 2),
                  (hx + head_r//2, hy - head_r - 10),
                  (hx, hy - 2)],
                 fill=(255, 140, 70, 255))

    # 眼睛（正面看）
    eye_y = hy - 3
    draw.ellipse([hx - head_r + 6, eye_y, hx - head_r + 14, eye_y + 6],
                 fill=(0, 0, 0, 255))
    draw.ellipse([hx + head_r - 14, eye_y, hx + head_r - 6, eye_y + 6],
                 fill=(0, 0, 0, 255))

    # 鼻子和嘴
    draw.ellipse([hx - 3, hy + 3, hx + 3, hy + 8], fill=(255, 90, 90, 255))
    draw.line([(hx, hy + 8), (hx, hy + 14)], fill=(0, 0, 0, 255), width=1)
    draw.arc([hx - 6, hy + 10, hx + 6, hy + 20], 0, 180,
             fill=(0, 0, 0, 255), width=1)

    # 胡须（鼻子两侧）
    whisker_y = hy + 6
    for dy in [-3, 0, 3]:
        draw.line([(hx - 6, whisker_y), (hx - head_r - 8, whisker_y + dy - 2)],
                  fill=(0, 0, 0, 80), width=1)
        draw.line([(hx + 6, whisker_y), (hx + head_r + 8, whisker_y + dy - 2)],
                  fill=(0, 0, 0, 80), width=1)

    if on_beat:
        draw.ellipse([hx - head_r + 8, eye_y + 1, hx - head_r + 11, eye_y + 4],
                     fill=(255, 255, 255, 220))
        draw.ellipse([hx + head_r - 11, eye_y + 1, hx + head_r - 8, eye_y + 4],
                     fill=(255, 255, 255, 220))

    # ===== 手臂（身体上方两侧，向下垂）=====
    shoulder_y = body_top + body_height // 3
    larm_origin = (body_x0 + 2, shoulder_y)
    rarm_origin = (body_x1 - 2, shoulder_y)

    def draw_limb(origin, angle, length, width, color):
        dx = int(math.cos(angle) * length)
        dy = int(math.sin(angle) * length)
        end = (origin[0] + dx, origin[1] + dy)
        draw.line([origin, end], fill=color, width=width)
        paw_r = width + 2
        draw.ellipse([end[0] - paw_r, end[1] - paw_r, end[0] + paw_r, end[1] + paw_r],
                     fill=color)

    # 手臂：上举摆动（模拟跳操挥手）
    la = larm_ang if larm_ang is not None else (math.pi * 0.1 - math.sin(frame_idx * 0.2) * 1.2)
    ra = rarm_ang if rarm_ang is not None else (math.pi * 0.9 + math.sin(frame_idx * 0.2) * 1.2)
    draw_limb(larm_origin, la, arm_len, limb_w, (255, 155, 75, 255))
    draw_limb(rarm_origin, ra, arm_len, limb_w, (255, 155, 75, 255))

    # ===== 腿（身体下方，向下站）=====
    hip_y = body_y1 - body_height // 3
    lleg_origin = (body_x0 + 3, hip_y)
    rleg_origin = (body_x1 - 3, hip_y)

    ll = lleg_ang if lleg_ang is not None else (math.pi * 0.5 + math.cos(frame_idx * 0.3) * 0.2)
    rl = rleg_ang if rleg_ang is not None else (math.pi * 0.5 + math.sin(frame_idx * 0.3) * 0.2)
    draw_limb(lleg_origin, ll, leg_len, limb_w + 1, (200, 120, 60, 255))
    draw_limb(rleg_origin, rl, leg_len, limb_w + 1, (200, 120, 60, 255))

    # ===== 尾巴（身体后方，翘起）=====
    tail_origin = (body_x1, body_y1 - body_height // 3)
    base_tail = math.pi * 0.1 + lean_x * 0.05
    sway = math.sin(frame_idx * 0.3) * 0.5
    tail_ang = base_tail + sway
    tail_pts = [tail_origin]
    tx, ty = tail_origin
    for i in range(3):
        tx += int(math.cos(tail_ang + i * 0.3) * arm_len * 0.5)
        ty += int(math.sin(tail_ang) * arm_len * 0.3 * (1 - i * 0.5))
        tail_pts.append((tx, ty))
    draw.line(tail_pts, fill=(255, 140, 70, 255), width=max(2, limb_w + 1),
              joint="curve")

    # ===== 运动光环 =====
    glow_cx, glow_cy = body_cx, (body_top + body_y1) // 2
    glow_r = max(body_width, body_height) + 8
    if on_beat:
        # 节拍时: 亮橙 → 金色渐变光环
        glow_colors = [(255, 200, 50), (255, 150, 30), (255, 100, 20)]
        glow_alpha = 180
    else:
        # 非节拍: 微弱蓝光环
        glow_colors = [(80, 160, 220)]
        glow_alpha = 60
    for gi, gc in enumerate(glow_colors):
        gr = glow_r - gi * 3
        draw.ellipse([glow_cx - gr, glow_cy - gr, glow_cx + gr, glow_cy + gr],
                     outline=(*gc, glow_alpha), width=max(1, size // 50))

    # ===== 节拍粒子爆发 =====
    if on_beat:
        particle_colors = [
            (255, 255, 100, 255), (255, 150, 100, 255), (255, 100, 200, 255),
            (100, 255, 200, 255), (100, 180, 255, 255), (255, 255, 255, 255),
        ]
        cx2, cy2 = body_cx, (body_top + body_y1) // 2
        n_particles = 8
        for p in range(n_particles):
            angle = math.pi * 2 * p / n_particles + frame_idx * 0.1
            dist = size * 0.2 + (hash((frame_idx * 100 + p) % 1000) % 20) * size * 0.015
            px = int(cx2 + math.cos(angle) * dist)
            py = int(cy2 + math.sin(angle) * dist)
            pr = max(2, size // 30 + p % 3)
            color = particle_colors[p % len(particle_colors)]
            draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=color)
            if p % 2 == 0:
                tx = int(px - math.cos(angle) * pr * 2)
                ty = int(py - math.sin(angle) * pr * 2)
                draw.line([(tx, ty), (px, py)], fill=color, width=max(1, pr - 1))

        # 表情回应 — 画彩色星形/火花
        emo_x = hx + head_r + 5
        emo_y = hy - head_r - 5
        r_star = size // 18
        star_colors = [(255, 200, 50, 255), (255, 100, 100, 255), (255, 255, 100, 255)]
        sc = star_colors[frame_idx % 3]
        # 画五角星
        pts = []
        for i in range(10):
            a = math.pi * i / 5 - math.pi / 2
            r = r_star if i % 2 == 0 else r_star // 2
            pts.append((int(emo_x + math.cos(a) * r), int(emo_y + math.sin(a) * r)))
        draw.polygon(pts, fill=sc)

    return img


class MascotStage:
    def run(self, ctx):
        if ctx.get("mascot_path") and path_exists(ctx.get("mascot_path")):
            print("    已存在，跳过")
            return

        cfg = ctx.config.get("mascot", {})
        if not cfg.get("enabled", False):
            return

        input_path = (ctx.get("watermark_path") or
                     ctx.get("energybar_path") or
                     ctx.get("beatflash_path") or
                     ctx.get("ghost_path") or
                     ctx.get("leadbox_path") or
                     str(ctx.input_path))
        if not input_path or not path_exists(input_path):
            print("    跳过: 无输入视频")
            return

        video_info = ctx.get("video_info")
        fps = video_info["fps"]
        max_frames = video_info.get("process_frames", video_info["frames"])

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {input_path}")
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        size = cfg.get("size", int(min(orig_w, orig_h) * 0.15))
        position = cfg.get("position", "bottom-left")

        # 读取关键点
        keypoints_cache = None
        cache_path = ctx.output_dir / f"{ctx.input_path.stem}_keypoints.json"
        if cache_path.exists():
            with open(cache_path) as f:
                data = json.load(f)
            keypoints_cache = data.get("keypoints", data)

        beat_frames = set(ctx.get("beat_frames", []))

        out_path = ctx.output_dir / f"{Path(input_path).stem}_mascot.mp4"
        tmpdir = Path(tempfile.mkdtemp(prefix="mc_"))

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
        print(f"    吉祥物(侧面): size={size}, pos={position}")

        cap = cv2.VideoCapture(input_path)
        frame_idx = 0

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            on_beat = any(abs(frame_idx - bf) <= 2 for bf in beat_frames)

            # 获取关键点
            kps = [(0, 0, 0)] * 17
            if keypoints_cache:
                entry = keypoints_cache.get(str(frame_idx))
                if entry and isinstance(entry, list) and len(entry) > 0:
                    best = max(entry, key=lambda p: sum(k[2] for k in p) if isinstance(p[0], list) else 0)
                    if isinstance(best[0], list):
                        kps = [(p[0], p[1], p[2]) for p in best[:17]]

            mascot_img = draw_side_cat(size, kps, frame_idx, on_beat)

            margin = 15
            bounce = int(size * 0.1) if on_beat else 0

            if position == "bottom-left":
                x, y = margin, orig_h - size - margin - bounce
            elif position == "bottom-right":
                x, y = orig_w - size - margin, orig_h - size - margin - bounce
            else:
                x, y = margin, orig_h - size - margin - bounce

            if 0 <= x and 0 <= y and x + size <= orig_w and y + size <= orig_h:
                mn = np.array(mascot_img)
                alpha = mn[:, :, 3] / 255.0
                for c in range(3):
                    roi = frame[y:y+size, x:x+size, c]
                    roi[:] = roi * (1 - alpha) + mn[:, :, c] * alpha
                    frame[y:y+size, x:x+size, c] = roi

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
            ctx.set("mascot_path", input_path)
            return

        ctx.set("mascot_path", str(out_path))
        print(f"    输出: {out_path.name}")
