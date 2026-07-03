#!/usr/bin/env python3
"""网红健身视频 → 换背景 + 换脸 (独立工具, 主管线零改动).

把任意「人物占满画面、相机基本静止」的源视频, 抠像贴到新背景 + 换成指定教练脸.
2026-07-01 泛化: 不再绑定丽丽/时代广场, 默认值 + 预设系统沉淀多轮迭代经验, 可处理类似视频.

对症两个老大难 (多轮迭代定稿, 详见 docs/BG_SWAP.md):
  1. 抠像边缘差/凹谷漏色 → **RVM (RobustVideoMatting) 高精度 per-pixel alpha** (默认 --matte,
     治本: 两腿间/腋下干净分离背景); --no-matte 回退 YOLOv8-seg 粗分割 (+despill/punch 补丁).
  2. 换脸不对/换到路人 → 复用 face_swap.swap_face(only_lead=True, lead_bbox=pose), 只换
     pose 锁定的网红脸, 不碰背景路人.

处理流程 (每帧, 流式不载全片):
  原帧 → ① pose (缓存) 算 lead 脸 bbox
       → ② RVM 人体 alpha mask (默认) / YOLOv8-seg (降级)
       → ③ 换脸 (only_lead, 只换网红脸)
       → ④ 色温匹配 + light wrap (治贴纸感/色温断层)
       → ⑤ 接地感增强 (治脚地两层, --grounding, 默认关)
       → ⑥ 静态背景合成 (默认冻结单帧; --dynamic-bg 切动态仅静态机位用)
       → ffmpeg rawvideo pipe (h264_nvenc 优先 libx264 兜底, 含原音频)

用法:
  # 基础 (用预设最省心, fitness = 丽丽时代广场定稿配置)
  python tools/bg_swap.py \
      --video "C:\\...\\网红.mp4" \
      --bg    "C:\\...\\背景.mp4" \
      --coach 丽丽 \
      --output output/bgswap/网红_丽丽_时代广场.mp4 \
      --preset fitness

  # 未知视频先用 clean 基线确认抠像+换脸, 再逐项开增强
  python tools/bg_swap.py --video ... --bg ... --coach ... --output ... --preset clean

常用 flag (完整见 --help, 预设见 presets/bgswap_*.yaml):
  [--preset fitness|clean|dance]   预设默认值 (CLI 显式值仍胜预设)
  [--matte / --no-matte]           RVM 高精度抠像 (默认开) / 回退 seg
  [--color-match 0.8]              全身色温匹配保L (治贴纸感)
  [--light-wrap 0.5]               边缘光融合 (治硬切)
  [--grounding 0.18]               接地感增强 (治脚地两层; 默认关, fitness 预设开)
  [--parallax 0.02]                视差纵深 ±2% (治人变地不变)
  [--shadow-strength 0]            接地阴影 (默认关, 阴影治不了浮反凸显两层)
  [--bg-frame 1.65]                静态背景取哪秒 (默认中间帧; 暖粉脚下区挑冷帧)
  [--no-faceswap] [--debug-only]   只换背景 / 只出 mask 检查图
  [--ffmpeg PATH]                  覆盖 ffmpeg 路径 (默认: BG_FFMPEG env > PATH > 已知好路径)

依赖: torch (RVM) + ultralytics (YOLOv8-seg 降级) + insightface (buffalo_l) + ffmpeg
      GPU: 三模型同进程需 onnxruntime cudnn_conv_algo_search=HEURISTIC + gpu_mem_limit 4GB
      (见 docs/BG_SWAP.md 坑 7 / memory face-swap-cudnn-fix)
"""

import argparse
import gc
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
from scipy.ndimage import binary_fill_holes

# 项目根入 sys.path → 能 import face_swap / lib / stages / student_closeup
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import face_swap  # 复用 get_face_analyser / get_swapper / extract_face_embedding /
#                  swap_face / find_lead_person / get_lead_bbox_from_pose
import student_closeup  # 复用 detect_pose / _sharpen / render 的 ffmpeg pipe 范式

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

def _rss_mb():
    """进程 RSS (MB), 诊断累积内存用."""
    return psutil.Process().memory_info().rss / 1024 / 1024

# ffmpeg 解析顺序: --ffmpeg CLI > BG_FFMPEG env > shutil.which(PATH) > 已知好路径 fallback.
# **保留已知好路径兜底**: CLAUDE.md 记 Winget 版 ffmpeg 有编码兼容 bug 会生成损坏 mp4,
# 本机优先用它; 换机器时走 PATH 或设 BG_FFMPEG env, 不再裸硬编码 (泛化 2026-07-01).
_KNOWN_GOOD_FFMPEG = r"C:/Users/18091/ffmpeg/ffmpeg.exe"


def _resolve_ffmpeg(override=None):
    """返回可用的 ffmpeg 可执行路径.

    解析顺序: --ffmpeg CLI > BG_FFMPEG env > 已知好路径 > PATH > 兜底字符串.
    **已知好路径优先于 PATH**: 本项目 Winget 版 ffmpeg (8.1-full) 有编码兼容 bug 会
    生成损坏 mp4 (CLAUDE.md), 已知好路径 C:/Users/18091/ffmpeg/ffmpeg.exe (8.1-essentials)
    实测稳定, 故只要它存在就用它. 换机器 (此路径不存在) 自动落到 PATH, 保证可移植.
    """
    if override:
        return override
    env = os.environ.get("BG_FFMPEG")
    if env:
        return env
    if os.path.isfile(_KNOWN_GOOD_FFMPEG):
        return _KNOWN_GOOD_FFMPEG
    found = shutil.which("ffmpeg")
    if found:
        return found
    return _KNOWN_GOOD_FFMPEG


FFMPEG = _resolve_ffmpeg()


# ============================================================
# 0. 教练换脸源照片 (复用 stages/37 find_coach_face 的优先级链)
# ============================================================
def find_coach_face(coach_name, tools_dir):
    """gfpgan 优先级链: {coach}_face_gfpgan.png > _gfpgan.png > _face.png > ...
    优先复用 stages/37_face_swap.find_coach_face (模块名 37_ 数字开头, 用 importlib 加载);
    加载失败则内联兜底 (优先级链一致)."""
    try:
        import importlib.util
        root = Path(__file__).resolve().parent.parent
        spec = importlib.util.spec_from_file_location(
            "_stage37_faceswap", str(root / "stages" / "37_face_swap.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        p = mod.find_coach_face(coach_name, tools_dir)
        if p:
            return p
    except Exception as e:
        print(f"  [find_coach_face] stages/37 加载失败({e}), 内联查找")
    if not coach_name:
        return None
    # alias 是历史 pinyin 桥 (旧教练脸文件用拼音名如 lili_gfpgan.png). **加新教练无需扩此 dict**:
    # priority chain 先试 coach_name 本名, 故 tools/{coach}_face_gfpgan.png 用中文名直接命中.
    alias = {"艳青": "yanqing", "丽丽": "lili", "建玲": "jianling",
             "小红豆": "xhd", "枫林红": "flh", "郭海军": "haijun"}.get(coach_name)
    names = [coach_name] + ([alias] if alias else [])
    for suf in ["_face_gfpgan.png", "_gfpgan.png", "_face.png", "_face.jpg",
                ".png", ".jpg", ".bmp"]:
        for nm in names:
            p = os.path.join(tools_dir, f"{nm}{suf}")
            if os.path.exists(p):
                return p
    return None


# ============================================================
# 1. Pose (复用 student_closeup.detect_pose, 独立缓存)
# ============================================================
def detect_pose(video_path, cache_path):
    """返回 (keypoints={fi: [[person_blaze33]*N]}, info={fps,width,height,frames})."""
    return student_closeup.detect_pose(video_path, cache_path)


# ============================================================
# 2. YOLOv8-seg 人体分割 + mask 精修 (对症"抠像边缘差")
# ============================================================
def load_seg_model(device="cuda:0", model_name="yolov8m-seg"):
    from ultralytics import YOLO
    model = YOLO(model_name)
    model.to(device)
    return model


def segment_person(model, frame, lead_bbox=None, conf=0.3):
    """YOLOv8-seg 取 person(class 0) instance mask (原图尺寸, 多边形填充).
    lead_bbox 给定时: 选包含 lead 脸中心的 instance (=网红); 否则选最大面积.
    返回 float mask (h×w, 0/1) 或 None(漏检)."""
    res = model(frame, verbose=False, conf=conf, classes=[0])[0]
    if res.masks is None:
        return None
    polys = getattr(res.masks, "xy", None)
    if not polys:
        return None
    h, w = frame.shape[:2]
    inst = []
    for poly in polys:
        if poly is None or len(poly) < 3:
            continue
        m = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1.0)
        inst.append(m)
    if not inst:
        return None
    if lead_bbox is not None:
        cx = int((lead_bbox[0] + lead_bbox[2]) / 2)
        cy = int((lead_bbox[1] + lead_bbox[3]) / 2)
        cx = max(0, min(w - 1, cx))
        cy = max(0, min(h - 1, cy))
        for m in inst:
            if m[cy, cx] > 0.5:
                return m
    return max(inst, key=lambda m: float(m.sum()))


def refine_mask(mask, feather=11, erode=4):
    """mask 精修: 二值化 → 闭运算填孔 → erode 收缩脏边缘(消除羽化区透色/光晕) → 高斯羽化.
    feather 默认 11 (窄化透色带), erode 默认 4 (收缩含原背景色的脏边缘像素)."""
    if mask is None:
        return None
    m = (mask > 0.5).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
    if erode > 0:  # 收缩边缘: 羽化半透明带往内移, 不让原边缘像素与背景混合透色
        ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (erode * 2 + 1, erode * 2 + 1))
        m = cv2.erode(m, ke, iterations=1)
    mf = m.astype(np.float32) / 255.0
    fk = int(feather)
    fk = max(3, fk if fk % 2 == 1 else fk + 1)
    mf = cv2.GaussianBlur(mf, (fk, fk), 0)
    return np.clip(mf, 0.0, 1.0)


# ============================================================
# 2.4b 打孔: 去分割 mask 误纳入的【原图背景】(粉地面凹谷 / 过分割边缘)
# 解决"两腿中间粉红": 原 bg 整体粉/鲑鱼色, mask 在凹谷(两腿间/腋下)和边缘误纳入
# 这块粉色背景 → 合成后硬核区保留原图粉色 (despill 只动软带, 治不了硬核).
# ============================================================
def captured_bg_mask(mask_raw, frame, rg_thresh=20, sat_thresh=40,
                     min_area=0, protect_bbox=None):
    """检测 mask 内【原图强粉背景】像素 = R-G>rg_thresh 且 饱和度>sat_thresh.
    ⚠️ 真正的判别量是 **R-G**(原图粉地面 R-G~36 vs 皮肤~12, 间隙大), 不是 S —— 实测
    这片粉地面 S 多在 50-65, 旧 sat>80 只覆盖最饱和的核(1.8-5.7%), 留下大片 S~55 粉
    原样保留 = 用户看到的"两腿间还是粉, 只有空洞漏灰". 故 sat 降到 40(只要 R-G>20 就判粉).
    两道保护防误打孔:
      - protect_bbox (x0,y0,x1,y1) 脸区(已 expand)内不打孔 —— 防"脸花了"
        (脸的嘴唇/腮红/暖高光 R-G 可 >20, 换脸后脸透出 bg). 主力保护 (与 sat/min_area 无关).
      - min_area 去小连通块 (默认 0=关).
    返回 bool 打孔区."""
    if frame is None or mask_raw is None:
        return None
    f = frame.astype(np.int16)
    rg = f[..., 2] - f[..., 1]                       # BGR: R-G
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cap = (mask_raw > 0.5) & (rg > rg_thresh) & (hsv[..., 1] > sat_thresh)
    if protect_bbox is not None:                     # 脸区保护
        x0, y0, x1, y1 = protect_bbox
        cap[max(0, y0):y1, max(0, x0):x1] = False
    if cap.any() and min_area > 0:                   # 去小连通块 (默认关)
        n, labels = cv2.connectedComponents(cap.astype(np.uint8), connectivity=8)
        if n > 1:
            sizes = np.bincount(labels.ravel())
            keep = sizes >= min_area
            keep[0] = False                          # 背景 label0 不算
            cap = cap & keep[labels]
    return cap


def build_mask(mask_raw, frame, feather=11, erode=4, punch=True,
               rg_thresh=20, sat_thresh=40, guard=7, protect_bbox=None, min_area=0):
    """完整 mask 构建 (替代裸 refine_mask):
      ① pre-punch  打掉大凹谷(两腿间)的原图粉背景 (脸区 protect_bbox 保护, 去小 speck)
      ② refine_mask  close 填小孔 / erode / feather
      ③ post-punch  去掉 ②里 CLOSE 重填的粉 speck + feather 越过粉背景的软带
                    (dilate guard px 护边, 不让 feather 把粉背景再混进来)
      ④ 软化打孔边界
    实测: 单 pre-punch 残留漏色 R-G +18~20 (CLOSE 重填 + feather 粉带);
    加 post-punch+护边 → 漏色 0. punch=False 退回裸 refine_mask."""
    if mask_raw is None:
        return None
    if not punch or frame is None:
        return refine_mask(mask_raw, feather, erode)
    cap = captured_bg_mask(mask_raw, frame, rg_thresh, sat_thresh,
                           min_area=min_area, protect_bbox=protect_bbox)
    m = (mask_raw > 0.5).astype(np.uint8).copy()
    if cap is not None and cap.any():
        m[cap] = 0                                    # ① pre-punch
    mf = refine_mask(m.astype(np.float32), feather, erode)   # ②
    if cap is not None and cap.any():
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (guard, guard))
        cap_dil = cv2.dilate(cap.astype(np.uint8), k, iterations=1).astype(bool)
        mf = mf * (~cap_dil).astype(np.float32)      # ③ post-punch + 护边
        fk = feather if feather % 2 == 1 else feather + 1
        mf = cv2.GaussianBlur(mf, (fk, fk), 0)       # ④ 软化
        mf = np.clip(mf, 0.0, 1.0)
    return mf


# ============================================================
# 2-RVM. Robust Video Matting 高精度抠像 (治本: 真正 per-pixel alpha)
# ------------------------------------------------------------
# 替代上面 YOLOv8-seg 粗分割. 痛根: seg 是二值粗掩码, 凹谷(两腿间/腋下/指缝)
# 误纳入原图粉红地面 → 合成后漏色; despill/punch/protect 一路打补丁仍残留.
# RVM 输出 per-pixel alpha (0-1 float), 凹谷干净分离背景, 根治漏色.
# 逐帧流式 (recurrent state rec, 常量显存, 任意长视频). 探针实测每帧清掉 seg
# 误纳的粉背景 44251~49832px. 默认开 (--matte), --no-matte 回退 seg.
# torch 2.6 坑: F.interpolate scale_factor 不再接 Tensor → downsample_ratio 用 float.
# ============================================================
def load_matte_model(device="cuda:0"):
    """加载 RVM mobilenetv3. GPU 有则 cuda+half (RTX 4070 ~150fps @1080×1920 dsr=0.25),
    否则 CPU float. torch.hub.load 首次下载, 后续走缓存."""
    import torch
    model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3")
    if torch.cuda.is_available():
        model = model.cuda().half()
    else:
        model = model.cpu().float()
    model.eval()
    return model


class MatteStream:
    """RVM 逐帧 alpha 流. rec=4 个 recurrent state 跨帧连续 → 时序稳定.
    顺序读帧连续调 alpha(); seek/跳帧后 reset() 清状态 (debug_sheet 抽帧用)."""
    def __init__(self, model, downsample_ratio=0.25):
        import torch
        self._torch = torch
        self.model = model
        self.dsr = float(downsample_ratio)   # torch 2.6: scale_factor 只接 float
        self.rec = [None] * 4
        param = next(model.parameters())
        self.device = param.device
        self.half = (param.dtype == torch.float16)

    def reset(self):
        self.rec = [None] * 4

    def alpha(self, bgr):
        """BGR uint8 (h,w,3) → alpha float32 (h,w) 0-1."""
        torch = self._torch
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        src = (torch.from_numpy(rgb).permute(2, 0, 1).float()
               .div(255.0).unsqueeze(0).to(self.device))
        if self.half:
            src = src.half()
        with torch.no_grad():
            fgr, pha, *rest = self.model(src, *self.rec, self.dsr)
            self.rec = rest
        return pha[0, 0].float().cpu().numpy()


# ============================================================
# 2.5 抠像去溢色 despill (原网红背景红砖暖橙 → 换冷灰背景后边缘红色光晕)
# ============================================================
def despill_frame(frame, mask, strength=1.0):
    """mask 边缘带去溢色 (defringe/despill): 羽化带采样到原图【粉红地面/红砖】
    暖色, 透到冷灰时代广场后成红色光晕. 检测 R 高出 max(G,B) 的暖红溢出, 在
    边缘带压 R 通道到中性. strength 0=关 1=完全压平; 硬核区(α≈1)不动 → 脸/身体不褪色.
    实测: strength=0.6 羽化带残留合成后 R-G≈+8.8 (用户仍见红晕); =1.0 压到中性.
    根因由用户确认: 原图地面是粉红色, 非影子 (源视频无影)."""
    if strength <= 0:
        return frame
    f = frame.astype(np.float32)
    r = f[..., 2]; g = f[..., 1]; b = f[..., 0]  # BGR
    spill = np.clip(r - np.maximum(g, b), 0.0, None)  # 暖红溢出量 (粉红/红砖)
    if float(spill.max()) <= 1.0:
        return frame
    # 整条羽化带 (含外缘采到的原粉红地面), 不是只取半透明中段; 9x9 膨胀覆盖整条脏区
    band = ((mask > 0.03) & (mask < 0.995)).astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    band = cv2.dilate(band, k)
    factor = np.clip(strength * band, 0.0, 1.0)  # (h,w)
    f[..., 2] = np.clip(r - spill * factor, 0, 255)  # 压 R 通道去暖溢
    return f.astype(np.uint8)


def despill_to_bg(out, mask, bg, strength=1.0):
    """合成后把【过渡带】暖溢色拉向【同位置背景】色温 (非压到中性).
    旧 despill_frame 压 R→max(G,B)=中性, 但背景本身 R-G≈+1.5 微暖 → 过校成青边/
    欠校成粉边, 帧间漂移 (analyze 实测 t8 暖冷交替 fringe / t2 残淡粉). 改成边缘
    R-G → 背景 R-G (per-pixel), 边缘天然匹配背景无色差, 不过校不残留. 只降 R (永不增),
    保人物亮度/形状. 根因由用户确认: 原图地面粉红 (非影子), 透到冷灰背景成暖边."""
    if strength <= 0 or bg is None:
        return out
    f = out.astype(np.float32)
    fb = bg.astype(np.float32)
    r, g = f[..., 2], f[..., 1]                        # BGR
    br, bgc = fb[..., 2], fb[..., 1]
    excess = np.clip((r - g) - (br - bgc), 0.0, None)  # 边缘比背景暖多少
    if float(excess.max()) <= 1.0:
        return out
    band = ((mask > 0.05) & (mask < 0.92)).astype(np.float32)  # 外过渡带, 不动硬核区
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    band = cv2.dilate(band, k)
    factor = np.clip(strength * band, 0.0, 1.0)
    f[..., 2] = np.clip(r - excess * factor, 0, 255)   # 降 R 拉向背景色温
    return f.astype(np.uint8)


# ============================================================
# 3. 背景预处理 (运镜背景 → 静态单帧 + 运镜跟随; 静态机位 → 动态对齐视频)
# ============================================================
# estimate_camera_motion 背景特征点采样区 (假设健身居中构图: 顶部建筑带 + 底部地面左右,
# 避开中间人体). --follow-cam 默认关, 此几何罕调, 故不暴露 CLI (泛化 2026-07-01).
CAM_MASK_TOP_PCT = 0.18         # 顶部建筑带高度占比
CAM_MASK_BOT_PCT = 0.90         # 底部地面带起始行占比
CAM_MASK_BOT_SIDE_PCT = 0.30    # 底部左右两侧地面宽度占比 (右侧起点 = 1 - 此值)


def estimate_camera_motion(video, total, smooth=5, track_win=40):
    """特征点光流(LK + forward-backward check)在背景mask区估相机运动轨迹 (traj_x, traj_y).
    背景 mask = 顶部建筑带 + 底部地面左右 (避开中间人体) → 只追真实背景特征点, 人体动作不污染.
    traj[t] = 第 t 帧相对锚帧的背景位移 = 相机运动 (手持抖/缓慢平移).
    **为何要它**: 原视频人体带相机抖动被 RVM 抠出, 静态背景不跟 → 人体晃背景死 = 滑动/漂浮感
    (用户: "原视频地面背景都是动的"). oversize 背景按 traj 逐帧裁切跟随 → 人体与背景来自同一
    相机运动 → 天然同步 + 保留手持动感. track_win 帧重锚防长追踪漂移/特征点丢失.
    (旧 phaseCorrelate 整图版把人体动作当运镜→乱裁脚滑; 特征点+mask+FB check 根治.)
    网红广场视频实测: 相机以垂直手持抖动为主 (swing_y~15px), 水平基本静止 (swing_x~5px).
    smooth=5 只去 LK 估计噪声, 保留手持抖动感 (smooth 过大会抹平→背景死, =foot-track 老问题).
    返回 (traj_x, traj_y, swing_x, swing_y) — swing=单边峰值, 定 oversize margin."""
    cap = cv2.VideoCapture(str(video))
    ok, f0 = cap.read()
    if not ok:
        cap.release()
        return (np.zeros(total, dtype=np.float32), np.zeros(total, dtype=np.float32), 0, 0)
    h, w = f0.shape[:2]
    anchor_gray = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
    mask = np.zeros((h, w), np.uint8)            # 背景区: 顶部建筑 + 底部地面左右 (避人体)
    mask[:int(h * CAM_MASK_TOP_PCT), :] = 255
    mask[int(h * CAM_MASK_BOT_PCT):, :int(w * CAM_MASK_BOT_SIDE_PCT)] = 255
    mask[int(h * CAM_MASK_BOT_PCT):, int(w * (1 - CAM_MASK_BOT_SIDE_PCT)):] = 255
    anchor_pts = cv2.goodFeaturesToTrack(anchor_gray, mask=mask, maxCorners=300,
                                         qualityLevel=0.08, minDistance=10)
    traj_x = np.zeros(total, dtype=np.float32)
    traj_y = np.zeros(total, dtype=np.float32)
    fi = 1; anchor_i = 0
    while fi < total:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        if anchor_pts is not None and len(anchor_pts) >= 10:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(anchor_gray, g, anchor_pts, None)
            back, stb, _ = cv2.calcOpticalFlowPyrLK(g, anchor_gray, nxt, None)
            fb = np.abs(back - anchor_pts).reshape(-1, 2).sum(1)
            good = (st.reshape(-1) == 1) & (stb.reshape(-1) == 1) & (fb < 1.0)
            if int(good.sum()) > 10:
                disp = (nxt[good] - anchor_pts[good]).reshape(-1, 2)
                traj_x[fi] = traj_x[anchor_i] + float(np.median(disp[:, 0]))
                traj_y[fi] = traj_y[anchor_i] + float(np.median(disp[:, 1]))
            else:
                traj_x[fi] = traj_x[fi - 1]; traj_y[fi] = traj_y[fi - 1]
        else:
            traj_x[fi] = traj_x[fi - 1]; traj_y[fi] = traj_y[fi - 1]
        if (fi - anchor_i) >= track_win:         # 重锚: 防长追踪漂移 + 补失效特征点
            anchor_i = fi; anchor_gray = g
            anchor_pts = cv2.goodFeaturesToTrack(g, mask=mask, maxCorners=300,
                                                 qualityLevel=0.08, minDistance=10)
        fi += 1
    cap.release()
    fill = float(traj_x[fi - 1]) if fi > 0 else 0.0
    traj_x[fi:] = fill                           # 读流早断尾巴补齐
    traj_y[fi:] = fill
    if smooth and smooth > 1 and total > smooth:
        k = np.ones(smooth, dtype=np.float32) / smooth
        traj_x = np.convolve(traj_x, k, mode="same").astype(np.float32)
        traj_y = np.convolve(traj_y, k, mode="same").astype(np.float32)
    swing_x = max(abs(float(traj_x.min())), abs(float(traj_x.max())))
    swing_y = max(abs(float(traj_y.min())), abs(float(traj_y.max())))
    print(f"  [运镜-LK] traj_x [{traj_x.min():+.1f},{traj_x.max():+.1f}] swing={swing_x:.1f}px; "
          f"traj_y [{traj_y.min():+.1f},{traj_y.max():+.1f}] swing={swing_y:.1f}px "
          f"→ {'背景将跟随相机运动' if swing_x > 6 or swing_y > 6 else '基本静止, 无需跟随'})")
    return traj_x, traj_y, int(swing_x) + 1, int(swing_y) + 1


def compute_sync_track(pose, total, w, h, cam_x, cam_y, foot_smooth=31, foot_frac=1.0):
    """脚钉砖缝 + 手持感: traj = 脚低频(重心转移) + cam高频(手持抖动).
    **治脚-砖缝低频漂移滑**: 实测脚水平低频 42px(重心转移) vs cam_x 仅 5.8px → 纯 camfollow
    残差 40px = 用户报"脚和砖缝相对运动". 背景跟脚低频(脚钉)消除该残差; cam 高频保留手持晃动感
    (用户认可 camfollow "晃动一致"); 脚高频踩踏(8px)不跟(否则砖缝逐帧抖) = 真实步态踩不同砖.
    远景建筑随脚低频漂(运镜跟随主体感, 低频缓慢不显假). foot_frac<1 克制(残差换自然).
    返回 (traj_x, traj_y, swing_x, swing_y)."""
    fx = np.full(total, np.nan); fy = np.full(total, np.nan)
    for fi in range(total):
        ps = pose.get(fi, [])
        if not ps:
            continue
        best = None; bn = -1
        for p in ps:
            pt = np.asarray(p, float); v = pt[pt[:, 2] > 0.3]
            if len(v) > bn:
                bn = len(v); best = pt
        if best is None or bn < 6:
            continue
        foot = [best[idx] for idx in [15, 16, 17, 18, 19, 20]
                if idx < len(best) and best[idx][2] > 0.3]    # ankle/heel/toe 中点
        if foot:
            foot = np.array(foot)
            fx[fi] = foot[:, 0].mean() * w; fy[fi] = foot[:, 1].mean() * h
    g = np.where(~np.isnan(fx))[0]
    if len(g) < 2:
        return cam_x.copy(), cam_y.copy(), int(max(abs(cam_x).max(), abs(cam_x).min())) + 1, \
               int(max(abs(cam_y).max(), abs(cam_y).min())) + 1
    fx = np.interp(np.arange(total), g, fx[g])
    fy = np.interp(np.arange(total), g, fy[g])

    def cwin(a, win):
        k = np.ones(win) / win; return np.convolve(a, k, 'same')
    foot_full_x = (fx - fx[0]) * foot_frac             # 水平完整跟脚(含踩踏/迈步高频)=脚钉死砖缝, 残差0
    foot_low_y = cwin(fy - fy[0], foot_smooth) * foot_frac   # 垂直只跟低频(跳蹲高频跟了建筑抖)
    cam_high_x = cam_x - cwin(cam_x, foot_smooth)
    cam_high_y = cam_y - cwin(cam_y, foot_smooth)
    traj_x = (foot_full_x + cam_high_x).astype(np.float32)
    traj_y = (foot_low_y + cam_high_y).astype(np.float32)
    swing_x = max(abs(float(traj_x.min())), abs(float(traj_x.max())))
    swing_y = max(abs(float(traj_y.min())), abs(float(traj_y.max())))
    fxsw = foot_full_x.max() - foot_full_x.min()
    print(f"  [脚钉+手持] traj_x swing={swing_x:.1f}px (脚完整水平{fxsw:.1f}+cam高频); "
          f"traj_y swing={swing_y:.1f}px → 水平脚钉死砖缝(残差0)+垂直低频+手持晃动")
    return traj_x, traj_y, int(swing_x) + 1, int(swing_y) + 1


def _cover_resize(img, tw, th, interp=cv2.INTER_LANCZOS4, crop_y_ratio=0.5):
    """等比缩放覆盖 (tw,th) + 按竖向偏移裁切 (不变形). 背景与输出宽高比不一致时
    (如竖屏 720x1280 背景塞横屏 1280x720 输出) 避免强制 resize 拉宽变形 —
    用户报'小推车拉宽变形厉害'即旧 cv2.resize((W,H)) 强拉伸所致 (2026-07-01).
    crop_y_ratio: 竖向裁切位置 0=顶 0.5=中心(默认) 1=底. 竖屏背景塞横屏且上部有
    遮挡物(天棚/建筑)时调高 (如 0.61) 让裁切窗下移避开 → 胳膊举起落在干净地面背景,
    RVM 抠胳膊准 (不消失/不虚化), 天棚降到画面外."""
    h, w = img.shape[:2]
    if w == 0 or h == 0:
        return img
    s = max(tw / w, th / h)                       # 等比放大到刚好覆盖 target
    nw = max(1, int(round(w * s)))
    nh = max(1, int(round(h * s)))
    img = cv2.resize(img, (nw, nh), interpolation=interp)
    ox = max(0, (nw - tw) // 2)                   # 横向中心裁切
    oy_avail = max(0, nh - th)
    oy = max(0, min(oy_avail, int(round(oy_avail * crop_y_ratio))))
    return img[oy:oy + th, ox:ox + tw]


def prepare_bg(bg_path, out_w, out_h, fps, tmp_dir, sharpen_bg=True,
               static=False, bg_frame_sec=None, margin_x=0, margin_y=0,
               bg_crop_y=0.5):
    """背景预处理 → 返回路径 (png=静态单帧 / mp4=动态对齐视频).
    static=True: 抽单帧 (bg_frame_sec 指定秒, 默认中间帧构图最饱满) → resize+锐化 → png.
                 **运镜背景必须用静态**: 动态背景逐帧推进, 静态抠像人物贴固定位置会相对
                 地面滑动 (phaseCorrelate 实测时代广场背景 25s 累计右移 1834px, dy≈0).
    static=False: 原逻辑, 整段视频 resize+重采样对齐成 mp4 逐帧推进 (仅静态机位背景用)."""
    if static:
        cap = cv2.VideoCapture(str(bg_path))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        bg_fps = cap.get(cv2.CAP_PROP_FPS) or fps
        target = (int(bg_frame_sec * bg_fps) if bg_frame_sec is not None
                  else n // 2)
        target = max(0, min(max(0, n - 1), target))
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, bg_img = cap.read()
        cap.release()
        if not ok or bg_img is None:
            print(f"  [背景][FAIL] 抽帧失败 f={target}")
            return None
        W_big = out_w + 2 * margin_x        # 运镜跟随: 四周留 margin 供逐帧裁切位移
        H_big = out_h + 2 * margin_y
        bg_img = _cover_resize(bg_img, W_big, H_big,
                               crop_y_ratio=bg_crop_y)  # 等比覆盖+竖向偏移裁切(避天棚)
        if sharpen_bg:  # 上采样(720→1080)补清晰度: unsharp
            blur = cv2.GaussianBlur(bg_img, (0, 0), 2.0)
            bg_img = cv2.addWeighted(bg_img, 1.5, blur, -0.5, 0)
        tag = f"{bg_frame_sec:g}s" if bg_frame_sec is not None else "mid"
        if margin_x > 0 or margin_y > 0:
            png = tmp_dir / f"bg_oversize_{W_big}x{H_big}_{tag}_cy{bg_crop_y:.2f}.png"
            note = f"oversize {W_big}x{H_big} (margin {margin_x}x{margin_y}px), 运镜跟随逐帧裁切"
        else:
            png = tmp_dir / f"bg_static_{out_w}x{out_h}_{tag}_cy{bg_crop_y:.2f}.png"
            note = "静态单帧, 冻结"
        cv2.imwrite(str(png), bg_img)
        print(f"  [背景] 静态单帧 f={target} (t={target / bg_fps:.1f}s) → {png.name} [{note}]")
        return png
    aligned = tmp_dir / f"bg_aligned_{out_w}x{out_h}_{int(round(fps))}fps.mp4"
    if not aligned.exists():
        # force_original_aspect_ratio=increase 等比放大覆盖 + crop 竖向偏移 = 不变形 cover
        vf = (f"scale={out_w}:{out_h}:force_original_aspect_ratio=increase"
              f":flags=lanczos,crop={out_w}:{out_h}:0:'(ih-{out_h})*{bg_crop_y:.3f}'")
        if sharpen_bg:
            vf += ",unsharp=5:5:0.6:5:5:0.0"
        vf += ",format=yuv420p"
        cmd = [FFMPEG, "-y", "-i", str(bg_path),
               "-vf", vf, "-r", f"{fps:.3f}", "-an",
               "-c:v", "libx264", "-preset", "fast", "-crf", "16",
               "-pix_fmt", "yuv420p", str(aligned)]
        print(f"  [背景] 预处理对齐 {out_w}x{out_h}@{fps:.2f}fps "
              f"(锐化={'on' if sharpen_bg else 'off'}) → {aligned.name} "
              f"[动态背景, 仅静态机位用]")
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0 or not aligned.exists():
            print(f"  [背景][FAIL] ffmpeg 预处理失败: "
                  f"{r.stderr.decode('utf-8','replace')[-500:]}")
            return None
    else:
        print(f"  [背景] 命中缓存: {aligned.name}")
    return aligned


# ============================================================
# 4. 换脸 (复用 face_swap.swap_face only_lead + pose lead_bbox)
# ============================================================
def _detect_lead_face_bbox(app, frame, pose_bbox, w, h):
    """insightface 检测 lead 脸的【紧 bbox】(int [x0,y0,x1,y1]) 给 punch 脸区保护用.
    ROI 用 pose_bbox(若有), 否则全图; 选最接近 ROI 中心的脸(同 swap_face lead 选法).
    比 pose lead bbox 紧得多(只罩脸, 不含肩/头周围背景). 检测不到返回 None."""
    if pose_bbox is not None:
        x1, y1, x2, y2 = pose_bbox
        x1 = max(0, min(w, x1)); y1 = max(0, min(h, y1))
        x2 = max(0, min(w, x2)); y2 = max(0, min(h, y2))
        if x2 - x1 < 20 or y2 - y1 < 20:
            pose_bbox = None
    # ROI 太小时 insightface 漏检 → 上采样到 512 (同 swap_face 策略)
    roi = frame
    rh, rw = frame.shape[:2]
    scaled = False
    sx = sy = 0
    if pose_bbox is not None:
        roi0 = frame[y1:y2, x1:x2].copy()
        rh0, rw0 = roi0.shape[:2]
        sx, sy = x1, y1
        if max(rh0, rw0) < 512:
            roi0 = cv2.resize(roi0, (512, 512), interpolation=cv2.INTER_CUBIC)
            scaled = True
        roi = roi0
    try:
        faces = app.get(roi)
    except Exception:
        return None
    if not faces:
        return None
    if pose_bbox is not None:
        rh, rw = roi.shape[:2]
        roi_cx, roi_cy = rw / 2.0, rh / 2.0
        def lead_score(f):
            b = f.bbox
            cx = (b[0] + b[2]) / 2.0; cy = (b[1] + b[3]) / 2.0
            dist = ((cx - roi_cx) ** 2 + (cy - roi_cy) ** 2) ** 0.5
            area = (b[2] - b[0]) * (b[3] - b[1])
            return (dist, -area, -f.det_score)
        best = min(faces, key=lead_score)
    else:
        best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    bx0, by0, bx1, by1 = best.bbox
    if scaled:
        # 反算回原图坐标
        bx0 = bx0 / 512 * rw0; bx1 = bx1 / 512 * rw0
        by0 = by0 / 512 * rh0; by1 = by1 / 512 * rh0
        rh, rw = rh0, rw0
    elif pose_bbox is not None:
        rh, rw = frame[y1:y2, x1:x2].shape[:2]
    bx0 = int(bx0 + sx); by0 = int(by0 + sy)
    bx1 = int(bx1 + sx); by1 = int(by1 + sy)
    bx0 = max(0, min(w, bx0)); bx1 = max(0, min(w, bx1))
    by0 = max(0, min(h, by0)); by1 = max(0, min(h, by1))
    if bx1 - bx0 < 10 or by1 - by0 < 10:
        return None
    return (bx0, by0, bx1, by1)


def swap_lead_face(swapper, src_face, app, frame, persons, w, h, swap_all=False):
    """对单帧换网红脸. 返回 (frame_swapped, swapped_bool, bbox, orient).
    swap_all=True: 全图检测, 把丽丽脸套到每个检测到的人 (多人场景, 如 3 人复制人构图).
        lead_bbox=None + only_lead=False → face_swap 走全图分支换所有脸.
    默认 False: 只换 lead (pose 锁中间真人), 不碰背景路人脸."""
    if swap_all:
        # 全图多人换脸: 全图检测每张脸 → 对每张逐个 ROI 上采样换脸 (复用 swap_face
        # lead_bbox 分支, 自带远景小脸 ROI→512 上采样 + LAB 色温匹配). 比 only_lead=False
        # 全图直接换好: (1) 全图分支 min_face_area=0.001 会过滤掉 22px 远景复制人脸;
        # (2) 即便不过滤, 22px 脸直接喂 inswapper (128输入) 会糊; ROI 上采样后清晰.
        # 无脸帧 swapped 仍记 True (计数=调用次数, 验证靠抽帧 embedding/像素不靠 swap_count);
        # 多人模式不判朝向 orient='multi', back_skip 不增. bbox 返 None: RVM 默认路径不
        # 依赖 lead bbox (grounding 用整帧 mask 脚底, punch n/a).
        h, w = frame.shape[:2]
        img_area = h * w
        out = frame
        for fc in app.get(frame):
            if fc.det_score < 0.3:
                continue
            b = fc.bbox
            if (b[2] - b[0]) * (b[3] - b[1]) / img_area < 0.0002:
                continue   # 过滤噪声小点 (时代广场静态背景无路人, 阈值放到 0.02%)
            cx = (b[0] + b[2]) / 2.0; cy = (b[1] + b[3]) / 2.0
            fw = b[2] - b[0]; fh = b[3] - b[1]
            # 脸框扩 ~0.9×宽 1.1×高 给 inswapper 上下文 (近似 get_lead_bbox_from_pose 肩宽)
            x1 = int(max(0, cx - fw * 0.9)); y1 = int(max(0, cy - fh * 1.1))
            x2 = int(min(w, cx + fw * 0.9)); y2 = int(min(h, cy + fh * 1.1))
            if x2 - x1 < 20 or y2 - y1 < 20:
                continue
            out = face_swap.swap_face(swapper, src_face, out, app,
                                      only_lead=True, lead_bbox=(x1, y1, x2, y2),
                                      color_match_strength=0.8)
        return out, True, None, "multi"
    if not persons:
        return frame, False, None, "unknown"
    lead = face_swap.find_lead_person(persons, w, h)
    if lead is None:
        return frame, False, None, "unknown"
    bbox, orient = face_swap.get_lead_bbox_from_pose(lead, w, h)
    if bbox is None or orient == "back":
        return frame, False, bbox, orient
    out = face_swap.swap_face(swapper, src_face, frame, app,
                              only_lead=True, lead_bbox=bbox,
                              color_match_strength=0.8)
    return out, True, bbox, orient


# ============================================================
# 4.5 脚下接地阴影 (修"贴纸感/脚底滑")
# ============================================================
def _contact_shadow(mask, h, w, strength=0.5, ground_y=None):
    """脚下软椭圆接地阴影. 按当前 mask 找脚底 (alpha>0.5 最低行) + 脚宽,
    在脚底正下方地面画扁椭圆 (横向 dilate + 大核 GaussianBlur = 软地面投影),
    返回 0-1 darkening 层 (None=无有效脚区).

    根因: RVM 干净抠像把人从原地板上切下来, 原始接地阴影 (属于原地板光照) 一起丢了
    → 人合成到静态背景上没有投影 → 视觉读成"浮/贴纸/脚底滑" (即使脚位几何正确).
    软椭圆暗影锚定接地, 跟脚逐帧 (抬腿/走位自然跟随)."""
    m = mask.astype(np.float32)
    ys, xs = np.where(m > 0.5)
    if len(ys) < 50:
        return None
    foot_y = int(ys.max())
    person_h = foot_y - int(ys.min())
    # 脚区 = 脚底往上 ~12% 人高: 取 cx + 宽 (跟脚逐帧, 抬腿交替自然跟随)
    band_lo = foot_y - max(8, int(person_h * 0.12))
    bxs = xs[ys > band_lo]
    if len(bxs) < 5:
        return None
    foot_cx = int(np.clip((int(bxs.min()) + int(bxs.max())) / 2, 0, w - 1))
    foot_w = max(int(bxs.max() - bxs.min()), int(w * 0.10))
    # 椭圆 Y 中心 = 接地线本身 (不前移). ground_y 给则锚基线 (不跟脚蹦=治"浮"), 否则脚底.
    # 脚离地越高 (lift>0) 影按比例变淡 (空中俯视投影弱, 跳起影自然虚).
    if ground_y is not None:
        cy = int(ground_y)
        lift = max(0.0, float(ground_y) - foot_y)             # 脚离地高度 px (>0=悬空)
        lift_f = float(np.clip(lift / (h * 0.12), 0.0, 1.0))   # 0=踩实, 1=高跳
        eff = strength * (1.0 - 0.55 * lift_f)                 # 跳起影淡 ~45%
    else:
        cy = foot_y
        eff = strength
    # umbra: 扁窄核心锚脚跟 (半宽=脚宽/2+3%w, 半高 0.6%h), 轻 blur. (2026-06-30 用户要
    # "脚印只在脚跟下一点, 前脚掌贴合": 半高减半=不铺到前脚掌/脚背, 半宽收窄=不外溢成片)
    umb = np.zeros((h, w), np.float32)
    u_ax = foot_w // 2 + int(w * 0.03)
    u_ay = max(4, int(h * 0.006))
    cv2.ellipse(umb, (foot_cx, cy), (u_ax, u_ay), 0, 0, 360, 1.0, -1)
    umb = cv2.GaussianBlur(umb, (0, 0), max(3.0, h * 0.010))
    # penumbra: 软外溢 (半宽=脚宽/2+5%w, 半高 1.0%h), 同心 (不前移), 中核 blur.
    pen = np.zeros((h, w), np.float32)
    p_ax = foot_w // 2 + int(w * 0.05)
    p_ay = max(6, int(h * 0.010))
    cv2.ellipse(pen, (foot_cx, cy), (p_ax, p_ay), 0, 0, 360, 1.0, -1)
    pen = cv2.GaussianBlur(pen, (0, 0), max(6.0, h * 0.018))
    layer = np.clip(umb * eff + pen * (0.45 * eff), 0.0, 1.0)
    # 前向衰减 (钉死, 别删 — 否则回退成脚尖前黑水洼):
    # 真实正面光下影落脚跟/正下, 脚尖前(镜头方向)几乎没有. 旧对称椭圆+前移半影铺到脚尖前
    # → front_near(foot_y+15..+40) V=40 全片最暗 = 假水洼 = "浮/不合理" (用户 2026-06-30 报,
    # measure_shadow 实测确认). 接地线及后方(<=cy+4)全强; 前方 cy+4→cy+34 线性衰到 0.12.
    yy = np.arange(h, dtype=np.float32)
    # 前向衰减更快 (2026-06-30): cy+4→cy+15 线性衰到 0.05, 前脚掌区(脚尖方向)近无影,
    # 前脚掌直接踩实地砖=贴合; 阴影只留脚跟正下/后方一小点.
    f0, f1 = cy + 4.0, cy + 15.0
    vatt = np.where(yy <= f0, 1.0,
            np.where(yy <= f1, 1.0 - 0.95 * (yy - f0) / (f1 - f0), 0.05))
    layer *= vatt[:, None]
    return np.clip(layer, 0.0, 1.0).astype(np.float32)


# ============================================================
# 4.6 全身色温/光照匹配 + 边缘光融合 (治"贴纸感/脚浮"头号成因)
# ============================================================
# 根因 (2026-06-30 LAB 实测): 人物比亮暖户外背景偏暗 ΔL=-28.5 偏冷 Δb=-8.6 →
# 纯 alpha 合成后人物像贴上去/浮起. 接地阴影只解决"影", 解决不了"色温不匹配".
# 行业公式: 色温统一 + 光照匹配 + light wrap(边缘混入环境光) 才是真治本.
# 单帧验证: t=0.6 把 ΔL -28.5→-11.6, Δb -8.6→-3.9 (人物本就比直射地面暗些, 合理),
# 模型确认人物融入背景/肤色自然/脚踩实. 这两个函数只动人物像素, 不碰背景.
def _clean_alpha(alpha, erode=3, feather=6):
    """去 RVM alpha 边缘 halo / 浅色残留 (原图浅地板在 alpha 过渡带的残留 = 鞋边白边,
    "脚浮"视觉源之一): erode 吃掉边缘 1-3px 浅残留, feather 软化免锯齿. 全身小 erode
    不毁主体 (RVM alpha 已足够细, 3px 对全身人物不明显). 合成时用 clean mask 替代原始 alpha."""
    a = alpha.astype(np.float32)
    if erode > 0:
        k = np.ones((erode * 2 + 1, erode * 2 + 1), np.uint8)
        a = cv2.erode(a, k)
    if feather > 0:
        a = cv2.GaussianBlur(a, (feather * 2 + 1, feather * 2 + 1), 0)
    return np.clip(a, 0, 1)


def _color_match_to_bg(fg, mask, bg_frame, t=0.8):
    """全身色温匹配 (mean-shift a/b, **保 L 不动**): 只把人物色温 (a/b 均值) 平移向
    背景环境光 (bg 下半部 y>0.4h 地+墙, 不含天空防被蓝天拉冷), **不迁移 L 亮度 / 不缩放
    方差** = 黑衣服保黑不变灰 (旧 Reinhard 同时缩 L 方差把黑衣服从 L10 拉到 L27 变灰,
    2026-06-30 用户报"颜色整体变灰"后改). 单帧实测: 黑衣服 L 原片 10.3 / Reinhard 27.2
    变灰 / 保L 10.4 保黑; 色温 Δb 保L -3.1 还优于 Reinhard -3.9. 只动人物像素不碰背景.
    t: 色温平移强度 0~1 (0=原样, 1=完全对齐背景色温)."""
    fg_mask = mask > 0.5
    if int(fg_mask.sum()) < 200:
        return fg
    h, w = fg.shape[:2]
    flab = cv2.cvtColor(fg, cv2.COLOR_BGR2LAB).astype(np.float64)
    blab = cv2.cvtColor(bg_frame, cv2.COLOR_BGR2LAB).astype(np.float64)
    af = flab[:, :, 1] - 128; bf = flab[:, :, 2] - 128
    ab = blab[:, :, 1] - 128; bb = blab[:, :, 2] - 128
    fam, fbm = float(af[fg_mask].mean()), float(bf[fg_mask].mean())
    bg_region = np.zeros((h, w), dtype=bool)
    bg_region[int(h * 0.4):, :] = True
    bam, bbm = float(ab[bg_region].mean()), float(bb[bg_region].mean())
    # mean-shift a/b (色温平移), L 完全不动 (保黑衣服对比度, 治'变灰')
    ao = af + (bam - fam) * t
    bo = bf + (bbm - fbm) * t
    flab[:, :, 1] = np.clip(ao + 128, 0, 255)
    flab[:, :, 2] = np.clip(bo + 128, 0, 255)
    return cv2.cvtColor(flab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _light_wrap(fg, bg_frame, mask, strength=0.5, edge=0.18):
    """边缘光融合: 在人物 alpha 羽化过渡带 (al>0.3, 权重随 (1-al)/edge 上升) 把
    背景色 (高斯模糊 σ=10) 混入人物 = 边缘不再硬切"贴纸"感, 受环境光染色.
    只作用边缘带, 人物核心像素 (al≈1) 不动."""
    al = mask.astype(np.float32)
    w = np.clip((1.0 - al) / edge, 0, 1) * strength
    w = w * (al > 0.3)
    bgblur = cv2.GaussianBlur(bg_frame, (0, 0), 10)
    w3 = w[:, :, None]
    return (fg.astype(np.float32) * (1 - w3) +
            bgblur.astype(np.float32) * w3).clip(0, 255).astype(np.uint8)


def _grounding(bg_frame, frame_sw, mask, h, w, ground_y=None, ao=0.18):
    """接地感增强 (2026-07-01, 治脚地分层; 区别于 _contact_shadow 失败 6 轮的单向硬阴影).
    memory 第 11 点定论: 滑动=真实脚步移动+合成丢真实接地接触, 阴影是死路 (凸显脚地两层).
    本函数换思路 = **极弱柔和 + 脚下局部精确融合**, 非明显暗影:
      (A) 脚下局部 light wrap (治硬切分层): 脚底 alpha 羽化带 (0.15-0.85, alpha≈0.5 峰) 混
          【脚下局部地面色】(bg 脚正下后方纯地面 mean). 全身 _light_wrap 用全局高斯模糊背景
          (σ10 会混入远处建筑色), 脚下要精确局部地面色才真实融合. 限脚下区域 (y >= cy-0.08h).
      (B) 极弱接地 AO (物理遮挡感): 脚正下后方小椭圆 bg gradient 微暗 (ao~0.18, vs 硬阴影
          0.5; 柔和大核 σ=h*0.030 无形状感), 前向衰减脚尖前无响应 (防前水洼).
    跳起 (lift>0) 按比例减弱 (空中不接地). 返回 (bg_frame_new, frame_sw_new)."""
    m = mask.astype(np.float32)
    ys, xs = np.where(m > 0.5)
    if len(ys) < 50:
        return bg_frame, frame_sw
    foot_y = int(ys.max())
    person_h = foot_y - int(ys.min())
    band_lo = foot_y - max(8, int(person_h * 0.12))
    bxs = xs[ys > band_lo]
    if len(bxs) < 5:
        return bg_frame, frame_sw
    foot_cx = int(np.clip((int(bxs.min()) + int(bxs.max())) / 2, 0, w - 1))
    foot_w = max(int(bxs.max() - int(bxs.min())), int(w * 0.10))
    cy = int(ground_y) if ground_y is not None else foot_y
    lift = max(0.0, float(cy) - foot_y)
    lift_f = float(np.clip(lift / (h * 0.12), 0.0, 1.0))
    eff_ao = ao * (1.0 - 0.55 * lift_f)
    wrap = ao * 2.8                                   # 脚下局部 wrap 强度 (随 ao)
    eff_wrap = wrap * (1.0 - 0.55 * lift_f)

    bgf = bg_frame.astype(np.float32)
    # (B) 极弱接地 AO: 脚正下后方椭圆, 柔和大核, 前向衰减 (脚尖前无响应, 防前水洼)
    if eff_ao > 0:
        resp = np.zeros((h, w), np.float32)
        cv2.ellipse(resp, (foot_cx, cy + 2),
                    (foot_w // 2 + int(w * 0.05), max(6, int(h * 0.014))),
                    0, 0, 360, 1.0, -1)
        resp = cv2.GaussianBlur(resp, (0, 0), max(10.0, h * 0.030))   # 极柔和 gradient 无形状
        yy = np.arange(h, dtype=np.float32)
        f0, f1 = cy + 4.0, cy + 18.0
        vatt = np.where(yy <= f0, 1.0,
                np.where(yy <= f1, 1.0 - 0.95 * (yy - f0) / (f1 - f0), 0.05))
        resp *= vatt[:, None]
        bgf *= (1.0 - eff_ao * resp[:, :, None])

    # (A) 脚下局部 light wrap: 脚底 alpha 羽化带混【脚下局部地面色】(非全局模糊背景)
    fsw = frame_sw.astype(np.float32)
    if eff_wrap > 0:
        gx0 = max(0, foot_cx - int(w * 0.12)); gx1 = min(w, foot_cx + int(w * 0.12))
        gy0 = max(0, cy + 6); gy1 = min(h, cy + int(h * 0.06))
        region_a = m[gy0:gy1, gx0:gx1]
        region_bg = bgf[gy0:gy1, gx0:gx1]
        gmask = region_a < 0.3                                   # 纯地面像素 (避开脚)
        ground_col = (region_bg[gmask].mean(axis=0).astype(np.float32)
                      if int(gmask.sum()) > 5 else np.array([120.0, 120.0, 120.0]))
        foot_top = int(cy - h * 0.08)                            # 脚下区域起点 (脚底往上 8%h)
        w_band = np.clip(1.0 - np.abs(m - 0.5) / 0.35, 0, 1)     # alpha≈0.5 峰过渡带
        ymask = (np.arange(h)[:, None] >= foot_top).astype(np.float32)
        w_band = w_band * ymask * eff_wrap
        fsw = (fsw * (1 - w_band[:, :, None]) +
               ground_col[None, None, :] * w_band[:, :, None])

    return (bgf.clip(0, 255).astype(np.uint8),
            fsw.clip(0, 255).astype(np.uint8))


# ============================================================
# 4.65 pose core-matte 撑实 (治 RVM 软抠像对细/快动胳膊低 alpha → 胳膊虚化/原背景渗出)
# ------------------------------------------------------------
# 根因 (2026-07-02): RVM 是软抠像, 对**细 (胳膊) + 快动**结构系统性低估 alpha —— 整条
# 胳膊 alpha 常只有 0.3-0.6 而非 1.0 → 合成时该区域半透明 + 新背景渗出. 这是 soft matting
# 固有软肋, 非实现 bug; 新一代抠像 (MatAnyone "core-area supervision" / VideoMaMa) 存在
# 的全部理由就是治这个 (详见 docs/BG_SWAP.md 坑 9 / memory bg-swap-arm-bleed-core-matte).
# 本工具不换模型, 用 **VFX core+edge matte 拆分**后处理达到同等效果:
#   pose 骨架包络 = 硬 core matte (保证身体/胳膊内部 alpha→1, 实心不透)
#   RVM alpha     = 软 edge matte (管真实轮廓边)
#   合成用 alpha_bolstered = max(alpha_rvm, envelope) → 包络覆盖处(含胳膊)强制高 alpha,
#   消虚化/渗出; RVM 仍管包络外的轮廓. max() 只抬不降 → 不会损伤已有的好 alpha.
# ============================================================
def _pose_core_matte(persons, w, h, scale=1.0, conf=0.3):
    """从 YOLOv8-pose 骨架建 core envelope (**所有检测到的人并集**; 多人场景全员撑实).
    返回 (envelope float32 (h,w) 0-1, shoulder_w px [最大人体, 供 gate 核尺度]);
    无有效骨架返回 (None, 0).

    COCO-17 段: 臂(肩5,6→肘7,8→腕9,10) / 躯干(肩-肩, 肩-髋11,12, 髋-髋) /
    腿(髋→膝13,14→踝15,16). **不含头** (头大 RVM 已准; 避干扰换脸区). **不含胳膊围成的
    圈内洞** (洞里无骨架 → 包络不覆盖 → 保持背景, 正确; 圈起的胳膊本身被包络撑实 → 治渗出).
    段直径按各人肩宽 scale: 臂 0.42 / 躯干 0.62 / 腿 0.46 (臂全覆盖上臂宽, 治虚化主目标).
    scale 调整体厚度 (1.0=默认; 0=关; >1 更厚适合宽松长袖/远景细胳膊). 多人各建包络取 max 并集.
    envelope 轻度 blur 平滑边界 (内部仍 1.0, 边缘软过渡与 RVM 软 alpha 接合不硬切)."""
    if not persons:
        return None, 0.0
    canvas = np.zeros((h, w), dtype=np.float32)
    max_sw = 0.0
    seen = 0
    for person in persons:
        kp = np.asarray(person, dtype=np.float32)
        if kp.ndim != 2 or kp.shape[0] < 13:
            continue
        valid = kp[:, 2] > conf
        if int(valid.sum()) < 6:                       # 关键点太少, 骨架不可靠
            continue
        px = kp[:, 0] * w
        py = kp[:, 1] * h
        if valid[5] and valid[6]:
            sw = float(np.hypot(px[5] - px[6], py[5] - py[6]))
        else:
            vi = np.where(valid)[0]
            coords = np.stack([px[vi], py[vi]], axis=1)
            sw = float(np.linalg.norm(coords[:, None] - coords[None], axis=-1).max()) * 0.45
        if sw < 12:
            continue
        max_sw = max(max_sw, sw)

        def seg(a, b, diam):
            if not (valid[a] and valid[b]):
                return
            t = max(3, int(round(diam)))
            cv2.line(canvas, (int(px[a]), int(py[a])), (int(px[b]), int(py[b])),
                     1.0, t, cv2.LINE_AA)
            r = max(2, t // 2)
            cv2.circle(canvas, (int(px[a]), int(py[a])), r, 1.0, -1)
            cv2.circle(canvas, (int(px[b]), int(py[b])), r, 1.0, -1)

        for a, b in [(5, 7), (7, 9), (6, 8), (8, 10)]:           # 臂 (治虚化主目标)
            seg(a, b, sw * 0.42 * scale)
        for a, b in [(5, 6), (5, 11), (6, 12), (11, 12)]:        # 躯干
            seg(a, b, sw * 0.62 * scale)
        for a, b in [(11, 13), (13, 15), (12, 14), (14, 16)]:    # 腿
            seg(a, b, sw * 0.46 * scale)
        seen += 1
    if seen == 0 or max_sw < 12:
        return None, 0.0
    # 轻度 blur 平滑包络边界 (核 ~4% 最大肩宽): 内部仍 1.0, 边缘软过渡
    ksz = max(5, int(round(max_sw * 0.04))) | 1
    env = cv2.GaussianBlur(canvas, (ksz, ksz), ksz / 6.0)
    return np.clip(env, 0.0, 1.0), max_sw


# ============================================================
# 4.66 arm-only core-matte (2026-07-03, 治 RVM 软抠对快动胳膊低 alpha 的确定性解)
# ------------------------------------------------------------
# 背景: A/B 实测 (memory `matanyone-ab-test-negative`) 证 MatAnyone 等软抠模型治
# 不了 2.56肩宽/帧运动模糊胳膊 (α 跌到 0.3-0.5). 唯一确定性解 = pose 骨架胳膊核心
# 强制 α→1 (max(rvm, env)). 旧 `_pose_core_matte` + `--core-bolster` 全身版被弃用,
# **且实测发现它在生产里根本没生效** — 缓存 blaze33 缺肘/腕 (映射漏) + 函数用 COCO
# 索引读 blaze 数据 = 双重错位, 画的包络是垃圾 (用户当时嫌 v3 "脏" 即此, 非 bolster 理念问题).
# 本函数 = arm-only + 正确 blaze 索引修双 bug:
#   ① 仅臂段 (肩→肘→腕), 不抬躯干/腿 → 胳膊是内部条带不碰轮廓, 避 v3 全身包络越界显脏;
#   ② 用 BlazePose-33 索引 (detect_pose 缓存格式): 11/12肩 13/14肘 15/16腕.
# 合成 **max(rvm, env) 无逐像素 alpha 门控** (2026-07-03 clip3 像素实测定稿):
#   - 旧 `--core-bolster` 用 `env × dilate(alpha>0.05)` 门控 → RVM 对快动胳膊给 α≈0 的那
#     31% 像素被门关在门外, bolster 顶死 0.69 上不去. 治虚化的全部意义恰是填这些 α≈0 像素.
#   - 去 alpha 门控改 max(rvm, env): 高潮 avg 0.74→0.954, min 0.395→0.654, **渗出 Δα=0**
#     (臂包络细带永远落在人体轮廓内, 不会越界到干净背景).
#   - 改按人**躯干存在门控** (双肩中点 rvm α>0.15): 多人源里背景路人 α≈0 自动跳过, 只撑前景.
#   - **motion 门控已弃**: 实测静止/快动帧 bolster 收益无差 (臂内部不碰轮廓, 全帧满抬也不脏),
#     motion weight 反而把快动帧 (最需治的) 压低 (arm_motion_weight 已删).
# 直径 0.42×肩宽×scale (scale 默认 1.5 覆盖运动模糊; A/B: 1.0→2.5 高潮 avg 0.91→0.99 零渗出).
# ============================================================
def _pose_arm_core_matte(persons, w, h, rvm_alpha=None, scale=1.5, conf=0.3, sho_thr=0.15):
    """胳膊核心包络 (**仅臂段**, 治 RVM 软抠对快动胳膊低 α). BlazePose-33 索引.
    返回 (envelope float32 (h,w) 0-1, shoulder_w px); 无有效臂骨架返回 (None, 0).

    合成端用 ``max(rvm_alpha, env)`` — 只抬不降零风险. 臂是内部条带不碰轮廓 → 不显脏
    (避 v3 全身包络越界). 直径 0.42×肩宽×scale.
    rvm_alpha 给定 → **按人躯干存在门控**: 双肩中点周围 rvm α 均值 < sho_thr 跳过其臂
    (RVM 没抠到 = 背景路人, 不该撑实; 只撑前景主角). 单人源无影响."""
    if not persons:
        return None, 0.0
    canvas = np.zeros((h, w), dtype=np.float32)
    max_sw = 0.0
    seen = 0
    for person in persons:
        kp = np.asarray(person, dtype=np.float32)
        if kp.ndim != 2 or kp.shape[0] < 17:
            continue
        valid = kp[:, 2] > conf
        if not (valid[11] and valid[12]):          # 需双肩定肩宽
            continue
        px = kp[:, 0] * w
        py = kp[:, 1] * h
        sw = float(np.hypot(px[11] - px[12], py[11] - py[12]))
        if sw < 12:
            continue
        # 躯干存在门控: 双肩中点周围 rvm α (前景主角躯干 α 高; 路人/背景 α≈0 跳过)
        if rvm_alpha is not None:
            mx, my = (px[11] + px[12]) / 2.0, (py[11] + py[12]) / 2.0
            r = max(8, int(sw * 0.5))
            x0, y0 = max(0, int(mx - r)), max(0, int(my - r))
            x1, y1 = min(w, int(mx + r)), min(h, int(my + r))
            if x1 > x0 and y1 > y0 and rvm_alpha[y0:y1, x0:x1].mean() < sho_thr:
                continue
        max_sw = max(max_sw, sw)
        t = max(3, int(round(sw * 0.42 * scale)))

        def seg(a, b):
            if valid[a] and valid[b]:
                cv2.line(canvas, (int(px[a]), int(py[a])), (int(px[b]), int(py[b])),
                         1.0, t, cv2.LINE_AA)
                r = max(2, t // 2)
                cv2.circle(canvas, (int(px[a]), int(py[a])), r, 1.0, -1)
                cv2.circle(canvas, (int(px[b]), int(py[b])), r, 1.0, -1)

        for a, b in [(11, 13), (13, 15), (12, 14), (14, 16)]:   # 肩-肘, 肘-腕 (左右)
            seg(a, b)
        seen += 1
    if seen == 0 or max_sw < 12:
        return None, 0.0
    ksz = max(5, int(round(max_sw * 0.04))) | 1
    env = cv2.GaussianBlur(canvas, (ksz, ksz), ksz / 6.0)
    return np.clip(env, 0.0, 1.0), max_sw


# ============================================================
# 4.7 视差缩放曲线 (从 pose 预算, 居中平滑 = 零延迟, 治 EMA "慢一拍")
# ============================================================
def compute_parallax_scale(pose, total, w, parallax, fast_win=9, base_win=60, gain=0.4):
    """从 pose keypoint 宽度预算逐帧 bg_scale. **居中平滑 (前后各 ±fast_win 帧) = 零相位延迟**,
    治 EMA 单向平滑导致的"慢一拍" (人动了背景才跟上来).
    人表观宽度 (keypoint x 跨度 = 转体/前后走/展臂) → bg 同向微缩放 = 视差纵深.
    全片预算 (pose 已缓存, 不跑 matte) → render 直接查表, 零帧内计算.
    fast = 居中短窗 (±fast_win, 响应当前尺寸), base = 居中长窗 (±base_win, 局部趋势基线);
    dev = (fast-base)/base; bg_scale = 1 + clip(gain*dev, ±parallax)."""
    widths = np.full(total, np.nan, dtype=np.float64)
    for fi in range(total):
        persons = pose.get(fi, [])
        if not persons:
            continue
        best, best_n = None, -1
        for p in persons:
            pts = np.asarray(p, dtype=np.float64)
            v = pts[pts[:, 2] > 0.3]
            if len(v) > best_n:
                best_n, best = len(v), pts
        if best is not None and best_n >= 6:
            v = best[best[:, 2] > 0.3]
            widths[fi] = (float(v[:, 0].max()) - float(v[:, 0].min())) * w
    good = np.where(~np.isnan(widths))[0]
    if len(good) < 2:
        return np.ones(total, dtype=np.float32)
    widths = np.interp(np.arange(total), good, widths[good])

    def cwin(a, win):
        out = np.empty(len(a))
        for i in range(len(a)):
            lo, hi = max(0, i - win), min(len(a), i + win + 1)
            out[i] = a[lo:hi].mean()
        return out
    fast = cwin(widths, fast_win)
    base = cwin(widths, base_win)
    dev = (fast - base) / np.maximum(base, 1.0)
    bg = 1.0 + np.clip(gain * dev, -parallax, parallax)
    print(f"  [视差] pose 宽度 mean={widths.mean():.0f}px "
          f"dev[{dev.min():+.3f},{dev.max():+.3f}] → "
          f"bg_scale[{bg.min():.4f},{bg.max():.4f}] std={bg.std():.4f} "
          f"(居中平滑 ±{fast_win}帧, 零延迟)")
    return bg.astype(np.float32)


def compute_foot_track(pose, total, w, base_frame=37, smooth_win=20, frac=1.0):
    """脚水平接地位置**低频**轨迹 → 背景水平跟随 (用户要"地面和脚尖一起动").
    schema 无关: 取每帧可见 keypoint 中 y 最大(最下=脚区) 的若干点 x 中点 = 脚水平位置.
    居中平滑(±smooth_win)= 低频(身体整体晃/走位), **减掉高频踩踏**(跟了背景逐帧抖).
    背景按 frac 同向跟随 = 脚低频漂移时背景跟着动 → 脚相对砖缝稳定不漂.

    ⚠ 物理权衡 (诚实):
    - 只跟低频: 高频踩踏(左右脚交替 0.93px/帧)不跟, 否则背景逐帧抖;
    - 真实连续走步跟随后=跑步机效应(地跟人动); 本片 foot_cx 68%来回震荡=身体晃非走,
      跟随≈模拟手持跟随晃动, 可接受; 若源里人真的大幅横走会显假.
    - 治标(减低频漂移滑动感), 不治本(真实接地接触仍缺). base_frame 对齐静态背景基帧."""
    cx = np.full(total, np.nan, dtype=np.float64)
    for fi in range(total):
        persons = pose.get(fi, [])
        if not persons:
            continue
        best, best_n = None, -1
        for p in persons:
            pts = np.asarray(p, dtype=np.float64)
            v = pts[pts[:, 2] > 0.3]
            if len(v) > best_n:
                best_n, best = len(v), pts
        if best is not None and best_n >= 6:
            v = best[best[:, 2] > 0.3]
            # 最下 ~30% keypoint (脚/小腿区) 的 x 中点 = 脚水平位置
            v = v[np.argsort(-v[:, 1])[:max(2, len(v) // 3)]]
            cx[fi] = float(v[:, 0].mean()) * w
    good = np.where(~np.isnan(cx))[0]
    if len(good) < 2:
        return np.zeros(total, dtype=np.float32)
    cx = np.interp(np.arange(total), good, cx[good])

    def cwin(a, win):
        out = np.empty(len(a))
        for i in range(len(a)):
            lo, hi = max(0, i - win), min(len(a), i + win + 1)
            out[i] = a[lo:hi].mean()
        return out
    low = cwin(cx, smooth_win)
    base = int(min(max(base_frame, 0), total - 1))
    traj = (low - low[base]) * frac
    print(f"  [脚跟] 脚水平低频 range [{traj.min():+.1f},{traj.max():+.1f}]px "
          f"(居中平滑±{smooth_win}帧, frac={frac}, base=f{base}) — 背景同向跟随减漂移")
    return traj.astype(np.float32)


# ============================================================
# 5. 渲染: 逐帧 (换脸→mask→背景合成) pipe 到 ffmpeg (含原音频, nvenc)
# ============================================================
def render(video, bg_aligned, pose, seg_model, swapper, app, src_face,
           output, info, feather=11, erode=4, do_faceswap=True, despill=0.6,
           punch=True, matte=None, traj_x=None, traj_y=None, shadow_strength=0.0,
           parallax=0.0, bg_scale_arr=None, color_match_t=0.8, light_wrap_s=0.5,
           foot_track=None, grounding_strength=0.0, pink_rg=20, pink_sat=40,
           swap_all=False, core_bolster=0.0, arm_grow=0, mask_thresh=0.0,
           mask_mode='rvm', yolo_seg_model=None):
    w, h = info["width"], info["height"]
    fps = info["fps"]
    total = info["frames"]
    cap = cv2.VideoCapture(str(video))
    # 背景可能是静态单帧 png (运镜背景冻结) 或动态对齐 mp4
    bg_is_static = (bg_aligned is not None and
                    Path(str(bg_aligned)).suffix.lower() == ".png")
    bg_static_img = None
    bg_big = None
    px_mx = px_my = 0
    if bg_is_static:
        bg_static_img = cv2.imread(str(bg_aligned))
        bg_cap = None
        bg_total = 1
        if parallax > 0 and bg_static_img is not None:
            # 视差纵深: 给静态背景四周留 7% 边距 (BORDER_REFLECT_101 镜像填充) 供逐帧微缩放.
            # s=1 取中央 w×h = 与原图完全一致构图; 缩放锚定脚接地线 → 近景(脚)不动远景动 = 视差.
            px_mx = int(w * 0.07); px_my = int(h * 0.07)
            bg_big = cv2.copyMakeBorder(bg_static_img, px_my, px_my, px_mx, px_mx,
                                        cv2.BORDER_REFLECT_101)
        elif foot_track is not None and bg_static_img is not None:
            # 脚跟: 背景水平跟随脚低频漂移 (治脚-砖缝滑动). 留 8% 横向边距供平移.
            px_mx = int(w * 0.08); px_my = 0
            bg_big = cv2.copyMakeBorder(bg_static_img, 0, 0, px_mx, px_mx,
                                        cv2.BORDER_REFLECT_101)
    else:
        bg_cap = cv2.VideoCapture(str(bg_aligned)) if bg_aligned else None
        bg_total = (int(bg_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if bg_cap is not None else 0)

    # 探 nvenc, 不可用退 libx264
    enc = "h264_nvenc"
    probe = subprocess.run([FFMPEG, "-hide_banner", "-encoders"],
                           capture_output=True, text=True)
    if "h264_nvenc" not in (probe.stdout or ""):
        enc = "libx264"
    if enc == "h264_nvenc":
        vcmd = ["-c:v", "h264_nvenc", "-preset", "p4", "-rc", "vbr",
                "-b:v", "10M", "-pix_fmt", "yuv420p"]
    else:
        vcmd = ["-c:v", "libx264", "-preset", "fast", "-crf", "17",
                "-pix_fmt", "yuv420p"]
    print(f"  [渲染] {w}x{h}@{fps:.2f}fps, 编码 {enc}, "
          f"背景={'静态单帧' if bg_is_static else '动态'}, "
          f"抠像={'RVM高精度alpha' if matte is not None else 'YOLOv8-seg'}, "
          f"羽化 {feather}, erode {erode}, despill {despill}, "
          f"punch {punch if matte is None else 'n/a'}, 换脸 {do_faceswap}"
          f"{'(全图多人)' if swap_all else ''}, "
          f"视差={'±%.1f%%' % (parallax * 100) if parallax > 0 else '关'}, "
          f"色温匹配 {color_match_t}, light wrap {light_wrap_s}")

    cmd = [FFMPEG, "-y",
           "-f", "rawvideo", "-vcodec", "rawvideo",
           "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", f"{fps:.3f}",
           "-i", "pipe:0",
           "-i", str(video),
           "-map", "0:v:0", "-map", "1:a:0?",
           *vcmd, "-c:a", "aac", "-b:a", "160k", "-shortest",
           "-movflags", "+faststart", str(output)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    fi = 0
    last_mask = None
    swap_ok = back_skip = no_pose = mask_miss = bg_loop = 0
    core_ok = 0   # pose core-matte 撑实命中帧 (验证: 抽帧看胳膊 alpha, 不靠此计数)
    arm_ok = 0    # arm-only bolster 命中帧 (2026-07-03, 治快动胳膊虚化; 同上不靠此计数判断效果)
    ground_samples = []      # foot_y 历史 → 固定接地基线 (影不随抬腿蹦)
    ground_y = None
    width_fast = width_slow = None   # 人表观宽度 EMA (快/慢) → 视差缩放驱动信号
    bg_scale = 1.0
    t0 = time.time()
    if matte is not None:
        matte.reset()   # 渲染从第 0 帧顺序读, 清 rec 状态
    while fi < total:
        ret, frame = cap.read()
        if not ret:
            break
        persons = pose.get(fi, [])
        if not persons:
            no_pose += 1

        # ① lead 脸 bbox (pose)
        lead = face_swap.find_lead_person(persons, w, h) if persons else None
        bbox, orient = (face_swap.get_lead_bbox_from_pose(lead, w, h)
                        if lead else (None, "unknown"))

        # ② mask
        if matte is not None:
            # RVM 高精度 alpha: 真正 per-pixel 抠像, 凹谷(两腿间/腋下/指缝)干净分离
            # 背景, 根治 YOLO-seg 误纳粉地面漏色. 纯 alpha 合成, 不需 punch/protect.
            mask = matte.alpha(frame)
            # 2026-07-03 mask_mode=intersect 治 RVM 远处半透真人鬼影 (新版 RVM 把远处真人
            # 当前景画 = "3 人身后站一个不动的人"). 用 YOLO-seg person mask 与 RVM α
            # 取交集: YOLO 边缘锐利剔除 RVM 远处半透区, RVM 内容保留填充 YOLO 内部.
            # 单帧视觉验证: 交集 mask 鬼影完全消失, 3 真人完整保留, 边缘略锯齿 (RVM α 平滑).
            if mask_mode == 'intersect':
                if yolo_seg_model is None:
                    raise ValueError("mask_mode=intersect 需要 yolo_seg_model (--yolo-seg-model / 加载 seg_model)")
                yolo_mask = segment_person(yolo_seg_model, frame, lead_bbox=None, conf=0.3)
                if yolo_mask is not None:
                    mask = mask * yolo_mask  # RVM α × YOLO person mask (0/1)
                # else: YOLO 漏检, fallback 纯 RVM (避免误删前景)
        else:
            # YOLO-seg 粗掩码 + build_mask(punch 打掉误纳粉背景 + 紧脸框 protect).
            # lead_bbox 选网红 instance; 漏检用上一帧兜底.
            mask_raw = segment_person(seg_model, frame, lead_bbox=bbox)
            if mask_raw is None:
                mask_miss += 1
                mask_raw = last_mask
            if mask_raw is not None:
                last_mask = mask_raw
            # 脸区保护 bbox: punch 不打脸 (脸的暖皮肤/唇/腮红 R-G>20 会被误判粉背景).
            # 用 insightface 紧脸框, 不用 pose lead bbox (基于肩+头估算本就宽, expand 会
            # 罩住头两侧/上方粉背景 → 那片粉被保护漏打孔 = "头脸/后腿腰间还有粉").
            protect = None
            if punch:
                det = _detect_lead_face_bbox(app, frame, bbox, w, h)
                if det is not None:
                    fx0, fy0, fx1, fy1 = det
                    fcx, fcy = (fx0 + fx1) / 2.0, (fy0 + fy1) / 2.0
                    fw, fh = (fx1 - fx0), (fy1 - fy0)
                    ew, eh = fw * 1.25 / 2.0, fh * 1.35 / 2.0  # 含脸颊/发际/下颌, 不碰地板
                    protect = (int(max(0, fcx - ew)), int(max(0, fcy - eh)),
                               int(min(w, fcx + ew)), int(min(h, fcy + eh)))
            mask = build_mask(mask_raw, frame, feather, erode, punch=punch,
                              protect_bbox=protect, rg_thresh=pink_rg, sat_thresh=pink_sat)

        # ②.5 pose core-matte 撑实 (2026-07-02, 治 RVM 软抠像对细/快动胳膊低 alpha → 胳膊
        # 虚化/原背景渗出; 详见 _pose_core_matte 注释 + docs/BG_SWAP.md 坑 9):
        # pose 骨架包络=硬 core, RVM alpha=软 edge; mask = max(rvm, env*gate).
        # gate = RVM 已感人体 (alpha>0.05) 邻域 dilate (核~0.25 肩宽) → 包络只在真实人体附近
        # 激活, 不会因 pose 误定位在干净背景里造"幻肢" (粘贴原背景). max() 只抬不降.
        # 注: env 实测仅覆盖画面~3% (骨架细带), 故 core-matte 只可能影响这 3% 区, 不是
        # 全片"不干净"的来源 (2026-07-02 诊断澄清, env 覆盖统计见 _temp 实测).
        if core_bolster > 0 and mask is not None and persons:
            env, sw = _pose_core_matte(persons, w, h, scale=core_bolster)
            if env is not None:
                gk = max(9, int(round(sw * 0.25))) | 1
                gate = cv2.dilate((mask > 0.05).astype(np.uint8),
                                  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (gk, gk)), 1)
                mask = np.maximum(mask.astype(np.float32),
                                  env * gate.astype(np.float32)).astype(np.float32)
                core_ok += 1

        # ②.6 arm-grow (2026-07-03, 替代 arm-bolster; 治过渡环而非核心管).
        # 原理: arm-bolster 只撑实核心管 (env scale 1.5), 但用户看到的渗出在核心管**外**
        # 的过渡环 (scale 1.5→3.0, RVM α 0.3-0.7 半透明带) = 99.8% 帧有 >2000 渗出像素.
        # D+grow = (a) inner (a>0.15) 填洞治斑驳 → (b) 在 RVM 自信前景 (a>0.05) 内 grow 3px/iter
        # 到真实边缘 → (c) max(rvm, smoothed_solid). key insight: grow 必须用 RVM α 门控
        # 否则扩到背景 (A 方案 halo 389%). 模拟 n=7488: 治愈 99.8% halo 2.5% (grow=1, 3px).
        # 教训: 别只测核心管好看就当治住了, 必须测用户看到的过渡环. 详见 memory
        # `bg-swap-core-matte-arm-bleed` + docs/BG_SWAP.md 坑 9.bis.
        if arm_grow > 0 and mask is not None and persons:
            env_zone, _sw = _pose_arm_core_matte(persons, w, h, rvm_alpha=mask,
                                                 scale=1.5)   # 臂区范围, 与 bolster 旧版同
            if env_zone is not None:
                # (a) inner = RVM 在臂内感到前景的区域 (含斑驳孔洞)
                inner = (mask > 0.15) & (env_zone > 0.5)
                # (b) outer = RVM 感到前景的过渡区 (背景 a<0.05 自动被剔, 防扩到背景)
                outer = (mask > 0.05) & (env_zone > 0.5)
                if inner.any() and outer.any():
                    # 填洞治斑驳 (不外扩 → halo 低)
                    solid = binary_fill_holes(inner).astype(np.uint8)
                    # 在 RVM 自信前景内 grow N×3px (grow=1=3px, 2=6px, 3=9px; 推荐 1)
                    solid_g = cv2.dilate(solid, np.ones((3, 3), np.uint8),
                                         iterations=int(arm_grow))
                    solid_g = solid_g & outer.astype(np.uint8)   # 关键: RVM 门控防撑背景
                    solid_smooth = cv2.GaussianBlur(
                        solid_g.astype(np.float32), (7, 7), 7 / 6.0)
                    mask = np.ascontiguousarray(mask, dtype=np.float32)
                    np.maximum(mask, solid_smooth, out=mask)   # in-place, 配合 gc 治 RAM
                    arm_ok += 1

        # 地面基线 (固定接地线 = 脚落地点的高分位 p92). 影锚定此线不随抬腿蹦 → 治"浮".
        # 全片站立时 foot_y 聚在高值(踩地), 跳起为低值(脚离地); p92 取站立线忽略跳起谷.
        if mask is not None:
            _ys, _xs = np.where(mask > 0.5)
            if len(_ys) > 50:
                ground_samples.append(int(_ys.max()))
                if len(ground_samples) >= 12:
                    ground_y = int(np.percentile(ground_samples, 92))
                else:
                    ground_y = int(ground_samples[-1])   # 前几帧用当前 foot_y 兜底
                # 视差纵深驱动信号: 人表观宽度 (转体/前后走 → 表观尺寸变). 预算的 bg_scale_arr
                # 用 pose 宽度 + 居中平滑 (零延迟, 治 EMA "慢一拍"); 无预算才退逐帧 EMA (有滞后).
                # 人变大→背景同向微缩放 (远景凑近) = 假造视差纵深, 治"人变地不变"贴纸感.
                if parallax > 0 and bg_scale_arr is not None:
                    bg_scale = float(bg_scale_arr[min(fi, len(bg_scale_arr) - 1)])
                elif parallax > 0:
                    bw = float(_xs.max() - _xs.min())
                    width_fast = bw if width_fast is None else 0.08 * bw + 0.92 * width_fast
                    width_slow = bw if width_slow is None else 0.015 * bw + 0.985 * width_slow
                    dev = (width_fast - width_slow) / max(width_slow, 1.0)
                    bg_scale = 1.0 + float(np.clip(0.4 * dev, -parallax, parallax))

        # ③ 换脸 (only_lead, 不碰背景路人)
        frame_sw = frame
        if do_faceswap:
            frame_sw, swapped, _, orient2 = swap_lead_face(
                swapper, src_face, app, frame, persons, w, h, swap_all=swap_all)
            if swapped:
                swap_ok += 1
            elif orient2 == "back":
                back_skip += 1

        # ④ 背景合成 (静态单帧=每帧同一图; 动态=逐帧推进, 比网红短则循环)
        bg_frame = None
        if bg_is_static:
            if parallax > 0 and bg_big is not None:
                # 视差纵深: bg_big (oversize) 按 bg_scale 缩放, 锚定脚接地线 (ground_y).
                # 锚点处背景不动 (脚不滑), 远离锚点的远景按 bg_scale 位移 = 视差.
                # WARP_INVERSE_MAP: M 直接是 dst→src 映射; s=1 时退化为中央裁切 = 原构图.
                anchor_y = float(ground_y) if ground_y is not None else h * 0.90
                inv_s = 1.0 / float(bg_scale)
                ax_out = w / 2.0
                M = np.array([[inv_s, 0.0, (w / 2.0 + px_mx) - ax_out * inv_s],
                              [0.0, inv_s, (anchor_y + px_my) - anchor_y * inv_s]],
                             dtype=np.float32)
                bg_frame = cv2.warpAffine(bg_big, M, (w, h),
                            flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                            borderMode=cv2.BORDER_REFLECT_101)
            else:
                if foot_track is not None and bg_big is not None:
                    # 脚跟: oversize 背景按脚水平低频偏移裁 (水平同向跟随, 脚相对砖缝稳定不漂).
                    # bg_big 横向留 8% 边距, 高度不变 (px_my=0) → oy=0, 只横移.
                    H_big, W_big = bg_big.shape[:2]
                    base_x = (W_big - w) // 2
                    tx = float(foot_track[fi]) if fi < len(foot_track) else 0.0
                    ox = max(0, min(W_big - w, int(round(base_x - tx))))
                    bg_frame = bg_big[:h, ox:ox + w]
                else:
                    H_big, W_big = bg_static_img.shape[:2]
                    if traj_x is not None and (W_big > w or H_big > h):
                        # 运镜跟随: 从 oversize 背景按 traj 偏移裁 (w,h).
                        # ox = base - traj_x: 背景内容呈现位移 = traj (与人物运镜同步) → 脚钉背景固定点.
                        base_x = (W_big - w) // 2
                        base_y = (H_big - h) // 2
                        tx = float(traj_x[fi]) if fi < len(traj_x) else 0.0
                        ty = float(traj_y[fi]) if fi < len(traj_y) else 0.0
                        ox = max(0, min(W_big - w, int(round(base_x - tx))))
                        oy = max(0, min(H_big - h, int(round(base_y - ty))))
                        bg_frame = bg_static_img[oy:oy + h, ox:ox + w]
                    else:
                        bg_frame = bg_static_img
        elif bg_cap is not None:
            okb, bg_frame = bg_cap.read()
            if not okb:
                bg_loop += 1
                bg_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                okb, bg_frame = bg_cap.read()
        # 脚下接地阴影: 按当前 mask 算脚位画软椭圆暗影到 bg (人合成在其上, 影外溢可见 = 接地锚定).
        # 修 RVM 干净抠像丢原阴影 → 人浮在静态背景上像贴纸/脚底滑.
        if (shadow_strength > 0 and mask is not None and bg_frame is not None
                and bg_frame.shape[:2] == (h, w)):
            sh = _contact_shadow(mask, h, w, shadow_strength, ground_y=ground_y)
            if sh is not None:
                bgf = bg_frame.astype(np.float32)
                bgf *= (1.0 - sh[:, :, None])
                bg_frame = bgf.clip(0, 255).astype(np.uint8)

        # ④.6 全身色温/光照匹配 + light wrap (治"贴纸感/脚浮"头号成因, 见 4.6 注释):
        # 人物偏暗偏冷贴亮暖背景 = 像贴纸浮起. 在 alpha 合成前对人物像素做 LAB Reinhard
        # 迁移 + 边缘混入背景光. 必须在换脸后(脸也统一色温) / 合成前(只动人物像素).
        if (color_match_t > 0 and mask is not None and bg_frame is not None
                and bg_frame.shape[:2] == (h, w)):
            frame_sw = _color_match_to_bg(frame_sw, mask, bg_frame, color_match_t)
            if light_wrap_s > 0:
                frame_sw = _light_wrap(frame_sw, bg_frame, mask, light_wrap_s)

        # ④.65 接地感增强 (2026-07-01, 治脚地分层; 区别失败 6 轮的硬阴影, 见 _grounding 注释):
        # 脚下局部 light wrap (脚底羽化带混脚下局部地面色治硬切) + 极弱接地 AO (脚正下微暗).
        if (grounding_strength > 0 and mask is not None and bg_frame is not None
                and bg_frame.shape[:2] == (h, w)):
            bg_frame, frame_sw = _grounding(bg_frame, frame_sw, mask, h, w,
                                            ground_y=ground_y, ao=grounding_strength)

        # 用原始 RVM alpha 合成. (_clean_alpha erode+feather 曾试治脚浮halo, 但导致
        # "人体忽然变薄"——erode/feather 缩边+虚化边缘, 叠加自然宽度变化转身帧显眼;
        # 且 halo_score 实测 erode 并未真正降 halo(浅残留非alpha边缘问题). 2026-06-30 回退.)
        # 2026-07-03 加 mask_thresh 治 RVM 远处半透真人 (α 0.3-0.5 区被 render 当前景画
        # = "鬼影"). 设 0.4-0.5 让远处降为背景; 设 0.0 = 维持 RVM 原 α (默认).
        if mask is not None and bg_frame is not None and bg_frame.shape[:2] == (h, w):
            if mask_thresh > 0:
                mask_use = np.where(mask > mask_thresh, mask, 0.0).astype(np.float32)
            else:
                mask_use = mask
            m3 = mask_use[:, :, None].astype(np.float32)
            out = (frame_sw.astype(np.float32) * m3 +
                   bg_frame.astype(np.float32) * (1.0 - m3)).astype(np.uint8)
        else:
            out = frame_sw  # 无 mask/背景 → 退原帧(已换脸)

        # ④.5 合成后去溢色 (直接打可见光晕像素, 最稳): 合成后过渡带是
        # 0.5*人物暖边 + 0.5*冷灰背景 的混合, R-G 仍偏高 = 可见暖边/光晕.
        # despill_to_bg 把过渡带 R-G 拉向【同位置背景】R-G (非中性), 边缘天然匹配
        # 背景无色差, 不过校(青边)不残留(粉边). 这是唯一根治红晕的步骤.
        if despill > 0 and mask is not None and bg_frame is not None:
            out = despill_to_bg(out, mask, bg_frame, despill)

        try:
            proc.stdin.write(out.tobytes())
        except (BrokenPipeError, OSError):
            print(f"  [渲染] pipe 断在第 {fi} 帧")
            break
        # 显式 del 还帧级临时数组给 refcount=0 (Windows committed memory 不及时归还 → 累积 OOM).
        # 2026-07-03 三次崩 (500/900/6200 帧) 都是 numpy 申请 1-10 MiB 失败; 治本得让大临时 refcount=0.
        # 名单: 帧级大临时 (10.5 MiB/张) + 每帧新建的小数组 (persons, 持续在 kp 缓存里).
        del frame_sw, out
        if mask is not None:
            try:
                del m3
            except NameError:
                pass
        if shadow_strength > 0:
            try:
                del bgf
            except NameError:
                pass
        fi += 1
        if fi % 30 == 0:   # 2026-07-03 30 帧 (旧 100 帧不够, 累积到 900 帧就崩)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()   # onnxruntime arena 跟着释放
            dt = time.time() - t0
            rss_mb = _rss_mb()
            print(f"  [渲染] {fi}/{total} ({fi*100//total}%) "
                  f"{fi/max(dt,0.001):.1f}fps | 换脸{swap_ok} 背面跳{back_skip} "
                  f"mask漏{mask_miss} core撑实{core_ok} arm撑实{arm_ok} "
                  f"rss={rss_mb:.0f}MB", flush=True)
        elif fi % 100 == 0:   # 兼容旧 100 帧进度 (silent, 不打 RSS)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            dt = time.time() - t0
            print(f"  [渲染] {fi}/{total} ({fi*100//total}%) "
                  f"{fi/max(dt,0.001):.1f}fps | 换脸{swap_ok} 背面跳{back_skip} "
                  f"mask漏{mask_miss} core撑实{core_ok} arm撑实{arm_ok}", flush=True)

    cap.release()
    if bg_cap is not None:
        bg_cap.release()
    try:
        proc.stdin.close()
    except Exception:
        pass
    rc = proc.wait()
    stat = dict(swap_ok=swap_ok, back_skip=back_skip, no_pose=no_pose,
                mask_miss=mask_miss, bg_loop=bg_loop, core_ok=core_ok,
                arm_ok=arm_ok, frames=fi)
    if rc != 0 or not output.exists():
        print(f"  [渲染][FAIL] ffmpeg rc={rc}")
        return False, stat
    dt = time.time() - t0
    print(f"  [渲染] 完成 {fi} 帧 / {dt:.1f}s ({fi/max(dt,0.001):.1f}fps): {output}")
    return True, stat


# ============================================================
# 6. debug 检查图 (采样 n 帧, 叠 mask 轮廓红 + lead bbox 黄, 拼网格)
# ============================================================
def debug_sheet(video, pose, out_png, w, h, seg_model=None, matte=None,
                n=8, feather=21, erode=2):
    cap = cv2.VideoCapture(str(video))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    times = [int(total * (i + 0.5) / n) for i in range(n)]
    thumbs = []
    for fi in times:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok:
            continue
        persons = pose.get(fi, [])
        lead = face_swap.find_lead_person(persons, w, h) if persons else None
        bbox, orient = (face_swap.get_lead_bbox_from_pose(lead, w, h)
                        if lead else (None, "unknown"))
        if matte is not None:
            matte.reset()   # 抽帧不连续, 单帧推理清 rec
            mask = matte.alpha(frame)
        else:
            mask = build_mask(segment_person(seg_model, frame, lead_bbox=bbox), frame,
                              feather, erode, punch=True)
        vis = frame.copy()
        if mask is not None:
            contours, _ = cv2.findContours((mask > 0.5).astype(np.uint8) * 255,
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, contours, -1, (0, 0, 255), 3)
        if bbox is not None:
            cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                          (0, 255, 255), 3)
        cv2.rectangle(vis, (0, 0), (240, 50), (0, 0, 0), -1)
        cv2.putText(vis, f"f={fi} {orient}", (8, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        thumbs.append(cv2.resize(vis, (w // 3, h // 3),
                                 interpolation=cv2.INTER_LANCZOS4))
    cap.release()
    if not thumbs:
        return None
    mh = max(t.shape[0] for t in thumbs)
    padded = []
    for t in thumbs:
        if t.shape[0] < mh:
            t = np.vstack([t, np.zeros((mh - t.shape[0], t.shape[1], 3),
                                       dtype=np.uint8)])
        padded.append(t)
    while len(padded) % 2:
        padded.append(np.zeros_like(padded[0]))
    rows = [np.hstack(padded[i:i + 2]) for i in range(0, len(padded), 2)]
    grid = np.vstack(rows)
    cv2.imencode(".png", grid)[1].tofile(str(out_png))
    print(f"  [debug] 检查图: {out_png} (红=mask轮廓, 黄=lead脸框)")
    return out_png


# ============================================================
# main
# ============================================================
def load_bgswap_preset(name):
    """读 presets/bgswap_<name>.yaml 的 bg_swap: 段 (flat dict). 不存在则退出.
    预设只设默认值, CLI 显式值仍胜 (两阶段 parse 实现, 见 main 开头)."""
    p = Path(__file__).resolve().parent.parent / "presets" / f"bgswap_{name}.yaml"
    if not p.exists():
        sys.exit(f"[ERROR] 预设不存在: bgswap_{name} "
                 f"(查找 presets/bgswap_*.yaml; 可选 fitness/clean/dance)")
    import yaml
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return data.get("bg_swap", {})


def main():
    # 两阶段 parse: mini-parser 先取 --preset, 用预设值设 argparse 默认; CLI 显式值仍胜预设.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset")
    pre_args, _ = pre.parse_known_args()
    preset = load_bgswap_preset(pre_args.preset) if pre_args.preset else {}
    if pre_args.preset:
        print(f"[预设] bgswap_{pre_args.preset}: {preset}")

    ap = argparse.ArgumentParser(description="网红视频换背景+换脸 (独立工具)")
    ap.add_argument("--video", required=True, help="网红源视频")
    ap.add_argument("--bg", required=True, help="背景 (动态视频 mp4 或静态图片)")
    ap.add_argument("--coach", required=True,
                    help="换脸目标教练 (find_coach_face 找 tools/{coach}_gfpgan.png)")
    ap.add_argument("--output", required=True, help="输出 mp4")
    ap.add_argument("--dynamic-bg", action="store_true",
                    default=preset.get("dynamic_bg", False),
                    help="用动态背景视频逐帧推进 (仅静态机位背景; 运镜背景会让人物滑动, 默认静态单帧)")
    ap.add_argument("--bg-crop-y", type=float, default=preset.get("bg_crop_y", 0.5),
                    help="背景 cover 竖向裁切位置 0=顶 0.5=中心(默认) 1=底. 竖屏背景塞横屏且"
                         "上部有遮挡物(天棚/建筑)时调高(如0.61)下移避开 → 胳膊举起落干净地面背景, "
                         "RVM抠胳膊准不消失/不虚化, 天棚降到画面外 (2026-07-01 网红多人案例)")
    ap.add_argument("--follow-cam", action="store_true",
                    default=preset.get("follow_cam", False),
                    help="运镜跟随 (默认关, 实验性): phaseCorrelate 扫源相机轨迹, oversize 背景"
                         "按轨迹逐帧裁. ⚠ 多数网红健身视频相机静止 + 人物占满画面 → phaseCorrelate "
                         "把人舞蹈/走位当运镜 → 背景凭空滚动 = 脚底滑动 (本工具默认静态已验证最稳). "
                         "仅人物小、背景纹理多、真有镜头平移的源才考虑试.")
    ap.add_argument("--bg-frame", type=float, default=preset.get("bg_frame"),
                    help="静态背景取哪秒的帧 (默认中间帧, 构图最饱满)")
    ap.add_argument("--feather", type=int, default=preset.get("feather", 11),
                    help="mask 边缘羽化核 (奇数, 默认11; 越大越柔但透色带越宽)")
    ap.add_argument("--erode", type=int, default=preset.get("erode", 4),
                    help="mask 边缘收缩像素 (收缩脏边缘/消透色, 默认4; 太大会吃脚底致悬空)")
    ap.add_argument("--despill", type=float, default=preset.get("despill"),
                    help="边缘去溢色强度 0-1 (默认: matte 模式 0=关 因 RVM alpha 干净无需; "
                         "seg 模式 1.0 去 YOLO 羽化带粉地面透色)")
    ap.add_argument("--shadow-strength", type=float, default=preset.get("shadow_strength", 0.0),
                    help="脚下接地阴影强度 0-1 (默认0=关. 2026-06-30 实测: 脚深相机静止0.33px, "
                         "脚-砖缝来回变化=真实脚步移动(非相机抖动), 滑动感来自合成丢真实接地接触+背景冻结, "
                         "非缺阴影; 人造阴影不追踪真实接触反凸显'脚地两层'分层感 → 默认关. "
                         "0.5=双层umbra+penumbra锚接地(历史值, 一般不需要)")
    ap.add_argument("--no-contact-shadow", action="store_true",
                    help="关掉脚下接地阴影")
    ap.add_argument("--grounding", type=float, default=preset.get("grounding", 0.0),
                    help="接地感增强强度 0-1 (默认0=关; 2026-07-01 用户选 C 方案再试接地感). "
                         "和 --shadow-strength 硬阴影区别: 不画明显暗影, 改用 (A) 脚下局部 light "
                         "wrap (脚底羽化带混脚下局部地面色治硬切分层) + (B) 极弱接地 AO (脚正下 "
                         "gradient 微暗). 0.18=推荐起步 (vs 硬阴影 0.5). 治'脚地两层'贴纸感.")
    ap.add_argument("--no-grounding", action="store_true",
                    help="关掉接地感增强 (脚下局部融合 + 极弱 AO)")
    ap.add_argument("--parallax", type=float, default=preset.get("parallax", 0.02),
                    help="视差纵深: 背景随人表观尺寸微缩放幅度 (默认0.02=±2%%). 人变大→背景同向"
                         "微缩放=假造视差纵深, 治'人变地不变'贴纸感. 锚定脚接地线, 远景动近景不动不滑."
                         "源相机静止无可复制的真运镜, 这是风格化假造 (幅度大易显假). 0=关")
    ap.add_argument("--no-parallax", action="store_true", help="关掉视差纵深缩放")
    ap.add_argument("--foot-track", action="store_true",
                    default=preset.get("foot_track", False),
                    help="背景水平跟随脚的低频漂移 (用户要'地面和脚尖一起动'). "
                         "只跟低频(身体整体晃/走位), 不跟高频踩踏(否则背景逐帧抖). "
                         "减脚-砖缝低频漂移滑动感; ⚠ 真实连续横走会显跑步机效应(地跟人动).")
    ap.add_argument("--foot-track-frac", type=float, default=preset.get("foot_track_frac", 1.0),
                    help="脚跟强度 0-1 (默认1.0=完全跟随低频漂移). 0.5=半跟, 更克制不易显假")
    ap.add_argument("--feet-seam", action="store_true",
                    default=preset.get("feet_seam", False),
                    help="脚钉砖缝模式 (配合 --follow-cam): traj=脚低频(重心转移)+cam高频(手持抖). "
                         "治脚-砖缝低频漂移滑(脚水平低频42px vs cam仅5.8px, 纯camfollow残差40px=滑). "
                         "脚钉砖缝+保留手持晃动; 脚高频踩踏(8px)不跟=真实步态踩不同砖.")
    ap.add_argument("--feet-seam-frac", type=float, default=preset.get("feet_seam_frac", 1.0),
                    help="脚钉强度 0-1 (默认1.0=完全跟脚低频, 脚钉死砖缝). "
                         "0.6=部分跟, 脚-砖缝残差换远景建筑少漂(更自然)")
    ap.add_argument("--color-match", type=float, default=preset.get("color_match", 0.8),
                    help="全身色温匹配强度 0-1 (默认0.8: 只平移人物 a/b 色温均值向背景环境光, "
                         "**保 L 不动**=黑衣服保黑不变灰 (旧 Reinhard 缩 L 方差把黑拉灰, 已弃). "
                         "治'贴纸感/脚浮'(人物偏冷蓝贴暖背景). 0=关. 只动 a/b 不毁对比度, 高到 1.0 也安全)")
    ap.add_argument("--no-color-match", action="store_true", help="关掉全身色温匹配")
    ap.add_argument("--light-wrap", type=float, default=preset.get("light_wrap", 0.5),
                    help="边缘光融合强度 0-1 (默认0.5: 人物 alpha 边缘带混入背景色, 边缘不再硬切"
                         "'贴纸'感, 受环境光染色. 0=关. 配合 --color-match 用")
    ap.add_argument("--no-light-wrap", action="store_true", help="关掉边缘光融合")
    ap.add_argument("--no-punch", action="store_true",
                    help="关掉打孔(默认开): 去 mask 误纳入的原图粉背景(两腿间凹谷/过分割边缘), "
                         "否则两腿中间保留原图粉色. 仅人物着装非饱和粉红时安全")
    ap.add_argument("--no-faceswap", action="store_true", help="只换背景不换脸")
    ap.add_argument("--swap-all", dest="swap_all", action="store_true",
                    help="换全图所有人脸 (多人场景如 3 人复制人构图, 默认只换 lead 真人)")
    ap.add_argument("--matte", dest="matte", action="store_true",
                    default=preset.get("matte", True),
                    help="RVM 高精度抠像 (默认开, 治本: 真 per-pixel alpha, 根治粉背景漏色)")
    ap.add_argument("--no-matte", dest="matte", action="store_false",
                    help="关掉 RVM, 回退 YOLOv8-seg 粗分割 (+despill/punch/protect 补丁)")
    ap.add_argument("--dsr", type=float, default=preset.get("dsr", 0.25),
                    help="RVM 内部降采样比 0.1-1.0 (默认0.25 快; **举手时胳膊虚/两臂间残留原图背景**"
                         "= dsr 太低 alpha 分辨不出细胳膊/凹陷洞, 调 0.4-0.5 锐化边缘+填准凹陷. "
                         "720p 多人细胳膊推荐 0.5; 1080p 单人全身 0.25 够. 越高越慢≈线性")
    ap.add_argument("--core-bolster", type=float, default=preset.get("core_bolster", 0.0),
                    help="pose core-matte 撑实强度, 治 RVM 胳膊虚化/原背景渗出 (2026-07-02). "
                         "RVM 软抠对细/快动胳膊低 alpha → 半透明+渗出; pose 骨架建 core 包络, "
                         "mask=max(rvm, 包络*gate) 把胳膊撑实. **默认 0=关** (2026-07-02 反转: v3 全片实测 "
                         "骨架带每帧硬抬 alpha 让人物轮廓显脏, 用户'基本都这样', 弃用回 v2 软边); "
                         "1.0=全覆盖上臂宽 (需治渗出时手动开, 接受边缘偏硬); >1 更厚. "
                         "根因+外部佐证(MatAnyone core-supervision)见 docs/BG_SWAP.md 坑 9.")
    ap.add_argument("--no-core-bolster", action="store_true",
                    help="关掉 pose core-matte 撑实 (回退纯 RVM alpha, 胳膊可能虚化/渗出)")
    ap.add_argument("--arm-grow", type=int, default=preset.get("arm_grow", 0),
                    help="arm 治渗出 (填洞+alpha 门控 grow), 治 RVM 对胳膊低 alpha 过渡环虚化. "
                         "**默认 0=关** (opt-in). grow N = 核心管外扩 N×3px 到真实边缘 (在 RVM 自信前景内). "
                         "推荐 1 (3px) — 模拟 n=7488 治愈 99.8% halo 2.5% (grow=2/3 略高 halo). "
                         "替代旧 --arm-bolster (治了核心管没治环, 用户拍板). 详见 docs/BG_SWAP.md 坑 9.bis.")
    ap.add_argument("--no-arm-grow", action="store_true",
                    help="关掉 arm-grow (回退纯 RVM alpha)")
    ap.add_argument("--mask-thresh", type=float, default=0.0,
                    help="合成前 RVM α 阈值 (默认 0=原 α; 设 0.4-0.5 让 RVM 远处半透真人"
                         "α 0.3-0.5 降为 0=背景, 治 2026-07-03 d_grow1 '鬼影' 问题 (新版 RVM"
                         "把远处真人当前景画). 0.5=严格前景 only; 0.3=保留更多半透前景; "
                         "0=不阈 (治 v3 core-matte 时代问题回退路径).")
    ap.add_argument("--mask-mode", choices=['rvm', 'intersect'], default='rvm',
                    help="合成 mask 来源 (默认 rvm=RVM α; intersect=RVM α × YOLO-seg person mask, "
                         "治 2026-07-03 RVM 远处半透真人 '鬼影' 问题. intersect 加 --yolo-seg-model 加载 "
                         "YOLOv8-seg. 单帧视觉验证: 鬼影完全消失, 3 真人完整保留.")
    ap.add_argument("--yolo-seg-model", default='yolov8n-seg.pt',
                    help="intersect 模式用的 YOLO-seg 模型路径 (默认 yolov8n-seg.pt, 6.7MB 轻量)")
    ap.add_argument("--no-sharpen-bg", action="store_true", help="背景不做锐化")
    ap.add_argument("--debug-only", action="store_true",
                    help="只出 mask 检查图不渲染 (先核对抠像质量)")
    ap.add_argument("--preset",
                    help="预设名 fitness/clean/dance (presets/bgswap_<name>.yaml); 设默认值, CLI 显式值仍胜")
    ap.add_argument("--ffmpeg", help="覆盖 ffmpeg 路径 (默认: BG_FFMPEG env > PATH > 已知好路径)")
    ap.add_argument("--pink-thresh-rg", type=int, default=20,
                    help="seg 回退路径 captured_bg_mask 的 R-G 粉检测阈值 (默认20; 仅 --no-matte 用)")
    ap.add_argument("--pink-thresh-sat", type=int, default=40,
                    help="seg 回退路径 captured_bg_mask 的饱和度粉检测阈值 (默认40; 仅 --no-matte 用)")
    args = ap.parse_args()

    # ffmpeg 可移植: --ffmpeg 覆盖 > env > PATH > 已知好路径 (泛化 2026-07-01)
    global FFMPEG
    FFMPEG = _resolve_ffmpeg(args.ffmpeg)

    video = Path(args.video)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path("_temp")
    tmp.mkdir(exist_ok=True)
    cache = tmp / f"{video.stem}_bgswap_keypoints.json"
    tools_dir = str(Path(__file__).resolve().parent)

    print(f"=== 网红换背景+换脸 ===")
    print(f"源: {video.name}  背景: {Path(args.bg).name}  教练: {args.coach}")

    # 1) pose
    print("[1/5] Pose 检测 (缓存复用)...")
    pose, info = detect_pose(str(video), cache)
    w, h, fps, total = info["width"], info["height"], info["fps"], info["frames"]
    print(f"  {w}x{h}@{fps:.2f}fps {total} 帧")

    # 2) 换脸源 (debug-only 跳过换脸模型加载, 最快核对 mask 质量)
    app = swapper = src_face = None
    do_faceswap = not args.no_faceswap and not args.debug_only
    if do_faceswap:
        print("[2/5] 加载换脸源 + 模型...")
        src_path = find_coach_face(args.coach, tools_dir)
        if not src_path:
            print(f"  [WARN] 找不到 {args.coach} 的换脸照片 "
                  f"(tools/{args.coach}_gfpgan.png 等), 跳过换脸")
            do_faceswap = False
        else:
            print(f"  换脸源: {Path(src_path).name}")
            app = face_swap.get_face_analyser()
            src_face = face_swap.extract_face_embedding(app, src_path)
            swapper = face_swap.get_swapper()
    if not do_faceswap:
        print("[2/5] 跳过换脸")

    # 3) 运镜跟随 (默认关): 网红健身视频相机基本静止, phaseCorrelate 在人物占满画面时把人
    #    动作当运镜 → 背景乱裁滚动 = 脚滑. 静态背景(冻结单帧)最稳, 脚钉背景固定点.
    follow_cam = args.follow_cam
    traj_x = traj_y = None
    margin_x = margin_y = 0
    if follow_cam and not args.dynamic_bg:
        print("[3/5] 运镜跟随: 特征点光流扫相机轨迹 + 脚低频(脚钉砖缝)...")
        cam_x, cam_y, swing_x, swing_y = estimate_camera_motion(str(video), total)
        if args.feet_seam:
            traj_x, traj_y, swing_x, swing_y = compute_sync_track(
                pose, total, w, h, cam_x, cam_y, foot_frac=args.feet_seam_frac)
        else:
            traj_x, traj_y = cam_x, cam_y
        pad = 32
        margin_x = swing_x + pad
        margin_y = swing_y + pad
    print("[3/5] 背景预处理对齐...")
    bg_aligned = prepare_bg(args.bg, w, h, fps, tmp,
                            sharpen_bg=not args.no_sharpen_bg,
                            static=not args.dynamic_bg,
                            bg_frame_sec=args.bg_frame,
                            margin_x=margin_x, margin_y=margin_y,
                            bg_crop_y=args.bg_crop_y)

    # 4) 抠像模型 + debug 检查图
    matte = None
    seg_model = None
    if args.matte:
        try:
            print("[4/5] RVM 高精度抠像 (RobustVideoMatting mobilenetv3)...")
            matte_model = load_matte_model()
            matte = MatteStream(matte_model, downsample_ratio=args.dsr)
            print(f"  RVM 就绪 (device={matte.device}, half={matte.half}, dsr={matte.dsr})")
        except Exception as e:
            print(f"  [WARN] RVM 加载失败 ({repr(e)[:200]}), 回退 YOLOv8-seg")
            matte = None
    if matte is None:
        print("[4/5] YOLOv8-seg 人体分割...")
        seg_model = load_seg_model()
    # 2026-07-03 mask_mode=intersect 模式加载 yolov8-seg 二次确认 (治 RVM 远处半透真人鬼影)
    yolo_seg_model = None
    if args.mask_mode == 'intersect':
        try:
            from ultralytics import YOLO
            # YOLO 强制 CPU: 避开与 RVM/buffalo_l/inswapper 三个 GPU 模型争 4GB onnxruntime
            # arena (face-swap-cudnn-fix 已知三模型 HEURISTIC+4GB 才能跑). 4 模型同 GPU
            # 加载 buffalo_l 1k3d68.onnx 'bad allocation' 已实测. yolov8n-seg 6.7MB
            # CPU 推理 ~50ms/帧 (intersect 仅需 person mask, 不需高精度), 720×1280 单帧可接受.
            yolo_seg_model = YOLO(args.yolo_seg_model)
            yolo_seg_model.to('cpu')
            print(f"[4/5] YOLO-seg 二次确认就绪: {args.yolo_seg_model} (CPU, mask_mode=intersect)")
        except Exception as e:
            print(f"[FAIL] --mask-mode intersect 需要 yolov8-seg ({args.yolo_seg_model}), 加载失败: {e}")
            return
    dbg = tmp / f"{video.stem}_bgswap_debug.png"
    debug_sheet(str(video), pose, dbg, w, h, seg_model=seg_model, matte=matte,
                feather=args.feather, erode=args.erode)

    if args.debug_only:
        print("[debug-only] 只出了检查图, 看一眼再渲染")
        return

    # 5) 渲染
    despill_val = args.despill if args.despill is not None else (0.0 if matte is not None else 1.0)
    parallax_val = 0.0 if args.no_parallax else args.parallax
    if args.foot_track:
        parallax_val = 0.0   # 脚跟优先, 占用 bg_big 做水平平移 (与视差缩放互斥)
    bg_scale_arr = None
    if parallax_val > 0:
        print("[视差] 从 pose 预算零延迟缩放曲线 (居中平滑, 治 EMA 慢一拍)...")
        bg_scale_arr = compute_parallax_scale(pose, total, w, parallax_val)
    foot_traj = None
    if args.foot_track:
        bf_sec = args.bg_frame if args.bg_frame is not None else (total / fps) / 2
        base_fr = int(min(max(round(bf_sec * fps), 0), total - 1))
        print("[脚跟] 从 pose 预算脚水平低频轨迹 (背景同向跟随减漂移)...")
        foot_traj = compute_foot_track(pose, total, w, base_frame=base_fr,
                                       smooth_win=20, frac=args.foot_track_frac)
    print("[5/5] 渲染 (换脸+mask+背景合成 → ffmpeg pipe)...")
    shadow_strength = 0.0 if args.no_contact_shadow else args.shadow_strength
    color_match_t = 0.0 if args.no_color_match else args.color_match
    light_wrap_s = 0.0 if args.no_light_wrap else args.light_wrap
    grounding_strength = 0.0 if args.no_grounding else args.grounding
    core_bolster_val = 0.0 if args.no_core_bolster else args.core_bolster
    arm_grow_val = 0 if args.no_arm_grow else args.arm_grow
    ok, stat = render(str(video), bg_aligned, pose, seg_model, swapper, app, src_face,
                      output, info, feather=args.feather, erode=args.erode,
                      despill=despill_val, do_faceswap=do_faceswap,
                      swap_all=args.swap_all,
                      punch=not args.no_punch, matte=matte,
                      traj_x=traj_x, traj_y=traj_y, shadow_strength=shadow_strength,
                      parallax=parallax_val, bg_scale_arr=bg_scale_arr,
                      color_match_t=color_match_t, light_wrap_s=light_wrap_s,
                      foot_track=foot_traj, grounding_strength=grounding_strength,
                      pink_rg=args.pink_thresh_rg, pink_sat=args.pink_thresh_sat,
                      core_bolster=core_bolster_val, arm_grow=arm_grow_val,
                      mask_thresh=args.mask_thresh,
                      mask_mode=args.mask_mode, yolo_seg_model=yolo_seg_model)
    if not ok:
        print(f"[FAIL] 渲染失败, 看 {dbg} 诊断. stat={stat}")
        return
    print(f"\n=== 完成 ===")
    print(f"视频: {output}")
    print(f"检查图: {dbg}")
    print(f"统计: 换脸成功 {stat['swap_ok']}/{stat['frames']} 帧, "
          f"背面跳过 {stat['back_skip']}, 无pose {stat['no_pose']}, "
          f"mask漏检 {stat['mask_miss']}, 背景循环 {stat['bg_loop']}, "
          f"core撑实 {stat.get('core_ok', 0)}, arm撑实 {stat.get('arm_ok', 0)}")


if __name__ == "__main__":
    main()
