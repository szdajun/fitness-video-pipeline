#!/usr/bin/env python3
"""换脸工具：将目标视频的人脸替换为源教练的人脸

依赖: insightface + inswapper_128.onnx

用法:
  python tools/face_swap.py --source 教练照片.jpg --target 目标视频.mp4 --output 输出.mp4

流程:
  1. 从 source 提取人脸特征(embedding)
  2. 逐帧检测 target 视频中的人脸
  3. 用 inswapper 将 source 人脸换到 target 上
  4. 编码输出视频（含原音频）
"""

import cv2, numpy as np, argparse, os, subprocess, sys, tempfile, shutil, json
from pathlib import Path

# 确保项目根在 sys.path (直接 `python tools/face_swap.py` 跑时根不在 path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.utils import resolve_ffmpeg

# 换脸默认用 GPU (inswapper + GFPGAN 都吃 GPU, 比 CPU 快十几倍).
# 仅当显式 FACE_SWAP_FORCE_CPU=1 时才回退 CPU (老机器显存吃紧时用).
# NOTE: 之前无条件禁 CUDA 是换脸慢到 22 分钟/片的元凶, 且把 GFPGAN 也拖到 CPU.
if os.environ.get("FACE_SWAP_FORCE_CPU", "0") == "1":
    os.environ["ORT_DISABLE_GPU"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

SWAPPER_MODEL = os.path.expanduser("~/.insightface/models/inswapper_128.onnx")
FFMPEG = resolve_ffmpeg()  # 可移植解析 (override/env/已知好路径/PATH), 不裸硬编码

# GFPGAN 换脸后修复: inswapper_128 输出仅 128px, 必须再过 GFPGAN 才有美颜质感.
# 权重搜索路径 (按优先级, 找到即用).
_GFPGAN_WEIGHT_CANDIDATES = [
    r"F:/wkspace/ComfyUI/models/gfpgan/GFPGANv1.4.pth",
    os.path.expanduser("~/gfpgan_weights/GFPGANv1.4.pth"),
    os.path.join(os.path.dirname(__file__), "..", "cloud_gpu", "gfpgan_weights", "GFPGANv1.4.pth"),
]


def _ensure_cuda_dlls():
    """让 onnxruntime-gpu 找到 CUDA 12 + cuDNN 9 DLL — 借用 torch 自带的 (省得单独装 cuDNN).

    onnxruntime-gpu 1.19 需要 cudart64_12 / cublas / cudnn64_9 等 DLL, torch wheel (cu124) 全自带在
    torch/lib/. 运行时把它加进 DLL 搜索路径即可. 必须在 insightface 建 session 之前调用.
    """
    try:
        import torch as _t
        lib = os.path.join(os.path.dirname(_t.__file__), "lib")
        if os.path.isdir(lib):
            try:
                os.add_dll_directory(lib)
            except (AttributeError, OSError):
                pass
            os.environ["PATH"] = lib + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


_ensure_cuda_dlls()


def _build_onnx_session_options():
    """构造 onnxruntime SessionOptions, 设 arena 策略 + 显存上限

    2026-06-27 修复: 长跑 + 多 stage GPU 共享导致 CUDNN_STATUS_EXECUTION_FAILED.
    - arena_extend_strategy='kSameAsRequested' 避免 kNextPowerOfTwo 的显存碎片
    - gpu_mem_limit=8GB 防止 onnxruntime 抢占所有显存
    """
    try:
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 注: onnxruntime 1.19 Python API 不能直接设 arena_extend_strategy,
        # 要通过 CUDAExecutionProvider 的 provider_options 传 (下面使用)
        return opts
    except Exception as e:
        print(f"  [WARN] CUDA provider_options 构建失败: {e}")
        return None


def _cuda_provider_options():
    """onnxruntime CUDA EP 的 provider_options (arena 策略 + 显存上限)"""
    return [{
        "device_id": 0,
        "arena_extend_strategy": "kSameAsRequested",  # 避免 kNextPowerOfTwo 碎片
        "gpu_mem_limit": 4 * 1024 * 1024 * 1024,      # 4GB 上限 (4070 12GB; inswapper 模型仅 52MB 够用,
        # 留余量给同进程 RVM torch + buffalo_l 检测器. CUDNN 稳定性靠 arena 策略, 非此上限)
        # HEURISTIC 而非 EXHAUSTIVE: bg_swap 同进程跑 RVM(torch)+buffalo_l+inswapper 三 GPU 模型,
        # EXHAUSTIVE 搜大 workspace algo 瞬时撑爆 12GB → Conv_62 起即 OOM (2026-06-30). HEURISTIC 省 workspace.
        "cudnn_conv_algo_search": "HEURISTIC",
        "do_copy_in_default_stream": True,
    }]


def get_swapper():
    import insightface
    if not os.path.exists(SWAPPER_MODEL):
        raise FileNotFoundError(f"模型未找到: {SWAPPER_MODEL}\n请先下载: python _download_inswapper.py")
    # 优先 GPU (CUDA), 不可用时降级 CPU
    # 2026-06-27: 加 provider_options (arena 策略 + 显存上限) 修 CUDNN 碎片化
    providers = [
        ("CUDAExecutionProvider", _cuda_provider_options()[0]),
        "CPUExecutionProvider",
    ]
    return insightface.model_zoo.get_model(SWAPPER_MODEL, providers=providers)


def get_face_analyser():
    import insightface
    # 优先 GPU (CUDAExecutionProvider, ctx_id=0), 不可用时降级 CPU
    # 2026-06-27: 加 provider_options (arena 策略 + 显存上限) 修 CUDNN 碎片化
    providers = [
        ("CUDAExecutionProvider", _cuda_provider_options()[0]),
        "CPUExecutionProvider",
    ]
    app = insightface.app.FaceAnalysis(name="buffalo_l", providers=providers)
    # 尝试 GPU (ctx_id=0), 失败回退 CPU (ctx_id=-1)
    try:
        app.prepare(ctx_id=0, det_size=(640, 640))
    except Exception as e:
        print(f"  [WARN] insightface GPU prepare 失败, 回退 CPU: {e}")
        app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


# ============================================================
# GFPGAN 换脸后修复 (inswapper → GFPGAN 标准两步)
# ============================================================
_GFPGAN_PATCHED = False
_GFPGAN_CACHE = {"model": None, "cascade": None, "device": None}


def _patch_torchvision_compat():
    """新版 torchvision 删了 functional_tensor, 旧版 gfpgan/basicsr 依赖它 — 补回来."""
    global _GFPGAN_PATCHED
    if _GFPGAN_PATCHED:
        return
    try:
        import types
        import torchvision.transforms.functional as _F
        m = types.ModuleType("torchvision.transforms.functional_tensor")
        m.rgb_to_grayscale = _F.rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = m
        _GFPGAN_PATCHED = True
    except Exception:
        pass


def _load_gfpgan(device="cuda", model_path=None):
    """惰性加载 GFPGANv1.4 + OpenCV Haar 人脸检测器. 找不到权重返回 (None,None,None)."""
    if _GFPGAN_CACHE["model"] is not None:
        return _GFPGAN_CACHE["model"], _GFPGAN_CACHE["cascade"], _GFPGAN_CACHE["device"]
    _patch_torchvision_compat()
    import torch
    if model_path is None:
        for p in _GFPGAN_WEIGHT_CANDIDATES:
            if os.path.exists(p):
                model_path = p
                break
    if not model_path or not os.path.exists(model_path):
        return None, None, None
    dev = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
    model = GFPGANv1Clean(
        out_size=512, num_style_feat=512, channel_multiplier=2,
        decoder_load_path=None, fix_decoder=False, num_mlp=8,
        input_is_latent=True, different_w=True, narrow=1, sft_half=True,
    )
    sd = torch.load(str(model_path), map_location="cpu")
    if "params_ema" in sd:
        sd = sd["params_ema"]
    model.load_state_dict(sd, strict=True)
    model.eval().to(dev)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    _GFPGAN_CACHE.update(model=model, cascade=cascade, device=dev)
    print(f"  GFPGAN 加载完成 (device={dev}, weight={os.path.basename(model_path)})")
    return model, cascade, dev


def gfpgan_restore_frame(frame, model, cascade, device, strength=0.5):
    """对帧内最大脸跑 GFPGAN 修复 + 强度混合 + 边缘羽化. 无脸/失败返回原帧.

    strength: 0=不修, 1=全用 GFPGAN 输出. swapped 脸建议 0.5 (保留身份又补美颜).
    """
    if model is None or strength <= 0:
        return frame
    import torch
    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.15, minNeighbors=5, minSize=(60, 60))
    if len(faces) == 0:
        return frame
    # 取最大脸, 扩 1.5x 确保含完整五官
    x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
    cx, cy = x + fw // 2, y + fh // 2
    hw_, hh_ = int(fw * 1.5 / 2), int(fh * 1.5 / 2)
    x1, y1 = max(0, cx - hw_), max(0, cy - hh_)
    x2, y2 = min(w, cx + hw_), min(h, cy + hh_)
    if x2 - x1 < 20 or y2 - y1 < 20:
        return frame
    roi = frame[y1:y2, x1:x2]
    oh, ow = roi.shape[:2]
    # 推理: resize 512 → GFPGAN → 回原尺寸
    f512 = cv2.resize(roi, (512, 512), interpolation=cv2.INTER_CUBIC)
    t = torch.from_numpy(f512.astype(np.float32) / 127.5 - 1.0).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
        if isinstance(out, (tuple, list)):
            out = out[0]
    out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out = np.clip((out + 1.0) * 127.5, 0, 255).astype(np.uint8)
    out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    out = cv2.resize(out, (ow, oh), interpolation=cv2.INTER_AREA)
    # 强度混合
    blended = cv2.addWeighted(roi, 1.0 - strength, out, strength, 0)
    # 边缘羽化 (中心全 GFPGAN, 边缘渐变回原图, 避免贴片感)
    mask = np.zeros((oh, ow), dtype=np.float32)
    im_ = max(5, min(oh, ow) // 12)
    cv2.rectangle(mask, (im_, im_), (ow - im_, oh - im_), 1.0, -1)
    bk = max(3, min(oh, ow) // 6)
    bk += bk % 2 == 0
    mask = cv2.GaussianBlur(mask, (bk, bk), 0)
    m3 = np.stack([mask] * 3, axis=-1)
    frame[y1:y2, x1:x2] = (
        roi.astype(np.float32) * (1.0 - m3) + blended.astype(np.float32) * m3
    ).astype(np.uint8)
    return frame


# ============================================================
# 源照质量门控 + 自动 GFPGAN 增强 (不合格照片自动修好, 存 _gfpgan.png 长期复用)
# ============================================================
def _imread_unicode(path):
    """中文路径安全读图 (Windows 下 cv2.imread 对中文路径返回 None)."""
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"  [WARN] 图像解码失败({path}): {e}")
        return None


def _imwrite_unicode(path, img):
    """中文路径安全写图."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
    return ok


# 源照质量阈值 (任一不达标 → 触发 GFPGAN 增强)
SRC_FACE_MIN_PX = 160      # 人脸 bbox 短边像素下限 (换脸 embedding 质量关键)
SRC_FACE_MIN_BLUR = 60     # 人脸区域拉普拉斯方差下限 (模糊度)
SRC_FACE_MIN_DET = 0.50    # insightface 检测置信度下限


def assess_source_quality(img, app):
    """评估换脸源照质量. 返回 (ok, reason, best_face).
    ok=True 合格可直接换脸; ok=False 建议先增强."""
    faces = _detect_with_fallback(app, img)
    if not faces:
        return False, "未检测到人脸", None
    best = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    fw = best.bbox[2] - best.bbox[0]
    fh = best.bbox[3] - best.bbox[1]
    face_short = min(fw, fh)
    x1, y1 = max(0, int(best.bbox[0])), max(0, int(best.bbox[1]))
    x2, y2 = int(best.bbox[2]), int(best.bbox[3])
    roi = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY) if x2 > x1 and y2 > y1 else None
    blur = float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi is not None and roi.size > 0 else 0.0
    if face_short < SRC_FACE_MIN_PX:
        return False, f"脸太小({face_short:.0f}px<{SRC_FACE_MIN_PX})", best
    if blur < SRC_FACE_MIN_BLUR:
        return False, f"脸模糊(清晰度{blur:.0f}<{SRC_FACE_MIN_BLUR})", best
    if best.det_score < SRC_FACE_MIN_DET:
        return False, f"置信度低({best.det_score:.2f}<{SRC_FACE_MIN_DET})", best
    return True, f"合格(脸{face_short:.0f}px,清晰度{blur:.0f})", best


def enhance_source_photo(img, model, cascade, device):
    """对源照全强度 GFPGAN 增强人脸 → 高清美颜照.
    换脸源照核心是人脸清晰度, GFPGAN(修脸+2x) 比通用 Real-ESRGAN(整图超分, 放大噪点) 更对口."""
    return gfpgan_restore_frame(img, model, cascade, device, strength=1.0)


def ensure_source_photo(source_path, coach_name, app=None, force=False, out_dir=None):
    """换脸源照质量门控 + 自动增强. 返回最终源照路径.

    - 文件名已含 _gfpgan → 已增强, 幂等返回 (除非 force).
    - 检测质量: 合格→原路; 不合格→GFPGAN 增强→存 {coach}_gfpgan.png→返回新路.
    - 生成的 _gfpgan.png 会被 find_coach_face 下次优先命中 → 长期复用, 零重复算力.
    """
    base = os.path.basename(source_path)
    if "_gfpgan" in base and not force:
        return source_path
    if out_dir is None:
        out_dir = os.path.dirname(source_path) or "tools"
    img = _imread_unicode(source_path)
    if img is None:
        print(f"  源照读取失败: {base}, 用原图")
        return source_path
    if app is None:
        app = get_face_analyser()
    ok, reason, _ = assess_source_quality(img, app)
    if ok and not force:
        print(f"  源照{reason}, 无需增强: {base}")
        return source_path
    print(f"  源照不合格({reason}), GFPGAN 增强中: {base}")
    model, cascade, device = _load_gfpgan()
    if model is None:
        print(f"  GFPGAN 不可用, 用原图")
        return source_path
    enhanced = enhance_source_photo(img, model, cascade, device)
    out_path = os.path.join(out_dir, f"{coach_name}_gfpgan.png")
    _imwrite_unicode(out_path, enhanced)
    # NOTE: 增强后不再用 blur 二次卡门 — GFPGAN 美颜照高频方差天然偏低 (皮肤光滑),
    # 拉普拉斯"清晰度"对美颜照不公 (实测增强前后均 ~23); GFPGANv1.4 输出即成熟修复脸, 直接信任复用.
    eh, ew = enhanced.shape[:2]
    print(f"  已 GFPGAN 增强 → {os.path.basename(out_path)} ({ew}x{eh}), 下次自动复用")
    return out_path


def _detect_with_fallback(app, img):
    """insightface 检测 + det_size 降级 (640→320).
    640 在某些角度/遮挡照片漏检 (实测 yanqing 短发仰角图),
    降级到 320 触发多尺度滑动窗口, 能救回 ~80% 的 640 漏检.
    返回 faces list (空 list = 都没检到).
    """
    faces = app.get(img)
    if faces:
        return faces
    # 降级: 临时把 det-size 改小
    original_size = getattr(app.det_model, "input_size", (640, 640))
    try:
        app.det_model.input_size = (320, 320)
        faces = app.get(img)
        if faces:
            print(f"  [降级] det_size=320 救回 {len(faces)} 张脸 (640 漏检)")
            return faces
    finally:
        app.det_model.input_size = original_size
    return []


def extract_face_embedding(app, image_path):
    """从源图片提取人脸特征"""
    img = _imread_unicode(image_path)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    faces = _detect_with_fallback(app, img)
    if not faces:
        raise ValueError(f"未检测到人脸: {image_path}")
    # 取面积最大的人脸
    best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    print(f"  源人脸: bbox={best.bbox.astype(int)}, det_score={best.det_score:.2f}")
    return best


def get_lead_bbox_from_pose(person, frame_w, frame_h, lead_orientation_threshold=0.10):
    """从 BlazePose 33 kp 算出领操人脸的 bbox + 朝向.

    Returns:
        (bbox, orientation): bbox=(x1,y1,x2,y2) 或 None; orientation in 'front'/'side'/'back'/'unknown'

    规则:
        - kp 0 = nose, kp 11/12 = 左右肩
        - 鼻子 conf < 0.3 → 'back'
        - 鼻子-肩膀中点偏移 > lead_orientation_threshold (归一化) → 'back'
        - bbox 大小 = max(肩宽 × 1.5, 160px) (保证脸完整在 ROI 内, 让 insightface 找到)
    """
    if not person or len(person) < 13:
        return None, "unknown"
    nose = person[0]
    ls = person[11]
    rs = person[12]

    # 背面: 鼻子不可见
    if nose[2] < 0.3:
        return None, "back"

    # 计算朝向
    if ls[2] > 0.3 and rs[2] > 0.3:
        scx = (ls[0] + rs[0]) / 2
        offset = abs(nose[0] - scx)
        orientation = "front" if offset < 0.03 else ("side" if offset < lead_orientation_threshold else "back")
    else:
        # 缺肩膀数据 → 假设正面
        orientation = "front"
        offset = 0

    if orientation == "back":
        return None, "back"

    # 算 bbox
    cx = nose[0] * frame_w
    cy = nose[1] * frame_h

    if ls[2] > 0.3 and rs[2] > 0.3:
        shoulder_w = abs(ls[0] - rs[0]) * frame_w
    else:
        shoulder_w = 100

    # ROI 大小: max(肩宽 × 1.5, 160px). 收紧避免远处小脸混入
    # ROI 大小: max(肩宽 × 1.5, 160px). 收紧避免远处小脸混入
    face_size = max(int(shoulder_w * 1.5), 160)
    half = face_size // 2
    x1 = max(0, int(cx - half))
    y1 = max(0, int(cy - half))
    x2 = min(frame_w, int(cx + half))
    y2 = min(frame_h, int(cy + half))

    return (x1, y1, x2, y2), orientation


def find_lead_person(keypoints_this_frame, frame_w, frame_h):
    """从 pose keypoints 中找领操人.

    策略: 优先 cx 接近 0.5 + bbox 较大的 (近景, 通常是领操人).
    bbox 估算: kp 11 (l_shoulder) 和 12 (r_shoulder) 之间的距离.

    Returns: person kp list (33 个) 或 None
    """
    if not keypoints_this_frame:
        return None
    candidates = []
    for person in keypoints_this_frame:
        valid = [k for k in person if len(k) > 2 and k[2] > 0.3]
        if len(valid) < 5:
            continue
        xs = [k[0] for k in valid[:13]]
        cx_norm = sum(xs) / len(xs)
        # 估算身体宽度 (用肩膀距离)
        if len(person) >= 13 and person[11][2] > 0.3 and person[12][2] > 0.3:
            body_w = abs(person[11][0] - person[12][0])
        else:
            # fallback: 用所有可见关键点 x 范围
            xs_all = [k[0] for k in valid]
            body_w = max(xs_all) - min(xs_all) if xs_all else 0
        candidates.append((cx_norm, body_w, person))

    if not candidates:
        return None

    # 评分: cx 居中度 (-|cx-0.5|) + 身体大小 (越大越可能是近景领操人)
    # 归一化身体大小: 除以最大
    max_body = max(c[1] for c in candidates) or 1
    scored = []
    for cx_norm, body_w, person in candidates:
        cx_score = 1 - abs(cx_norm - 0.5) * 4  # cx=0.5 得 1, cx=0.25 得 0
        size_score = body_w / max_body
        score = cx_score * 0.4 + size_score * 0.6  # 大小更重要
        scored.append((score, person))
    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


def swap_face(swapper, source_face, target_img, app, multi_src_faces=None,
              only_lead=True, min_face_area=0.001, lead_bbox=None,
              color_match_strength=0.8):
    """多脸换脸: 默认只换领操人 (面积最大的人脸)

    Args:
        swapper: inswapper 模型
        source_face: 默认源脸 (单脸模式)
        target_img: 当前帧
        app: face detector
        multi_src_faces: 多张源脸列表 (按 x 排序分配) — 优先于 source_face
        only_lead: True=只换最大脸 (领操人), False=换所有脸 (旧行为)
        min_face_area: 跳过小于此面积比 (相对画面) 的人脸, 默认 0.001 (0.1%)
            (健身视频全身镜头里领操人脸常只占 0.5-1%, 旧默认 0.02 会把领操人过滤掉)
        lead_bbox: (可选) 用 pose 关键点算出的领操人脸 bbox (x1,y1,x2,y2).
            若提供: 在 ROI 内做 swap (绕开 insightface 在全身/仰头/侧脸时的漏检).
            若 None: 用 app.get() 全图检测 (旧行为).
    """
    h, w = target_img.shape[:2]

    # === 分支 1: lead_bbox 已知 → 在 ROI 内 swap (pose-driven) ===
    if lead_bbox is not None:
        x1, y1, x2, y2 = lead_bbox
        x1 = max(0, min(w, x1)); x2 = max(0, min(w, x2))
        y1 = max(0, min(h, y1)); y2 = max(0, min(h, y2))
        if x2 - x1 < 20 or y2 - y1 < 20:
            return target_img
        roi0 = target_img[y1:y2, x1:x2].copy()
        rh0, rw0 = roi0.shape[:2]
        # 2026-06-29: 远景小脸修复. insightface det_size=640 只把大图缩到 640, 不会放大小图.
        # 健身全身镜头里 lead ROI 常仅 160px, 脸 ~40px → buffalo_l 漏检 (日志 "人脸:0帧",
        # swap 计数空涨但 inswapper 没真跑, 输出原帧未换). 上采样 ROI 到 ≥512 让脸到 ~128px
        # 可检可换, 完成后下采样贴回原位.
        UPSCALE_TARGET = 512
        upscaled = False
        roi = roi0
        if max(rh0, rw0) < UPSCALE_TARGET:
            roi = cv2.resize(roi0, (UPSCALE_TARGET, UPSCALE_TARGET), interpolation=cv2.INTER_CUBIC)
            upscaled = True
        roi_faces = app.get(roi)
        if not roi_faces:
            return target_img
        # 选 lead: 优先 cx 接近 ROI 中心的脸 (领操人), 否则最大脸.
        # 距离 ROI 中心越近越优先, 再按面积. 避免远处大婶被误选.
        rh, rw = roi.shape[:2]
        roi_cx = rw / 2
        roi_cy = rh / 2

        def lead_score(f):
            bbox = f.bbox
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            dist = ((cx - roi_cx) ** 2 + (cy - roi_cy) ** 2) ** 0.5
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            return (dist, -area, -f.det_score)  # 距离小 > 面积大 > 置信度高

        roi_face = min(roi_faces, key=lead_score)
        # 备份原【脸部区域】用于色温迁移 (2026-06-28: 改用纯脸框, 不用整个 ROI——
        # 整个 ROI 含黑头发/暗背景会拉低均值, 把换上的脸拉暗成"深色面膜")
        orig_face_roi = None
        if color_match_strength > 0:
            fx1, fy1, fx2, fy2 = [int(v) for v in roi_face.bbox]
            fx1 = max(0, fx1); fy1 = max(0, fy1)
            fx2 = min(roi.shape[1], fx2); fy2 = min(roi.shape[0], fy2)
            if fx2 - fx1 > 5 and fy2 - fy1 > 5:
                orig_face_roi = roi[fy1:fy2, fx1:fx2].copy()
        try:
            roi_swapped = swapper.get(roi, roi_face, source_face, paste_back=True)
            # 色温迁移: 把换后脸肤色拉回原场景
            if color_match_strength > 0 and orig_face_roi is not None:
                roi_swapped = color_match_face(
                    roi_swapped, roi_face.bbox, orig_face_roi, color_match_strength)
            if upscaled:
                roi_swapped = cv2.resize(roi_swapped, (rw0, rh0), interpolation=cv2.INTER_AREA)
            target_img[y1:y2, x1:x2] = roi_swapped
        except Exception:
            pass
        return target_img

    # === 分支 2: 全图检测 (旧逻辑) ===
    faces = app.get(target_img)
    if not faces:
        return target_img
    img_area = h * w

    # 过滤: 置信度 + 面积
    candidates = []
    for f in faces:
        if f.det_score < 0.3:
            continue
        bbox = f.bbox
        face_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        if face_area / img_area < min_face_area:
            continue
        candidates.append(f)
    if not candidates:
        return target_img

    if only_lead:
        # 只换领操人: 取面积最大的 1 张脸
        faces_sorted = [max(candidates, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))]
    else:
        # 多脸模式: 按 x 排序 (左→右)
        faces_sorted = sorted(candidates, key=lambda f: f.bbox[0])
    # 多脸模式: 按位置分配
    src_faces_list = multi_src_faces if multi_src_faces else [source_face]
    for fi_idx, face in enumerate(faces_sorted):
        # 第 0 张脸 → src_faces[0], 第 1 张 → src_faces[1], ...
        face_idx = min(fi_idx, len(src_faces_list) - 1)
        try:
            target_img = swapper.get(target_img, face, src_faces_list[face_idx], paste_back=True)
        except Exception:
            pass
    return target_img


def swap_background(frame, mask_fg, bg_img):
    """在前景 mask 上换背景 (简单 alpha blending, 不做运镜匹配)

    Args:
        frame: 原帧
        mask_fg: 前景 mask (0~1 浮点, 2D 或 3D)
        bg_img: 背景图 (已 resize 到帧尺寸)
    """
    if mask_fg is None or bg_img is None:
        return frame
    # 保证 mask 是 (h, w) 2D
    if mask_fg.ndim == 3:
        mask_fg = mask_fg[:, :, 0] if mask_fg.shape[2] == 1 else mask_fg.mean(axis=2)
    # 软化
    mask_soft = cv2.GaussianBlur(mask_fg, (11, 11), 5)
    # 限制 0~1
    mask_soft = np.clip(mask_soft, 0, 1)
    # 3 通道 for broadcast
    m3 = mask_soft[:, :, np.newaxis].astype(np.float32)
    out = (frame.astype(np.float32) * m3 + bg_img.astype(np.float32) * (1 - m3)).astype(np.uint8)
    return out


def get_bg_mask_simple(frame, faces):
    """简单背景 mask: 用脸部位置向外扩展 (估计人体 bounding box)

    不调用 SAM2, 适合无 SAM2 环境或快速预览.
    """
    if not faces:
        return None
    h, w = frame.shape[:2]
    # 找到最小/最大脸位置
    xs_min = min(f.bbox[0] for f in faces)
    ys_min = min(f.bbox[1] for f in faces)
    xs_max = max(f.bbox[2] for f in faces)
    ys_max = max(f.bbox[3] for f in faces)
    # 扩展: 上下扩展 2x face_h, 水平扩展 1x face_w
    face_h = ys_max - ys_min
    face_w = xs_max - xs_min
    y1 = max(0, int(ys_min - face_h * 2))
    y2 = min(h, int(ys_max + face_h * 1.5))
    x1 = max(0, int(xs_min - face_w * 0.5))
    x2 = min(w, int(xs_max + face_w * 0.5))
    mask = np.zeros((h, w), dtype=np.float32)
    mask[y1:y2, x1:x2] = 1.0
    return mask


def color_match_face(frame, bbox, ref_roi, strength=0.8):
    """把 frame[bbox] 换脸区域的色彩向 ref_roi(换脸前的原肤色)迁移, 消除偏色/过白.

    用 LAB 空间 Reinhard 迁移 (匹配均值+方差), 只改颜色不改五官/纹理 → 保留换脸身份,
    但肤色/光感回归原场景. strength: 0=不迁移, 1=完全用原肤色. 默认 0.8.
    2026-06-28: 关键在 ref 必须是【纯脸框】(换脸前原脸, 不含黑头发/背景)——
    旧版 ref=整个ROI 时黑头发拉低 L 均值, 全迁会把脸拉暗成"深色面膜".
    ref 改脸框后 L 均值=真实原肤色, L/a/b 全迁安全且必要 (只迁 a/b 不迁 L 会
    "白皮+中等色调"发灰发青). 见 swap_face() 中 orig_face_roi 的取法.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = y2 - y1, x2 - x1
    if h < 10 or w < 10 or ref_roi is None:
        return frame
    tgt = frame[y1:y2, x1:x2]
    ref = cv2.resize(ref_roi, (w, h))
    tgt_lab = cv2.cvtColor(tgt, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    # 2026-06-28: ref 已是纯脸框(换脸前原脸), L 均值=真实原肤色, L/a/b 全迁安全且必要
    out = tgt_lab.copy()
    for c in range(3):
        tm, ts = tgt_lab[..., c].mean(), tgt_lab[..., c].std() + 1e-6
        rm, rs = ref_lab[..., c].mean(), ref_lab[..., c].std() + 1e-6
        migrated = (tgt_lab[..., c] - tm) / ts * rs + rm
        out[..., c] = tgt_lab[..., c] * (1 - strength) + migrated * strength
    out_bgr = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    # 羽化 mask (中心全迁移, 边缘渐变回换脸结果)
    mask = np.zeros((h, w), dtype=np.float32)
    im_ = max(5, min(h, w) // 12)
    cv2.rectangle(mask, (im_, im_), (w - im_, h - im_), 1.0, -1)
    bk = max(3, min(h, w) // 6); bk += bk % 2 == 0
    mask = cv2.GaussianBlur(mask, (bk, bk), 0)
    m3 = np.stack([mask] * 3, axis=-1)
    frame[y1:y2, x1:x2] = (
        tgt.astype(np.float32) * (1.0 - m3) + out_bgr.astype(np.float32) * m3
    ).astype(np.uint8)
    return frame


def process_video(source_path, target_path, output_path, max_frames=0, every_n=1,
                  multi_sources=None, bg_path=None, only_lead=True, gfpgan_strength=0.5,
                  min_face_area=0.001, color_match_strength=0.8,
                  keypoints_file=None, lead_orientation_threshold=0.10):
    """逐帧处理视频换脸 (支持多脸 + 简单换背景 + GFPGAN 美颜修复)

    Args:
        source_path: 默认源脸 (单脸模式)
        target_path: 目标视频
        output_path: 输出
        max_frames: 限帧
        every_n: 已废弃 (历史参数, 保留兼容). 现在每帧都换脸, 避免奇偶帧交替闪烁.
        multi_sources: 多张源脸路径 (按 x 排序分配给视频里从左到右的脸)
        bg_path: 背景图路径 (None=不换背景)
        gfpgan_strength: 换脸后 GFPGAN 修复强度 0~1 (0=关; 默认 0.5). inswapper 输出仅
            128px, 必须再过 GFPGAN 才有美颜质感, 否则换出来的脸偏糊/无美颜.
        min_face_area: 脸占画面面积比下限, 小于此值跳过 (防远处群众脸误换).
            默认 0.001 (0.1%) — 宽景/全身镜头里领操人脸常只占 0.2%, 旧默认 0.02 会把
            所有脸过滤掉导致换脸完全不执行!
        color_match_strength: 换脸后把脸肤色迁移回原场景肤色的强度 0~1 (0=关; 默认 0.8).
            消除换脸偏色/过白发死 (源照偏冷时换上去会发青). 只改颜色, 保留换脸五官.
    """
    print(f"加载模型...")
    app = get_face_analyser()
    swapper = get_swapper()
    source_face = extract_face_embedding(app, source_path)

    # 多脸模式: 预提取所有源脸
    multi_src_faces = None
    if multi_sources:
        multi_src_faces = []
        for fp in multi_sources:
            if not os.path.exists(fp):
                print(f"  [警告] 多脸源不存在: {fp}, 跳过")
                continue
            sf = extract_face_embedding(app, fp)
            multi_src_faces.append(sf)
        if multi_src_faces:
            print(f"  多脸源加载: {len(multi_src_faces)} 张")

    # 背景
    bg_img = None
    if bg_path:
        # OpenCV 在 Windows 上不识别中文路径, 复制到短路径临时读
        import shutil
        if any(ord(c) > 127 for c in bg_path):
            tmp_bg = os.path.join(os.path.dirname(target_path), "_bg_tmp.jpg")
            shutil.copy(bg_path, tmp_bg)
            bg_img = cv2.imread(tmp_bg)
            os.remove(tmp_bg)
        else:
            bg_img = cv2.imread(bg_path)
        if bg_img is None:
            print(f"  [警告] 背景图读不到: {bg_path}")
        else:
            print(f"  背景: {bg_path}")

    # GFPGAN 美颜修复 (换脸后增强脸区域 — inswapper_128 输出仅 128px, 必须补这一步)
    gfpgan_model = gfpgan_cascade = gfpgan_device = None
    if gfpgan_strength and gfpgan_strength > 0:
        gfpgan_model, gfpgan_cascade, gfpgan_device = _load_gfpgan()
        if gfpgan_model is not None:
            print(f"  GFPGAN 美颜修复已启用 (strength={gfpgan_strength})")
        else:
            print(f"  [警告] GFPGAN 权重未找到, 跳过美颜修复 (换出来的脸会是 128px 原生态)")

    cap = cv2.VideoCapture(target_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(3))
    h = int(cap.get(4))

    if max_frames > 0:
        total = min(total, max_frames)

    if bg_img is not None:
        bg_img = cv2.resize(bg_img, (w, h))
    print(f"处理 {total} 帧 @ {fps:.1f}fps, 每帧换脸 + GFPGAN 修复...")

    # 管道输出到F盘临时文件，不写PNG序列
    tmp_vid = os.path.join(os.path.dirname(output_path), "_tmp_vid.mp4")
    # Clean up old tmp file
    if os.path.exists(tmp_vid):
        try:
            os.remove(tmp_vid)
        except OSError:
            pass

    ffmpeg_cmd = [
        FFMPEG, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-an",
        tmp_vid
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)

    fi = 0
    out_fi = 0
    swap_count = 0
    face_count = 0
    skip_back = 0
    skip_no_pose = 0

    # 加载 keypoints (若提供)
    keypoints = None
    if keypoints_file and os.path.exists(keypoints_file):
        try:
            with open(keypoints_file, encoding="utf-8") as f:
                kp_data = json.load(f)
            keypoints = kp_data.get("keypoints", kp_data)
            # 2026-06-28: 空 keypoints (缓存毒化 / pose 失败) 必须显式降级 + 警告,
            # 否则空 dict is not None 仍走 pose 分支但每帧领操人=None → 静默 fallback 全图检测,
            # 多人广角画面会漏检/误选领操人 (本 bug 曾导致艳青1 "看不出换脸")
            non_empty = sum(1 for v in keypoints.values() if v) if isinstance(keypoints, dict) else 0
            if non_empty == 0:
                print(f"  [警告] keypoints 为空 ({os.path.basename(keypoints_file)}), 降级 fallback 全图检测 — 请重跑 pose")
                keypoints = None
            else:
                print(f"  pose keypoints 加载: {keypoints_file} ({len(keypoints)} 帧, 非空 {non_empty})")
        except Exception as e:
            print(f"  [警告] keypoints 加载失败: {e}")
            keypoints = None

    while fi < total:
        ret, frame = cap.read()
        if not ret:
            break

        # === 优先: 用 pose 定位领操人 + 朝向判断 ===
        lead_bbox = None
        use_pose = False
        if keypoints is not None:
            persons = keypoints.get(str(fi), [])
            lead_person = find_lead_person(persons, w, h)
            if lead_person is not None:
                bbox, orientation = get_lead_bbox_from_pose(lead_person, w, h,
                                                            lead_orientation_threshold)
                if orientation == "back":
                    skip_back += 1
                    # 背面: 直接写原帧, 不换脸
                    proc.stdin.write(frame.tobytes())
                    out_fi += 1
                    fi += 1
                    continue
                if bbox is not None:
                    lead_bbox = bbox
                    use_pose = True

        if not use_pose:
            # fallback: 老逻辑, 用 app.get() 选最大脸
            faces_before = app.get(frame)
            if not faces_before:
                skip_no_pose += 1
                proc.stdin.write(frame.tobytes())
                out_fi += 1
                fi += 1
                continue
            face_count += 1
            lead = max(faces_before, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            lx1, ly1, lx2, ly2 = [int(v) for v in lead.bbox]
            orig_face_roi = frame[ly1:ly2, lx1:lx2].copy()

        # 换脸 (pose 模式传 lead_bbox; 旧模式不传)
        frame = swap_face(swapper, source_face, frame, app, multi_src_faces,
                          only_lead=only_lead, min_face_area=min_face_area,
                          lead_bbox=lead_bbox,
                          color_match_strength=color_match_strength)
        swap_count += 1

        # GFPGAN 美颜修复 (默认关闭)
        if gfpgan_model is not None:
            frame = gfpgan_restore_frame(frame, gfpgan_model, gfpgan_cascade,
                                         gfpgan_device, gfpgan_strength)
        # 色温迁移: 换脸+GFPGAN 后脸会偏色/发白, 把肤色拉回原场景 → 自然不吓人
        if color_match_strength > 0 and 'orig_face_roi' in dir() and orig_face_roi.size > 0:
            frame = color_match_face(frame, lead.bbox, orig_face_roi,
                                     color_match_strength)
        # 换背景 (在脸位置周围简单 mask)
        if bg_img is not None and 'faces_before' in dir():
            mask = get_bg_mask_simple(frame, faces_before)
            if mask is not None:
                frame = swap_background(frame, mask, bg_img)

        try:
            proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            print(f"    pipe broken at frame {fi}")
            break
        out_fi += 1
        fi += 1

        if fi % 50 == 0:
            print(f"  进度: {fi}/{total} ({fi*100//total}%) swap:{swap_count} back:{skip_back} 人脸:{face_count}帧", flush=True)

    cap.release()
    try:
        proc.stdin.close()
    except Exception:
        pass
    retcode = proc.wait()
    if retcode != 0:
        print(f"    ffmpeg pipe failed: rc={retcode}")
    print(f"  换脸完成: {out_fi} 帧 (swap={swap_count}, 背面跳过={skip_back}, 无pose={skip_no_pose}), 混入音频...")

    # 混入原音频（如果 target 无音频流则跳过，直接复制无声视频）
    r = subprocess.run([
        FFMPEG, "-y", "-i", tmp_vid, "-i", target_path,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0?",  # ?: 1:a:0 不存在时不报错
        "-shortest",
        output_path
    ], check=False, capture_output=True, timeout=120)
    if r.returncode != 0 or not os.path.exists(output_path):
        # 混音失败（target 无音频流/超时），直接复制无声视频
        print(f"    混音失败 (可能无音频流), 直接复制无声视频")
        r2 = subprocess.run([
            FFMPEG, "-y", "-i", tmp_vid,
            "-c:v", "copy", "-an",
            output_path
        ], check=False, capture_output=True, timeout=60)
        if r2.returncode != 0:
            print(f"    复制视频也失败: {r2.stderr[-200:]}")

    # 只在 output 成功生成后才删 tmp_vid（之前无论成功与否都删导致数据丢失）
    if os.path.exists(output_path):
        os.remove(tmp_vid)
        print(f"  输出: {output_path}")
    else:
        # 保留 tmp_vid 供上层 stage fallback 使用
        print(f"  换脸视频未生成, 保留临时文件: {tmp_vid}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="教练换脸工具 (支持多脸 + 简单换背景)")
    parser.add_argument("--source", required=True, help="默认源脸 (单脸模式)")
    parser.add_argument("--target", required=True, help="目标视频路径")
    parser.add_argument("--output", required=True, help="输出视频路径")
    parser.add_argument("--multi-source", nargs="+", default=None,
                        help="多张源脸 (按 x 排序分配给视频里从左到右的脸, 例: --multi-source a.jpg b.jpg)")
    parser.add_argument("--bg", default=None, help="背景图路径 (替换脸部周围背景)")
    parser.add_argument("--max-frames", type=int, default=300, help="最大处理帧数(默认300)")
    parser.add_argument("--every-n", type=int, default=1, help="(已废弃)历史参数, 现每帧换脸")
    parser.add_argument("--gfpgan-strength", type=float, default=0.5,
                        help="换脸后GFPGAN美颜修复强度0~1(默认0.5, 0=关)")
    parser.add_argument("--min-face-area", type=float, default=0.001,
                        help="脸占画面面积比下限(默认0.001=0.1%%; 宽景全身镜头别调高, 否则换脸不执行)")
    parser.add_argument("--color-match-strength", type=float, default=0.8,
                        help="换脸后肤色迁移回原场景的强度0~1(默认0.8; 0=关, 消除偏色/过白)")
    args = parser.parse_args()

    process_video(args.source, args.target, args.output,
                  max_frames=args.max_frames, every_n=args.every_n,
                  multi_sources=args.multi_source, bg_path=args.bg,
                  gfpgan_strength=args.gfpgan_strength,
                  min_face_area=args.min_face_area,
                  color_match_strength=args.color_match_strength)
