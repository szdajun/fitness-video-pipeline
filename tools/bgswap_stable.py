#!/usr/bin/env python3
"""SAM2 背景替换（运镜匹配版）
1. SAM2 抠前景 → 排除人体干扰
2. 背景区 ORB 特征 + RANSAC 单应性 → 纯摄像机运动
3. 累积变换驱动静态背景 → 人景同步不滑

用法: python tools/bgswap_stable.py --target 视频 --bg 背景 --face 美颜照 --output 输出
"""
import cv2, numpy as np, os, subprocess, argparse, gc, torch, tempfile, shutil

FFMPEG = "C:/Users/18091/ffmpeg/ffmpeg.exe"
SAM2_M = "F:/wkspace/sam2/checkpoints/sam2_hiera_small.pt"
SAM2_C = "F:/wkspace/sam2/sam2/configs/sam2/sam2_hiera_s.yaml"


def estimate_camera_motion(frames, fg_masks):
    """在背景区域检测 ORB 特征 → RANSAC 单应性 → 累积摄像机运动矩阵"""
    if len(frames) < 2:
        return [np.eye(3, dtype=np.float32)] * len(frames)

    orb = cv2.ORB_create(2000)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    transforms = [np.eye(3, dtype=np.float32)]

    prev_bg_mask = 1 - (fg_masks[0] > 0.5).astype(np.uint8)
    prev_kp, prev_des = orb.detectAndCompute(frames[0], prev_bg_mask)

    for i in range(1, len(frames)):
        bg_mask = 1 - (fg_masks[i] > 0.5).astype(np.uint8)
        kp, des = orb.detectAndCompute(frames[i], bg_mask)

        if (prev_des is not None and des is not None
                and len(prev_kp) > 8 and len(kp) > 8):
            matches = matcher.match(prev_des, des)
            if len(matches) > 12:
                src_pts = np.float32([prev_kp[m.queryIdx].pt for m in matches])
                dst_pts = np.float32([kp[m.trainIdx].pt for m in matches])
                H, inliers = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
                if H is not None and inliers is not None and inliers.sum() > 8:
                    transforms.append(H @ transforms[-1])
                else:
                    transforms.append(transforms[-1])
            else:
                transforms.append(transforms[-1])
        else:
            transforms.append(transforms[-1])

        prev_des, prev_kp = des, kp
        if i % 100 == 0:
            print(f"  运动分析: {i}/{len(frames)}")

    return transforms


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--bg", required=True)
    p.add_argument("--face", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    import insightface
    from sam2.build_sam import build_sam2_video_predictor

    print("加载模型...")
    predictor = build_sam2_video_predictor(SAM2_C, SAM2_M, device="cuda")
    detector = insightface.app.FaceAnalysis(name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    detector.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model(
        os.path.expanduser("~/.insightface/models/inswapper_128.onnx"))
    # GPU 优先，自动回退 CPU
    src_face = max(detector.get(cv2.imread(args.face)),
                   key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

    # 读帧 (内存保护: 限制帧数 + 步长采样)
    cap = cv2.VideoCapture(args.target)
    fps, total = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4))
    frame_mem_mb = (w * h * 3) / 1024 / 1024
    # 限制 300 帧 (JPEG 临时目录时 SAM2 全量加载, 防 OOM)
    limit = min(300, total)
    step = max(1, total // limit) if total > limit else 1
    frames = []
    frame_indices = []
    for fi in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, f = cap.read()
        if not ret: break
        frames.append(f)
        frame_indices.append(fi)
        if len(frames) >= limit: break
    cap.release()
    total = len(frames)
    print(f"  全帧加载 {total} 帧 (步长 {step}, 约 {total*frame_mem_mb:.0f}MB)")

    # ── 1. SAM2 前景分割 ──
    print("SAM2 抠像...")
    tmp_dir = tempfile.mkdtemp(prefix="bgs_")
    for i, f in enumerate(frames):
        cv2.imwrite(f"{tmp_dir}/{i:06d}.jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 92])

    state = predictor.init_state(video_path=tmp_dir, offload_video_to_cpu=True)
    predictor.reset_state(state)

    # 三点锚定: 头(+)、胯(+)、脚(-) 覆盖全身
    f0_faces = detector.get(frames[0])
    skip_swap = False
    if f0_faces:
        b = max(f0_faces, key=lambda x: x.det_score).bbox.astype(int)
        cx = (b[0] + b[2]) // 2
        cy = (b[1] + b[3]) // 2  # 脸中心
        print(f"  检测到人脸: score={max(f0_faces,key=lambda x:x.det_score).det_score:.2f}")
    else:
        for si in range(1, min(5, len(frames))):
            ff = detector.get(frames[si])
            if ff:
                b = max(ff, key=lambda x: x.det_score).bbox.astype(int)
                cx = (b[0] + b[2]) // 2
                cy = (b[1] + b[3]) // 2
                skip_swap = False
                print(f"  检测到人脸(帧{si}): score={max(ff,key=lambda x:x.det_score).det_score:.2f}")
                break
        else:
            print("  未检测到人脸, 用画面中轴 + 跳过换脸")
            cx, cy = w // 2, h // 4
            skip_swap = True
    # 三点覆盖全身: 头(h*0.15) + 胯(h*0.55) + 脚底(h-5,负标签=地面)
    head_y = max(10, int(h * 0.15))
    hip_y = int(h * 0.55)
    foot_y = h - 5
    predictor.add_new_points_or_box(
        inference_state=state, frame_idx=0, obj_id=0,
        points=np.array([[cx, head_y],      # 头顶区域(+)
                         [cx, hip_y],       # 胯部(+)
                         [cx, foot_y]],     # 脚底(-, 地面)
        dtype=np.float32),
        labels=np.array([1, 1, 0], dtype=np.int32))

    masks = []
    for _out_fi, _out_ids, out_logits in predictor.propagate_in_video(state):
        masks.append((torch.sigmoid(out_logits[0]) > 0.5)
                     .cpu().numpy().squeeze(0).astype(np.float32))
    shutil.rmtree(tmp_dir)

    # ── 2. 背景运动估计（mask 排除人体） ──
    print("分析背景运动...")
    matrices = estimate_camera_motion(frames, masks)

    # ── 3. 合成 ──
    print("合成...")
    bg_orig = cv2.resize(cv2.imread(args.bg), (w, h))
    tmp_vid = args.output.replace(".mp4", "_tmp.mp4")
    proc = subprocess.Popen([
        FFMPEG, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "pipe:0", "-c:v", "libx264", "-preset", "fast",
        "-crf", "20", "-pix_fmt", "yuv420p", "-an", tmp_vid
    ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    for fi in range(total):
        frame = frames[fi].copy()
        mask = masks[min(fi, len(masks) - 1)]
        H = matrices[min(fi, len(matrices) - 1)]

        bg_moved = cv2.warpPerspective(bg_orig, H, (w, h),
                                       borderMode=cv2.BORDER_REPLICATE)

        if mask.sum() > 100:
            mask_soft = cv2.GaussianBlur(mask, (11, 11), 5)[:, :, np.newaxis]
            frame = (frame * mask_soft + bg_moved * (1 - mask_soft)).astype(np.uint8)

        # 换脸
        faces = detector.get(frame)
        if faces:
            best = max(faces, key=lambda f: f.det_score)
            if best.det_score > 0.3:
                try:
                    frame = swapper.get(frame, best, src_face, paste_back=True)
                except Exception:
                    pass

        proc.stdin.write(frame.tobytes())
        if fi % 100 == 0:
            print(f"  {fi}/{total}")

    proc.stdin.close()
    proc.wait()

    # 混音
    subprocess.run([FFMPEG, "-y", "-i", tmp_vid, "-i", args.target,
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
                    "-map", "0:v:0", "-map", "1:a:0", "-shortest", args.output],
                   check=True, capture_output=True, timeout=120)
    os.remove(tmp_vid)
    del predictor, detector, swapper
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  完成: {args.output}")


if __name__ == "__main__":
    main()
