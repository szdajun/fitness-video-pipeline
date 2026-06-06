#!/usr/bin/env python3
"""SAM2 视频背景替换 — 首帧框选人像 → 自动追踪全片 → 换背景+换脸
用法: python tools/sam2_bg_swap.py --target 视频 --bg 背景 --face 美颜照 --output 输出
"""
import cv2, numpy as np, os, sys, subprocess, argparse, gc, torch

FFMPEG = "C:/Users/18091/ffmpeg/ffmpeg.exe"
SAM2_MODEL = "F:/wkspace/sam2/checkpoints/sam2_hiera_small.pt"
SAM2_CFG = "sam2_hiera_s.yaml"


def load_sam2():
    from sam2.build_sam import build_sam2_video_predictor
    from sam2.sam2_video_predictor import SAM2VideoPredictor
    cfg_path = f"F:/wkspace/sam2/sam2/configs/sam2/{SAM2_CFG}"
    predictor = build_sam2_video_predictor(cfg_path, SAM2_MODEL, device="cuda")
    return predictor


def process(target_path, bg_path, face_path, output_path, max_frames=0):
    import insightface, tempfile, os as _os

    # 读视频元数据
    cap = cv2.VideoCapture(target_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = int(cap.get(3)), int(cap.get(4))

    # 全帧加载 (短视频 < 1000 帧; 长视频限制 300 帧)
    if total <= 1000:
        limit = total
        step = 1
    else:
        limit = min(300, int(8000 / frame_mem_mb))
        step = max(1, total // limit)
    print(f"  全帧加载 {limit}/{total} 帧 (步长 {step}, 约 {limit*frame_mem_mb:.0f}MB)")

    # 抽取帧到临时目录 (SAM2 需要目录输入)
    tmpdir = tempfile.mkdtemp(prefix="sam2_frames_")
    all_frames = []
    frame_indices = []
    for fi in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret: break
        all_frames.append(frame)
        frame_indices.append(fi)
        cv2.imwrite(_os.path.join(tmpdir, f"{len(all_frames)-1:05d}.jpg"), frame)
        if len(all_frames) >= limit:
            break
    cap.release()
    total = len(all_frames)
    if total == 0:
        print("  无帧可处理")
        return

    print("加载 SAM2...")
    predictor = load_sam2()

    print("加载换脸模型...")
    detector = insightface.app.FaceAnalysis(name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    detector.prepare(ctx_id=0, det_size=(640, 640))
    swapper = insightface.model_zoo.get_model(
        _os.path.expanduser("~/.insightface/models/inswapper_128.onnx"),
        providers=["CPUExecutionProvider"])

    src_img = cv2.imread(face_path)
    src_faces = detector.get(src_img)
    if not src_faces:
        print("  美颜照未检测到人脸")
        return
    source_face = max(src_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    print(f"  源脸: {source_face.det_score:.2f}")

    bg_img = cv2.resize(cv2.imread(bg_path), (w, h))

    # SAM2 初始化: 传帧目录
    print("SAM2 推理...")
    inference_state = predictor.init_state(video_path=tmpdir, offload_video_to_cpu=True)
    predictor.reset_state(inference_state)

    # 首帧人脸检测 (搜索前 5 帧)
    face_frame_idx = 0
    best_face = None
    for search_i in range(min(5, total)):
        faces = detector.get(all_frames[search_i])
        if faces:
            best_face = max(faces, key=lambda f: f.det_score)
            if best_face.det_score > 0.3:
                face_frame_idx = search_i
                print(f"  检测到人脸: 帧{frame_indices[search_i]} (score={best_face.det_score:.2f})")
                break
    if best_face is None:
        print("  未检测到人脸, 用画面中心 + 跳过换脸")
        # 无人脸时: 画面中心点作为前景锚点, 只做背景替换不做换脸
        cx, cy = w // 2, h // 3
        head_h, head_w = h // 6, w // 6
        x1, y1, x2, y2 = cx - head_w//2, cy - head_h, cx + head_w//2, cy
        skip_swap = True
    else:
        x1, y1, x2, y2 = best_face.bbox.astype(int)
        skip_swap = False
    head_h = y2 - y1
    head_w = x2 - x1
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    body_cx = cx
    body_cy = min(y1 + head_h * 4, h - 10)
    points = np.array([[cx, cy], [body_cx, body_cy], [body_cx, h-5]], dtype=np.float32)
    labels = np.array([1, 1, 0], dtype=np.int32)

    # 从检测到人脸的帧开始传播
    for fi in range(face_frame_idx):
        predictor.propagate_in_video(inference_state, start_frame_idx=fi, max_frame_num_to_track=1)

    _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
        inference_state=inference_state, frame_idx=face_frame_idx,
        obj_id=0, points=points, labels=labels)

    # 视频遮罩传播 + 换背景 + 换脸
    tmp_vid = output_path.replace(".mp4", "_tmp.mp4")
    proc = subprocess.Popen([
        FFMPEG, "-y", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "pipe:0", "-c:v", "libx264", "-preset", "fast",
        "-crf", "20", "-pix_fmt", "yuv420p", "-an", tmp_vid
    ], stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    mask_cache = {}
    for fi in range(total):
        frame = all_frames[fi].copy()
        out_frame_idx = fi

        if fi not in mask_cache:
            try:
                for out_frame_idx2, out_obj_ids2, out_mask_logits2 in predictor.propagate_in_video(
                    inference_state, start_frame_idx=max(0, fi-1), max_frame_num_to_track=1):
                    if out_frame_idx2 >= fi:
                        ml = out_mask_logits2[0]
                        mask_cache[fi] = (torch.sigmoid(ml) > 0.5).cpu().numpy().squeeze(0).astype(np.float32)
                        mask_cache[fi] = cv2.GaussianBlur(mask_cache[fi], (11, 11), 5)[:, :, np.newaxis]
                        break
            except:
                mask_cache[fi] = mask_cache.get(fi-1, np.ones((h, w, 1), dtype=np.float32))

        mask = mask_cache.get(fi, np.ones((h, w, 1), dtype=np.float32))
        frame = (frame * mask + bg_img * (1 - mask)).astype(np.uint8)

        # 换脸 (无人脸时跳过)
        if not skip_swap:
            faces_i = detector.get(frame)
            if faces_i:
                best_i = max(faces_i, key=lambda f: f.det_score)
                if best_i.det_score > 0.3:
                    try: frame = swapper.get(frame, best_i, source_face, paste_back=True)
                    except: pass

        proc.stdin.write(frame.tobytes())
        if fi % 30 == 0: print(f"  {fi}/{total}")

    proc.stdin.close()
    proc.wait()
    _cleanup(tmpdir)
    del predictor; gc.collect(); torch.cuda.empty_cache()

    subprocess.run([FFMPEG, "-y", "-i", tmp_vid, "-i", target_path,
        "-c:v", "copy", "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0", "-shortest", output_path],
        check=True, capture_output=True, timeout=60)
    _os.remove(tmp_vid)
    print(f"  完成: {output_path}")


def _cleanup(d):
    import shutil
    try: shutil.rmtree(d, ignore_errors=True)
    except: pass


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True, dest="target_path")
    p.add_argument("--bg", required=True, dest="bg_path")
    p.add_argument("--face", required=True, dest="face_path")
    p.add_argument("--output", required=True, dest="output_path")
    p.add_argument("--max-frames", type=int, default=0)
    a = p.parse_args()
    process(a.target_path, a.bg_path, a.face_path, a.output_path, a.max_frames)


if __name__ == "__main__":
    main()
