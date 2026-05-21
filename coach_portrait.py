"""
Coach Portrait Extractor + Enhancer
1. Extract best face frames from fitness videos
2. GFPGAN face enhancement
3. Output high-quality coach portraits
"""

import os, sys, cv2, time, argparse, glob
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "coach_portraits")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_portrait(video_path, coach_name=None, sample_every=3, top_n=3):
    """
    Sample video frames, find the clearest face shots, enhance them.
    Returns list of output paths.
    """
    if coach_name is None:
        base = os.path.splitext(os.path.basename(video_path))[0]
    else:
        base = coach_name

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total / fps if fps > 0 else 0

    print(f"[Portrait] {os.path.basename(video_path)}: {total}frames, {duration:.0f}s")

    # Use YOLOv8-pose (already in project) - works better for distant fitness shots
    import torch
    from ultralytics import YOLO
    yolo_path = os.path.join(os.path.dirname(__file__), "yolov8n-pose.pt")
    model = YOLO(yolo_path)
    if torch.cuda.is_available():
        model.to('cuda')

    candidates = []
    for i in range(0, total, sample_every * int(max(fps, 1))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, verbose=False)
        num_persons = 0
        for r in results:
            if r.boxes is None or len(r.boxes) == 0: continue
            num_persons += len(r.boxes)
            boxes = r.boxes
            if boxes is None or len(boxes) == 0: continue
            # Pick largest person box
            areas = [(b.xyxy[0][2]-b.xyxy[0][0])*(b.xyxy[0][3]-b.xyxy[0][1]) for b in boxes.cpu()]
            best = np.argmax(areas)
            boxes_cpu = boxes.cpu()
            box = boxes_cpu[best].xyxy[0].numpy()
            x1,y1,x2,y2 = box.astype(int)
            person_h = y2-y1
            if person_h < 200: continue  # accept smaller people (wide shots)

            # Face region = upper 1/3 of person box
            face_y2 = y1 + int(person_h * 0.3)
            fx1 = x1 + int((x2-x1)*0.25)
            fx2 = x2 - int((x2-x1)*0.25)
            face_region = frame[y1:face_y2, fx1:fx2]
            if face_region.size < 1000: continue

            ratio = (x2-x1)*(y2-y1) / (frame.shape[0]*frame.shape[1])
            score = ratio * float(boxes_cpu[best].conf[0]) * 100
            if ratio > 0.001:  # very permissive for wide shots
                candidates.append((score, i, face_region, None))

    cap.release()
    print(f"  Found {len(candidates)} face frames (>{len(candidates)//2 if candidates else 0} good)")

    if not candidates:
        print("  No suitable face detected - skip")
        return []

    # Pick top N by score
    candidates.sort(key=lambda x: -x[0])
    best = candidates[:top_n]

    # Enhance each cropped face
    results = []
    for rank, (score, frame_idx, cropped, _) in enumerate(best):
        # cropped is already the face region from YOLO detection

        # Save raw crop
        raw_path = os.path.join(OUTPUT_DIR, f"{base}_{rank+1}_raw.jpg")
        cv2.imwrite(raw_path, cropped)

        # GFPGAN enhance
        enhanced = enhance_face(cropped)
        out_path = os.path.join(OUTPUT_DIR, f"{base}_{rank+1}.jpg")
        if enhanced is not None:
            cv2.imwrite(out_path, enhanced)
        else:
            cv2.imwrite(out_path, cropped)
            out_path = raw_path

        h, w = enhanced.shape[:2] if enhanced is not None else cropped.shape[:2]
        print(f"  [{rank+1}] score={score:.1f}, {w}x{h} -> {out_path}")
        results.append(out_path)

    return results


def enhance_face(img_bgr):
    """GFPGAN face enhancement - monkey-patched for torchvision compat"""
    import types
    ft = types.ModuleType('torchvision.transforms.functional_tensor')
    ft.rgb_to_grayscale = lambda x: x.mean(dim=-3, keepdim=True)
    sys.modules.setdefault('torchvision.transforms.functional_tensor', ft)

    from gfpgan import GFPGANer
    import torch

    restorer = GFPGANer(
        model_path=r'F:\wkspace\ComfyUI\models\gfpgan\GFPGANv1.4.pth',
        upscale=2,
        arch='clean',
        channel_multiplier=2,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    )
    _, _, restored = restorer.enhance(img_bgr, has_aligned=False, only_center_face=False, paste_back=True)
    return restored


def batch_extract(video_dir=None):
    """Extract portraits from all videos in source dir"""
    src = video_dir or r"C:\Users\18091\Desktop\短视频素材"
    videos = []
    for ext in ["*.mp4", "*.MP4"]:
        videos.extend(glob.glob(os.path.join(src, ext)))
    videos = sorted(set(videos))

    for v in videos:
        name = os.path.splitext(os.path.basename(v))[0]
        try:
            extract_portrait(v, coach_name=name)
        except Exception as e:
            print(f"  FAIL: {e}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Video file to extract from")
    parser.add_argument("--batch", action="store_true", help="Process all source videos")
    parser.add_argument("--coach", default=None, help="Coach name for output filename")
    args = parser.parse_args()

    if args.batch:
        batch_extract()
    elif args.video:
        extract_portrait(args.video, coach_name=args.coach)
    else:
        batch_extract()
