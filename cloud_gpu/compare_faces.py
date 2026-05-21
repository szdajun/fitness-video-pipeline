"""对比分析：原视频 vs GFPGAN 增强后的人脸质量"""
import cv2, sys, os
import numpy as np
from pathlib import Path

def analyze_face_quality(orig_path, enhanced_path, output_dir='face_compare'):
    """提取人脸区域对比 + 清晰度指标"""
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True)

    cap1 = cv2.VideoCapture(orig_path)
    cap2 = cv2.VideoCapture(enhanced_path)

    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )

    frame_idx = 0
    metrics_log = []

    # 采样中间帧（第 300、600、900、1200 帧对比）
    sample_frames = [300, 600, 900, 1200, 1500, 1800]

    print(f"{'帧':>6} | {'原图清晰度':>10} | {'增强清晰度':>10} | {'提升%':>6}")
    print("-" * 40)

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break

        if frame_idx in sample_frames or frame_idx % 500 == 0:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            faces1 = cascade.detectMultiScale(gray1, 1.15, 5, minSize=(60, 60))
            faces2 = cascade.detectMultiScale(gray2, 1.15, 5, minSize=(60, 60))

            # 清晰度指标：Laplacian 方差（越高越清晰）
            lap1 = cv2.Laplacian(gray1, cv2.CV_64F).var()
            lap2 = cv2.Laplacian(gray2, cv2.CV_64F).var()
            improvement = (lap2 - lap1) / lap1 * 100 if lap1 > 0 else 0
            metrics_log.append((frame_idx, lap1, lap2, improvement))

            print(f"{frame_idx:>6} | {lap1:>8.1f}  | {lap2:>8.1f}  | {improvement:>+5.1f}%")

            # 有检测到人脸时，保存对比图
            if len(faces1) > 0 or len(faces2) > 0:
                # 取最大人脸
                def _largest_face(faces):
                    if len(faces) == 0:
                        return None
                    return max(faces, key=lambda f: f[2]*f[3])

                f1 = _largest_face(faces1)
                f2 = _largest_face(faces2)

                # 创建对比图
                h, w = frame1.shape[:2]
                compare = np.zeros((h, w * 2 + 10, 3), dtype=np.uint8)
                compare[:, :w] = frame1
                compare[:, w+10:] = frame2

                # 分隔线
                compare[:, w:w+10] = 128

                # 标注
                cv2.putText(compare, f"原图 (lap={lap1:.0f})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(compare, f"GFPGAN 增强 (lap={lap2:.0f})", (w+20, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                compare_path = out_dir / f"compare_frame_{frame_idx}.jpg"
                cv2.imwrite(str(compare_path), compare)
                print(f"  对比图已保存: {compare_path}")

                # 如果检测到人脸，保存人脸特写对比
                face_crops = []
                for f, label in [(f1, "orig"), (f2, "enhanced")]:
                    if f is not None:
                        x, y, fw, fh = f
                        # 扩大范围
                        margin = int(max(fw, fh) * 0.3)
                        x1 = max(0, x - margin)
                        y1 = max(0, y - margin)
                        x2 = min(w * 2 + 10, x + fw + margin)
                        if label == "enhanced":
                            x1_adj = x1 + w + 10
                            x2_adj = min(compare.shape[1], x2 + w + 10)
                            crop = compare[y1:y+fh+margin, x1_adj:x2_adj]
                        else:
                            crop = compare[y1:y+fh+margin, x1:x+fw+margin]
                        face_crops.append(crop)

                if len(face_crops) >= 2:
                    # 拼成人脸特写横向对比
                    max_h = max(c.shape[0] for c in face_crops)
                    face_row = []
                    for c in face_crops:
                        if c.shape[0] < max_h:
                            pad = np.zeros((max_h - c.shape[0], c.shape[1], 3), dtype=np.uint8)
                            c = np.vstack([c, pad])
                        face_row.append(c)
                    face_compare = np.hstack(face_row)

                    face_path = out_dir / f"face_compare_frame_{frame_idx}.jpg"
                    cv2.imwrite(str(face_path), face_compare)
                    print(f"  人脸特写已保存: {face_path}")

        frame_idx += 1

    cap1.release()
    cap2.release()

    # 汇总
    print("\n" + "=" * 40)
    print("清晰度汇总 (Laplacian 方差)")
    print("=" * 40)
    if metrics_log:
        avg_orig = np.mean([m[1] for m in metrics_log])
        avg_enh = np.mean([m[2] for m in metrics_log])
        avg_imp = np.mean([m[3] for m in metrics_log])
        print(f"平均原图清晰度:   {avg_orig:.1f}")
        print(f"平均增强清晰度:   {avg_enh:.1f}")
        print(f"平均提升:         {avg_imp:+.1f}%")
        print(f"采样帧数:         {len(metrics_log)}")
    else:
        print("无数据")

    print(f"\n对比图已保存到: {out_dir.resolve()}")


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("用法: python compare_faces.py 原视频.mp4 增强视频.mp4")
        sys.exit(1)
    analyze_face_quality(sys.argv[1], sys.argv[2])
