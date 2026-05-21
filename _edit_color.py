"""Temporary script to add face_sharpen to color_grade stage"""
with open('stages/06_color_grade.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add face_sharpen config reading after adaptive_contrast
old1 = 'adaptive_contrast = cfg.get("adaptive_contrast", 0)'
new1 = old1 + '\n        face_sharpen = cfg.get("face_sharpen", 0)'
content = content.replace(old1, new1, 1)

# 2. Add keypoints loading before frame_idx
old2 = 'tmpdir_short = to_short(str(tmpdir))\n\n        frame_idx = 0'
new2 = old2 + '\n\n        # load keypoints for face-localized sharpening\n        face_keypoints = None\n        if face_sharpen > 0:\n            import json\n            kp_path = ctx.output_dir / f"{ctx.input_path.stem}_keypoints.json"\n            if kp_path.exists():\n                try:\n                    with open(kp_path, encoding="utf-8") as f:\n                        raw = json.load(f)\n                    face_keypoints = raw.get("keypoints", raw)\n                except Exception:\n                    pass'
content = content.replace(old2, new2, 1)

# 3. Add face sharpen processing after first-frame sharpen
old3 = '# 首帧：全局锐化\n                    frame = cv2.addWeighted(frame, 1.0 + sharpen, blurred, -sharpen, 0)\n\n            # 12. 时间平滑'
new3 = old3 + '\n\n            # 12. 脸部局部锐化（利用 pose 关键点定位脸部）\n            if face_sharpen > 0 and face_keypoints is not None:\n                h_f, w_f = frame.shape[:2]\n                frame_kps = face_keypoints.get(str(frame_idx))\n                if frame_kps:\n                    for person_kps in frame_kps:\n                        kps_arr = np.array(person_kps)\n                        face_idx = [0, 1, 2, 3, 4]\n                        pts = [(int(kps_arr[i][0] * w_f), int(kps_arr[i][1] * h_f))\n                               for i in face_idx if i < len(kps_arr) and kps_arr[i][2] > 0.3]\n                        if len(pts) >= 3:\n                            xs = [p[0] for p in pts]\n                            ys = [p[1] for p in pts]\n                            cx = int(np.mean(xs))\n                            cy = int(np.mean(ys))\n                            radius = max(max(xs) - min(xs), max(ys) - min(ys)) // 2 + 30\n                            mask = np.zeros((h_f, w_f), dtype=np.float32)\n                            cv2.circle(mask, (cx, cy), radius, 1.0, -1)\n                            mask = cv2.GaussianBlur(mask, (31, 31), 15)\n                            blurred_face = cv2.GaussianBlur(frame, (0, 0), 2)\n                            face_enhanced = cv2.addWeighted(\n                                frame, 1.0 + face_sharpen, blurred_face, -face_sharpen, 0)\n                            mask_3ch = np.stack([mask] * 3, axis=-1)\n                            frame = (frame.astype(np.float32) * (1 - mask_3ch) +\n                                     face_enhanced.astype(np.float32) * mask_3ch).astype(np.uint8)\n                        break  # only first detected person\n'

content = content.replace(old3, new3, 1)

with open('stages/06_color_grade.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done')
