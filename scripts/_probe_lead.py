# -*- coding: utf-8 -*-
"""抽 30 帧看领操人 cx + body_h (参数化版)"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
from ultralytics import YOLO
import cv2, numpy as np

m = YOLO('yolov8n-pose.pt')
for f in sys.argv[1:]:
    cap = cv2.VideoCapture(f)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cx_list, bh_list = [], []
    for i in range(30):
        ok, frame = cap.read()
        if not ok: break
        r = m.predict(frame, verbose=False, conf=0.3)
        if not r or len(r[0].boxes)==0: continue
        a = r[0].boxes.xyxy.cpu().numpy()
        areas = (a[:,2]-a[:,0])*(a[:,3]-a[:,1])
        x1,y1,x2,y2 = a[int(np.argmax(areas))]
        cx_list.append((x1+x2)/2/w); bh_list.append((y2-y1)/h)
    cap.release()
    print(f'=== {f} ===')
    if cx_list:
        print(f'  n={len(cx_list)} cx=[{min(cx_list):.2f},{max(cx_list):.2f}] median={np.median(cx_list):.2f}  body_h=[{min(bh_list):.2f},{max(bh_list):.2f}] median={np.median(bh_list):.2f}')
    else:
        print('  no pose')
    print()