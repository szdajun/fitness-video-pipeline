# -*- coding: utf-8 -*-
"""彩娥1+2 三件套 4 bug 像素验证"""
import sys, subprocess, os, json
sys.stdout.reconfigure(encoding='utf-8')

VID = 'output/2026-07-10/彩娥1_2_merged_full_16x9_1920x1080'
PROBE_DIR = '_temp/caie_verify'
os.makedirs(PROBE_DIR, exist_ok=True)

def ffprobe_dur(p):
    r = subprocess.run(['ffprobe','-v','quiet','-print_format','json','-show_format',p],
                       capture_output=True, text=True, encoding='utf-8')
    return float(json.loads(r.stdout)['format']['duration'])

def grab(p, t):
    out = f'{PROBE_DIR}/frame_{t}.png'
    subprocess.run(['ffmpeg','-y','-ss',str(t),'-i',p,'-frames:v','1','-update','1',out],
                   capture_output=True, encoding='utf-8')
    return out

def count_rgb(img, lo, hi):
    import cv2, numpy as np
    b = cv2.imread(img); h,w = b.shape[:2]
    arr = b.reshape(-1,3)
    r,g,bb = arr[:,2], arr[:,1], arr[:,0]
    m = (r>=lo[0])&(r<=hi[0])&(g>=lo[1])&(g<=hi[1])&(bb>=lo[2])&(bb<=hi[2])
    return int(m.sum()), w*h

def audio_db(p, t0, t1):
    r = subprocess.run(['ffmpeg','-y','-i',p,'-ss',str(t0),'-to',str(t1),
                        '-af','volumedetect','-f','null','-'],
                       capture_output=True, text=True, encoding='utf-8')
    for line in r.stderr.splitlines():
        if 'mean_volume' in line:
            return line.strip()
    return 'n/a'

ys = VID+'_yt_shorts.mp4'
dy = VID+'_douyin.mp4'
ln = VID+'.mp4'

# 1) hook 🔥
for name, p, t in [('yt_shorts', ys, 2.0), ('douyin', dy, 2.0)]:
    img = grab(p, t)
    n,tot = count_rgb(img, (220,40,0), (255,120,60))
    print(f'[{name} hook 🔥 t={t}s] 橙红={n} (期望>2000)')

# 2) opening 黄字
img = grab(ys, 5.0); n,_ = count_rgb(img, (200,170,0), (255,240,110))
print(f'[yt_shorts opening 黄 t=5s] 像素={n} (期望>2000)')

# 3) PIP 白边
for t in (11.5, 20.0):
    img = grab(ys, t); n,_ = count_rgb(img, (240,240,240), (255,255,255))
    print(f'[yt_shorts PIP 白 t={t}s] 像素={n} (期望>500)')
for t in (11.5, 60.0):
    img = grab(dy, t); n,_ = count_rgb(img, (240,240,240), (255,255,255))
    print(f'[douyin PIP 白 t={t}s] 像素={n} (期望>500)')

# 4) hook 静音
print(f'[hook 静音 0-4s] {audio_db(ys, 0, 4)}')

# 5) CTA
ys_dur = ffprobe_dur(ys)
img = grab(ys, ys_dur-2); n,_ = count_rgb(img, (200,170,0), (255,240,110))
print(f'[yt_shorts CTA 黄 t={ys_dur-2:.1f}s] 像素={n} (期望>2000)')