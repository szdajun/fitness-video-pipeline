# -*- coding: utf-8 -*-
"""枫林红1+2 三件套 4 bug 像素验证（与张杰/小飞侠同方法）"""
import sys, subprocess, os, json
sys.stdout.reconfigure(encoding='utf-8')

VID = 'output/2026-07-10/枫林红1_2_merged_full_16x9_1920x1080'
PROBE_DIR = '_temp/flh_verify'
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

# 1) yt_shorts hook 橙红(🔥) t=2s
ys = VID+'_yt_shorts.mp4'
dur = ffprobe_dur(ys)
hook_t = 2.0
img = grab(ys, hook_t)
n,tot = count_rgb(img, (220,40,0), (255,120,60))
print(f'[yt_shorts hook 🔥 t={hook_t}s] 橙红像素={n} / total={tot} (期望>2000)')

# 2) douyin hook 橙红 t=2s
dy = VID+'_douyin.mp4'
img = grab(dy, 2.0)
n,tot = count_rgb(img, (220,40,0), (255,120,60))
print(f'[douyin hook 🔥 t=2.0s] 橙红像素={n} / total={tot} (期望>2000)')

# 3) yt_shorts opening 黄字 t=5s
img = grab(ys, 5.0)
n,tot = count_rgb(img, (200,170,0), (255,240,110))
print(f'[yt_shorts opening 黄 t=5s] 像素={n} (期望>2000)')

# 4) yt_shorts PIP 白边 t=11.5/20s
for t in (11.5, 20.0):
    img = grab(ys, t)
    n,tot = count_rgb(img, (240,240,240), (255,255,255))
    print(f'[yt_shorts PIP 白边 t={t}s] 像素={n} (期望>500)')

# 5) douyin PIP 白边 t=11.5/60s
for t in (11.5, 60.0):
    img = grab(dy, t)
    n,tot = count_rgb(img, (240,240,240), (255,255,255))
    print(f'[douyin PIP 白边 t={t}s] 像素={n} (期望>500)')

# 6) hook 静音 0-4s
print(f'[hook 静音 0-4s] {audio_db(ys, 0, 4)}')

# 7) final 爆燃 (long)
ln = VID+'.mp4'
ln_dur = ffprobe_dur(ln)
burst_t = ln_dur - 25  # 末段
img = grab(ln, burst_t)
n,tot = count_rgb(img, (220,40,0), (255,120,60))
print(f'[long 爆燃 t={burst_t:.1f}s] 橙红像素={n} (期望>2000 出现红字)')

# 8) final 汉印/时间戳（长视频末段左上）
for t in (5.0, 30.0, ln_dur-15, ln_dur-5):
    img = grab(ln, t)
    b = subprocess.run(['ffmpeg','-y','-ss',str(t),'-i',ln,'-frames:v','1','-update','1',f'{PROBE_DIR}/sealt_{t}.png'],
                       capture_output=True, encoding='utf-8').returncode
    # 红圆印 region: x=0-260 y=95-115
    import cv2, numpy as np
    img_arr = cv2.imread(f'{PROBE_DIR}/sealt_{t}.png')
    crop = img_arr[95:115, 0:260]
    red = (crop[:,:,2]>=180)&(crop[:,:,1]<80)&(crop[:,:,0]<80)
    print(f'[long 汉印左上 t={t}s] 像素={int(red.sum())}')

# 9) douyin/yt_shorts CTA (末段)
ys_dur = ffprobe_dur(ys)
img = grab(ys, ys_dur-2)
n,_ = count_rgb(img, (200,170,0), (255,240,110))
print(f'[yt_shorts CTA 黄字 t={ys_dur-2:.1f}s] 像素={n} (期望>2000)')