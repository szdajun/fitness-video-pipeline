# -*- coding: utf-8 -*-
"""ffprobe 看任意视频元数据 (参数化版)"""
import subprocess, json, os, sys
sys.stdout.reconfigure(encoding='utf-8')

paths = sys.argv[1:]
for f in paths:
    p = subprocess.run(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', f],
                       capture_output=True, text=True, encoding='utf-8')
    j = json.loads(p.stdout)
    v = next(s for s in j['streams'] if s['codec_type']=='video')
    a = next((s for s in j['streams'] if s['codec_type']=='audio'), None)
    sz = os.path.getsize(f)/1024/1024
    print(f'=== {f} ({sz:.1f} MB) ===')
    print(f'  v: {v["codec_name"]} {v["width"]}x{v["height"]} {v.get("r_frame_rate","?")} {v.get("pix_fmt","?")} nb={v.get("nb_frames","?")} dur={float(j["format"]["duration"]):.2f}s')
    if a:
        print(f'  a: {a["codec_name"]} {a.get("sample_rate","?")} {a.get("channels","?")}ch')
    print()