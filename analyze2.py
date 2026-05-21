"""更精确的分析：先提取完整WAV再分段"""
import subprocess
from pathlib import Path

output = "output/2026-04-27/艳青4_final_16x9.mp4"

# Step 1: Full decode to WAV (avoids AAC seeking issues)
print("=== 1. 完整解码到 WAV ===")
r = subprocess.run(['ffmpeg','-y','-i',output,'-ac','2','-ar','48000','full_audio.wav'],
                   capture_output=True, text=True)
wav_size = Path('full_audio.wav').stat().st_size
r2 = subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0','full_audio.wav'],
                    capture_output=True, text=True)
print(f"  WAV: {wav_size/1024/1024:.0f}MB, dur={r2.stdout.strip()}s")

# Step 2: Check each 1-second segment from the WAV
print("\n=== 2. 从 WAV 逐秒检测音量 ===")
for t in range(0, 78):
    r = subprocess.run(['ffmpeg','-i','full_audio.wav','-ss',str(t),'-t','1',
                        '-af','volumedetect','-f','null','/dev/null'],
                       capture_output=True, text=True)
    for line in r.stderr.split('\n'):
        if 'mean_volume' in line:
            db = line.split(':')[1].strip().replace(' dB','')
            marker = " <--- 延伸段" if t >= 68 else (" <--- 过渡区" if t >= 66 else "")
            print(f"  t={t:2d}s: {db:>7} dB{marker}")
            break

# Step 3: Check key transitions
print("\n=== 3. 关键过渡段 ===")
for ss,dur,label in [(64,4,'64-68s 过渡前'),(68,8,'68-76s 延伸段全段')]:
    r = subprocess.run(['ffmpeg','-i','full_audio.wav','-ss',str(ss),'-t',str(dur),
                        '-af','volumedetect','-f','null','/dev/null'],
                       capture_output=True, text=True)
    for line in r.stderr.split('\n'):
        if 'mean_volume' in line:
            print(f"  {label}: {line.strip()}")
