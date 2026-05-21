"""全面分析最终输出文件的音频"""
import subprocess
import sys
from pathlib import Path

output_path = "output/2026-04-27/艳青4_final_16x9.mp4"
source_path = "C:/Users/18091/Desktop/短视频素材/艳青4.mp4"

print("=" * 60)
print("1. 输出文件基本信息")
print("=" * 60)
r = subprocess.run(['ffprobe','-v','error','-show_entries','format=duration,size','-of','json',output_path], capture_output=True, text=True)
print(r.stdout)

print("2. 音频流信息")
r = subprocess.run(['ffprobe','-v','error','-select_streams','a:0','-show_entries','stream=index,codec_name,duration,start_time,sample_rate,channels','-of','json',output_path], capture_output=True, text=True)
print(r.stdout)

print("=" * 60)
print("3. 分段音量检测（逐秒）")
print("=" * 60)
for t in range(0, 77):
    r = subprocess.run(['ffmpeg','-i',output_path,'-ss',str(t),'-t','1','-af','volumedetect','-f','null','/dev/null'], capture_output=True, text=True)
    for line in r.stderr.split('\n'):
        if 'mean_volume' in line:
            # Extract float value
            parts = line.split(':')
            if len(parts) >= 2:
                db = parts[1].strip().replace(' dB', '')
                marker = " <<<" if t >= 68 else ""
                print(f"  t={t:2d}s: {db:>7} dB{marker}")
            break

print()
print("=" * 60)
print("4. 源视频 vs 输出音频 逐段提取对比")
print("=" * 60)
# Extract the last 10 seconds from source
print("\n4a. 提取源视频最后10秒 (58-68s)...")
r = subprocess.run(['ffmpeg','-y','-i',source_path,'-ss','58','-t','10','-ac','2','src_last10s.wav'], capture_output=True, text=True)
if r.returncode != 0: print(f"  ERROR: {r.stderr[-200:]}")

# Extract the extension part from output (68-76s)
print("4b. 提取输出文件扩展部分 (68-76s)...")
r = subprocess.run(['ffmpeg','-y','-i',output_path,'-ss','68','-t','8','-ac','2','out_ext_8s.wav'], capture_output=True, text=True)
if r.returncode != 0: print(f"  ERROR: {r.stderr[-200:]}")

# Extract the intro part from output
print("4c. 提取输出文件开头 (0-4s)...")
r = subprocess.run(['ffmpeg','-y','-i',output_path,'-ss','0','-t','4','-ac','2','out_intro_4s.wav'], capture_output=True, text=True)
if r.returncode != 0: print(f"  ERROR: {r.stderr[-200:]}")

# Extract the 64-68s transition zone
print("4d. 提取过渡区域 (64-72s)...")
r = subprocess.run(['ffmpeg','-y','-i',output_path,'-ss','64','-t','8','-ac','2','out_transition_8s.wav'], capture_output=True, text=True)
if r.returncode != 0: print(f"  ERROR: {r.stderr[-200:]}")

# Check file sizes
for f in ['src_last10s.wav','out_ext_8s.wav','out_intro_4s.wav','out_transition_8s.wav']:
    p = Path(f)
    if p.exists():
        size_kb = p.stat().st_size / 1024
        dur_r = subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',f], capture_output=True, text=True)
        dur = dur_r.stdout.strip()
        # Volume
        vol_r = subprocess.run(['ffmpeg','-i',f,'-af','volumedetect','-f','null','/dev/null'], capture_output=True, text=True)
        vol_lines = [l for l in vol_r.stderr.split('\n') if 'mean_volume' in l or 'max_volume' in l]
        print(f"  {f}: {size_kb:.0f}KB, dur={dur}s")
        for vl in vol_lines:
            print(f"    {vl}")
    else:
        print(f"  {f}: NOT FOUND")

print()
print("=" * 60)
print("5. 源视频 60-68s 原始音频内容检测")
print("=" * 60)
r = subprocess.run(['ffmpeg','-y','-i',source_path,'-ss','60','-t','8','-ac','2','src_60_68.wav'], capture_output=True, text=True)
if r.returncode == 0:
    vol_r = subprocess.run(['ffmpeg','-i','src_60_68.wav','-af','volumedetect','-f','null','/dev/null'], capture_output=True, text=True)
    for line in vol_r.stderr.split('\n'):
        if 'mean_volume' in line or 'max_volume' in line:
            print(f"  {line.strip()}")

print()
print("=" * 60)
print("6. 检查过渡区域是否有异常（波形样本对比）")
print("=" * 60)
# Check if the transition 66-68s matches in source vs output
# Source 66-68s
subprocess.run(['ffmpeg','-y','-i',source_path,'-ss','66','-t','2','-ac','2','src_66_68.wav'], capture_output=True)
# Output 66-68s
subprocess.run(['ffmpeg','-y','-i',output_path,'-ss','66','-t','2','-ac','2','out_66_68.wav'], capture_output=True)

for f in ['src_66_68.wav','out_66_68.wav']:
    if Path(f).exists():
        vol_r = subprocess.run(['ffmpeg','-i',f,'-af','volumedetect','-f','null','/dev/null'], capture_output=True, text=True)
        for line in vol_r.stderr.split('\n'):
            if 'mean_volume' in line or 'max_volume' in line:
                print(f"  {f}: {line.strip()}")

print()
print("=" * 60)
print("分析完成！")
print("=" * 60)
