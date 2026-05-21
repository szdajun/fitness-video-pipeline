"""Check audio at 68-76s in each test file"""
import subprocess

def check_segment(filepath, label):
    r = subprocess.run(['ffmpeg','-y','-i',filepath,'-ss','68','-t','8','-af','volumedetect','-f','null','/dev/null'],
                       capture_output=True, text=True)
    for line in r.stderr.split('\n'):
        if 'mean_volume' in line:
            print(f"  {label}: {line.strip()}")
            return
    # If no volumedetect output, check size
    from pathlib import Path
    p = Path(filepath)
    print(f"  {label}: {p.stat().st_size/1024:.0f}KB")

print("=== 各测试文件 68-76s 音频检测 ===")
check_segment('t1_stereo.mkv', '立体声源(原始)')
check_segment('t2_ln.mkv', 'loudnorm立体声')
check_segment('t3_mono.mkv', 'loudnorm单声道')
check_segment('t5_fallback.mkv', 'apad静音填充')

print("\n=== 输出 final 文件 68-76s ===")
check_segment('output/2026-04-27/艳青4_final_16x9.mp4', '艳青4_final')
