from pydub import AudioSegment
import subprocess, shutil, os

orig = AudioSegment.from_file('C:/Users/18091/Desktop/短视频素材/丽丽3.mp4')
print(f'原始音频: {len(orig)/1000:.1f}s')

looped = orig * 2
looped = looped[:73700]
print(f'循环后: {len(looped)/1000:.1f}s')

OUT = 'F:/wkspace/fitness-video-pipeline/output/2026-04-22'
FFMPEG = shutil.which('ffmpeg') or 'C:/Users/18091/ffmpeg/ffmpeg.exe'

looped_path = os.path.join(OUT, '_looped.wav')
looped.export(looped_path, format='wav')
print(f'WAV已生成: {os.path.getsize(looped_path)/1024:.0f}KB')

mp3 = os.path.join(OUT, '_audio_faded.mp3')
r = subprocess.run([FFMPEG, '-y', '-i', looped_path,
                    '-af', 'afade=t=in:st=0:d=2.5,afade=t=out:st=64:d=7.2,volume=6',
                    '-c:a', 'libmp3lame', '-b:a', '128k', mp3],
                   capture_output=True, text=True, encoding='utf-8', errors='replace')
if r.returncode == 0:
    print(f'音频完成: {os.path.getsize(mp3)/1024:.0f}KB')
else:
    print(f'失败: {r.stderr[-200:]}')
os.unlink(looped_path)