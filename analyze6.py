"""定位acrossfade在何时失败"""
import subprocess
import numpy as np
from pathlib import Path

combined = "output/2026-04-27/_combined.mp4"
source = "C:/Users/18091/Desktop/短视频素材/艳青4.mp4"

# Get durations
src_dur = float(subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',source],
                               capture_output=True, text=True).stdout.strip())
total_sec = float(subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',combined],
                                 capture_output=True, text=True).stdout.strip())
fill_dur = total_sec - src_dur
ext_start = max(0, src_dur - fill_dur)
print(f"src_dur={src_dur:.3f}s total_sec={total_sec:.3f}s fill_dur={fill_dur:.3f}s ext_start={ext_start:.3f}s")

def check_wav_audio(wav_path, label):
    """Check if WAV has real audio content"""
    try:
        import wave
        with wave.open(str(wav_path), 'rb') as wf:
            nchannels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            nframes = wf.getnframes()
            data = wf.readframes(nframes)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            if len(samples) == 0:
                print(f"  {label}: 空文件")
                return
            rms = np.sqrt(np.mean(samples**2))
            db = 20 * np.log10(rms / 32768) if rms > 1 else -100
            nonzero = np.count_nonzero(np.abs(samples) > 100) / len(samples) * 100
            print(f"  {label}: {len(samples)}样本, {nchannels}ch, {framerate}Hz, RMS={db:.1f}dB, 非静音={nonzero:.1f}%")
    except Exception as e:
        print(f"  {label}: 读取失败 - {e}")

print("\n=== 测试1: 原始立体声源 ===")
fc1 = (
    f'[1:a]asplit=2[orig][ext_src];'
    f'[ext_src]atrim=start={ext_start}:duration={fill_dur},asetpts=N/SR/TB[ext];'
    f'[orig][ext]acrossfade=d=2[full];'
    f'[full]apad=whole_dur={total_sec}[padded];'
    f'[padded]afade=type=out:st={total_sec-3}:d=3[a]'
)
r = subprocess.run(['ffmpeg','-y','-i',combined,'-i',source,
    '-filter_complex',fc1,'-map','0:v','-map','[a]',
    '-c:v','libx264','-preset','ultrafast','-crf','51','-c:a','pcm_s16le',
    't1_stereo.mkv'], capture_output=True, text=True)
subprocess.run(['ffmpeg','-y','-i','t1_stereo.mkv','-ss','68','-t','8','t1_68_76.wav'], capture_output=True)
check_wav_audio('t1_68_76.wav', '立体声源68-76s')

print("\n=== 测试2: 仅应用loudnorm（不改变声道）===")
subprocess.run(['ffmpeg','-y','-i',source,'-af','loudnorm=I=-14:LRA=11:TP=-1.5',
    '-c:a','pcm_s16le','-ac','2','ln_stereo.wav'], capture_output=True)
print("  loudnorm立体声WAV创建完成")
fc2 = (
    f'[1:a]asplit=2[orig][ext_src];'
    f'[ext_src]atrim=start={ext_start}:duration={fill_dur},asetpts=N/SR/TB[ext];'
    f'[orig][ext]acrossfade=d=2[full];'
    f'[full]apad=whole_dur={total_sec}[padded];'
    f'[padded]afade=type=out:st={total_sec-3}:d=3[a]'
)
subprocess.run(['ffmpeg','-y','-i',combined,'-i','ln_stereo.wav',
    '-filter_complex',fc2,'-map','0:v','-map','[a]',
    '-c:v','libx264','-preset','ultrafast','-crf','51','-c:a','pcm_s16le',
    't2_ln.mkv'], capture_output=True)
subprocess.run(['ffmpeg','-y','-i','t2_ln.mkv','-ss','68','-t','8','t2_68_76.wav'], capture_output=True)
check_wav_audio('t2_68_76.wav', 'loudnorm立体声68-76s')

print("\n=== 测试3: loudnorm + 强制单声道 ===")
subprocess.run(['ffmpeg','-y','-i',source,'-af','loudnorm=I=-14:LRA=11:TP=-1.5',
    '-c:a','pcm_s16le','-ac','1','ln_mono.wav'], capture_output=True)
print("  loudnorm单声道WAV创建完成")
fc3 = (
    f'[1:a]asplit=2[orig][ext_src];'
    f'[ext_src]atrim=start={ext_start}:duration={fill_dur},asetpts=N/SR/TB[ext];'
    f'[orig][ext]acrossfade=d=2[full];'
    f'[full]apad=whole_dur={total_sec}[padded];'
    f'[padded]afade=type=out:st={total_sec-3}:d=3[a]'
)
subprocess.run(['ffmpeg','-y','-i',combined,'-i','ln_mono.wav',
    '-filter_complex',fc3,'-map','0:v','-map','[a]',
    '-c:v','libx264','-preset','ultrafast','-crf','51','-c:a','pcm_s16le',
    't3_mono.mkv'], capture_output=True)
subprocess.run(['ffmpeg','-y','-i','t3_mono.mkv','-ss','68','-t','8','t3_68_76.wav'], capture_output=True)
check_wav_audio('t3_68_76.wav', 'loudnorm单声道68-76s')

print("\n=== 测试4: 直接acrossfade单声道音频文件（不加视频）===")
subprocess.run(['ffmpeg','-y','-i','ln_mono.wav',
    '-filter_complex','[0:a]asplit=2[orig][ext_src];[ext_src]atrim=start=60:duration=8,asetpts=N/SR/TB[ext];[orig][ext]acrossfade=d=2[full]',
    '-map','[full]','-ac','2','t4_mono_only.wav'], capture_output=True)
check_wav_audio('t4_mono_only.wav', '纯单声道acrossfade')

print("\n=== 测试5: 原始音频+静音填充（不跨段，最简单方案）===")
fc5 = (
    f'[1:a]afade=type=out:st={src_dur-1}:d=1,'
    f'apad=whole_dur={total_sec},'
    f'afade=type=out:st={total_sec-1}:d=1[a]'
)
subprocess.run(['ffmpeg','-y','-i',combined,'-i',source,
    '-filter_complex',fc5,'-map','0:v','-map','[a]',
    '-c:v','libx264','-preset','ultrafast','-crf','51','-c:a','pcm_s16le',
    't5_fallback.mkv'], capture_output=True)
subprocess.run(['ffmpeg','-y','-i','t5_fallback.mkv','-ss','68','-t','8','t5_68_76.wav'], capture_output=True)
check_wav_audio('t5_68_76.wav', 'apad静音填充68-76s')

print("\n=== 结论 ===")
for name in ['t1_stereo.mkv','t2_ln.mkv','t3_mono.mkv','t4_mono_only.wav','t5_fallback.mkv']:
    p = Path(name)
    if p.exists():
        print(f"  {name}: {p.stat().st_size/1024:.0f}KB")
