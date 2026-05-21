"""找出acrossfade为何不工作 - 逐步调试"""
import subprocess, numpy as np, wave
from pathlib import Path

combined = "output/2026-04-27/_combined.mp4"
source_video = "C:/Users/18091/Desktop/短视频素材/艳青4.mp4"

# Simulate what the audio stage produces
print("=== 步骤1: 创建模拟处理后的单声道AAC（匹配管线音频输出）===")
subprocess.run(['ffmpeg','-y','-i',source_video,
    '-af','loudnorm=I=-14:LRA=11:TP=-1.5,afade=t=out:st=67.078:d=1.0',
    '-c:a','aac','-b:a','128k','-ac','1','-ar','96000','-shortest',
    'sim_mono.aac'], capture_output=True)

# Get durations
src_dur = float(subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0','sim_mono.aac'],
                               capture_output=True, text=True).stdout.strip())
total_sec = float(subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',combined],
                                 capture_output=True, text=True).stdout.strip())
fill_dur = total_sec - src_dur
ext_start = max(0, src_dur - fill_dur)
print(f"  sim_mono.aac: dur={src_dur:.3f}s, total={total_sec:.3f}s")

# Step-by-step filter test
print("\n=== 步骤2: 只做atrim，验证[ext]是否有内容 ===")
subprocess.run(['ffmpeg','-y','-i','sim_mono.aac',
    f'-af','atrim=start={ext_start}:duration={fill_dur},asetpts=N/SR/TB',
    '-ac','2','-ar','48000','step2_ext.wav'], capture_output=True)
with wave.open('step2_ext.wav','rb') as w:
    d = w.readframes(w.getnframes())
    s = np.frombuffer(d, dtype=np.int16).astype(np.float32)
    m = s.reshape(-1,w.getnchannels()).mean(axis=1)
    rms = np.sqrt(np.mean(m**2))
    db = 20 * np.log10(rms/32768) if rms>1 else -100
    nz = np.count_nonzero(np.abs(m)>100)/len(m)*100
    print(f"  [ext] atrim={ext_start:.1f}+{fill_dur:.1f}s: RMS={db:.1f}dB, non-silent={nz:.1f}%")

print("\n=== 步骤3: asplit → atrim → acrossfade (输出到WAV检查) ===")
subprocess.run(['ffmpeg','-y','-i','sim_mono.aac',
    '-filter_complex',
    f'[0:a]asplit=2[orig][ext_src];'
    f'[ext_src]atrim=start={ext_start}:duration={fill_dur},asetpts=N/SR/TB[ext];'
    f'[orig][ext]acrossfade=d=2[full]',
    '-map','[full]','-ac','2','-ar','48000','step3_acrossfade.wav'], capture_output=True)

# Check full WAV duration and content
with wave.open('step3_acrossfade.wav','rb') as w:
    n = w.getnframes()
    sr = w.getframerate()
    d = w.readframes(n)
    s = np.frombuffer(d, dtype=np.int16).astype(np.float32)
    m = s.reshape(-1,w.getnchannels()).mean(axis=1)
    dur = len(m)/sr
    print(f"  acrossfade输出时长: {dur:.3f}s")

    # Check segments
    for t in [0, 66, 67, 68, 69, 70, 73]:
        start = int(t*sr)
        end = int(min((t+1)*sr, len(m)))
        if start >= len(m):
            print(f"  t={t}s: 超出范围({dur:.1f}s)")
            continue
        chunk = m[start:end]
        rms = np.sqrt(np.mean(chunk**2))
        db = 20*np.log10(rms/32768) if rms>1 else -100
        nz = np.count_nonzero(np.abs(chunk)>100)/len(chunk)*100
        marker = " <<< 延伸段" if t >= 68 else ""
        print(f"  t={t}s: RMS={db:.1f}dB non-silent={nz:.1f}%{marker}")

print("\n=== 步骤4: 同样操作但用立体声源（对照）===")
subprocess.run(['ffmpeg','-y','-i',source_video,
    '-filter_complex',
    f'[0:a]asplit=2[orig][ext_src];'
    f'[ext_src]atrim=start={60}:duration={8},asetpts=N/SR/TB[ext];'
    f'[orig][ext]acrossfade=d=2[full]',
    '-map','[full]','-ac','2','-ar','48000','step4_stereo.wav'], capture_output=True)

with wave.open('step4_stereo.wav','rb') as w:
    d = w.readframes(w.getnframes())
    s = np.frombuffer(d, dtype=np.int16).astype(np.float32)
    m = s.reshape(-1,w.getnchannels()).mean(axis=1)
    dur = len(m)/w.getframerate()
    print(f"  acrossfade输出时长: {dur:.3f}s")
    for t in [0, 66, 67, 68, 69, 70, 73]:
        start = int(t*w.getframerate())
        end = int(min((t+1)*w.getframerate(), len(m)))
        if start >= len(m):
            continue
        chunk = m[start:end]
        rms = np.sqrt(np.mean(chunk**2))
        db = 20*np.log10(rms/32768) if rms>1 else -100
        nz = np.count_nonzero(np.abs(chunk)>100)/len(chunk)*100
        print(f"  t={t}s: RMS={db:.1f}dB non-silent={nz:.1f}%")

print("\n=== 步骤5: 直接测试 - 把单声道通过aformat转为立体声后再acrossfade ===")
subprocess.run(['ffmpeg','-y','-i','sim_mono.aac',
    '-filter_complex',
    f'[0:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,asplit=2[orig][ext_src];'
    f'[ext_src]atrim=start={ext_start}:duration={fill_dur},asetpts=N/SR/TB[ext];'
    f'[orig][ext]acrossfade=d=2[full]',
    '-map','[full]','-ac','2','-ar','48000','step5_convert.wav'], capture_output=True)

with wave.open('step5_convert.wav','rb') as w:
    d = w.readframes(w.getnframes())
    s = np.frombuffer(d, dtype=np.int16).astype(np.float32)
    m = s.reshape(-1,w.getnchannels()).mean(axis=1)
    dur = len(m)/w.getframerate()
    print(f"  acrossfade输出时长: {dur:.3f}s")
    for t in [0, 66, 67, 68, 69, 70, 73]:
        start = int(t*w.getframerate())
        end = int(min((t+1)*w.getframerate(), len(m)))
        if start >= len(m):
            continue
        chunk = m[start:end]
        rms = np.sqrt(np.mean(chunk**2))
        db = 20*np.log10(rms/32768) if rms>1 else -100
        nz = np.count_nonzero(np.abs(chunk)>100)/len(chunk)*100
        print(f"  t={t}s: RMS={db:.1f}dB non-silent={nz:.1f}%")
