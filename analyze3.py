"""检查延伸段音频的实际内容"""
import subprocess
import numpy as np
from pathlib import Path

output = "output/2026-04-27/艳青4_final_16x9.mp4"
source = "C:/Users/18091/Desktop/短视频素材/艳青4.mp4"

# 1. Extract key segments as WAV for content analysis
print("=== 1. 提取关键段为WAV ===")

# Source: 58-68s (the last 10 seconds of original)
subprocess.run(['ffmpeg','-y','-i',source,'-ss','58','-t','10','-ac','2','-ar','48000','src_58_68.wav'],
               capture_output=True)
src_size = Path('src_58_68.wav').stat().st_size
print(f"  源58-68s: {src_size/1024:.0f}KB")

# Output: 66-76s (the transition and extension)
subprocess.run(['ffmpeg','-y','-i',output,'-ss','66','-t','10','-ac','2','-ar','48000','out_66_76.wav'],
               capture_output=True)
out_size = Path('out_66_76.wav').stat().st_size
print(f"  输出66-76s: {out_size/1024:.0f}KB")

# Output: 0-4s (intro)
subprocess.run(['ffmpeg','-y','-i',output,'-ss','0','-t','4','-ac','2','-ar','48000','out_0_4.wav'],
               capture_output=True)
intro_size = Path('out_0_4.wav').stat().st_size
print(f"  输出0-4s: {intro_size/1024:.0f}KB")

print("\n=== 2. 计算每个段落的RMS能量（原始样本级检测）===")
# Use Python to read WAV samples and compute RMS directly
import wave

def analyze_wav(path):
    try:
        with wave.open(str(path), 'rb') as wf:
            n = wf.getnframes()
            data = wf.readframes(n)
            samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            # RMS in dB
            rms = np.sqrt(np.mean(samples**2)) if len(samples) > 0 else 0
            if rms > 0:
                db = 20 * np.log10(rms / 32768)
            else:
                db = -100
            peak = np.max(np.abs(samples)) / 32768
            peak_db = 20 * np.log10(peak) if peak > 0 else -100
            # Check for non-zero samples
            nonzero = np.count_nonzero(np.abs(samples) > 100) / len(samples) * 100 if len(samples) > 0 else 0
            return db, peak_db, nonzero, len(samples)
    except Exception as e:
        return None, None, None, None

for name in ['src_58_68.wav','out_66_76.wav','out_0_4.wav']:
    db, peak_db, nonzero_pct, n = analyze_wav(name)
    if db is not None:
        status = "有音频" if db > -60 else "静音"
        print(f"  {name}: RMS={db:.1f}dB, Peak={peak_db:.1f}dB, 非静音样本={nonzero_pct:.1f}%, samples={n} -> {status}")

print("\n=== 3. 把输出66-76s分段检查 ===")
subprocess.run(['ffmpeg','-y','-i','out_66_76.wav','-f','wav','-ac','2','-ar','48000','out_66_76_full.wav'],
               capture_output=True)
with wave.open('out_66_76_full.wav', 'rb') as wf:
    n = wf.getnframes()
    data = wf.readframes(n)
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    sr = wf.getframerate()
    # Split into 0.5s chunks
    chunk_size = sr // 2  # 0.5 seconds
    for i in range(0, len(samples), chunk_size):
        chunk = samples[i:i+chunk_size]
        if len(chunk) < 100:
            break
        rms = np.sqrt(np.mean(chunk**2))
        db = 20 * np.log10(rms / 32768) if rms > 0 else -100
        t_start = i / sr
        t_end = (i + len(chunk)) / sr
        bar = '#' * max(0, int((db + 60) / 2)) if db > -60 else '.'
        print(f"  {t_start:5.1f}-{t_end:4.1f}s: {db:6.1f} dB |{bar}")

print("\n=== 4. 对比源60-68s和输出延伸段(68-76s)的波形相似度 ===")
# Source 60-68s
subprocess.run(['ffmpeg','-y','-i',source,'-ss','60','-t','8','-ac','2','-ar','48000','src_60_68.wav'],
               capture_output=True)
# Output 68-76s
subprocess.run(['ffmpeg','-y','-i','full_audio.wav','-ss','68','-t','8','-ac','2','-ar','48000','out_68_76.wav'],
               capture_output=True)

# Compare both
for name in ['src_60_68.wav','out_68_76.wav']:
    with wave.open(name, 'rb') as wf:
        n = wf.getnframes()
        data = wf.readframes(n)
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        rms = np.sqrt(np.mean(samples**2))
        db = 20 * np.log10(rms / 32768) if rms > 0 else -100
        peak = np.max(np.abs(samples)) / 32768
        peak_db = 20 * np.log10(peak) if peak > 0 else -100
        print(f"  {name}: RMS={db:.1f}dB, Peak={peak_db:.1f}dB")

# If both have content, check the cross-correlation
src_wav = wave.open('src_60_68.wav','rb')
src_data = np.frombuffer(src_wav.readframes(src_wav.getnframes()), dtype=np.int16).astype(np.float32)
src_wav.close()
out_wav = wave.open('out_68_76.wav','rb')
out_data = np.frombuffer(out_wav.readframes(out_wav.getnframes()), dtype=np.int16).astype(np.float32)
out_wav.close()

min_len = min(len(src_data), len(out_data))
if min_len > 1000:
    # Simple RMS comparison
    src_rms = np.sqrt(np.mean(src_data[:min_len]**2))
    out_rms = np.sqrt(np.mean(out_data[:min_len]**2))
    ratio = out_rms / src_rms if src_rms > 0 else 0
    print(f"\n  输出/源 RMS比率: {ratio:.2f}x ({20*np.log10(ratio):.1f}dB)")

print("\n=== 分析完毕 ===")
print("源视频最后有音频:", Path('src_58_68.wav').stat().st_size > 10000)
print("输出延伸段有音频:", Path('out_66_76.wav').stat().st_size > 10000)
