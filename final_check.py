"""最终确认：直接从完整WAV读取样本数据"""
import subprocess
import numpy as np
import wave

# Extract FULL audio to reliable WAV
r = subprocess.run(['ffmpeg','-y','-i','output/2026-04-27/艳青4_final_16x9.mp4',
    '-ac','2','-ar','48000','-c:a','pcm_s16le','full_check.wav'],
    capture_output=True)
if r.returncode != 0:
    print("WAV extraction failed")
    exit(1)

# Read WAV samples directly
with wave.open('full_check.wav','rb') as w:
    nframes = w.getnframes()
    sr = w.getframerate()
    ch = w.getnchannels()
    data = w.readframes(nframes)
    print(f"WAV: {nframes} frames, {sr}Hz, {ch}ch, {len(data)} bytes")

samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)

# Convert to mono
mono = samples.reshape(-1, ch).mean(axis=1)
print(f"Total samples: {len(mono)}, expected duration: {len(mono)/sr:.3f}s")

print("\n=== 关键时间点逐秒检查 ===")
for t in range(64, 77):
    start = int(t * sr)
    end = int(min((t+1) * sr, len(mono)))
    if start >= len(mono):
        print(f"  t={t}s: 超出范围")
        continue
    chunk = mono[start:end]
    rms = np.sqrt(np.mean(chunk**2))
    db = 20 * np.log10(rms/32768) if rms > 1 else -100
    nz = np.count_nonzero(np.abs(chunk) > 100) / len(chunk) * 100
    marker = " <<< 延伸段" if t >= 68 else ""
    bar = '#' * max(0, int((db + 60) / 3)) if db > -50 else '.'
    print(f"  t={t:2d}s: RMS={db:6.1f}dB non-silent={nz:5.1f}% |{bar}{marker}")

# boundary check
print("\n=== 边界采样精度检查 ===")
for t in [67.5, 67.8, 67.9, 68.0, 68.1, 68.5]:
    start = int(t * sr)
    end = int(min((t+0.1) * sr, len(mono)))
    if start >= len(mono):
        continue
    chunk = mono[start:end]
    rms = np.sqrt(np.mean(chunk**2))
    db = 20 * np.log10(rms/32768) if rms > 1 else -100
    max_s = np.max(np.abs(chunk))
    print(f"  t={t:.1f}s: RMS={db:.1f}dB max_sample={max_s}")
