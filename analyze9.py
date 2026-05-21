"""直接用字节读取WAV，验证Python读法和FFmpeg读法是否一致"""
import subprocess
import numpy as np
from pathlib import Path

# Use FFmpeg to extract 68-76s from the final output
r = subprocess.run(['ffmpeg','-y','-i','output/2026-04-27/艳青4_final_16x9.mp4',
    '-ss','68','-t','8','-c:a','pcm_s16le','-ar','48000','-ac','2',
    'seg_68_76_ffmpeg.wav'], capture_output=True, text=True)
p = Path('seg_68_76_ffmpeg.wav')
print(f"FFmpeg提取68-76s文件大小: {p.stat().st_size} bytes")

# Read with Python
with open('seg_68_76_ffmpeg.wav', 'rb') as f:
    raw = f.read()

# Find "data" chunk (skip any headers before it)
data_start = raw.find(b'data')
if data_start >= 0:
    data_size = int.from_bytes(raw[data_start+4:data_start+8], 'little')
    data = raw[data_start+8:data_start+8+data_size]
    print(f"找到 data chunk 在偏移 {data_start}, 数据大小={data_size}")
else:
    data = raw[44:]  # fallback
    data_size = len(data)

# Get format info from fmt chunk
fmt_start = raw.find(b'fmt ')
if fmt_start >= 0:
    audio_format = int.from_bytes(raw[fmt_start+8:fmt_start+10], 'little')
    num_channels = int.from_bytes(raw[fmt_start+10:fmt_start+12], 'little')
    sample_rate = int.from_bytes(raw[fmt_start+12:fmt_start+16], 'little')
    bits_per_sample = int.from_bytes(raw[fmt_start+22:fmt_start+24], 'little')
    print(f"  format={audio_format}, channels={num_channels}, rate={sample_rate}, bits={bits_per_sample}")
else:
    # Defaults for standard WAV
    audio_format = 1
    num_channels = 2
    sample_rate = 48000
    bits_per_sample = 16
    data = raw[44:]
    data_size = len(raw) - 44

# Calculate total samples
bytes_per_sample = bits_per_sample // 8
frame_size = num_channels * bytes_per_sample
total_frames = len(data) // frame_size
total_samples = total_frames * num_channels
print(f"  frame_size={frame_size}, total_frames={total_frames}, data_usable={total_frames * frame_size} of {len(data)}")

# Trim to full frames
data = data[:total_frames * frame_size]

# Read samples (16-bit little-endian)
samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
print(f"  总样本数: {len(samples)}")

# Reshape for channels
if num_channels == 2:
    samples_2ch = samples.reshape(-1, 2)
    mono = np.mean(samples_2ch, axis=1)
else:
    mono = samples

# Check audio content
rms = np.sqrt(np.mean(mono**2))
db = 20 * np.log10(rms / 32768) if rms > 1 else -100
nonzero = np.count_nonzero(np.abs(mono) > 100) / len(mono) * 100
print(f"\nPython直接读取68-76s: RMS={db:.1f}dB, 非静音={nonzero:.1f}%")

max_sample = np.max(np.abs(mono))
max_db = 20 * np.log10(max_sample / 32768) if max_sample > 0 else -100
print(f"  最大样本: {max_sample}, Peak={max_db:.1f}dB")

# Check each half-second
chunk_sec = sample_rate // 2
print(f"\n逐0.5秒检查:")
for i in range(0, len(mono), chunk_sec):
    chunk = mono[i:i+chunk_sec]
    if len(chunk) < 100:
        break
    rms_c = np.sqrt(np.mean(chunk**2))
    db_c = 20 * np.log10(rms_c / 32768) if rms_c > 1 else -100
    nz = np.count_nonzero(np.abs(chunk) > 100) / len(chunk) * 100
    t = i / sample_rate
    bar = '#' * max(0, int((db_c + 60) / 2)) if db_c > -50 else '.'
    print(f"  t={t:.1f}s: RMS={db_c:6.1f}dB nz={nz:4.1f}% |{bar}")

# Now use FFmpeg volumedetect on the same extracted file
r2 = subprocess.run(['ffmpeg','-i','seg_68_76_ffmpeg.wav','-af','volumedetect','-f','null','/dev/null'],
                    capture_output=True, text=True)
print(f"\nFFmpeg volumedetect on same file:")
for line in r2.stderr.split('\n'):
    if 'mean_volume' in line or 'max_volume' in line:
        print(f"  {line.strip()}")
