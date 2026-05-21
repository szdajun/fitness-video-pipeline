"""终极检测：直接读取full_audio.wav的样本数据，按字节位置判断各时间段是否有音频"""
import wave
import numpy as np
import subprocess
from pathlib import Path

print("=" * 60)
print("确认 full_audio.wav 是否存在")
print("=" * 60)
wav_path = Path('full_audio.wav')
print(f"full_audio.wav 大小: {wav_path.stat().st_size} bytes, 存在: {wav_path.exists()}")

print("\n" + "=" * 60)
print("打开 WAV 获取元数据")
print("=" * 60)
with wave.open(str(wav_path), 'rb') as wf:
    nchannels = wf.getnchannels()
    sampwidth = wf.getsampwidth()
    framerate = wf.getframerate()
    nframes = wf.getnframes()
    print(f"  通道: {nchannels}, 位宽: {sampwidth*8}bit, 采样率: {framerate}Hz, 总帧数: {nframes}")
    print(f"  理论时长: {nframes/framerate:.3f}s")
    print(f"  数据大小: {nframes * nchannels * sampwidth} bytes")

    print("\n" + "=" * 60)
    print("直接读取每1秒区间的原始样本并计算RMS")
    print("=" * 60)

    # Read the entire audio data
    data = wf.readframes(nframes)
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    print(f"  总样本数: {len(samples)}, 形状: {samples.shape}")

    # Reshape for multi-channel (interleaved)
    samples_2ch = samples.reshape(-1, nchannels)
    # Average to mono for analysis
    mono = np.mean(samples_2ch, axis=1)

    # Check each second
    frames_per_sec = framerate
    for t in range(78):
        start = t * frames_per_sec
        end = min((t + 1) * frames_per_sec, len(mono))
        if start >= len(mono):
            print(f"  t={t:2d}s: 超出范围 (总{len(mono)/framerate:.1f}s)")
            continue
        chunk = mono[start:end]

        # RMS
        rms = np.sqrt(np.mean(chunk**2))
        db = 20 * np.log10(rms / 32768) if rms > 1 else -100

        # Percentage of non-silent samples (|sample| > 100)
        nonzero = np.count_nonzero(np.abs(chunk) > 100) / len(chunk) * 100

        # Peak
        peak = np.max(np.abs(chunk)) / 32768
        peak_db = 20 * np.log10(peak) if peak > 0.001 else -100

        marker = " <<< 延伸段" if t >= 68 else (" <<< 过渡区" if t >= 66 else "")
        status = "正常" if db > -40 else ("微弱" if db > -60 else "静音")
        bar = '#' * max(0, int((db + 60) / 2)) if db > -50 else '.'

        if t >= 65 or t <= 5:
            print(f"  t={t:2d}s: RMS={db:6.1f}dB Peak={peak_db:5.1f}dB 非静音={nonzero:5.1f}% {status:4s} |{bar}{marker}")

    print("\n" + "=" * 60)
    print("视频关键区间的RMS分布")
    print("=" * 60)

    # 66-68s (transition)
    start_66 = 66 * frames_per_sec
    end_68 = 68 * frames_per_sec
    chunk_66_68 = mono[start_66:end_68]
    rms = np.sqrt(np.mean(chunk_66_68**2))
    db = 20 * np.log10(rms / 32768) if rms > 1 else -100
    nonzero = np.count_nonzero(np.abs(chunk_66_68) > 100) / len(chunk_66_68) * 100
    print(f"  66-68s (过渡区): RMS={db:.1f}dB 非静音={nonzero:.1f}%")

    # 68-76s (extension)
    start_68 = 68 * frames_per_sec
    end_76 = 76 * frames_per_sec
    chunk_68_76 = mono[start_68:end_76]
    rms = np.sqrt(np.mean(chunk_68_76**2))
    db = 20 * np.log10(rms / 32768) if rms > 1 else -100
    nonzero = np.count_nonzero(np.abs(chunk_68_76) > 100) / len(chunk_68_76) * 100
    print(f"  68-76s (延伸段): RMS={db:.1f}dB 非静音={nonzero:.1f}%")

    # 70-76s (fade out)
    start_70 = 70 * frames_per_sec
    chunk_70_76 = mono[start_70:end_76]
    rms = np.sqrt(np.mean(chunk_70_76**2))
    db = 20 * np.log10(rms / 32768) if rms > 1 else -100
    nonzero = np.count_nonzero(np.abs(chunk_70_76) > 100) / len(chunk_70_76) * 100
    print(f"  70-76s (淡出区): RMS={db:.1f}dB 非静音={nonzero:.1f}%")

    print("\n" + "=" * 60)
    print("检查 70-76s 逐秒")
    print("=" * 60)
    for t in range(70, 77):
        start = t * frames_per_sec
        end = min((t + 1) * frames_per_sec, len(mono))
        if start >= len(mono):
            continue
        chunk = mono[start:end]
        rms = np.sqrt(np.mean(chunk**2))
        db = 20 * np.log10(rms / 32768) if rms > 1 else -100
        nonzero = np.count_nonzero(np.abs(chunk) > 100) / len(chunk) * 100
        bar = '#' * max(0, int((db + 60) / 2)) if db > -50 else '.'
        print(f"  t={t:2d}s: RMS={db:6.1f}dB 非静音={nonzero:5.1f}% |{bar}")
