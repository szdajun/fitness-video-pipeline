"""测试 acrossfade 在不同输入格式下的表现"""
import subprocess
import wave
import numpy as np
from pathlib import Path

print("=" * 60)
print("测试1: 原始视频作为音频源（之前用这个方法测试成功）")
print("=" * 60)
source = "C:/Users/18091/Desktop/短视频素材/艳青4.mp4"
combined = "output/2026-04-27/_combined.mp4"

# Get durations
src_dur = float(subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',source],
                               capture_output=True, text=True).stdout.strip())
total_sec = float(subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0',combined],
                                 capture_output=True, text=True).stdout.strip())
fill_dur = total_sec - src_dur
ext_start = max(0, src_dur - fill_dur)
audio_fade_d = 3.0
xfade_dur = 2.0
print(f"源视频: src_dur={src_dur:.3f}s, total_sec={total_sec:.3f}s")
print(f"fill_dur={fill_dur:.3f}s, ext_start={ext_start:.3f}s, xfade={xfade_dur}s")

fc = (
    f'[1:a]asplit=2[orig][ext_src];'
    f'[ext_src]atrim=start={ext_start}:duration={fill_dur},asetpts=N/SR/TB[ext];'
    f'[orig][ext]acrossfade=d={xfade_dur}[full];'
    f'[full]apad=whole_dur={total_sec}[padded];'
    f'[padded]afade=type=out:st={total_sec - audio_fade_d}:d={audio_fade_d}[a]'
)

# Test with original video as source (stereo)
subprocess.run(['ffmpeg','-y','-i',combined,'-i',source,
                '-filter_complex',fc,
                '-map','0:v','-map','[a]',
                '-c:v','libx264','-preset','ultrafast','-crf','51',
                '-c:a','pcm_s16le',
                'test_stereo.mkv'], capture_output=True)

# Check audio at 68-76s
subprocess.run(['ffmpeg','-y','-i','test_stereo.mkv','-ss','68','-t','8','-ac','2','stereo_68_76.wav'], capture_output=True)
with wave.open('stereo_68_76.wav','rb') as wf:
    data = wf.readframes(wf.getnframes())
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    rms = np.sqrt(np.mean(samples**2))
    db = 20 * np.log10(rms / 32768) if rms > 1 else -100
    nonzero = np.count_nonzero(np.abs(samples) > 100) / len(samples) * 100
    print(f"  立体声源 68-76s: RMS={db:.1f}dB, 非静音={nonzero:.1f}%")

print("\n" + "=" * 60)
print("测试2: 模拟处理后的单声道M4A作为音频源")
print("=" * 60)

# Create a mono AAC file from the original (simulate what audio stage does)
subprocess.run(['ffmpeg','-y','-i',source,
                '-af','loudnorm=I=-14:LRA=11:TP=-1.5,afade=t=out:st=67.078:d=1.0',
                '-c:a','aac','-b:a','128k',
                '-ac','1',  # Force mono
                '-ar','96000',  # Force 96kHz
                '-shortest','test_mono_source.aac'], capture_output=True)

mono_src_dur = float(subprocess.run(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0','test_mono_source.aac'],
                                    capture_output=True, text=True).stdout.strip())
print(f"单声道源文件时长: {mono_src_dur}s")

fc_mono = (
    f'[1:a]asplit=2[orig][ext_src];'
    f'[ext_src]atrim=start={ext_start}:duration={fill_dur},asetpts=N/SR/TB[ext];'
    f'[orig][ext]acrossfade=d={xfade_dur}[full];'
    f'[full]apad=whole_dur={total_sec}[padded];'
    f'[padded]afade=type=out:st={total_sec - audio_fade_d}:d={audio_fade_d}[a]'
)

subprocess.run(['ffmpeg','-y','-i',combined,'-i','test_mono_source.aac',
                '-filter_complex',fc_mono,
                '-map','0:v','-map','[a]',
                '-c:v','libx264','-preset','ultrafast','-crf','51',
                '-c:a','pcm_s16le',
                'test_mono.mkv'], capture_output=True)

subprocess.run(['ffmpeg','-y','-i','test_mono.mkv','-ss','68','-t','8','-ac','2','mono_68_76.wav'], capture_output=True)
with wave.open('mono_68_76.wav','rb') as wf:
    data = wf.readframes(wf.getnframes())
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    rms = np.sqrt(np.mean(samples**2))
    db = 20 * np.log10(rms / 32768) if rms > 1 else -100
    nonzero = np.count_nonzero(np.abs(samples) > 100) / len(samples) * 100
    print(f"  单声道源 68-76s: RMS={db:.1f}dB, 非静音={nonzero:.1f}%")

print("\n" + "=" * 60)
print("测试3: 测试acrossfade在不改变声道数下直接输出")
print("=" * 60)
# Try with explicit aformat to maintain stereo
fc_stereo = (
    f'[1:a]aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,asplit=2[a_orig][a_ext_src];'
    f'[a_ext_src]atrim=start={ext_start}:duration={fill_dur},asetpts=N/SR/TB[a_ext];'
    f'[a_orig][a_ext]acrossfade=d={xfade_dur}[a_full];'
    f'[a_full]apad=whole_dur={total_sec}[a_padded];'
    f'[a_padded]afade=type=out:st={total_sec - audio_fade_d}:d={audio_fade_d}[a]'
)
subprocess.run(['ffmpeg','-y','-i',combined,'-i','test_mono_source.aac',
                '-filter_complex',fc_stereo,
                '-map','0:v','-map','[a]',
                '-c:v','libx264','-preset','ultrafast','-crf','51',
                '-c:a','pcm_s16le',
                'test_stereo_force.mkv'], capture_output=True)

subprocess.run(['ffmpeg','-y','-i','test_stereo_force.mkv','-ss','68','-t','8','-ac','2','stereo_force_68_76.wav'], capture_output=True)
with wave.open('stereo_force_68_76.wav','rb') as wf:
    data = wf.readframes(wf.getnframes())
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    rms = np.sqrt(np.mean(samples**2))
    db = 20 * np.log10(rms / 32768) if rms > 1 else -100
    nonzero = np.count_nonzero(np.abs(samples) > 100) / len(samples) * 100
    print(f"  强制立体声 68-76s: RMS={db:.1f}dB, 非静音={nonzero:.1f}%")

print("\n" + "=" * 60)
print("测试4: 不跨段，只用apad填充静音（原始音频直接淡出+静音填充）")
print("=" * 60)
fc_fallback = (
    f'[1:a]afade=type=out:st={src_dur - 1.0}:d=1.0,'
    f'apad=whole_dur={total_sec},'
    f'afade=type=out:st={total_sec - 1.0}:d=1.0[a]'
)
subprocess.run(['ffmpeg','-y','-i',combined,'-i','test_mono_source.aac',
                '-filter_complex',fc_fallback,
                '-map','0:v','-map','[a]',
                '-c:v','libx264','-preset','ultrafast','-crf','51',
                '-c:a','pcm_s16le',
                'test_fallback.mkv'], capture_output=True)
subprocess.run(['ffmpeg','-y','-i','test_fallback.mkv','-ss','68','-t','8','-ac','2','fallback_68_76.wav'], capture_output=True)
with wave.open('fallback_68_76.wav','rb') as wf:
    data = wf.readframes(wf.getnframes())
    samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
    rms = np.sqrt(np.mean(samples**2))
    db = 20 * np.log10(rms / 32768) if rms > 1 else -100
    nonzero = np.count_nonzero(np.abs(samples) > 100) / len(samples) * 100
    print(f"  静音填充 68-76s: RMS={db:.1f}dB, 非静音={nonzero:.1f}%")
