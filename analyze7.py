"""用FFmpeg检查WAV内容（替代Python wave）"""
import subprocess

def check_audio_ffmpeg(filepath, label):
    r = subprocess.run(['ffprobe','-v','error','-show_entries','stream=codec_name,sample_rate,channels','-of','csv=p=0',filepath],
                       capture_output=True, text=True)
    print(f"  {label}: {r.stdout.strip()}")
    r2 = subprocess.run(['ffmpeg','-i',filepath,'-af','volumedetect','-f','null','/dev/null'],
                        capture_output=True, text=True)
    for line in r2.stderr.split('\n'):
        if 'mean_volume' in line or 'max_volume' in line:
            print(f"    {line.strip()}")
    # Also extract to standard WAV
    outpath = filepath.replace('.wav','_std.wav').replace('.mkv','_extract.wav')
    subprocess.run(['ffmpeg','-y','-i',filepath,'-ac','2','-ar','48000','-c:a','pcm_s16le',outpath],
                   capture_output=True)
    s = subprocess.run(['ffmpeg','-i',outpath,'-af','volumedetect','-f','null','/dev/null'],
                       capture_output=True, text=True)
    for line in s.stderr.split('\n'):
        if 'mean_volume' in line:
            print(f"    -> {label}: {line.strip()}")

print("=== 检查各测试文件音频 ===")
check_audio_ffmpeg('t1_stereo.mkv', '立体声源')
check_audio_ffmpeg('t2_ln.mkv', 'loudnorm立体声')
check_audio_ffmpeg('t3_mono.mkv', 'loudnorm单声道')
check_audio_ffmpeg('ln_mono.wav', '源: loudnorm单声道WAV')
check_audio_ffmpeg('t5_fallback.mkv', 'apad静音填充')
