"""自动计算视频总时长，生成匹配音频，合并成最终视频"""
import subprocess, shutil, os
from pathlib import Path
from pydub import AudioSegment

FFMPEG = "C:/Users/18091/ffmpeg/ffmpeg.exe"
FFPROBE = "C:/Users/18091/ffmpeg/ffprobe.exe"
TEMP = os.environ.get('TEMP', 'F:/wkspace/temp')

def to_short(p):
    import ctypes
    GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
    GetShortPathNameW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint]
    GetShortPathNameW.restype = ctypes.c_uint
    buf_size = GetShortPathNameW(str(p), None, 0)
    if buf_size == 0: return str(p)
    buf = ctypes.create_unicode_buffer(buf_size)
    GetShortPathNameW(str(p), buf, buf_size)
    return buf.value

def get_duration(path):
    r = subprocess.run([FFPROBE, '-v', 'error', '-show_entries', 'format=duration',
                       '-of', 'csv=p=0', str(path)], capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except:
        return 0.0

def make_audio(src_mp4, dst_mp3, dur):
    orig = AudioSegment.from_file(src_mp4, format='mp4')
    looped = orig * 3
    looped = looped[:int(dur * 1000)]
    looped = looped.fade_in(2500)  # Only pydub fade_in
    tmp_wav = Path(TEMP) / '_audio_tmp.wav'
    looped.export(str(tmp_wav), format='wav')
    # Only use ffmpeg afade for both in and out
    fade_start = dur - 2.5
    r = subprocess.run([FFMPEG, '-y', '-i', str(tmp_wav),
                       '-af', f'afade=t=in:st=0:d=2.5,afade=t=out:st={fade_start:.1f}:d=2.5',
                       '-ar', '44100', '-ac', '2', '-b:a', '192k', dst_mp3],
                      capture_output=True, text=True, encoding='utf-8', errors='replace')
    os.remove(tmp_wav)
    return r.returncode == 0

def process(name, out_dir, src_video):
    OUT = Path(out_dir)
    intro = OUT / f'{name}_h_intro.mp4'
    main = OUT / f'{name}_h_main.mp4'
    outro = OUT / f'{name}_h_outro.mp4'
    final = OUT / f'{name}_h_final.mp4'

    t_intro = get_duration(intro)
    t_main = get_duration(main)
    t_outro = get_duration(outro)
    t_total = t_intro + t_main + t_outro
    print(f'{name}: intro={t_intro:.1f}s main={t_main:.1f}s outro={t_outro:.1f}s total={t_total:.1f}s')

    audio = OUT / f'{name}_h_audio.mp3'
    if not make_audio(src_video, str(audio), t_total):
        print(f'{name}: audio FAIL')
        return

    # Step 1: xfade intro+main (0.3s crossfade)
    xfade_dur = 0.3
    intro_main_offset = t_intro - xfade_dur
    xfade_file = OUT / f'{name}_h_xfade.mp4'
    filter_xfade = (
        f'[0:v][1:v]xfade=transition=fade:duration={xfade_dur}:offset={intro_main_offset:.3f}[vo];'
        f'[0:a][1:a]acrossfade=d={xfade_dur}[ao]'
    )
    r = subprocess.run([FFMPEG, '-y',
                        '-r', '30', '-i', str(intro),
                        '-r', '30', '-i', str(main),
                        '-filter_complex', filter_xfade,
                        '-map', '[vo]', '-map', '[ao]',
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                        '-pix_fmt', 'yuv420p',
                        '-c:a', 'aac', '-b:a', '192k',
                        to_short(str(xfade_file))],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    if r.returncode != 0:
        print(f'{name}: xfade FAIL: {r.stderr[-200:]}')
        return

    # Step 2: concat xfade+outro
    filter_concat = '[0:v:0]format=yuv420p[v0];[1:v:0]format=yuv420p[v1];[v0][0:a:0][v1][1:a:0]concat=n=2:v=1:a=1[outv][outa]'
    no_audio = OUT / f'{name}_h_noaudio.mp4'
    r = subprocess.run([FFMPEG, '-y',
                        '-i', str(xfade_file),
                        '-i', str(outro),
                        '-filter_complex', filter_concat,
                        '-map', '[outv]', '-map', '[outa]',
                        '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                        '-an',
                        to_short(str(no_audio))],
                       capture_output=True, text=True, encoding='utf-8', errors='replace')
    os.remove(xfade_file)
    if r.returncode != 0:
        print(f'{name}: concat FAIL: {r.stderr[-200:]}')
        return

    r = subprocess.run([FFMPEG, '-y', '-i', str(no_audio), '-i', str(audio),
                       '-map', '0:v', '-map', '1:a',
                       '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
                       str(final)],
                      capture_output=True, text=True, encoding='utf-8', errors='replace')
    os.remove(no_audio)
    if r.returncode == 0:
        print(f'{name}: DONE {final.name} ({os.path.getsize(final)/1024/1024:.1f}MB)')
    else:
        print(f'{name}: merge FAIL: {r.stderr[-200:]}')

if __name__ == '__main__':
    process('小飞侠3', 'output/小飞侠3_2026-04-24',
            'C:/Users/18091/Desktop/短视频素材/小飞侠3.mp4')
    process('小红豆2', 'output/小红豆2_2026-04-24',
            'C:/Users/18091/Desktop/短视频素材/小红豆2.mp4')
    process('李刚3', 'output/李刚3_2026-04-24',
            'C:/Users/18091/Desktop/短视频素材/李刚3.mp4')