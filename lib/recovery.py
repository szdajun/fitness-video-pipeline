"""抢救 long final audio 工具 (per memory recovery-audio-sting-isolation v22)

核心原则 (user 2026-07-13 钉死):
  - 原始视频图像和音乐绝对对齐
  - 编辑不能挤占/延迟/提前主体视频的音乐
  - main 段 sample 不能改 PTS, 必须在 main 段内嵌 volume filter 做末尾渐弱
"""
import os
import subprocess
from pathlib import Path
from typing import Optional


def _ffprobe_duration(path: str) -> float:
    """ffprobe 查 mp4 时长"""
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
         '-of', 'default=noprint_wrappers=1:nokey=1', path],
        capture_output=True, text=True, encoding='utf-8', errors='replace'
    ).stdout.strip()
    return float(out)


def _volumedetect(wav_path: str) -> dict:
    """volumedetect 1 段 wav 返回 {mean_volume, max_volume}"""
    out = subprocess.run(
        ['ffmpeg', '-hide_banner', '-i', wav_path, '-af', 'volumedetect',
         '-f', 'null', '-'],
        capture_output=True, text=True, encoding='utf-8', errors='replace'
    ).stderr
    result = {}
    for line in out.splitlines():
        if 'mean_volume:' in line:
            result['mean_volume'] = float(line.split('mean_volume:')[1].strip().split()[0])
        if 'max_volume:' in line:
            result['max_volume'] = float(line.split('max_volume:')[1].strip().split()[0])
    return result


def _wav_segment(mp4_path: str, ss: float, dur: float, out_wav: str) -> str:
    """抽 mp4 一段 wav"""
    subprocess.run(
        ['ffmpeg', '-y', '-hide_banner', '-i', mp4_path, '-ss', str(ss),
         '-t', str(dur), '-vn', '-c:a', 'pcm_s16le', out_wav],
        capture_output=True, encoding='utf-8', errors='replace'
    )
    return out_wav


def rebuild_long_audio(
    input_long: str,
    source_merged: str,
    output: str,
    intro_sting: Optional[str] = None,
    intro_dur: float = 4.0,
    outro_dur: float = 5.0,
    fade_dur: float = 8.0,
    overwrite: bool = True,
) -> dict:
    """抢救 long final audio (v22 模板)

    严格原始对齐 + 末尾 8s 渐弱. 不动 main 段 PTS, 在 main 段内嵌 volume filter.

    Args:
        input_long: 抢救的 long video (从 _combined.mp4 来的, 无 audio)
        source_merged: 源合并 mp4 (含完整 audio)
        output: 输出路径
        intro_sting: sting wav 路径 (默认 music_library/intro_sting/intro_ref1.wav)
        intro_dur: intro 段时长 (默认 4s)
        outro_dur: outro 段时长 (默认 5s)
        fade_dur: 末尾渐弱时长 (默认 8s)
        overwrite: 覆盖已存在输出 (默认 True)

    Returns:
        dict {success, output, video_dur, audio_dur, intro_db, main_db, fade_curve[]}
        失败 raise RuntimeError
    """
    if intro_sting is None:
        intro_sting = str(
            Path(__file__).parent.parent / 'music_library' / 'intro_sting' / 'intro_ref1.wav'
        )

    if not os.path.exists(input_long):
        raise FileNotFoundError(f'input_long not found: {input_long}')
    if not os.path.exists(source_merged):
        raise FileNotFoundError(f'source_merged not found: {source_merged}')
    if not os.path.exists(intro_sting):
        raise FileNotFoundError(f'intro_sting not found: {intro_sting}')

    src_dur = _ffprobe_duration(source_merged)
    video_dur = _ffprobe_duration(input_long)
    total_dur = intro_dur + src_dur + outro_dur
    fade_start = src_dur - fade_dur  # main 段内嵌渐弱起点 (sample 152.85 for 160.85 source)

    # v22 filter_complex: main atrim 0:SOURCE_DUR 不动 PTS, 段内嵌 volume 8s 渐弱
    fc = (
        f"[1:a]aresample=48000,atrim=0:{intro_dur}[intro];"
        f"[2:a]atrim=0:{src_dur},"
        f"volume=eval=frame:volume='if(lt(t,{fade_start}),1.0,"
        f"max(0.0,1.0-(t-{fade_start})/{fade_dur}))'[main];"
        f"anullsrc=r=48000:cl=stereo[si];"
        f"[si]atrim=0:{outro_dur}[outro];"
        f"[intro][main][outro]concat=n=3:v=0:a=1[fulla]"
    )

    cmd = [
        'ffmpeg', '-y', '-i', input_long, '-i', intro_sting, '-i', source_merged,
        '-map', '0:v', '-filter_complex', fc, '-map', '[fulla]',
        '-c:v', 'copy', '-c:a', 'aac', '-b:a', '128k',
        '-movflags', '+faststart', '-t', str(total_dur), output,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.returncode != 0:
        raise RuntimeError(f'ffmpeg 失败: {result.stderr[-500:]}')

    # 4 步验证 (per memory recovery-audio-unstable-root-cause)
    import tempfile
    metrics = {
        'output': output,
        'video_dur': _ffprobe_duration(output),
        'src_dur': src_dur,
        'total_dur': total_dur,
    }
    with tempfile.TemporaryDirectory() as tmp:
        # 1. 长度对齐 (audio 自动抽)
        out_audio_dur = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'a',
             '-show_entries', 'stream=duration', '-of', 'default=noprint_wrappers=1:nokey=1', output],
            capture_output=True, text=True, encoding='utf-8', errors='replace'
        ).stdout.strip()
        metrics['audio_dur'] = float(out_audio_dur)
        if abs(metrics['video_dur'] - metrics['audio_dur']) > 0.5:
            raise RuntimeError(
                f'video/audio 长度差 {abs(metrics["video_dur"] - metrics["audio_dur"]):.2f}s > 0.5s'
            )

        # 2. intro 段 (0-4s) 音量 ≈ -16dB (sting)
        intro_wav = _wav_segment(output, 0, intro_dur, str(Path(tmp) / 'intro.wav'))
        intro_vd = _volumedetect(intro_wav)
        metrics['intro_db'] = intro_vd.get('mean_volume', -999)
        if not (-18.0 <= metrics['intro_db'] <= -14.0):
            raise RuntimeError(
                f'intro 段音量 {metrics["intro_db"]}dB 不在 -18~-14 范围 (sting 应该是 -16dB)'
            )

        # 3. main 段 (intro+1s) 满音量, 跟 intro 不同
        main_wav = _wav_segment(output, intro_dur, 4.0, str(Path(tmp) / 'main.wav'))
        main_vd = _volumedetect(main_wav)
        metrics['main_db'] = main_vd.get('mean_volume', -999)
        if metrics['main_db'] > -10.0:
            raise RuntimeError(
                f'main 段音量 {metrics["main_db"]}dB 过高 (主体满音量应该 -10~-14dB)'
            )
        if abs(metrics['main_db'] - metrics['intro_db']) < 3.0:
            raise RuntimeError(
                f'main 段音量 {metrics["main_db"]}dB 跟 intro {metrics["intro_db"]}dB 太接近 (intro 跟 main 应该是不同音乐)'
            )

        # 4. 末尾 5 段 1s peak 渐弱 (覆盖 fade_dur 范围)
        # 抽 main 段后 fade_dur 段, 每 fade_dur/5 抽 1s
        # main 段在 fulla 流 src_dur 位置 (intro 0-src_dur, outro src_dur+)
        # 渐弱 T=fade_start=fade_start_in_main=src_dur-fade_dur
        # fulla 流 sample 位置 = main sample + intro_dur
        # 抽 5 段位置: fulla 流 src_dur-fade_dur+0 到 src_dur (覆盖渐弱 8s)
        fade_curve = []
        seg_count = 5
        for i in range(seg_count):
            t_offset = intro_dur + fade_start + (i * fade_dur / seg_count)  # main 段内渐弱起点到 main 末尾
            seg_wav = _wav_segment(output, t_offset, 1.0, str(Path(tmp) / f'seg_{i}.wav'))
            seg_vd = _volumedetect(seg_wav)
            fade_curve.append(seg_vd.get('max_volume', -999))
        metrics['fade_curve'] = fade_curve
        # 检查曲线单调下降 (允许末段 -inf)
        for i in range(len(fade_curve) - 1):
            if fade_curve[i + 1] > fade_curve[i] + 0.5:
                raise RuntimeError(
                    f'末尾渐弱曲线在第 {i+1} 段回升: {fade_curve}'
                )

    metrics['success'] = True
    return metrics


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='抢救 long final audio (v22 模板)')
    ap.add_argument('input_long', help='抢救的 long video 路径 (从 _combined.mp4 来)')
    ap.add_argument('source_merged', help='源合并 mp4 路径 (含完整 audio)')
    ap.add_argument('output', help='输出 mp4 路径')
    ap.add_argument('--intro-sting', help='sting wav 路径 (默认 music_library/intro_sting/intro_ref1.wav)')
    ap.add_argument('--intro-dur', type=float, default=4.0)
    ap.add_argument('--outro-dur', type=float, default=5.0)
    ap.add_argument('--fade-dur', type=float, default=8.0)
    args = ap.parse_args()
    try:
        m = rebuild_long_audio(
            args.input_long, args.source_merged, args.output,
            intro_sting=args.intro_sting,
            intro_dur=args.intro_dur, outro_dur=args.outro_dur, fade_dur=args.fade_dur,
        )
        print(f'OK: {m}')
    except RuntimeError as e:
        print(f'FAIL: {e}', file=__import__('sys').stderr)
        __import__('sys').exit(1)
