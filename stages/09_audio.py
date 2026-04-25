"""阶段09: 音频处理

- 响度标准化 (LUFS loudnorm)
- 背景音乐混合
- 淡入淡出
"""

import subprocess
import shutil
from pathlib import Path


class AudioStage:
    def run(self, ctx):
        cfg = ctx.config.get("audio", {})
        enabled = cfg.get("enabled", False)
        if not enabled:
            print("    跳过: audio 未启用")
            ctx.set("audio_path", None)
            return

        # 增量跳过：输出已存在则跳过
        if ctx.get("audio_path") and path_exists(ctx.get("audio_path")):
            print("    已存在，跳过")
            return

        video_info = ctx.get("video_info")
        fps = video_info["fps"]
        total_frames = video_info.get("process_frames", video_info["frames"])
        total_sec = total_frames / fps

        # 原始视频路径（取音频用）
        video_path = ctx.input_path
        if not path_exists(video_path):
            print("    跳过: 原始视频不存在")
            ctx.set("audio_path", None)
            return

        ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"
        if not path_exists(ffmpeg):
            print("    跳过: FFmpeg 未安装")
            ctx.set("audio_path", None)
            return

        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        audio_out = (ctx.output_dir / f"{Path(video_path).stem}_audio.aac").resolve()

        # 目标 LUFS
        target_lufs = cfg.get("target_lufs", -14.0)
        bg_music = cfg.get("bg_music", None)
        bg_volume = cfg.get("bg_volume", 0.25)
        fade_in = cfg.get("fade_in", 0.5)
        fade_out = cfg.get("fade_out", 1.0)
        denoise = cfg.get("denoise", 0)    # 降噪强度 0=关闭, 4=轻, 12=普通, 20=强
        ducking = cfg.get("ducking", 0)    # Ducking 强度 0=关闭, 3=轻, 6=普通, 10=强

        print(f"    响度: {target_lufs} LUFS, 淡入{fade_in}s/淡出{fade_out}s"
              + (f", 背景音乐: {Path(bg_music).name}" if bg_music and path_exists(bg_music) else "")
              + (f", 降噪: {denoise}" if denoise > 0 else "")
              + (f", Ducking: {ducking}" if ducking > 0 else ""))

        filter_parts = []

        # 0. 降噪（FFT denoiser，去除呼吸声/风声）
        if denoise > 0:
            speech_parts = [f"afftdn=nr={denoise}"]
        else:
            speech_parts = []

        # 1. 响度标准化
        speech_parts.append(f"loudnorm=I={target_lufs}:LRA=11:TP=-1.5")

        speech_chain = ",".join(speech_parts)

        # 2. 背景音乐混合 + Ducking
        if bg_music and path_exists(bg_music):
            fade_out_start = max(0.1, total_sec - fade_out)

            if ducking > 0:
                # Ducking: 用语音信号作为边链压缩背景音乐
                # threshold=-20dB, ratio=4:1, attack=5ms, release=300ms
                # threshold: 线性值，越小越敏感（0.2=轻压, 0.06=强压）
                threshold = max(0.03, 0.25 - ducking * 0.02)
                filter_complex = (
                    f"[0:a]{speech_chain}[speech];"
                    f"[1:a]volume={bg_volume},afade=t=in:st=0:d={fade_in},"
                    f"afade=t=out:st={fade_out_start}:d={fade_out},"
                    f"acompressor=threshold={threshold}:ratio=4:attack=5:release=300"
                    f"[music_compressed];"
                    f"[speech][music_compressed]amix=inputs=2:duration=first:dropout_transition=2,"
                    f"alimiter=limit=0.95:attack=5[aout]"
                )
            else:
                filter_complex = (
                    f"[1:a]volume={bg_volume},afade=t=in:st=0:d={fade_in},"
                    f"afade=t=out:st={fade_out_start}:d={fade_out},"
                    f"aformat=sample_fmts=fltp:sample_rates=48000,"
                    f"loudnorm=I={target_lufs}:LRA=11:TP=-1.5"
                    f"[music];"
                    f"[0:a]{speech_chain}[speech];"
                    f"[speech][music]amix=inputs=2:duration=first:dropout_transition=2[aout]"
                )
            input_args = ["-i", str(video_path), "-i", str(bg_music)]
            map_args = ["-map", "[aout]"]
        else:
            # 只有原始音频：重建 speech_chain（含 denoise + loudnorm + fade）
            fade_out_start_s = max(0.1, total_sec - fade_out)
            final_parts = speech_parts[:]  # 已有 denoise + loudnorm
            if fade_in > 0:
                final_parts.append(f"afade=t=in:st=0:d={fade_in}")
            if fade_out > 0:
                final_parts.append(f"afade=t=out:st={fade_out_start_s}:d={fade_out}")
            speech_chain = ",".join(final_parts)
            filter_complex = f"[0:a]{speech_chain}[aout]"
            input_args = ["-i", str(video_path)]
            map_args = ["-map", "[aout]"]

        cmd = [
            ffmpeg, "-y",
            *input_args,
            "-filter_complex", filter_complex,
            *map_args,
            "-c:a", "aac", "-b:a", "128k",
            "-shortest",
            str(audio_out)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
        if result.returncode != 0:
            print(f"    音频处理失败: {result.stderr[-200:]}")
            ctx.set("audio_path", None)
            return

        ctx.set("audio_path", str(audio_out))
        print(f"    输出: {audio_out.name}")