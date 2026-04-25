"""阶段07: 合成输出

将处理后的视频与原始音频合并，输出最终 H.264 编码视频。
支持放大到 1080x1920 全高清竖版。
支持裁切重复片段（--cut 30-60,120-150）。
"""

import subprocess
import shutil
import ctypes
from pathlib import Path


class ExportStage:
    def run(self, ctx):
        # 按优先级找最终处理的视频
        processed_path = (ctx.get("energybar_path") or
                         ctx.get("highlight_path") or
                         ctx.get("beatflash_path") or
                         ctx.get("sync_path") or
                         ctx.get("heatmap_path") or
                         ctx.get("faceblur_path") or
                         ctx.get("ghost_path") or
                         ctx.get("leadbox_path") or
                         ctx.get("count_path") or
                         ctx.get("skeleton_path") or
                         ctx.get("color_path") or
                         ctx.get("warped_path") or
                         ctx.get("ken_burns_path") or
                         ctx.get("h2v_path") or
                         ctx.get("stabilized_path"))

        if not processed_path or not Path(processed_path).exists():
            print("    跳过: 无处理后的视频")
            return

        # ffmpeg 路径（提前定义，片头片尾拼接也需要）
        ffmpeg = shutil.which("ffmpeg") or "C:/Users/18091/ffmpeg/ffmpeg.exe"

        # 片头片尾拼接
        intro_path = ctx.get("intro_path")
        outro_path = ctx.get("outro_path")
        has_intro = intro_path and Path(intro_path).exists()
        has_outro = outro_path and Path(outro_path).exists()

        if has_intro or has_outro:
            concat_files = []
            if has_intro:
                concat_files.append(str(Path(intro_path).resolve()))
            concat_files.append(str(Path(processed_path).resolve()))
            if has_outro:
                concat_files.append(str(Path(outro_path).resolve()))

            combined_path = ctx.output_dir / "_combined.mp4"
            n = len(concat_files)

            # 用 concat demuxer 避免 filter_complex 导致的帧损坏问题
            list_path = ctx.output_dir / "_concat_list.txt"
            with open(list_path, "w", encoding="utf-8") as f:
                for fp in concat_files:
                    f.write(f"file '{fp}'\n")

            cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0",
                   "-i", str(list_path.resolve()),
                   "-c", "copy",
                   "-an",
                   str(combined_path.resolve())]
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode != 0:
                print(f"    片头片尾拼接失败: {r.stderr[-200:]}")
                combined_path = processed_path
            else:
                print(f"    片头片尾拼接完成 ({n}段)")
                processed_path = str(combined_path)

        video_path = ctx.input_path
        video_info = ctx.get("video_info")
        fps = video_info["fps"]
        is_preview = ctx.config.get("preview", False)

        # 输出配置
        output_cfg = ctx.config.get("output", {})
        out_w = output_cfg.get("width", None)
        out_h = output_cfg.get("height", None)
        crf = output_cfg.get("crf", 26)           # 默认用26，省体积（23太保守）

        # 自动检测输入视频方向，保持原方向不强制缩放
        import cv2
        cap_d = cv2.VideoCapture(processed_path)
        if cap_d.isOpened():
            in_w = int(cap_d.get(3))
            in_h = int(cap_d.get(4))
            cap_d.release()
            # 如果 config 的宽高与输入视频方向不一致，以输入为准
            if in_w > 0 and in_h > 0:
                if out_w and out_h:
                    # 方向不匹配时用输入尺寸
                    if (in_h > in_w and out_h < out_w) or (in_h < in_w and out_h > out_w):
                        out_w, out_h = in_w, in_h
                        print(f"    自动调整为 {out_w}x{out_h}（保持原方向）")
        preset = output_cfg.get("preset", "fast")  # 默认fast，不用medium
        audio_bitrate = output_cfg.get("audio_bitrate", "96k")  # 默认96k，不用128k
        video_fade_out = output_cfg.get("video_fade_out", 2.0)  # 视频淡出秒数
        intro_outro_cfg = ctx.config.get("intro_outro", {})
        audio_fade_d = intro_outro_cfg.get("audio_fade_out", 3.0)  # 音频淡出秒数
        cut_ranges = output_cfg.get("cut_ranges", [])

        ctx.output_dir.mkdir(parents=True, exist_ok=True)

        is_full_video = ctx.config.get("full_video", False)
        if is_preview:
            output_name = f"{video_path.stem}_preview.mp4"
        elif is_full_video:
            output_name = f"{video_path.stem}_full.mp4"
        else:
            output_name = f"{video_path.stem}_final.mp4"

        # 根据输出分辨率添加横竖版后缀
        if out_w and out_h:
            if out_h > out_w:  # 竖版 9:16
                output_name = output_name.replace(".mp4", "_9x16.mp4")
            else:  # 横版 16:9
                output_name = output_name.replace(".mp4", "_16x9.mp4")

        output_path = ctx.output_dir / output_name

        has_ffmpeg = Path(ffmpeg).exists() or shutil.which("ffmpeg")
        audio_path = ctx.get("audio_path")

        if has_ffmpeg:
            # 缩放滤镜 + 锐化（填满画面）
            sharpen = output_cfg.get("sharpen", 0.5)
            if out_w and out_h:
                # 填满：强制缩放填充整个画面（不保留原始比例）
                scale_filter = f"scale={out_w}:{out_h}:flags=bilinear"
            else:
                scale_filter = ""
            if sharpen > 0:
                if scale_filter:
                    scale_filter += f",unsharp=5:5:{sharpen}"
                else:
                    scale_filter = f"unsharp=5:5:{sharpen}"
            res_info = f"{out_w}x{out_h}" if out_w and out_h else "原始分辨率"

            # 视频总时长（秒）
            total_sec = video_info["frames"] / fps
            fade_start = max(0, total_sec - video_fade_out)

            if cut_ranges and not is_preview:
                # 有裁切: 用 filter_complex 同时裁视频+音频
                cut_ranges = sorted(cut_ranges)
                cut_exprs = [f"between(t,{s},{e})" for s, e in cut_ranges]
                cut_or = "+".join(cut_exprs)

                vf_parts = [f"select='not({cut_or})'", "setpts=N/FRAME_RATE/TB"]
                if scale_filter:
                    vf_parts.append(scale_filter)
                vf_parts.append(f"fade=t=out:st={fade_start:.2f}:d={video_fade_out}")
                vf = ",".join(vf_parts)

                total_cut = sum(e - s for s, e in cut_ranges)
                print(f"    FFmpeg 裁切输出 ({res_info}, CRF {crf}, preset={preset}, 裁掉 {len(cut_ranges)} 段, 共 {total_cut:.0f}s)")
                for s, e in cut_ranges:
                    print(f"      裁切: {s}s ~ {e}s")

                if audio_path:
                    # 音频已处理过（包含淡入淡出），直接copy
                    cmd = [ffmpeg, "-y", "-i", str(processed_path), "-i", str(audio_path)]
                    cmd.extend(["-filter_complex", f"[0:v]{vf}[v]"])
                    cmd.extend(["-map", "[v]", "-map", "1:a"])
                    cmd.extend(["-c:v", "libx264", "-preset", preset,
                                "-crf", str(crf), "-c:a", "copy"])
                else:
                    af = f"aselect='not({cut_or})',asetpts=N/SR/TB"
                    cmd = [ffmpeg, "-y", "-i", str(processed_path), "-i", str(video_path)]
                    cmd.extend(["-filter_complex",
                                f"[0:v]{vf}[v];[1:a]{af}[a]"])
                    cmd.extend(["-map", "[v]", "-map", "[a]"])
                    cmd.extend(["-c:v", "libx264", "-preset", preset,
                                "-crf", str(crf), "-c:a", "aac", "-b:a", audio_bitrate])
            else:
                # 无裁切: 直接合并
                print(f"    FFmpeg 合并输出 ({res_info}, CRF {crf}, preset={preset}, audio={audio_bitrate})...")

                # 音频淡出滤镜（使用intro_outro配置中的audio_fade_out）
                audio_fade_start = max(0, total_sec - audio_fade_d)
                audio_fade = f"afade=type=out:st={audio_fade_start:.2f}:d={audio_fade_d}" if audio_fade_d > 0 else ""
                vf_final = scale_filter  # 禁用 export 阶段的视频淡出（outro已有内置淡出）

                if audio_path:
                    # 视频用 intro/outro 合并后的，音频用原始（附加淡出滤镜）
                    cmd = [ffmpeg, "-y",
                           "-i", str(processed_path),  # combined video (intro+main+outro, no audio)
                           "-i", str(audio_path)]       # original audio
                    cmd.extend(["-map", "0:v:0", "-map", "1:a"])
                    if audio_fade:
                        cmd.extend(["-af", audio_fade])
                else:
                    cmd = [ffmpeg, "-y",
                           "-i", str(processed_path),  # combined video (no audio)
                           "-i", str(video_path)]       # original video (has audio)
                    cmd.extend(["-map", "0:v:0", "-map", "1:a:0?"])
                    if audio_fade:
                        cmd.extend(["-af", audio_fade])
                cmd.extend(["-vf", vf_final])
                cmd.extend(["-c:v", "libx264", "-preset", preset,
                            "-crf", str(crf),
                            "-c:a", "aac", "-b:a", audio_bitrate])

            if is_preview:
                cmd.extend(["-t", str(ctx.config.get("preview_seconds", 3))])

            cmd.append(str(output_path))

            result = subprocess.run(cmd, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace")
            if result.returncode != 0:
                stderr = result.stderr[-300:]
                print(f"    FFmpeg 失败: {stderr}")
                shutil.copy2(processed_path, output_path)
        else:
            print("    FFmpeg 未安装，直接复制")
            shutil.copy2(processed_path, output_path)

        ctx.set("final_path", str(output_path))

        # 显示输出文件大小
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"    输出: {output_path} ({size_mb:.1f} MB)")

        # 清理中间文件
        if not is_preview:
            self._cleanup_intermediates(ctx.output_dir, output_path)

    def _cleanup_intermediates(self, output_dir, final_path):
        """删除处理过程中的中间文件"""
        video_stem = final_path.stem.replace("_final", "").replace("_full", "").replace("_9x16", "").replace("_16x9", "")
        is_full = "_full" in final_path.name
        intermediates = [
            "_keypoints.json",
            "_stabilized.mp4",
            "_vectors.trf",
            "_h2v.mp4",
            "_warped.mp4",
            "_face.mp4",
            "_color.mp4",
            "_kenburns.mp4",
            "_audio.aac",
            "_skeleton.mp4",
            "_count.mp4",
            "_leadbox.mp4",
            "_ghost.mp4",
            "_faceblur.mp4",
            "_heatmap.mp4",
            "_sync.mp4",
            "_beatflash.mp4",
            "_energybar.mp4",
            "_highlight.mp4",
        ]
        removed = 0
        for suffix in intermediates:
            f = output_dir / f"{video_stem}{suffix}"
            # full_video 模式下保留 _highlight.mp4（引流版单独有用）
            if is_full and suffix in ("_highlight.mp4", "_energybar.mp4"):
                continue
            if f.exists() and f != final_path:
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
        if removed > 0:
            print(f"    清理: 删除 {removed} 个中间文件")
