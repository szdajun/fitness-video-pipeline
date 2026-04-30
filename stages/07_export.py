"""阶段07: 合成输出

将处理后的视频与原始音频合并，输出最终 H.264 编码视频。
支持放大到 1080x1920 全高清竖版。
支持裁切重复片段（--cut 30-60,120-150）。
"""

import subprocess
import shutil
import ctypes
import cv2
from lib.utils import path_exists
from lib.ai_upscale import AIUpscaler
from pathlib import Path

def _to_short(path_str):
    """转换到 Windows 短路径（中文路径兼容）"""
    buf_size = ctypes.windll.kernel32.GetShortPathNameW(str(path_str), None, 0)
    if buf_size == 0:
        return str(path_str)
    buf = ctypes.create_unicode_buffer(buf_size)
    ctypes.windll.kernel32.GetShortPathNameW(str(path_str), buf, buf_size)
    return buf.value



class ExportStage:
    def run(self, ctx):
        # 按优先级找最终处理的视频
        # face_beautify2 优先于 face_beautify（InsightFace vs MediaPipe）
        # face_beautify 优先于 beatflash_path（美颜效果更强）
        processed_path = (ctx.get("rife_path") or
                         ctx.get("face_beautify2_path") or
                         ctx.get("face_beautify_path") or
                         ctx.get("beatflash_path") or
                         ctx.get("energybar_path") or
                         ctx.get("highlight_path") or
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
                         ctx.get("stabilized_path") or
                         str(ctx.input_path))  # Fallback to original video

        if not processed_path or not cv2.VideoCapture(processed_path).isOpened():
            print("    跳过: 无处理后的视频")
            return

        # ffmpeg 路径（提前定义，片头片尾拼接也需要）
        # 注意：shutil.which("ffmpeg") 可能返回 WinGet Gyan 坏掉的版本，要优先用明确的路径
        ffmpeg_bin = Path("C:/Users/18091/ffmpeg/ffmpeg.exe")
        if ffmpeg_bin.exists():
            ffmpeg = str(ffmpeg_bin)
        else:
            ffmpeg = shutil.which("ffmpeg") or str(ffmpeg_bin)
        # ffprobe 路径（用于获取时长）
        ffprobe_bin = Path("C:/Users/18091/ffmpeg/ffprobe.exe")
        if ffprobe_bin.exists():
            ffprobe = str(ffprobe_bin)
        else:
            ffprobe = ffmpeg

        # 片头片尾拼接
        intro_path = ctx.get("intro_path")
        outro_path = ctx.get("outro_path")
        has_intro = intro_path and path_exists(intro_path)
        has_outro = outro_path and path_exists(outro_path)

        if has_intro or has_outro:
            # 纯视频拼接（音频在最后导出阶段从原片提取 + 填充静音）
            concat_files = []
            if has_intro:
                concat_files.append(str(Path(intro_path).resolve()))
            concat_files.append(str(Path(processed_path).resolve()))
            if has_outro:
                concat_files.append(str(Path(outro_path).resolve()))

            combined_path = ctx.output_dir / "_combined.mp4"
            n = len(concat_files)
            filter_parts = ''.join([f"[{i}:v]" for i in range(n)])
            filter_parts += f"concat=n={n}:v=1:a=0[outv]"

            cmd = [ffmpeg, "-y"]
            for fp in concat_files:
                cmd.extend(["-i", fp])
            cmd.extend(["-filter_complex", filter_parts,
                        "-map", "[outv]",
                        "-c:v", "libx264", "-preset", "fast", "-crf", "1",
                        str(combined_path.resolve())])
            r = subprocess.run(cmd, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
            if r.returncode != 0:
                print(f"    片头片尾拼接失败: {r.stderr[-200:]}")
                has_intro = has_outro = False
            else:
                print(f"    片头片尾拼接完成 ({n}段)")
                processed_path = str(combined_path)

        video_path = ctx.input_path
        video_info = ctx.get("video_info")
        fps = video_info["fps"]
        is_preview = ctx.config.get("preview", False)

        # 获取视频总时长（秒）— 如果有片头片尾合并，用合并后的实际时长
        if has_intro or has_outro:
            probe = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", str(processed_path)],
                capture_output=True, text=True, encoding="utf-8", errors="replace"
            )
            if probe.stdout.strip():
                total_sec = float(probe.stdout.strip())
            else:
                total_sec = video_info["frames"] / fps
        else:
            total_sec = video_info["frames"] / fps

        # 输出配置
        output_cfg = ctx.config.get("output", {})
        out_w = output_cfg.get("width", None)
        out_h = output_cfg.get("height", None)
        crf = output_cfg.get("crf", 26)           # 默认用26，省体积（23太保守）

        # 自动检测输入视频方向，保持原方向不强制缩放
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

        has_ffmpeg = path_exists(ffmpeg) or shutil.which("ffmpeg")
        audio_path = ctx.get("audio_path")

        if has_ffmpeg:
            # 缩放滤镜 — 根据方向自动选择最优算法
            sharpen = output_cfg.get("sharpen", 0.5)
            resize_filter = output_cfg.get("resize_filter", "lanczos")
            if out_w and out_h:
                # 自动选择缩放算法: 放大用lanczos, 缩小用area, 可选cubic
                if resize_filter == "auto":
                    if in_w > 0 and in_h > 0 and (out_w < in_w or out_h < in_h):
                        resize_filter = "area"  # 缩小用 area 抗锯齿
                    else:
                        resize_filter = "lanczos"  # 放大用 lanczos
                scale_flag = {"lanczos": "lanczos", "cubic": "bicubic",
                              "area": "area", "bilinear": "bilinear"}.get(resize_filter, "lanczos")
                scale_filter = f"scale={out_w}:{out_h}:flags={scale_flag}"
            else:
                scale_filter = ""
            if sharpen > 0:
                if scale_filter:
                    scale_filter += f",unsharp=5:5:{sharpen}"
                else:
                    scale_filter = f"unsharp=5:5:{sharpen}"
            res_info = f"{out_w}x{out_h}" if out_w and out_h else "原始分辨率"

            # ---- AI 超分（Real-ESRGAN, GPU 可选） ----
            if output_cfg.get("upscale_mode") == "realesrgan" and out_w and out_h:
                upscaler = AIUpscaler(
                    model_name=output_cfg.get("realesrgan_model", "realesrgan-x4plus"),
                    scale=output_cfg.get("realesrgan_scale", 2),
                    tile=output_cfg.get("realesrgan_tile", 256),
                    gpu=output_cfg.get("realesrgan_gpu", True),
                )
                if upscaler.is_available() and AIUpscaler.need_upscale(in_w, in_h, out_w, out_h):
                    print(f"    AI 超分: {in_w}x{in_h} → {out_w}x{out_h} ...")
                    import tempfile, os
                    tmpdir = Path(tempfile.mkdtemp(prefix="esrgan_"))
                    try:
                        cap_ai = cv2.VideoCapture(processed_path)
                        fi = 0
                        while True:
                            ret, frm = cap_ai.read()
                            if not ret:
                                break
                            up = upscaler.upscale(frm)
                            up = AIUpscaler.preprocess(up, out_w, out_h)
                            cv2.imwrite(str(tmpdir / f"f_{fi:06d}.png"), up)
                            fi += 1
                            if fi % 200 == 0:
                                print(f"    超分进度: {fi} 帧")
                        cap_ai.release()
                        # 编码超分后视频（无音频）
                        esrgan_video = ctx.output_dir / f"{video_path.stem}_esrgan.mp4"
                        short_in = _to_short(str(tmpdir))
                        short_out = _to_short(str(esrgan_video))
                        subprocess.run([
                            ffmpeg, "-y", "-framerate", str(fps),
                            "-i", f"{short_in}/f_%06d.png",
                            "-c:v", "libx264", "-preset", preset,
                            "-crf", str(crf), "-pix_fmt", "yuv420p", "-an",
                            short_out,
                        ], capture_output=True, check=True)
                        processed_path = str(esrgan_video)
                        in_w, in_h = out_w, out_h
                        # 替换 scale_filter 为空（已经超分到目标分辨率）
                        scale_filter = ""
                        if sharpen > 0:
                            scale_filter = f"unsharp=5:5:{sharpen}"
                        res_info = f"{out_w}x{out_h}(AI)"
                        print(f"    AI 超分完成: {fi} 帧")
                    except Exception as e:
                        print(f"    AI 超分失败: {e}，回退 lanczos")
                    finally:
                        shutil.rmtree(tmpdir, ignore_errors=True)
                elif not upscaler.is_available():
                    print(f"    提示: Real-ESRGAN 未安装，回退 {resize_filter} 缩放")

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
                # 注意：combined视频时长可能比原音频长，从原片提取音频 + apad填充静音 + 淡出
                vf_final = scale_filter  # 禁用 export 阶段的视频淡出（outro已有内置淡出）

                if has_intro or has_outro:
                    # 拼接后的视频是纯视频，从原片提取音频
                    audio_src = str(audio_path) if (audio_path and Path(audio_path).exists()) else str(ctx.input_path)

                    # 获取源音频时长
                    probe = subprocess.run(
                        [ffprobe, "-v", "error", "-show_entries", "format=duration",
                         "-of", "csv=p=0", audio_src],
                        capture_output=True, text=True, encoding="utf-8", errors="replace"
                    )
                    src_dur = float(probe.stdout.strip()) if probe.stdout.strip() else total_sec

                    # 使用源实际时长计算延伸段，避免 loudnorm 扩展的静音尾
                    content_dur = video_info["frames"] / fps
                    xfade_dur = 2.0
                    # 需要补的总长度 = (总时长 - 源内容) + crossfade重叠量
                    # 这样 acrossfade 输出恰好等于 total_sec，无需 apad 静音填充
                    need_content = total_sec - content_dur + xfade_dur

                    if need_content > 0.5 and has_intro:
                        # acrossfade：取音频内容前段做无缝循环延伸
                        ext_start = max(0, content_dur - need_content)
                        actual_fill = min(need_content, content_dur)
                        fc_parts = [
                            f"[1:a]atrim=0:{content_dur},asetpts=N/SR/TB[orig_content]",
                            f"[1:a]atrim=start={ext_start}:duration={actual_fill},asetpts=N/SR/TB[ext]",
                            f"[orig_content][ext]acrossfade=d={xfade_dur}[full]",
                        ]
                        total_filled = content_dur + actual_fill - xfade_dur

                        if total_filled > total_sec:
                            fc_parts.append(f"[full]atrim=0:{total_sec}[trimmed]")
                            fade_st = max(0, total_sec - audio_fade_d)
                            fc_parts.append(f"[trimmed]afade=type=out:st={fade_st:.3f}:d={audio_fade_d}[a]")
                        else:
                            fc_parts.append(f"[full]apad=whole_dur={total_sec}[padded]")
                            fade_st = max(0, total_sec - audio_fade_d)
                            fc_parts.append(f"[padded]afade=type=out:st={fade_st:.3f}:d={audio_fade_d}[a]")

                        filter_complex = ";".join(fc_parts)
                    else:
                        # 音频够长或没有片头，直接截断 + 填充 + 淡出
                        fade_st = max(0, total_sec - audio_fade_d)
                        af = f"atrim=0:{total_sec},apad=whole_dur={total_sec}"
                        if audio_fade_d > 0:
                            af += f",afade=type=out:st={fade_st:.3f}:d={audio_fade_d}"
                        filter_complex = f"[1:a]{af}[a]"

                    cmd = [ffmpeg, "-y",
                           "-i", str(processed_path),
                           "-i", audio_src,
                           "-filter_complex", filter_complex,
                           "-map", "0:v", "-map", "[a]"]
                else:
                    # 无片头片尾：直接从原片提取音频 + 淡出
                    audio_src = str(audio_path) if (audio_path and Path(audio_path).exists()) else str(ctx.input_path)
                    fade_st = max(0, total_sec - audio_fade_d)

                    if audio_fade_d > 0:
                        af = f"afade=type=out:st={fade_st:.3f}:d={audio_fade_d}"
                        cmd = [ffmpeg, "-y",
                               "-i", str(processed_path),
                               "-i", audio_src,
                               "-filter_complex", f"[1:a]{af}[a]",
                               "-map", "0:v", "-map", "[a]"]
                    else:
                        cmd = [ffmpeg, "-y",
                               "-i", str(processed_path),
                               "-i", audio_src,
                               "-map", "0:v", "-map", "1:a"]

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
            "_face_beautify.mp4",
            "_face_beautify2.mp4",
            "_rife.mp4",
        ]
        removed = 0
        for suffix in intermediates:
            f = output_dir / f"{video_stem}{suffix}"
            # full_video 模式下保留 _highlight.mp4（引流版单独有用）
            if is_full and suffix in ("_highlight.mp4", "_energybar.mp4"):
                continue
            if is_full and suffix in ("_beatflash.mp4", "_face_beautify.mp4"):
                continue
            if f.exists() and f != final_path:
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
        if removed > 0:
            print(f"    清理: 删除 {removed} 个中间文件")
