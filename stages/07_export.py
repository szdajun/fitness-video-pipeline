"""阶段07: 合成输出

将处理后的视频与原始音频合并，输出最终 H.264 编码视频。
支持放大到 1080x1920 全高清竖版。
支持裁切重复片段（--cut 30-60,120-150）。
GPU 加速: 自动检测 h264_nvenc，优先使用硬件编码。
"""

import subprocess
import shutil
import cv2
from lib.utils import path_exists, to_short as _to_short
from lib.ai_upscale import AIUpscaler
from pathlib import Path
import sys
# _make_shorts 在项目根目录, 子模块 sys.path 不含根, 加上
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))



class ExportStage:
    _nvenc_available = None  # lazy probe

    @staticmethod
    def has_audio_stream(video_path: str) -> bool:
        """检测视频文件是否包含音频流.
        2026-06-20 新增: 解决无音频视频 (如截取的测试片段) 导致 [1:a]afade 报错.
        """
        if not video_path or not Path(video_path).exists():
            return False
        ffprobe_bin = Path("C:/Users/18091/ffmpeg/ffprobe.exe")
        if not ffprobe_bin.exists():
            ffprobe_bin = Path(shutil.which("ffprobe") or "ffprobe")
        try:
            r = subprocess.run(
                [str(ffprobe_bin), "-v", "error",
                 "-select_streams", "a",
                 "-show_entries", "stream=codec_type",
                 "-of", "csv=p=0", str(video_path)],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=10
            )
            return "audio" in r.stdout
        except Exception:
            return False

    @classmethod
    def _probe_nvenc(cls) -> bool:
        """检测 h264_nvenc 编码器是否可用.
        必须在隔离的 ffmpeg 进程里真的编码几帧, 看 returncode 和 stderr.
        之前只看 returncode == 0 是错的, 因为有些 GPU 状态会让 NVENC 短暂失败.
        """
        ffmpeg = Path("C:/Users/18091/ffmpeg/ffmpeg.exe")
        if not ffmpeg.exists():
            ffmpeg = Path(shutil.which("ffmpeg") or "ffmpeg")
        # 真编码 5 帧到 null, 失败的话 stderr 含 'Error' / 'cannot'
        r = subprocess.run(
            [str(ffmpeg), "-y", "-hide_banner", "-f", "lavfi",
             "-i", "color=c=black:s=256x256:d=0.1:r=10",
             "-c:v", "h264_nvenc", "-preset", "p1", "-b:v", "1M",
             "-frames:v", "5", "-an", "-f", "null", "-"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15,
        )
        if r.returncode != 0:
            return False
        err = (r.stderr or "").lower()
        if "cannot" in err or "failed" in err or "no nvenc" in err:
            return False
        return True

    def _encoder_args(self, output_cfg: dict) -> list:
        """根据 GPU 可用性和配置返回编码器参数.

        策略变更 (修根因):
        - 默认 'auto' 走 CPU libx264, 稳, 不依赖 GPU
        - 'nvenc' 显式开启时探测 GPU 失败则报错不降级 (避免静默改 cmd 残留)
        - 'libx264' 显式指定 CPU
        """
        encoder = output_cfg.get("encoder", "auto")
        if encoder == "libx264":
            return self._libx264_args(output_cfg)
        if encoder == "nvenc":
            if self._probe_nvenc():
                return self._nvenc_args(output_cfg)
            raise RuntimeError("NVENC 不可用, 但配置 encoder=nvenc 强制 GPU")
        # auto: 默认 CPU (稳)
        if self._probe_nvenc() and output_cfg.get("prefer_gpu", False):
            return self._nvenc_args(output_cfg)
        return self._libx264_args(output_cfg)

    def _nvenc_args(self, output_cfg: dict) -> list:
        return ["-c:v", "h264_nvenc", "-preset", "p6",
                "-rc", "vbr", "-cq", "18", "-b:v", "0",
                "-spatial-aq", "1", "-aq-strength", "8"]

    def _libx264_args(self, output_cfg: dict) -> list:
        preset = output_cfg.get("preset", "fast")
        crf = output_cfg.get("crf", 26)
        deblock = output_cfg.get("deblock", "")
        args = ["-c:v", "libx264", "-preset", preset, "-crf", str(crf)]
        if deblock:
            args += ["-x264-params", f"deblock={deblock}".replace(":", "\\:")]
        return args
    def run(self, ctx):
        # 按优先级找最终处理的视频
        # face_beautify2 优先于 face_beautify（InsightFace vs MediaPipe）
        # face_beautify 优先于 beatflash_path（美颜效果更强）
        # smart_crop 最优先: 输出已是裁好的 9:16 视频, 后面的装饰 stage 都基于它叠加
        processed_path = (ctx.get("smart_crop_path") or
                  ctx.get("rife_path") or
                  ctx.get("face_beautify2_path") or
                         ctx.get("face_beautify_path") or
                         ctx.get("bgm_path") or
                         ctx.get("pip_path") or
                         ctx.get("burst_path") or
                         ctx.get("danmaku_path") or
                         ctx.get("mascot_path") or
                         ctx.get("watermark_path") or
                         ctx.get("energybar_path") or
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
        coldopen_path = ctx.get("coldopen_path")
        has_intro = intro_path and path_exists(intro_path)
        has_outro = outro_path and path_exists(outro_path)
        has_coldopen = coldopen_path and path_exists(coldopen_path)

        if has_intro or has_outro or has_coldopen:
            # 纯视频拼接（音频在最后导出阶段从原片提取 + 填充静音）
            concat_files = []
            if has_coldopen:
                concat_files.append(str(Path(coldopen_path).resolve()))
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
                        "-pix_fmt", "yuv444p",
                        str(combined_path.resolve())])
            r = subprocess.run(
            cmd,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=600)
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
        deblock = output_cfg.get("deblock", "")   # x264 deblock 参数，如 "-1:-1"

        # 自动检测输入视频方向，保持原方向不强制缩放
        # 优先用 ffprobe (更可靠, 不受 cv2 缓存/格式影响), cv2 兜底
        in_w, in_h = 0, 0
        try:
            probe = subprocess.run(
                [ffprobe, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height",
                 "-of", "csv=p=0", str(processed_path)],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10,
            )
            if probe.stdout.strip():
                w, h = probe.stdout.strip().split(",")
                in_w, in_h = int(w), int(h)
        except Exception as e:
            print(f"    警告: ffprobe 探测失败: {e}, 尝试 cv2 兜底")
        if in_w == 0 or in_h == 0:
            cap_d = cv2.VideoCapture(processed_path)
            if cap_d.isOpened():
                in_w = int(cap_d.get(3))
                in_h = int(cap_d.get(4))
                cap_d.release()
        print(f"    输入尺寸: {in_w}x{in_h}, 输出: {out_w}x{out_h}")
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

        # 根据输出分辨率 + preset 名生成有意义的 final 名
        # 2026-06-18 改: 命名加 preset 标识 + 比例 + 尺寸
        # 例: 海军3_douyin_9x16_1080x1920.mp4 / 海军3_xiaohongshu_3x4_1080x1440.mp4
        if out_w and out_h:
            preset_name = ctx.config.get("_preset_name", "")
            if out_h > out_w:  # 竖版
                ratio_w = 9 if abs(out_w/out_h - 9/16) < 0.01 else int(round(out_w / 64))
                ratio_h = 16 if abs(out_w/out_h - 9/16) < 0.01 else int(round(out_h / 64))
                # 更精确判断 3:4 vs 9:16
                from math import gcd
                g = gcd(out_w, out_h)
                rw, rh = out_w // g, out_h // g
                ratio_str = f"{rw}x{rh}"
            else:  # 横版
                ratio_str = "16x9"
            # preset 名映射到人话
            preset_label = {
                "douyin": "douyin",
                "xiaohongshu": "xiaohongshu",
                "shorts": "shorts",
                "youtube_shorts": "shorts",
                "youtube": "youtube",
            }.get(preset_name, preset_name)
            if preset_label:
                output_name = f"{video_path.stem}_{preset_label}_{ratio_str}_{out_w}x{out_h}.mp4"
            else:
                output_name = output_name.replace(".mp4", f"_{ratio_str}_{out_w}x{out_h}.mp4")

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
                # 横跨比例时以教练为中心裁切 (原设计: 突出教练, 不拉伸)
                # 16:9→9:16: 裁出 9:16 窗口跟踪教练位置 → scale 填满
                # 修复: 只有真正的 9:16 竖源(in_h > in_w) 才能走 '竖→横' 分支,
                #       否则 21:9 之类的超宽源会算出负 crop_y 失败
                in_aspect = in_w / in_h if in_h > 0 else 1.0
                out_aspect = out_w / out_h if out_h > 0 else 1.0
                is_vertical_source = in_w > 0 and in_h > 0 and in_h > in_w
                is_horizontal_target = out_w > out_h
                if is_vertical_source and is_horizontal_target:
                    # 真正的竖源(in_h > in_w) → 横版: 裁上下居中
                    crop_h = int(in_w * out_h / out_w)
                    crop_h = crop_h if crop_h % 2 == 0 else crop_h - 1
                    crop_y = max(0, (in_h - crop_h) // 2)  # 防止负 y
                    scale_filter = f"crop={in_w}:{crop_h}:0:{crop_y},scale={out_w}:{out_h}:flags={scale_flag}"
                elif out_h > out_w and in_w > 0 and in_h > 0:
                    # 横源 → 9:16 竖版: 9:16 窗口, 以教练 x 为中心
                    crop_w = int(in_h * out_w / out_h)
                    crop_w = crop_w if crop_w % 2 == 0 else crop_w - 1
                    crop_x = max(0, int(ctx.get("lead_cx", 0.5) * in_w - crop_w / 2))
                    crop_x = min(crop_x, in_w - crop_w)
                    scale_filter = f"crop={crop_w}:{in_h}:{crop_x}:0,scale={out_w}:{out_h}:flags={scale_flag}"
                else:
                    # 同方向或同比例, 直接 scale
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
                max_ai_frames = output_cfg.get("realesrgan_max_frames", 300)
                total_in_frames = int(cv2.VideoCapture(processed_path).get(cv2.CAP_PROP_FRAME_COUNT))
                if upscaler.is_available() and AIUpscaler.need_upscale(in_w, in_h, out_w, out_h):
                    if total_in_frames > max_ai_frames:
                        print(f"    AI 超分跳过: {total_in_frames} 帧 > {max_ai_frames} 上限，回退 {resize_filter}")
                    else:
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
                                "-crf", str(crf), "-pix_fmt", "yuv444p", "-an",
                                short_out,
                            ], capture_output=True, check=True)
                            processed_path = str(esrgan_video)
                            in_w, in_h = out_w, out_h
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
                vf_parts.append("deband=0.1:0.1:0.1:0.1:8")  # 强deband减轻色块
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
                    cmd.extend(self._encoder_args(output_cfg))
                    cmd.extend(["-c:a", "copy"])
                else:
                    af = f"aselect='not({cut_or})',asetpts=N/SR/TB"
                    cmd = [ffmpeg, "-y", "-i", str(processed_path), "-i", str(video_path)]
                    cmd.extend(["-filter_complex",
                                f"[0:v]{vf}[v];[1:a]{af}[a]"])
                    cmd.extend(["-map", "[v]", "-map", "[a]"])
                    cmd.extend(self._encoder_args(output_cfg))
                    cmd.extend(["-c:a", "aac", "-b:a", audio_bitrate])
            else:
                # 无裁切: 直接合并
                _enc_name = 'NVENC' if 'nvenc' in str(self._encoder_args(output_cfg)) else 'libx264'
                if _enc_name == 'NVENC':
                    print(f"    GPU 编码: h264_nvenc (preset p6, CQ 18, deband)")
                else:
                    print(f"    CPU 编码: libx264 (preset {preset}, CRF {crf})")
                print(f"    FFmpeg 合并输出 ({res_info}, audio={audio_bitrate})...")

                # 音频淡出滤镜（使用intro_outro配置中的audio_fade_out）
                # 注意：combined视频时长可能比原音频长，从原片提取音频 + apad填充静音 + 淡出
                vf_final = f"{scale_filter},deband=0.1:0.1:0.1:0.1:8"  # 强deband减轻色块
                # 防御: scale_filter 为空时 vf_final 会以 ",deband=..." 开头 (ffmpeg 报 Invalid argument)
                # 这种情况通常因为 in_w/in_h=0 探测失败, 强制兜底一个 scale
                if not scale_filter and out_w and out_h:
                    scale_flag = output_cfg.get("resize_filter", "lanczos")
                    if scale_flag == "auto":
                        scale_flag = "lanczos"
                    vf_final = f"scale={out_w}:{out_h}:flags={scale_flag},deband=0.1:0.1:0.1:0.1:8"
                    print(f"    [WARN] scale_filter 空, 兜底用 {vf_final}")
                # 防御: 移除 vf 字符串里可能因 scale_filter="" 产生的开头 "," 或孤立 ",,"
                vf_final = ",".join(s for s in vf_final.split(",") if s)

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

                    # 提前判断有无音频, 后续 if/elif/else 各分支都能用
                    has_audio = ExportStage.has_audio_stream(str(audio_src))
                    # 2026-06-29: 片头音乐 sting (秦腔). 只在 has_intro 主分支用, 替换 anullsrc 静音
                    has_sting = False
                    sting_path = None

                    if has_intro and has_audio:
                        # 2026-06-28 修复: intro 段音频 = 静音 (anullsrc), 主体段 = 原音频 0~content_dur
                        # 旧代码 atrim=0:{content_dur} 把原视频第 0s 音频直接塞到 final 视频第 0s,
                        # 但 final 第 0s 是片头画面 → 用户感知 "音乐提前了 intro_duration 秒"
                        # intro/outro 时长从 intro_outro config 拿, 与 stages/20_intro_outro.py 同步
                        intro_outro_cfg_audio = ctx.config.get("intro_outro", {})
                        intro_dur_for_audio = intro_outro_cfg_audio.get("intro_duration", 4.0)
                        outro_dur_for_audio = intro_outro_cfg_audio.get("outro_duration", 5.0)
                        # 2026-06-29: 片头音乐 sting. 原秦腔 (intro_qinqiang.wav) 用户要求换现代古风,
                        # 切到 intro_ref1.wav (同为 4s). 可经 intro_outro.intro_sting 覆盖; 秦腔文件保留可回退.
                        default_sting = Path(__file__).parent.parent / "music_library" / "intro_sting" / "intro_ref1.wav"
                        sting_path = Path(intro_outro_cfg_audio.get("intro_sting", str(default_sting)))
                        has_sting = sting_path.exists()
                        main_dur = content_dur                # 主体段音频长度
                        out_dur = total_sec                   # 最终音频长度
                        fade_st = max(0, out_dur - audio_fade_d)

                        # 简化: 用 adelay 给原音频延时 intro_dur 秒, 让 intro 段自然变静音
                        # 原音频通过 adelay 推到 intro_dur 开始, 时长 = main_dur + intro_dur
                        # 再 atrim 截取 0~out_dur
                        # 最后 afade out
                        # 2026-06-29: intro 段音频 = 秦腔 sting (有则用, 无则退化为静音, 兜底).
                        #   sting atrim 切长/apad 补短到 intro_dur 秒; [main_a]主体音频不变;
                        #   concat 硬切 → sting 在 intro_dur 戛然而止, 主体音乐即时接管, 不溢出.
                        #   sting 的 0.3s 淡入已 baked 进 wav, filter 里不再加.
                        # 2026-06-30: 片头音乐=片头画面对应的素材原声 (用户方案: "从原视频拿
                        # 合适的4秒音乐"). 配置 intro_outro.intro_music_from_clip=true 启用.
                        # 片头画面截自素材 best_start 起 intro_dur 秒 (20_intro_outro 存的
                        # intro_start_sec), 片头音频用同一段原声 → 音画完全同步 (原片高潮段
                        # 原汁原味原声). 主体音频=素材完整原声 0~main_dur (与主体画面同步,
                        # 口令动作对齐不穿帮). 消除独立 sting 风格突兀, 无错位无重复.
                        # sting 仍作 fallback 保留.
                        music_from_main = intro_outro_cfg_audio.get("intro_music_from_main", False)
                        music_from_clip = intro_outro_cfg_audio.get("intro_music_from_clip", False)
                        if music_from_main:
                            # 2026-06-30: 片头=主体"完全终止落点"前最后几小节 (截掉前面铺垫).
                            # 片头音频=主体[offset:phrase_end] (offset/end 由 20_intro_outro 算, phrase_end=8小节
                            # 完全终止). 结尾锁定落点 → 节拍完全落下且片头短. 主体段原声 0~main_dur 不变
                            # (与画面同步, 口令动作对齐不穿帮). 接缝(片头末尾完全终止→主体开头新乐段):
                            # 片头末尾 seam 秒淡出 + 主体开头 seam 秒淡入, 软化硬切.
                            # 用户: "接缝处理好, 片头前面可以随意截掉".
                            has_sting = False
                            intro_dur_a = float(ctx.get("intro_dur_aligned", intro_dur_for_audio))
                            intro_off = float(ctx.get("intro_music_offset", 0.0))
                            intro_end = float(ctx.get("intro_music_end", intro_dur_a))
                            seam = 0.35
                            fc_parts = [
                                # 片头段: 主体[offset:phrase_end] (截前面铺垫, 保留完全终止落点), 末尾淡出软化接缝
                                f"[1:a]atrim={intro_off:.3f}:{intro_end:.3f},asetpts=N/SR/TB,afade=type=out:st={max(0.0,intro_dur_a-seam):.3f}:d={seam}[intro_a]",
                                # 主体段: 完整原声 0~main_dur (不迁移, 与画面同步), 开头淡入软化接缝
                                f"[1:a]atrim=0:{main_dur},asetpts=N/SR/TB,afade=type=in:st=0:d={seam}[main_a]",
                                f"[intro_a][main_a]concat=n=2:v=0:a=1[full_a]",
                                f"[full_a]apad=whole_dur={out_dur},afade=type=out:st={fade_st:.3f}:d={audio_fade_d}[a]",
                            ]
                            print(f"    [片头音乐] 截前留落点 (主体[{intro_off:.2f}:{intro_end:.2f}]s, 片头{intro_dur_a:.2f}s, 接缝淡入淡出{seam}s)")
                        elif music_from_clip:
                            has_sting = False  # 不引入 sting input (input 只剩 0:v, 1:a)
                            intro_start = float(ctx.get("intro_start_sec", 0.0))
                            intro_clip_end = intro_start + intro_dur_for_audio
                            # 接缝软化: 片头原声末尾淡出 + 主体开头淡入, 把硬切变软过渡,
                            # 缓解"片头段→主体"音乐位置跳变的断裂感 (结构性跳变无法完全消除).
                            seam = 0.35
                            fc_parts = [
                                # 片头段: 素材原声 [intro_start:intro_clip_end], 末尾淡出
                                f"[1:a]atrim={intro_start:.3f}:{intro_clip_end:.3f},asetpts=N/SR/TB,afade=type=out:st={intro_dur_for_audio-seam:.3f}:d={seam}[intro_a]",
                                # 主体段: 素材完整原声 [0:main_dur], 开头淡入
                                f"[1:a]atrim=0:{main_dur},asetpts=N/SR/TB,afade=type=in:st=0:d={seam}[main_a]",
                                # 片头+主体 concat
                                f"[intro_a][main_a]concat=n=2:v=0:a=1[full_a]",
                                # 补到总时长 (片尾段静音) + 末尾淡出
                                f"[full_a]apad=whole_dur={out_dur},afade=type=out:st={fade_st:.3f}:d={audio_fade_d}[a]",
                            ]
                            print(f"    [片头音乐] 片头画面对应原声 (intro_music_from_clip) start={intro_start:.1f}s")
                        elif has_sting:
                            fc_parts = [
                                f"[2:a]atrim=0:{intro_dur_for_audio},apad=whole_dur={intro_dur_for_audio},asetpts=N/SR/TB[intro_silence]",
                                # 主体段: 原音频 0~main_dur (不变)
                                f"[1:a]atrim=0:{main_dur},asetpts=N/SR/TB[main_a]",
                                # intro 音乐 + 主体 concat (硬切)
                                f"[intro_silence][main_a]concat=n=2:v=0:a=1[full_a]",
                                # 整体淡出末尾
                                f"[full_a]afade=type=out:st={fade_st:.3f}:d={audio_fade_d}[a]",
                            ]
                        else:
                            fc_parts = [
                                # 兜底: sting 不存在 → 维持静音 (行为同 2026-06-28 之前)
                                f"anullsrc=r=48000:cl=stereo[si]",
                                f"[si]atrim=0:{intro_dur_for_audio},asetpts=N/SR/TB[intro_silence]",
                                # 主体段: 原音频 0~main_dur
                                f"[1:a]atrim=0:{main_dur},asetpts=N/SR/TB[main_a]",
                                # intro 静音 + 主体 concat
                                f"[intro_silence][main_a]concat=n=2:v=0:a=1[full_a]",
                                # 整体淡出末尾
                                f"[full_a]afade=type=out:st={fade_st:.3f}:d={audio_fade_d}[a]",
                            ]
                        filter_complex = ";".join(fc_parts)
                    elif need_content > 0.5 and has_intro:
                        # 旧逻辑兜底: 主体音频延伸 (acrossfade), 用于无音频源 + intro 模式
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
                        # 2026-06-20 修复: 检测音频流是否存在, 无音频时跳过音频处理
                        # (避免测试片段或无音视频报错 "Stream specifier ':a' matches no streams")
                        if ExportStage.has_audio_stream(str(audio_src)):
                            filter_complex = f"[1:a]{af}[a]"
                        else:
                            print(f"    [INFO] 源视频无音频流, 跳过音频滤镜")
                            filter_complex = None

                    if filter_complex:
                        cmd = [ffmpeg, "-y",
                               "-i", str(processed_path),
                               "-i", audio_src]
                        if has_sting:
                            # 2026-06-29: 片头音乐 sting → input index 2 ([2:a] 引用)
                            cmd.extend(["-i", str(sting_path)])
                            print(f"    [片头音乐] {sting_path.name}")
                        cmd.extend(["-filter_complex", filter_complex,
                                    "-map", "0:v", "-map", "[a]"])
                    else:
                        # 无音频, 直接复制
                        cmd = [ffmpeg, "-y",
                               "-i", str(processed_path),
                               "-i", audio_src,
                               "-map", "0:v", "-an"]
                else:
                    # 无片头片尾：直接从原片提取音频 + 淡出
                    audio_src = str(audio_path) if (audio_path and Path(audio_path).exists()) else str(ctx.input_path)
                    fade_st = max(0, total_sec - audio_fade_d)
                    has_audio = ExportStage.has_audio_stream(audio_src)

                    if audio_fade_d > 0 and has_audio:
                        af = f"afade=type=out:st={fade_st:.3f}:d={audio_fade_d}"
                        cmd = [ffmpeg, "-y",
                               "-i", str(processed_path),
                               "-i", audio_src,
                               "-filter_complex", f"[1:a]{af}[a]",
                               "-map", "0:v", "-map", "[a]"]
                    elif has_audio:
                        cmd = [ffmpeg, "-y",
                               "-i", str(processed_path),
                               "-i", audio_src,
                               "-map", "0:v", "-map", "1:a"]
                    else:
                        # 2026-06-20 修复: 无音频流, 直接复制视频不混音
                        print(f"    [INFO] 源视频无音频流, 输出无音频")
                        cmd = [ffmpeg, "-y",
                               "-i", str(processed_path),
                               "-c:v", "copy",
                               "-an"]

                cmd.extend(["-vf", vf_final])
                cmd.extend(self._encoder_args(output_cfg))
                cmd.extend(["-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-b:a", audio_bitrate])

            if is_preview:
                cmd.extend(["-t", str(ctx.config.get("preview_seconds", 3))])

            cmd.append(str(output_path))

            result = subprocess.run(
            cmd,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=600)
            if result.returncode != 0:
                # 不再瞎改 cmd 残留参数. _encoder_args 默认 libx264, 不会走 NVENC.
                # 这里任何 ffmpeg 失败是命令本身的问题 (filter / input / map), 不应静默兜底.
                stderr = result.stderr[-300:]
                # 完整打印 cmd 方便诊断
                print(f"    [FFMPEG FAIL] cmd: {' '.join(str(c) for c in cmd)}")
                print(f"    [FFMPEG FAIL] stderr: {stderr}")
                raise RuntimeError(f"07_export 失败: {stderr}")
        else:
            print("    FFmpeg 未安装，直接复制")
            shutil.copy2(processed_path, output_path)

        ctx.set("final_path", str(output_path))

        # 显示输出文件大小
        size_mb = output_path.stat().st_size / 1024 / 1024
        print(f"    输出: {output_path} ({size_mb:.1f} MB)")

        # ---- 多平台分发: 额外输出其他格式 (可选) ----
        # 配置: output.formats: ['9x16', '16x9']
        # 一次跑同时输出抖音 9:16 + YouTube 16x9
        # 主输出保持原格式 (width x height), 额外格式为副本
        extra_formats = ctx.config.get("output", {}).get("formats", [])
        if extra_formats and not is_preview:
            # 提取干净 stem: 去掉 _full_9x16 / _full_16x9 / _full / _final 等后缀
            base_stem = output_path.stem
            for suffix in ("_full_9x16", "_full_16x9", "_final_9x16", "_final_16x9",
                           "_full", "_final"):
                if base_stem.endswith(suffix):
                    base_stem = base_stem[:-len(suffix)]
                    break
            print(f"    多格式分发: {extra_formats}")
            # 重要: 多格式分发的源视频应该是横源 (stabilized/color/warped/h2v)
            # 不能是 final_9x16 (已被 smart_crop 裁成竖版, 再转 16:9 会双重裁切模糊)
            fmt_source_path = (ctx.get("stabilized_path") or
                               ctx.get("color_path") or
                               ctx.get("warped_path") or
                               ctx.get("h2v_path") or
                               str(ctx.input_path))
            for fmt in extra_formats:
                if fmt not in ("9x16", "16x9"):
                    continue
                fmt_w, fmt_h = (1080, 1920) if fmt == "9x16" else (1920, 1080)
                fmt_out = ctx.output_dir / f"{base_stem}_full_{fmt}.mp4"
                # 跨比例时裁切 (满屏, 字幕黑区被裁掉, 视频占满整个画布)
                #   - 9:16 → 16:9: 裁掉上下 30% (intro/outro 黑底 30%+字幕),
                #                  留中间 70% 视频内容, 横向拉伸到 16:9
                #   - 16:9 → 9:16: 裁掉左右 30%, 留中间 70%, 纵向拉伸到 9:16
                # 用 fmt_source 的实际尺寸做参考, 不是 final_9x16
                fmt_cap = cv2.VideoCapture(fmt_source_path)
                if not fmt_cap.isOpened():
                    fmt_cap.release()
                    fmt_filter = f"scale={fmt_w}:{fmt_h}:flags=lanczos"
                else:
                    fsw = int(fmt_cap.get(3))
                    fsh = int(fmt_cap.get(4))
                    fmt_cap.release()
                    if fmt == "16x9" and fsh > fsw:
                        # 9:16 → 16:9: 裁上下 30%, 横向拉伸
                        crop_h = int(fsh * 0.70)
                        crop_y = (fsh - crop_h) // 2
                        fmt_filter = f"crop={fsw}:{crop_h}:0:{crop_y},scale={fmt_w}:{fmt_h}:flags=lanczos"
                    elif fmt == "9x16" and fsw > fsh:
                        # 16:9 → 9:16: 裁左右 30%, 纵向拉伸
                        crop_w = int(fsw * 0.70)
                        crop_x = (fsw - crop_w) // 2
                        fmt_filter = f"crop={crop_w}:{fsh}:{crop_x}:0,scale={fmt_w}:{fmt_h}:flags=lanczos"
                    else:
                        fmt_filter = f"scale={fmt_w}:{fmt_h}:flags=lanczos"
                enc_args = self._encoder_args(ctx.config.get("output", {}))
                # 音频: 优先用 audio stage 产物, 没有则从原视频提取 (与 main final 一致)
                audio_src_for_fmt = str(audio_path) if (audio_path and Path(audio_path).exists()) else str(ctx.input_path)
                if Path(audio_src_for_fmt).exists():
                    cmd_fmt = [
                        ffmpeg, "-y",
                        "-i", fmt_source_path,
                        "-i", audio_src_for_fmt,
                        "-vf", fmt_filter,
                        "-map", "0:v",
                        "-map", "1:a",
                        *enc_args,
                        "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "96k",
                        "-shortest",
                        str(fmt_out),
                    ]
                else:
                    cmd_fmt = [
                        ffmpeg, "-y",
                        "-i", fmt_source_path,
                        "-vf", fmt_filter,
                        *enc_args,
                        "-pix_fmt", "yuv420p",
                        "-an",
                        str(fmt_out),
                    ]
                r_fmt = subprocess.run(
            cmd_fmt,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=600)
                if r_fmt.returncode == 0 and fmt_out.exists():
                    sz = fmt_out.stat().st_size / 1024 / 1024
                    print(f"    [{fmt}] {fmt_out.name} ({sz:.1f} MB, 源={Path(fmt_source_path).name})")
                    ctx.set(f"final_{fmt}_path", str(fmt_out))
                else:
                    print(f"    [{fmt}] FFmpeg 失败: {r_fmt.stderr[-200:]}")

        # 清理中间文件
        self._is_preview = is_preview  # 传 self 给 helper 函数
        if not is_preview:
            self._cleanup_intermediates(ctx.output_dir, output_path)

    def _cleanup_intermediates(self, output_dir, final_path):
        """删除处理过程中的中间文件"""
        video_stem = final_path.stem.replace("_final", "").replace("_full", "").replace("_9x16", "").replace("_16x9", "")
        is_full = "_full" in final_path.name
        intermediates = [
            # "_keypoints.json",  # keep for Shorts lead tracking
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

        # === 集成测试: 验证 Shorts 含诗词 (防 make_shorts 重写时漏掉) ===
        is_preview = getattr(self, "_is_preview", False)
        if not is_preview:
            try:
                import subprocess as _sp
                tests_dir = Path(__file__).parent.parent / "tests"
                for test_name, desc in [
                    ("test_shorts_poem.py", "Shorts 诗词"),
                    ("test_final_video.py", "final 视频元素"),
                ]:
                    test_script = tests_dir / test_name
                    if not test_script.exists():
                        continue
                    r = _sp.run(
                        ["python", str(test_script)],
                        capture_output=True, text=True,
                        encoding="utf-8", errors="replace", timeout=120,
                    )
                    if r.returncode != 0:
                        print(f"    [警告] {desc}测试失败! 检查对应 stage")
                        out = (r.stdout or "") + (r.stderr or "")
                        for line in out.strip().split("\n")[-5:]:
                            print(f"      {line}")
                    else:
                        print(f"    [OK] {desc}测试通过")
            except Exception as e:
                print(f"    [跳过] 集成测试: {e}")
