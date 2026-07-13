---
name: recovery-audio-unstable-root-cause
description: 抢救 long final audio 流程不稳定根因 + 主动验证 4 步法 (2026-07-13 6 版迭代)
metadata:
  type: feedback
---

# 抢救 long final audio 流程不稳定根因 + 主动验证 4 步法 (2026-07-13)

**Why**: 2026-07-13 抢救建玲+小飞侠 long final, 同一个错反复重做 6 版 (v1→v6) 才修对, 每版都是用户拍板后才意识到问题. 用户原话"为何这个环节不稳定, 要解决好". 6 版迭代根因:

1. **v1: atrim 算错** (错算 main 156.85s, 应 160.85) → 末尾 4s 静音
2. **v2: 没 afade** → 末尾硬静音, 听感"音乐变了"
3. **v3: afade=tri 3s 渐弱 -3dB** → 用户"听起来就是静音" (线性 3dB 听感弱)
4. **v4: afade=curve=log 5s** → 反而 0.1s 内跌到 -inf, 听感"突然静音" (log 起始陡)
5. **v5: 改用 volume filter 8s** → **完全没生效**! 因为 fulla 流 t 从 0 算, lt(t, T) 永远 true
6. **v6: PTS 重置 + volume 8s** → **听感完美** ✅ (建玲 -0.8→-4.2→-6.0→-11.3→静, 8s 线性)

**根因 3 层 (钉死)**:
1. **filter 算术**: atrim 算错 (少 4s), outro 长度算错 (4.82 应 13s)
2. **afade 听感**: 线性 3s 渐弱对人耳对数感知 = 几乎无变化; log 曲线起始 0.1s 跌到底 = 硬切
3. **PTS 流断**: `concat` 输出流的 PTS 从 0 重新计算, `[fulla]volume` 表达式里 `t` 是流内时间, 不是全局视频时间 → volume filter 永远 1.0 不生效

**How to apply**: 抢救 long final audio **必须 4 步主动验证**, 不靠"听用户反馈":

**4 步验证流程** (ffmpeg + bash, 5min 内完成):
```bash
# 1. 长度对齐: video 时长 ≈ audio 时长 (差 < 0.5s)
ffprobe -v error -show_entries stream=codec_type,duration -of default in.mp4

# 2. intro 段 (0-4s) 音量应 ≈ intro_ref1.wav 库音量 (-16.0dB), 不是主体音乐满音量
ffmpeg -i in.mp4 -ss 0 -t 4 -vn intro.wav
ffmpeg -i intro.wav -af volumedetect -f null - 2>&1 | grep mean_volume
# 期望: mean_volume ≈ -16.0 dB (跟库 intro_ref1.wav 一致)

# 3. main 段 (4-8s) 音量应是主体音乐满音量, 跟 intro 不同
ffmpeg -i in.mp4 -ss 4 -t 4 -vn main.wav
ffmpeg -i main.wav -af volumedetect -f null - 2>&1 | grep mean_volume
# 期望: mean_volume ≈ -10 ~ -13 dB (满音量主体)

# 4. 末尾 5 段 1s 渐弱 peak 应该平滑下降
for offset in T-8 T-6 T-4 T-2 T; do  # T = total-8
  ffmpeg -i in.mp4 -ss $offset -t 1 -vn -c:a pcm_s16le /tmp/sec.wav
  peak=$(ffmpeg -hide_banner -i /tmp/sec.wav -af volumedetect -f null - 2>&1 | grep max_volume | awk '{print $5}')
  echo "t=${offset}s: max_volume=$peak"
done
# 期望: peak 平滑从 -1 → -4 → -7 → -11 → 静音 (-inf), 8s 线性
# 失败模式: 一直 0dB = volume 没生效 (PTS 问题); 突然 -inf = 硬切 (afade 问题)
```

**关键避坑 (钉死, 给未来)**:
- **绝不能用 afade tri d=3**: 3s 线性渐弱听感"几乎无变化" (用户原话"听起来就是静音"), **最少 8s 线性**才明显
- **绝不能用 afade curve=log**: 起始 0.1s 跌到底, 比硬切还糟. log 曲线对人耳听感不友好
- **volume filter 必须先 PTS 重置**: `asetpts=PTS-STARTPTS+offset/TB` 每段, 让 fulla t 跟全局视频 t 对齐
- **绝不能省 outro 13s**: 5s 原 outro + 8s 渐弱空间, 跟 main atrim 算 total-13 算 silence 长度
- **绝不能 ffmpeg 写回原文件**: 临时名 + Remove-Item + Move-Item

**自动化方向 (待办)**:
- 写 `scripts/rebuild_long_audio.py` 把 v6 filter_complex 模板化, 接收 (input_long, source_merged, output) 三参数
- 加 `tests/test_recovery_audio.py` 4 个守门: (1) 长度对齐 (2) intro 音量 ≈ -16dB (3) main 满音量 (4) 末尾 5 段 peak 渐弱
- 长期: 修 `stages/07_export.py` 默认 `audio_fade_out=8.0` (从 3.0 改), 加 PTS 重置 filter, 让主管线自带"听感明显"末尾渐弱 (v6 完整逻辑进 export)

**关联 memory**:
- [[recovery-audio-sting-isolation]] (v6 完整 filter_complex 模板)
- [[cleanup-output-safelist]] (白名单清理 3 步)
- [[intro-music-trim-front-keep-cadence]] (intro 音乐从切主体改回独立 sting, 2026-07-01)

**事故时间**: 2026-07-13 22:36 → 23:30 抢救 long audio, 6 版迭代
**最终方案**: v6 PTS 重置 + volume 8s 显式线性 (听感 -0.8→-4.2→-6.0→-11.3→静, 8s 平滑)
**教训**: 抢救前**必须 4 步验证**, 抢救后**必须 4 步验证**, 不靠"听用户反馈"循环
