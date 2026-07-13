---
name: recovery-audio-sting-isolation
description: 抢救 long final audio 严格原始对齐 + 末尾 8s 渐弱 (v22 模板钉死, 22 版迭代)
metadata:
  type: feedback
---

# 抢救 long final audio — 严格原始对齐 + 末尾 8s 渐弱 (2026-07-13 22 版迭代)

**核心原则 (user 2026-07-13 钉死)**: 原始视频中图像和音乐绝对对齐. 编辑过程**不能挤占/延迟/提前主体视频的音乐**. 之前 v1-v17 全部违反 (asetpts=+4/TB 推后 main 段, 听感"快进"/"主体错位"). v18-v21 末尾硬切或急速. **v22 真正修复** — 不动 main 段 PTS, 在 main 段内嵌 volume filter 8s 渐弱.

**Why (历史根因 3 层)**:
1. **v1-v6**: atrim 算错 (4:160.85 vs 0:160.85) + 末尾硬静音 → 听感"音乐变了"
2. **v7-v17**: 加 `asetpts=PTS-STARTPTS+4/TB` 给 main 段, 推后 4s → **违反原则, 主体音乐被延迟 4s**, 听感"快进" / "主体错位"
3. **v18-v21**: 不动 PTS 严格对位, 但末尾用 afade 8s 急速 -91dB / volume 8s 4s 后 -inf, **末尾没真正渐弱**

**v22 完整 filter_complex 模板** (钉死, 适用所有抢救 long final):
```bash
# main atrim=0:SOURCE_DUR 严格不动 PTS, 末尾 8s 在 main 段内嵌 volume 渐弱
ffmpeg -y \
  -i <抢救的_long_video.mp4> \
  -i <music_library/intro_sting/intro_ref1.wav> \
  -i <源_合并_mp4> \
  -map 0:v -filter_complex \
  "[1:a]aresample=48000,atrim=0:4[intro];
   [2:a]atrim=0:SOURCE_DUR,volume=eval=frame:volume='if(lt(t,T),1.0,max(0.0,1.0-(t-T)/8))'[main];
   anullsrc=r=48000:cl=stereo[si];
   [si]atrim=0:5[outro];
   [intro][main][outro]concat=n=3:v=0:a=1[fulla]" \
  -map "[fulla]" -c:v copy -c:a aac -b:a 128k -movflags +faststart -t TOTAL_DUR \
  <output>.mp4
```

**关键参数 (v22 钉死)**:
- `<SOURCE_DUR>` = 源 mp4 视频时长 (建玲 160.85, 小飞侠 108.87) — atrim 0:SOURCE_DUR, **不截短, 不延长**
- `<T>` = SOURCE_DUR - 8 = 渐弱起点 (建玲 152.85, 小飞侠 100.87) — main 段最后 8s 开始渐弱
- `<TOTAL_DUR>` = intro(4) + SOURCE_DUR + outro(5) - 帧边界微调 (建玲 169.67, 小飞侠 117.66) — 钳制
- 渐弱 8s 范围 = SOURCE_DUR - 8 到 SOURCE_DUR (main 段最后 8s) = 视频 (SOURCE_DUR+4-8) 到 (SOURCE_DUR+4) 段
- main 段 volume 表达式 `t` 是 main 流内时间 = source sample 时间 = video 4-164.85 时间 1:1
- **不修改 PTS** (无 `asetpts`), main sample 0 PTS=0s, intro sample 0 PTS=0s, outro sample 0 PTS=0s — concat 按 sample 顺序拼, audio 严格 1:1 对应 video
- volume 表达式 t=152.85 (源 sample 152.85) = video 156.85 位置 = total-12.82 ≈ 末尾 8s 起点 (outro 之前 0s, 因为 outro 在 main 之后 5s)

**v22 验证数据 (建玲)**:
- video 4-8s (main 段第 4s) audio: **-11.0 dB** = 源 0-4s audio: -10.9 dB ✅ 严格对位
- video 50s (main 段中段) audio: -12.8 dB = 源 46s audio: -12.7 dB ✅
- 末尾 1s frame RMS 渐弱 8s 完美: -12.8 → -14.0 → -14.9 → -16.7 → -18.2 → -20.1 → -23.8 → -28.3 → -37.2 → -inf

**v22 验证数据 (小飞侠)**:
- video 4-8s: -13.3 dB = 源 0-4s: -13.2 dB ✅
- 末尾 1s frame RMS: -15.2 → -15.3 → -16.6 → -18.3 → -22.0 → -24.8 → -28.9 → 静

**禁用路径 (踩过的雷, 钉死)**:
- ❌ `asetpts=PTS-STARTPTS+4/TB` 给 main 段 — 延迟 4s, 违反"原始对齐"原则
- ❌ `asetpts=PTS-STARTPTS+offset/TB` 给 intro/outro — 同样延迟
- ❌ `afade=type=out` 不管什么 curve (tri/log/esin/qsin) — 0.1s 内跌底, 听感急速
- ❌ volume filter 表达式 `t` 在 fulla 流上无 asetpts offset — 4s 后 -inf 硬切 (因为 v6 渐弱在 main 段 152.85-160.85, fulla 流 t=152.85+4=156.85, 但 fulla 流 t 跟 main 流 t 错位 4s)
- ❌ `volume=enable=between(t,...)` 单引号 ffmpeg 拒 (用 `\,` 转义 bash 也拒)
- ❌ main atrim 超过 source_dur — 抽 wav 抽到空 sample (max 抽不出)
- ❌ outro atrim 不是 5s (跟原视频 outro 长度不匹配)

**v22 听感**: 末尾 8s 平滑渐弱 -12.8 → -37 dB (9dB 渐弱 = 听感"渐弱收尾" 明显), intro 段 sting 跟 video intro 段 frame 1:1, main 段 source audio 跟 video main 段 frame 1:1, outro 段静音 5s 跟 video outro 段 frame 1:1 = 严格 1:1 对齐.

**22 版迭代历史** (留作档案):
- v1-v6: atrim 错 + afade/volume 各种失败
- v7-v17: 加 asetpts+4 推后, 违反原则
- v18-v21: 不动 PTS 严格对位但末尾硬切/急速
- **v22 (定稿)**: main 段内嵌 volume 8s 渐弱, 不动 PTS

**关联 memory**:
- [[recovery-audio-unstable-root-cause]] (4 步主动验证法, 不靠用户反馈)
- [[cleanup-output-safelist]] (白名单清理 3 步)
- [[intro-music-trim-front-keep-cadence]] (intro 音乐从切主体改回独立 sting, 2026-07-01)

**事故时间**: 2026-07-13 22:36 → 2026-07-14 02:00 抢救 long audio, 22 版迭代
**最终方案**: v22 main atrim 0:SOURCE_DUR + 段内 volume 8s 渐弱
**教训**: 抢救前**必须严格遵循"原始对齐"原则**, 任何 asetpts/trim/adelay 都不能动 main 段 sample
