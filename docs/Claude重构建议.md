# Claude 对健身视频流水线重构的建议

## 背景

GPT-5 提出了5个PR的重构方案，我给了反馈：
- PR-3（Manifest）和 PR-4（H2V拆分）最值
- PR-1（配置schema）和 PR-2（stage契约）是过度设计
- PR-5（Metrics）可以顺带做

GPT后来出了精简版，收敛到只做3件事。我基本认可精简版的方向，但有补充。

---

## 我认为真正值得做的3件事

### 1. Manifest 增量恢复（必须做）

**现状问题：**
- 当前 engine 用 `_scan_existing_outputs()` 扫描输出目录，靠文件名猜测恢复缓存
- `*_new.mp4` 这类 fallback 文件经常导致误恢复
- 改配置后旧缓存可能被复用，结果和预期不符

**我的建议：**
- 新建 `pipeline/manifest.py`
- manifest 文件名：`<video_stem>_manifest.json`，放在输出目录
- 每次运行开始时检查 manifest 是否 compatible，stage 成功后更新
- 首批接入：`pose_detect`、`h2v_convert`、`body_warp`、`color_grade`、`ken_burns`

注意：GPT 只列了4个 stage，但 `ken_burns` 生成PNG写FFmpeg编码要 2-3 分钟，`beat_flash` 处理也要近1分钟，这两个其实最需要缓存，应该一起接入。

### 2. H2V 策略拆分（值得做）

**现状问题：**
- `03_h2v_convert.py` 超过 500 行，同时承担：track构建、lead识别、场景判定、片段合并、FFmpeg裁切、cropped keypoints回写
- 想改领操人识别逻辑要在大文件里翻
- 想调全景判定阈值不知道去哪找

**我的建议：**
- 新建 `lib/crop_strategy.py`
- 从 h2v_convert 抽出：`build_tracks()`、`score_track()`、`select_lead_track()`、`classify_frames()`、`merge_segments()`
- 当前行为尽量保持，只做结构迁移，不改算法
- 优先配置化的常量：`top_region_ratio`、`pano_person_threshold`、`min_segment_seconds`、`lead_crop_padding`、`track_match_threshold`

### 3. 轻量 Metrics（顺手做）

**现状问题：**
- 调参靠肉眼判断，效率低
- 不同 preset 的效果只能凭感觉对比

**我的建议：**
- 新建 `lib/quality_metrics.py`
- 输出 `run_metrics.json` 到输出目录
- 第一版只记录：
  - `pose_detect_rate`：关键点检测成功率
  - `avg_person_count`：平均人数
  - `lead_center_jitter`：领操人中心抖动程度（帧间标准差）
  - `output_frame_delta`：实际帧数 vs 预期帧数偏差
  - `video_duration_sec`：总时长（秒）

---

## 我建议暂缓的

### PR-1 配置大一统

理由：当前配置虽然分散，但基本能跑。迁移到 `config["stages"]["pose_detect"]["enabled"]` 需要改所有 preset 文件，工作量大但收益不直接。preset 本身是 yaml 文件，改结构后用户要重写所有 preset，体验差。

### PR-2 Stage 契约体系

理由：`requires`/`provides` 听起来美好，但每个 stage 都要维护 boilerplate。实际项目中 stage 依赖看代码就明白，不需要显式声明。当前主要瓶颈不在这里。

### Context 分层

理由：`runtime`/`assets`/`metrics` 分类边界不清晰。比如 `video_info` 放 runtime 但 `keypoints` 放 assets，这种分类本身就有点武断。维护成本高，收益不明显。

---

## 我额外建议做的（GPT没有提到的）

### 4. 更新 `视频生成流程.md`

这个文档描述的是老流程（`fitness_processor.py`、`loop_video.py`），但实际系统已经变成 `main.py` + 20+ stage 的 pipeline。文档和代码对不上，会让后来者困惑。

### 5. 清理根目录临时脚本

当前根目录有很多调试脚本和临时文件：
- `check_*.jpg` 一堆调试图片
- `dbg_*.jpg` 调试图
- `_*.py` 下划线开头的一次性脚本

建议：建 `tools/` 目录，临时脚本移进去，调试图片统一清理或忽略。

### 6. 输出目录加入 `.gitignore`

output目录里有视频成品和 `_keypoints.json`，不应该进入版本控制。

---

## 推荐实施顺序

1. **先做 Manifest** — 稳住缓存恢复，最快见到效果
2. **同步接入 `ken_burns` 和 `beat_flash`** — 这两个最耗时，缓存收益最大
3. **做 H2V 策略拆分** — 降低后续调参难度
4. **顺手补 Metrics** — 不需要单独花时间，在各 stage 里埋点就行
5. **更新过时文档** — `视频生成流程.md`
6. **整理根目录** — tools/ + 清理调试文件

---

## 关于"配置schema"的最终立场

如果后续真的要做的更完善的方向，我建议不要做全量schema迁移，而是：

- 保持当前 preset 的 yaml 结构不变
- 在 `pipeline/config.py` 里加一层 `normalize_legacy_config()`，让新旧字段都能用
- 不要要求用户重写 preset

这样既保留了配置统一的目标，又不需要用户大范围重写文件。代价是 config.py 里要有一些兼容层代码，但这是值得的。

---

## 总结

GPT的精简版方向是对的，但：
1. Manifest 应该多接入 `ken_burns` 和 `beat_flash`
2. 要更新过时的文档
3. 要清理根目录

这三件事不做，Pipeline 不会崩溃，但会一直有点脏。