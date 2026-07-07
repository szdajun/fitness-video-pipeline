# HANDOFF.md — 当前迭代状态（活文档）

> **新会话先读本文件**（见 `CLAUDE.md` 的"会话开局协议"）。
> 这里只记"现在在做什么 / 上次停在哪 / 下一步 / 待用户确认"，不重复架构（架构看 `docs/PROJECT_DESIGN.md`，规则看 `CLAUDE.md`，历史坑看 `memory/`）。
> **每次会话结束前更新本文件**——这是会话衔接的核心。

最后更新: 2026-07-08 01:48（**张杰1_2 final 修复重跑 + 新增"无源照自动抽源"功能 — 全元素 vision 验证通过 — 待用户拍板 YT 重传**）:

**【本轮任务 (两条线)】**:
1. 用户报张杰1_2 final (YT long pO5h9UXBtI0) **缺汉印/时间戳/爆燃文字** (弹幕正常) → 诊断根因 + 修复 + 重跑
2. 用户"关于换脸，如果没有提供美颜照，那就按照原来的策略自己在视频中找个合适的照片进行超分处理，留下来长期使用" → 新功能: 无源照自动抽帧+GFPGAN超分+长期复用

**【根因 — burst fallback 链缺 watermark_path】**:
- `stages/35_intensity_burst.py` 旧链 `smart_crop/mascot/face_swap/danmaku → energybar` 漏了 `watermark_path`. 张杰无源照→face_swap 跳过→mascot_path/face_swap_path 全 None→burst **跌穿到 energybar_path**(watermark 之前, 无汉印)→下游 danmaku/export 读 burst→final 丢汉印+时间戳.
- **修复**: 链中 `watermark_path` 放 `energybar_path` **之前** (L41). face_swap 缺席时 burst 也接力含汉印视频. 守门 `tests/test_burst_chain_watermark.py` (链顺序钉死).
- 关键认知 (钉死): **每个 stage 的 fallback 链必须含 watermark_path** (汉印/时间戳在 watermark stage 加). burst 当时漏了, danmaku/export 早有.

**【新功能 — 无源照自动抽源 `extract_source_from_video`】** (用户要求, memory face-swap-no-source-self-beautify 策略产品化):
- `tools/face_swap.py` 新增: 无源照时, pose keypoints 每帧 find_lead_person→算 lead 脸 ROI+朝向, 正脸分(nose_conf×肩宽) top_k 帧 → 实读像素 ROI 外扩1.3× `_detect_with_fallback`(det_size=320) 确认有脸 → 选 area×det_score 最大 → GFPGAN 全强度增强 → 复检增强后脸≠0(避 flh 坑) → 存 `tools/{coach}_face.png` 长期复用.
- `stages/37_face_swap.py` 接线: find_coach_face 失败时调 extract_source_from_video, 失败回落 skip 不阻塞.
- **为什么抽 ROI 不抽整帧**: GFPGAN align 帧内最大脸, 群体健身帧最大脸可能是近处路人; ROI 锁 lead→GFPGAN 必修对脸.
- 张杰实测: 抽第5帧 lead ROI(rank=2155)→GFPGAN→`tools/张杰_face.png` (208×208), **vision 确认男性正面清晰脸适合换脸源**. ensure_source_photo 再增强→`张杰_gfpgan.png`.

**【张杰1_2 重跑 (exit 0, 4423s, 增量 burst→shorts)】**:
- 删 6 个 stale 产物 (burst/danmaku/final/full/douyin/yt_shorts, **精确文件名** per 白名单原则), 保留上游 (energybar/watermark/keypoints/color/highlight/intro/outro)
- face_swap **首次跑** (旧版跳过): 4747帧 swap=2526 背面跳过=2221(领操转身) 无pose=0, **无 CUDNN 崩** (arena kSameAsRequested 生效)
- burst 输出名变 `..._faceswap_burst.mp4` (读 face_swap_path, 证 fix 触发, 非旧 energybar_burst)
- stage: face_swap 3056s / burst 446s / danmaku 546s / export 201s / shorts 171s
- 爆燃: **9 处峰值** (norm_i>0.7 候选 13 个, random/beat gate 过 9 个)

**【产物 vision 验证 (全过, 非像素阈值 per memory danmaku-yuv420p-subsampling)】** ✅:
- final@12s: 🔴汉印✓(左上红圆印) ⏰时间戳✓("2026-07-07") 弹幕✓("姐妹身材太好") 爆燃✗(非峰值帧)
- final@95s: 🔴汉印✓ ⏰时间戳✓ 弹幕✓("别放弃!") 爆燃✗
- **final@156s(帧4680, 峰值norm_i=1.139)**: 🔴汉印✓ ⏰时间戳✓ **🔥爆燃✓(中央红色大字"燃")** ← 三元素同帧全证
- (burst中间产物@4680 也确认"燃"红字, 88%置信度)

**【张杰三件套 (output/2026-07-07/, 01:25~01:48 重生)】**:
- `张杰1_2_merged_final_16x9_1920x1080.mp4` 337MB (含汉印+时间戳+爆燃+弹幕+**换脸**[新增]+片头片尾)
- `张杰1_2_merged_final_16x9_1920x1080_yt_shorts.mp4` 68MB (YT Shorts, 含hook)
- `张杰1_2_merged_final_16x9_1920x1080_douyin.mp4` 317MB (抖音, 含hook)
- `张杰1_2_merged_faceswap.mp4` 214MB (face_swap 中间产物, PIP源)

**【未提交工作树】** (per 全局规则"commit 只在用户要求时"): burst fix + auto-source + guard test + 张杰源照. 待用户说提交再 commit.

**【待用户拍板】** ⭐:
1. **YT 重传**: 旧 pO5h9UXBtI0 (缺陷版: 丢汉印/时间戳/无换脸) 是否删→重传新 final? (per memory no-auto-rerun-after-fix, 不自作主张重传; 新 final 远优于旧版)
2. 抖音新 douyin (含换脸) 用户手工传 (memory douyin-manual-upload)
3. 本轮代码改动 (burst fix + auto-source) 是否要 commit

**【下一步候选】**: 用户拍板 YT 重传 / 下一个视频 / Matting Studio.

---

最后更新: 2026-07-07 23:58（**张杰 YT 上传完成 (long+Shorts) — 抖音待用户手工**）:

**【本轮任务】**: 用户"张杰的上传，抖音的我上传" → 上传张杰 YT long + Shorts, 抖音用户手工.

**【阻塞→解决: YT OAuth token 过期 (invalid_grant)】**:
- 上传首次崩 `google.auth.exceptions.RefreshError: ('invalid_grant: Token has been expired or revoked.')` (memory youtube-upload-large-file-wrong-videoid 记的坑, refresh token 失效)
- ComfyUI `youtube_upload.py:get_authenticated_service` 缺陷: TOKEN_YANZHI 缺失时 fallback 到 TOKEN_FILE (可能别的账号), 且过期 token `creds.refresh()` 抛异常**不**落到浏览器重授权分支 → 直接崩
- **解法 (不碰 ComfyUI 代码)**: 新建 `tools/reauth_youtube_fitness.py` 显式跑 `InstalledAppFlow.run_local_server(prompt='consent')` 存到 `TOKEN_YANZHI` (channel='fitness' 正确文件). 旧过期 token 改名 `.expired_Jul` 留证. 用户浏览器登录胭脂虎账号授权 → 新 token 落 Jul 7 23:55 → 重跑上传成功. **token 会再过期, 以后直接跑这个 reauth 脚本**.

**【张杰 YT 上传结果 (public 立即发布, exit 0)】** ✅:
- **long**: https://www.youtube.com/watch?v=pO5h9UXBtI0 (321MB, 【神行太保】张杰持久有氧操 | 持久有氧跟练 | 细柳营健身)
  - verify 修正大文件误拿 videoid: youtube_upload 返回 8aFSdsV5ttg (search 误拿) → verify 双匹配 (标题+新鲜度) 修正为真实 **pO5h9UXBtI0** (第 2 次 search 命中). manifest 已记 pO5h9UXBtI0.
- **short**: https://www.youtube.com/watch?v=T8KQzYlc3AI (66MB, 【神行太保】张杰30秒持久有氧操 | 持久有氧挑战 | 细柳营健身 #Shorts). manifest 已记.
- 两视频均 public 立即发布 (per yt-long-video-publish-immediately).

**【抖音 — 用户手工】** (用户"抖音的我上传"):
- 文件: `output/2026-07-07/张杰1_2_merged_final_16x9_1920x1080_douyin.mp4` 322MB (含 hook + 片头诗词, 9:16 1080×1920, 2:42)
- 不自动传抖音 (memory douyin-manual-upload: 自动传被平台检测封号, 用户决策)

**【新增工具】**: `tools/reauth_youtube_fitness.py` (YT fitness token 过期重授权, 浏览器登录胭脂虎账号). 保留复用.

**【下一步候选】**:
1. 用户手工传抖音 (张杰 douyin 文件已就位)
2. (可选) 艳青1_2 重跑 shorts+douyin 生成 4-bug 修复版重传 (用户已传旧 bug 版; 需用户拍板, 非 agent 自作主张 per no-auto-rerun)
3. (可选) 张杰换脸源照 `tools/张杰.png` (想换脸时用户提供清晰照) → 重跑 face_swap+下游
4. 下一个视频 / Matting Studio Phase 2 升级

**【待用户拍板】**: 艳青1_2 是否要修复版重传.

---

最后更新: 2026-07-07 23:25（**4 bug 修复+张杰1_2 主管线完成+产物验证 4/4 过 — 待用户拍板上传**）:

**【本轮任务】**: 用户上传艳青1_2 后报 4 问题 → 修复后跑张杰1_2 验证"文档问题是否解决".
1. 抖音版无开头爆燃预警(hook)片段
2. YT Shorts hook 开头🔥变方框(tofu 未识别)
3. 竖屏 PIP 领操人背向时覆盖头一部分
4. 竖屏 PIP 和主视频不同步

**【4 bug 修复 (commit 3d45fc5, pre-commit 35 测试过, 全验证)】**:
- **Bug1 抖音无hook**: `39_shorts.py` douyin 调用补 `hook_enabled/hook_dur`; `short_vertical.py:846` hook gate `profile in ("yt_shorts","douyin")` (旧仅 yt_shorts). 现 yt_shorts + douyin 都有 hook.
- **Bug2 🔥tofu**: `render_short_overlay.py` msyhbd 无🔥(U+1F525)字形→方框; 加 `FONT_EMOJI=seguiemj.ttf` + `draw_emoji_cjk_centered` 用 seguiemj 渲🔥+msyhbd 渲"高燃预警"拼接(emoji 不加描边避糊色). 实测 6858px 火焰 vs 2484px 方框; 输出mp4 标题区火焰 34611px.
- **Bug3 PIP背向挡头**: 背向时脸kp(0-6)低置信度被过滤→bbox丢头→PIP压(后)脑检测不到. `compute_pip_rect` 脸不可见但双肩(11,12)可见时, 从肩宽推断头位(肩中点上方~1×肩宽,横±0.5×肩宽)补进 bbox. 新增背向守门测试, **7/7 pip 测试过**.
- **Bug4 PIP提前4s不同步** ⭐核心: `face_swap_path` 是 **workout-only**(stage37 跑在 export07 之前, export 才加 intro/outro; ffprobe 实测 **179.6s vs final 188.6s** 差正好 intro4+outro5). 旧代码 PIP 输入无条件 `-ss skip`(=intro4s)→face_swap 被多跳4s→PIP 比主画面**提前4s**. 修: `pip_seek = skip if pip_src==src_path(final含片头) else 0`. 端到端验证(艳青1_2 yt_shorts): PIP内容 vs face_swap@11.5s(对齐)**MSE 1274** vs @15.5s(旧bug)**MSE 3412**, 对齐帧匹配 2.7×.

**【关键认知 (钉死)】**:
- `face_swap_path` 是 **workout-only**(无片头无片尾). 任何复用它做时间对齐的逻辑**不能套用 final 的 intro skip** — 这正是 Bug4 根因.
- `msyhbd.ttc` **无 emoji 字形**, 渲 emoji 必须用 `seguiemj.ttf`.

**【张杰1_2 主管线 (完成 exit 0, 产物验证 4/4)】**:
- 合并: `张杰1.mp4(173MB)+张杰2.mp4(165MB)` → `source_videos/张杰1_2_merged.mp4` (158MB, 4747帧, ~158s, 1920×1080@30fps)
- 命令: `uv run python -u main.py process "source_videos/张杰1_2_merged.mp4" --preset youtube --shorts-coach 张杰` (后台 bavv4d7yg)
- stage: pose 109s / color(长) / watermark 679s / **face_swap 0.0s 跳过(张杰无源照, 保留本人脸)** / burst 503s / danmaku 542s / export 219s / shorts 171s
- **face_swap 跳过 → PIP 源降级 final_path** → pip_seek=skip(同源) → Bug4 同源同 seek 自动对齐. 张杰想换脸: 放 `tools/张杰.png` 重跑.
- coach_profile 张杰齐全: 花名**神行太保** + 判词"万里征途始于足下,飞毛腿疾如风,马拉松魂燃细柳营" + shorts_poem"天高云淡路远/帅哥美女争先/遥见一骑如烟/细柳营中张哥" + focus 持久有氧 + en ENDURANCE BURN.

**【张杰三件套 (output/2026-07-07/)】**:
- `张杰1_2_merged_final_16x9_1920x1080.mp4` 321MB (YT long, 含片头片尾+弹幕+爆燃+无换脸本人)
- `张杰1_2_merged_final_16x9_1920x1080_yt_shorts.mp4` 66MB (YT Shorts, **含 hook**)
- `张杰1_2_merged_final_16x9_1920x1080_douyin.mp4` 307MB (抖音完整版, **含 hook** ← Bug1 修复)

**【4 bug 张杰产物验证 (4/4 过)】** ✅:
1. **Bug1 抖音hook**: douyin@2s 火焰 28719px + hook 0-4s mean **-74.0dB** 静音(anullsrc). yt_shorts/douyin 都有 hook.
2. **Bug2 🔥emoji**: yt_shorts@2s 火焰 29317px(非方框). seguiemj 渲🔥生效.
3. **Bug3 PIP背向挡头**: 算法层 7/7 pip 测试(含背向补头); 张杰 PIP@(456,24).
4. **Bug4 PIP同步**: PIP vs final@15.5s **MSE 1294**(<5000 同步). (face_swap 跳过→PIP=final同源, seek=skip 对齐; 若启用换脸则 face_swap_path seek=0 对齐, 两路都验过)
- 时长: yt_shorts 34s(hook4+30), douyin 2:42(full+hook4).

**【下一步】**:
1. 待用户拍板上传张杰三件套 (long【神行太保】张杰燃脂跟练|持久有氧耐力燃脂|细柳营健身, **public 立即发布**; Shorts 同; 抖音手工)
2. (可选) 张杰想换脸: 用户提供清晰照 → `tools/张杰.png` → 重跑 face_swap+下游
3. (可选) 艳青1_2 重跑 shorts+douyin 生成修复版供重传(用户已上传旧bug版; 重跑需用户拍板, 非 agent 自作主张)

**【待用户拍板】**: 张杰上传; 艳青1_2 是否要修复版重传.

---

最后更新: 2026-07-07 21:14（**艳青1+2 合并→主管线→三件套 全齐 — hook 默认开首次实战验证 6/6 过**）:

**【本轮任务】**: 用户"有新视频艳青1，艳青2，合并后进行处理，验证一下新功能" (验证 hook 默认开).

**【合并】**: `scripts/merge_clips.py --clips 艳青1.mp4 艳青2.mp4 --output 艳青1_2_merged.mp4` → `source_videos/艳青1_2_merged.mp4` (148.5MB, 5388 帧, 179.6s, 1920×1080, 30fps).

**【主管线 (hook 默认开首跑)】** (`--preset youtube --shorts-coach 艳青`, 5211.5s ≈ 87min, exit 0):
- stage times: pose 124s / color 1333s / highlight 14s / energy_bar 704s / intro_outro 144s / watermark 799s / face_swap 479s / burst 551s / danmaku 671s / export 225s / shorts 165s
- **face_swap: swap 5293/5350 (99%), back:7, 人脸:0帧** — 艳青 换脸命中 `tools/yanqing_face.png` (memory face-swap-yanqing-gfpgan-bad 的修后美颜照), 仅 7 帧背面跳过
- **磁盘考验 (F: 31G→8.7G 谷底→25G 收)**: color_grade 峰 18G / watermark 峰 17G / danmaku 峰 19G, 各自 try/finally 清理后恢复; 谷底 8.7G (danmaku encode 期) 未崩, export/shorts 轻量安然过. 长 179s 视频磁盘确紧张但 survive.

**【最终产物 (output/2026-07-07/)】**:
- `艳青1_2_merged_final_16x9_1920x1080.mp4` 381MB 21:11 (YT long, 含片头片尾+弹幕+爆燃+换脸)
- `艳青1_2_merged_final_16x9_1920x1080_yt_shorts.mp4` 76MB 21:12 (YT Shorts, **含 hook**)
- `艳青1_2_merged_final_16x9_1920x1080_douyin.mp4` 381MB 21:14 (抖音, 无 hook 隔离)
- hook 隔离证据: `_hook_overlay_yt_shorts_*.png` 生成, **无** `_hook_overlay_douyin_*` ✓

**【hook 默认开实战验证 (艳青1_2, 6/6 全过)】** ✅:
1. hook 字幕 @2s: 橙红 25851px + 黄 2396px (高燃预警+先睹为快)
2. opening 诗词 @5s: 黄 30999px (concat 后正片 t 语义正确)
3. pip 白边框 @11.5s: 顶边横线 496px (600px 小窗, concat 未破坏 enable=between)
4. 时长 **34.000s** (hook 4 + 正片 30)
5. hook 段 0-4s 静音 mean -74.4 dB (anullsrc 真静音)
6. **帧精确零错位**: hook 帧 270 == nohook@150 (nonzero=0 逐字节同帧), 邻帧 nohook@155 diff 51%
- → **hook 默认开在真实主管线首跑验证通过** (非孤立 李刚1 test, 而是完整 87min 管线产物)

**【YT 标题 (CLAUDE 钉死 + coach_profiles 胭脂虎)】**:
- long: 【胭脂虎】艳青暴汗燃脂操 | 塑腰弯跟练 | 细柳营健身
- short: 【胭脂虎】艳青30秒暴汗燃脂操 | 塑腰弯挑战 | 细柳营健身 #Shorts

**【本轮 commits】**: 无 (纯跑管线, 代码已在上轮 afacaea/c2f3613 落地)

**【待用户拍板】**: 上传艳青1_2 (long + short, public 立即发布; 抖音手工). 与之前批次 (郭海军/彩娥/枫林红/李刚1) 同等待传队列.

**【下一步候选】**:
1. 用户拍板上传艳青1_2 + 之前批次
2. 下一个视频

---

最后更新: 2026-07-07（**高燃预览开场 hook 上线 — yt_shorts 前 N 秒拼全片最燃段(静音+字幕) + 正片零错位**）:

**【本轮任务】**: 引流功能 — YT Shorts 完播率前 3 秒决定 70%, 旧版固定裁前 30s 开场慢热起步. 用户"开头黄金三秒也是个不错的想法, 可以试试" + 约束"音频对齐不要错位 / 避免音频切换太突兀 / 时长灵活可多".

**【设计】**: yt_shorts 前拼一段**全片最燃窗** (默认 4s, 静音 + "🔥 高燃预警"橙红字幕 + "先睹为快"黄副标), 正片音频**零错位**保留. douyin 完整版零改动 (隔离). 窗口=复用 `35_intensity_burst` 逐帧 motion 食谱 (conf>0.3 关键点位移均值), 滑动窗取 mean-motion 最大起点, 排除首尾各 10%, 自身抗单帧尖刺.

**【实现 (5 文件 + 1 测试, commit afacaea)】**:
- `stages/short_vertical.py`: + `compute_hook_window(kp, crop_segments, fps, total_dur, hook_dur, skip_sec)` → `(hook_start, hook_crop_x)` 或 None; `make_vertical` +`hook_enabled/hook_dur` 参数, hook on 时插 step0 (hook 静音段+字幕PNG, 静态 crop) → step1 (正片不变) → step1.5 (concat demuxer -c copy 零重编码) → step2 音频合并; hook off 原路径
- `stages/render_short_overlay.py`: + `render_preview()` 🔥 高燃预警(橙红 255,80,30 110px bold) + 先睹为快(黄 48px), 中部半透明黑底 (y 38-56%)
- `stages/39_shorts.py`: 读 `cfg.shorts_hook/shorts_hook_dur`, yt_shorts 传 hook_enabled (douyin 不传=隔离)
- `main.py`: + `--with-hook/--no-hook` (默认关 opt-in) + `--hook-duration` (默认 4)
- `pipeline/config.py`: shorts_hook + shorts_hook_dur 加 known keys
- `tests/test_short_vertical_hook.py`: 10 测纯算法层 (高燃窗选择/尖刺不污染/首尾排除/crop 钳制/多段取对段/skip 映射/帧对齐/边界 None)

**【⚠ 音频错位坑 (核心, memory adelay-silence-gapless-strip)】**:
- step2 音频**不能用 `adelay={ms}`** — 它产生的前导静音被 AAC encoder 标记为 **gapless encoder_delay side data**, ffmpeg-based 解码器 (含 YouTube) 解码时整体丢弃 → 主音频从 t=0 越过预览播放 = **音视频错位**
- 诊断证据 (旧版): 容器层 -c copy 0-4s raw 是真静音帧, 但 decode = 30s (不是 34s), 0-4s 测出 -13.8dB 响 = 静音被丢
- **修复**: 改用 `anullsrc` lavfi 源产**真零样本静音** + `concat=n=2:v=0:a=1` 拼主音频 (anullsrc 不被 gapless 剥)

**【验证 (李刚1, 像素非肉眼, 全 6 过)】** ✅:
1. hook 字幕 @2s: 橙红(255,80,30)=25740px + 黄=2407px (高燃预警+先睹为快)
2. opening 诗词 @5s: 黄=23181px (concat 后正片 t 语义正确)
3. pip 白边框 @11.5s: 顶边横线 496px ≈ pip 宽 480 (concat -c copy 未破坏 enable=between)
4. 时长 34.059s (hook 4 + 正片 30)
5. hook 段 0-4s 静音 mean **-74.4 dB** (anullsrc 真静音, 解码后保留)
6. **帧精确零错位**: hook 帧 271 == nohook 帧 150 (nonzero=**0** 逐字节同帧), 邻帧 nohook@155 diff 38% (证明确实匹配同帧非相似内容). → concat 把 main 放在 hook 帧 121 (≈ hook_dur·fps), 零错位

**【默认值 / 调参】**:
- `hook_dur=4.0` (--hook-duration 可配, 范围 3-5; 用户"可以多")
- `shorts_hook` 默认 **True** (2026-07-07 用户拍板"功能稳定后要默认开"已执行; 想关加 `--no-hook`); `hook_dur` 可调 (`--hook-duration` / config `shorts_hook_dur`, 默认 4 范围 3-5)
- 字幕文案"🔥 高燃预警 / 先睹为快" (橙红+黄, 全教练统一, 不调 coach_profiles)

**【待用户拍板】**:
1. 功能默认关, 下次跑主管线时想带 hook 加 `--with-hook`; 旧视频 (李刚1/枫林红/彩娥/郭海军) 要补 hook 需重跑 shorts (per memory no-auto-rerun, 不主动重跑)
2. hook_dur 是否调 (4s 默认, 用户说"可以多多也可以")
3. ✅ 默认已改开 (用户 2026-07-07 拍板"功能稳定后要默认开, hook_dur 可调"; 39_shorts.py default False→True + main help 同步 + docs 同步, commit 本次)

**【本轮 commits】**: `afacaea` feat(shorts): 高燃预览开场 hook — 6 文件 +469/-19, pre-commit 35 passed 零回归

**【下一步候选】**:
1. 用户下一个视频 / 拍板上传之前批次 (郭海军/彩娥/枫林红/李刚1 四套三件套待传)
2. Matting Studio Phase 2 升级

---

最后更新: 2026-07-07（**竖屏画中画小窗 功能上线 — Shorts+抖音 右上 16:9 全景小窗**）:

**【本轮任务】**: 竖屏画中画 (用户「画中画用于竖屏产品, 展现全横幅 16:9 画面, 竖屏主画面以领操人为主题范围很小」+「诗词结束后右上出现, 全程存在, 位置通过计算得到」+「换脸后视频更好, 看难度/有合适产物就用」).

**【设计】**: 竖屏 9:16 从 16:9 裁切丢左右画面 → 右上叠 16:9 全景小窗补场景 (信息互补, 区别横屏 31_pip 冗余=永久关). 内容源降级链 `face_swap_path` (换脸·干净横屏无文字) > `final_path` > source. 时机: 诗词 `opening_end≈6.5s` 结束后全程常驻. 位置不写死: `compute_pip_rect` 用 pose kp 算领操人上半身 bbox 在竖屏分布, 右上贴边扫 y 找最靠上且"领操人覆盖帧 <8%"锚点.

**【实现 (5 文件 + 1 测试)】**:
- `stages/short_vertical.py`: + `compute_pip_rect` (位置计算, 复用 crop_segments+kp, 映射横屏归一化→竖屏像素); `make_vertical` +`pip_src/pip_enabled` 参数, step1_vf 加 pip overlay (`scale+drawbox 3px 白边+enable='between(opening_end,total)'`, pip input 带 `-ss skip` 对齐主画面)
- `stages/39_shorts.py`: ShortsStage 从 ctx 取 face_swap_path (降级 final) 传 pip_src; 读 `cfg.shorts_pip`
- `main.py`: +`--with-pip`/`--no-pip` (默认开) + overrides + `config["stages"]["shorts_pip"]`
- `pipeline/config.py`: shorts_pip + shorts 开关家族 (shorts_yt/douyin/duration/coach/intro_seconds) 加 known keys (避 --no-pip warning)
- `tests/test_short_vertical_pip.py`: 6 tests (compute_pip_rect 不变量: 无kp/无段 fallback / 16:9 / 边界 / 居中领操人→右上 / 多段不同 crop_x)

**【验证 (李刚1, 像素非肉眼)】** ✅:
- yt_shorts: `[pip] 小窗 480x270 at (576,24) 避开领操人`. 抽帧 t3(诗词中) 白边 0.00 (无小窗) / t10·t20 白边 0.76-0.78 (小窗出现+常驻)
- douyin 完整版 217MB: 同位置. t3 白边 0.00 / t10·t60 白边 0.76-0.77
- 领操人 cx≈540 居中 (crop_x=506), 上半身 y≈700-960 → 小窗 [576-1056, 24-294] 右上不挡
- 全量 **142 passed** (136+6 新) 零回归
- 产物在 `_temp/pip_test/` (yt_shorts 52MB + douyin 217MB + 抽帧 png, 不入 git, 供看效果)

**【CLAUDE.md 同步】**: 永久关 Stage 表 pip 行澄清 (横屏 31_pip 关 vs 竖屏 shorts_pip 开) + Post-2026-06-27 加 #9 竖屏画中画 + CLI flag 表加 `--with-pip`

**【待用户拍板】**:
1. ✅ 尺寸定 **600** (用户 2026-07-07 拍板「就定这个尺寸」). yt_shorts+douyin 600 版像素验证通过 (小窗 600x338@(456,24) 避开领操人, 诗词后全程常驻).
2. 功能默认开, 下次跑主管线所有新视频自动带 600 小窗; 旧视频 (李刚1/枫林红/彩娥/郭海军) 要补小窗需重跑 shorts (用户未要求, 不主动重跑 per memory no-auto-rerun).
3. 调参点 (备用): `compute_pip_rect` 默认 `target_w=600`/`overlap_thr=0.08`/`margin=24`; `make_vertical(pip_target_w=...)` 可逐视频传.

**【尺寸迭代 (2026-07-07 用户反馈"空白足够")】**: 默认 `target_w` 480→600 (占宽 44%→56%). `compute_pip_rect` 默认改 + `make_vertical +pip_target_w` 可传参 (方便以后调). 实测李刚1 小窗 **600x338@(456,24)**, y=24 仍靠上 (领操人中下不挡), 像素白边 0.76 同 480 版. test 去硬编码尺寸, 6 passed.

**【本轮 commits】**: `1b1d96c` feat 竖屏画中画 (480 baseline) + 尺寸 tune 480→600

---

最后更新: 2026-07-07 00:00（**李刚1 主管线→三件套 全齐**）:

**【本轮任务】**: 李刚1.mp4 主管线处理 (用户"继续处理李刚1视频").

**【主管线 (tested 李刚 face_swap 已有源照)】** (23:10 ~ 23:59, 50min, exit 0):
- 时序: 23:10 keypoints → 23:21 color/highlight/beatflash → 23:28 energybar (1.7G)
        → 23:29 intro/outro → 23:37 watermark → 23:42 face_swap (289s, 命中 `tools/李刚_face_gfpgan.png` 3.3MB 自美化源)
        → 23:48 burst → 23:54 danmaku → 23:57 export → 23:59 shorts/yt_shorts → 23:59 douyin
- stage times: color 652s / energybar 425s / intro_outro 102s / watermark 471s
        / face_swap 289s / intensity_burst 334s / danmaku 391s / export 154s / shorts 103s
- **总耗时 3025.7s ≈ 50min** (略短于枫林红1+2 65min 因为视频更短 119s vs 151s)

**【视觉验证 (t25/t50/t75s 抽帧)】** ✅:
- 李刚1 男性领操人脸自然 (没被硬换) - 李刚源照就是他自己 GFPGAN 自美化版本, 换脸就是用美化版的脸替换原版
- **旁人完全未触**: pose lead-only 锁脸生效, 周围十多人脸全部保留
- **弹幕全齐**: "新手友好!"粉色 + "小蛮腰养成中!"黄色 + "牛仔裤松了!"橙色 + "996斤"绿色
- **爆燃文字**: "牛仔裤松了"橙色显眼, 男士操配套
- **汉印 + 昵称水印**: "细柳营·胭脂虎 / 2026-07-06" 完整
- **能量条**: 右下绿条可见

**【最终产物 (output/2026-07-06/)】**:
- `李刚1_final_16x9_1920x1080.mp4` 259MB 23:57 (YT long, 128.1s, 含片头片尾+弹幕+爆燃+换脸)
- `李刚1_final_16x9_1920x1080_yt_shorts.mp4` 52MB 23:57 (YT Shorts)
- `李刚1_final_16x9_1920x1080_douyin.mp4` 214MB 23:59 (抖音, intro skip -ss 4s)
- + 之前已存在: 郭海军1_2_merged + 彩娥3 + 枫林红1_2_merged 三件套 (output/2026-07-06/ 7.1G)
- 用户待拍板上传 (按惯例手工传, 与枫林红+郭海军+彩娥同一批次)

**【YT 标题 (CLAUDE 钉死 + coach_profiles 拿昵称/focus)】**:
- 李刚1 long: 【胭脂虎】李刚力量塑形操 | 力量塑形跟练 | 细柳营健身
- 李刚1 short: 【胭脂虎】李刚30秒力量塑形操 | 力量塑形挑战 | 细柳营健身 #Shorts

**【本轮 commits】**: 无 (无代码/文档改动, 仅跑主管线产出)

**【下一步候选】**:
1. 用户拍板上传 → 跑 tools/upload_youtube.py (李刚1 long + short, public 立即发布; 抖音手工)
2. 用户下一视频
3. Matting Studio Phase 2 升级 (per 上轮 HANDOFF line 119-134)

---

最后更新: 2026-07-06 22:36（**枫林红1+2 合并→主管线→三件套 全齐 + 换脸源照入库**）:

**【本轮任务】**: 枫林红1.mp4+枫林红2.mp4 合并处理 (用户"跑主管线，处理枫林红1，枫林红2两个视频，合并为一个文件进行处理").

**【合并】**: `scripts/merge_clips.py --clips 枫林红1.mp4 枫林红2.mp4 --output 枫林红1_2_merged.mp4` (137MB, 4532 帧, 151.07s, 1920×1080, 30fps, 教练名在前).

**【换脸源照入库 — 关键决策】**:
- 用户原大头美颜照 `Desktop/短视频素材/枫林红大头美颜照.jpg` (11.5KB, 348×213) — cv2 能读 (PowerShell 上次 imread 失败是编码 bug 不影响 Python)
- `find_coach_face` 优先级 1 命中 `tools/枫林红_gfpgan.png`: GFPGAN 全强度增强 213×348 → 119.5KB, **insightface 直测 score=0.864 (避 flh 坑)**
- 原图 `tools/枫林红_face.jpg` 同时入库 (11.3KB, score=0.857), 作为 fallback
- commit `0ebb2e4 chore(coach): 枫林红 换脸源照入库` (35 passed 零回归)
- **用户"处理好的用来换脸的照片入库，以后直接用即可"** — 已 commit, 后续枫林红自动命中 gfpgan 首选

**【主管线 (tested 弹幕修复 + 新增 gfpgan)】** (21:32~22:36, 65min, exit 0):
- 时序: 21:32 keypoints → 21:47 color (903s) → 21:55 energybar (523s, 1.6GB) → 21:58 intro/outro
        → 22:09 watermark (610s) → 22:15 face_swap (377s, swap 完成) → 22:22 burst
        → 22:31 danmaku (519s) → 22:34 final export (203s) → 22:36 shorts/douyin
- stage times: color 903s / energybar 523s / intro_outro 121s / watermark 610s
        / face_swap 378s / intensity_burst 439s / danmaku 519s / export 203s / shorts 116s
- 三段裁切: `[crop] 逐段 crop_x (2段, fps=30.00): [0-2195]=545 [2195-4532]=1282` (段2 领操人 cx 右移正确跟随)

**【视觉验证 (t30/t60/t90s 抽帧)】** ✅:
- 换脸成功: 领操人(中间黑衣短裙)脸部是用户美颜照的年轻女性(瓜子脸/眼大/肤白嫩),源中年阿姨的脸完全替换
- 弹幕全齐: "姐妹身材太好了吧!" + "免疫JUP!" + "今天也要卷!" + "腹肌出来了!" + "运动是最好的药!"
- 爆燃文字全齐: 橙色"腹肌出来了" + 绿色"运动是最好的药" + 蓝色"眉颈不酸了"
- 汉印 + 昵称水印: "细柳营·胭脂虎 / 2026-07-06"
- 能量条: 右侧绿条可见
- 旁人脸未触: t60 周围多人脸不变 (pose lead-only 锁脸生效)

**【最终产物 (output/2026-07-06/)】**:
- `枫林红1_2_merged_final_16x9_1920x1080.mp4` 322MB 22:34 (YT long, 160.1s, 含片头片尾+弹幕+爆燃+换脸)
- `枫林红1_2_merged_final_16x9_1920x1080_yt_shorts.mp4` 54MB 22:35 (YT Shorts)
- `枫林红1_2_merged_final_16x9_1920x1080_douyin.mp4` 246MB 22:36 (抖音, 含 intro skip -ss 4s)
- + 郭海军1_2_merged 三件套 (312M/50M/225M, 04:50~04:52) + 彩娥3 三件套 (182M/58M/160M, 06:33~06:35)
- output/2026-07-06/ 总: 6.3G (三套三件套)
- 用户待拍板上传 YT (按惯例明早手工传)

**【YT 标题 (CLAUDE 钉死 + coach_profiles 拿昵称/focus)】**:
- 枫林红1+2 long: 【霸道总裁】枫林红高效有氧操 | 高效有氧跟练 | 细柳营健身 (上轮 枫林红2+3 同标题)
- 枫林红1+2 short: 【霸道总裁】枫林红30秒高效有氧操 | 高效有氧挑战 | 细柳营健身 #Shorts

**【本轮 commits】**:
- `0ebb2e4` chore(coach): 枫林红 换脸源照入库 (大头美颜照 0.857 + GFPGAN 增强 0.864)
- 工作树干净

**【下一步候选】**:
1. 用户拍板上传 → 跑 tools/upload_youtube.py (long + short, public 立即发布, 抖音手工)
2. 下一个健身视频
3. Matting Studio Phase 2 升级 (per 上轮 HANDOFF line 119-134)

---

最后更新: 2026-07-06 04:53（**弹幕从未进 final — 调换 stage 顺序 + 修 fallback 链根治 + commit 211937c**）:

**【本轮根因: 弹幕从未叠加到 final】** (用户 2026-07-06 报 "字幕没有"):
- 抽帧 t=20/30/50/60/100/120/140s 实证 final 视频**无任何弹幕文字** (汉印/标语/能量条之外全没).
- 范围: 艳青1/3_4、小飞侠1_2、丽丽4_5_6、枫林红2_3、郭海军1_2 全部受影响 (2026-06-20 修复埋的坑).
- **三处叠加根因**:
  1. main.py:416-418 stage 顺序 danmaku → burst → export, 弹幕跑过但被 burst 接力跳过
  2. stages/35_intensity_burst.py fallback 链 (84d39a2 钉死) mascot > face_swap > danmaku → burst 接力 face_swap (无弹幕) 输出
  3. stages/07_export.py fallback 链 burst > danmaku → export 接力 burst (无弹幕) 输出
- **修复 (3 处, commit 211937c)**:
  1. main.py 调换顺序: `intensity_burst → danmaku` (L418→L416), 让 danmaku 接力 burst 输出
  2. stages/34_danmaku.py fallback 链加 `burst_path` (在 mascot_path 之前), 让 danmaku 读 burst 输出画弹幕
  3. stages/07_export.py fallback 链把 `danmaku_path` 提到 `burst_path` 之前, export 接力 danmaku 输出
- **效果**: final = face_swap + 爆燃文字 + 弹幕 全部齐; 抽帧 t20s 实证"流汗就是燃脂!"绿色字 + "大哥太猛了!"白字 + "受不了"绿字 都在画面里
- **守门**: `tests/test_burst_danmaku_fallback.py` 7 新测试 (主 fallback 链守门 + 验证 main.py 顺序不能回退) + 11 原 84d39a2 测试 (mascot 优先, 仍 100% 兼容) = 18 passed; 完整 136 passed 零回归
- **保留钉死原则**: 84d39a2 的 `mascot_path > danmaku_path` 顺序 (face_swap 接力优先) 保留, test_burst_mascot_before_danmaku 仍 PASSED

**【验证重跑 (2026-07-06 04:03 ~ 04:53, ~50min)】**:
- 增量重跑 郭海军1_2 process, 03:48 keypoints 重新生成 → 04:03 color → 04:12 energybar → 04:25 watermark → 04:31 face_swap → 04:39 burst (`xxx_faceswap_burst.mp4`) → 04:47 danmaku (`xxx_faceswap_burst_danmaku.mp4` ← 接力 burst) → 04:50 export (`xxx_final_16x9_1920x1080.mp4` 311M) → 04:52 douyin
- stage timing 实测: color 890s / energybar 544s / watermark 630s / face_swap 387s / burst 426s / danmaku 520s / export 190s / shorts 119s (合计 ~65min 串行)

**【本会话所有改动】**:
- `260970c` chore(coach): 艳青换脸源照替换 GFPGAN 增强坏照 (上轮修复)
- `211937c` fix(stage_chain): 弹幕从未叠加到 final — 调换 burst/danmaku 顺序 + fallback 链修复

**【产物清单 (本会话最终)】**:
- output/2026-07-06/ 郭海军1_2_merged 三件套 (含弹幕+爆燃):
  - `郭海军1_2_merged_final_16x9_1920x1080.mp4` 312M 04:50 ← YT long
  - `郭海军1_2_merged_final_16x9_1920x1080_yt_shorts.mp4` 50M 04:51 ← YT Shorts
  - `郭海军1_2_merged_final_16x9_1920x1080_douyin.mp4` 225M 04:52 ← 抖音
- output/ 总大小: 17G → 0.4G (-16.6G, 4 轮清理)
- 用户明早手工上传 (用户 2026-07-06 拍板 "今天小飞侠/艳青的都没上传, 我明天早上上传")

**【YT 标题 (已钉, CLAUDE)】**:
- 郭海军1+2 long: 【老兵不老】郭海军力量燃脂操 | 刚劲塑形跟练 | 细柳营健身
- 郭海军1+2 short: 【老兵不老】郭海军30秒暴汗燃脂操 | 全身塑形挑战 | 细柳营健身 #Shorts

**【未提交工作树】** (本轮 HANDOFF 更新)

**【下一步候选】**:
1. 下一个健身视频
2. Matting Studio 升级 (per 上轮 HANDOFF line 32-37)

---

最后更新: 2026-07-06 06:40（**彩娥3 主管线处理 + 自美化源修复 + 误删恢复**）:

**【本轮任务】**: 彩娥3.mp4 主管线处理
- 输入 `source_videos/彩娥3.mp4` 174M, 81s, 1920×1080, 30fps (mtime 2026-07-06 01:33)

**【自美化源路线 (memory face-swap-no-source-self-beautify)】**:
- 抽帧 (每 30 帧) → 选最正脸 frame=690 t=23.0s, yaw=0.63° det_score=0.762
- 裁肖像 78x94 (含肩) → GFPGAN 全强度增强 → 1024x1024
- **insightface 直测 det_score=0.822 (非毁脸)**, 避 face-swap-gfpgan-ruins-photo 坑
- 存 `tools/彩娥_gfpgan.png` (860KB)

**【环境问题 + 修复 (2026-07-06 首次补装)】**:
- C 盘 100% 满 (244.7G/244.8G), uv pip install gfpgan 失败 (torch 2.10 wheel 写不下 C 盘缓存)
- 用户选: 清 CrashDumps 37M + Doubao 673M + Adobe 1.3G → 释放 ~4.7G, 升到 4.8G free
- 用户关豆包后清完整 Doubao 目录
- `uv pip install gfpgan basicsr facexlib` 成功 (gfpgan 1.3.8 / basicsr 1.4.2 / facexlib 0.3.0)
- _load_gfpgan (走 stub) 加载 GFPGANv1.4.pth device=cuda OK

**【主管线 (tested 弹幕修复 + 新增 gfpgan)】**:
- 时序: 05:19 keypoints → 05:27 color/highlight/beatflash → 05:32 energybar (1.1G)
        → 05:34 intro/outro → 05:40 watermark → 05:44 face_swap → 05:48 burst
        → 05:53 danmaku → 05:54 export final (182M) → 05:55 shorts/yt_shorts → 05:56 douyin
- stage times: color 990s / highlight 13s / energybar 530s / intro_outro 120s / watermark 660s
        / face_swap 388s / intensity_burst 238s / danmaku 275s / export 119s / shorts 84s
- 增量跳过了 keypoints (上次算过 42M)
- 弹幕+爆燃+换脸 全齐 (验证抽帧 t35s: "越跳越健康!" 绿字 + "每天30分远离医院!" 黄字)

**【失误 + 恢复 (2026-07-06)】** ⚠️:
- 清中间产物时用了 `find -name "彩娥3_*" -delete` 这个**误删三件套**的命令 (彩娥3_final_*.mp4 也匹配 "彩娥3_*")
- 后果: 彩娥3 三件套 + 全部中间产物 + manifest 全删, 仅剩郭海军1_2 三件套
- 恢复: 重跑 process 完整流程 (~28min, 06:07~06:35), 重新生成全部产物
- **教训 (写这里)**: 清中间产物**必须**用 `! -name "*_final_16x9_1920x1080.mp4" ! -name "*_final_16x9_1920x1080_yt_shorts.mp4" ! -name "*_final_16x9_1920x1080_douyin.mp4"` 这种**白名单**方式, 不能用 `彩娥3_*` 黑名单方式. 把"产品名前缀"当作中间产物删=灾难.

**【最终产物】**:
- `output/2026-07-06/彩娥3_final_16x9_1920x1080.mp4` 182M 06:33 (YT long, 含弹幕+爆燃+换脸)
- `output/2026-07-06/彩娥3_final_16x9_1920x1080_yt_shorts.mp4` 58M 06:34 (YT Shorts)
- `output/2026-07-06/彩娥3_final_16x9_1920x1080_douyin.mp4` 160M 06:35 (抖音)
- `output/2026-07-06/郭海军1_2_merged_*` 三件套 (312M/50M/225M, 04:50~04:52)
- output/2026-07-06/ 总: 943M (两套三件套)
- 用户明早手工上传 (用户 2026-07-06 拍板)

**【本轮 commits】**:
- `c7d1811` chore(coach): 彩娥3+建玲 换脸源照 (实际只 commit 彩娥_gfpgan.png, 建玲早已在 6e4ee83)
- 工作树干净

**【YT 标题 (CLAUDE 钉死)】**:
- 彩娥3 long: 【孤勇者】彩娥勇气燃脂操 | 勇气燃脂跟练 | 细柳营健身
- 彩娥3 short: 【孤勇者】彩娥30秒勇气燃脂操 | 勇气燃脂挑战 | 细柳营健身 #Shorts
  (从 coach_profiles 孤勇者/勇气燃脂 取, 同彩娥2 line 148)

---

最后更新: 2026-07-04（**bg_swap 多人/单人 验证 RVM 软抠天花板 → 暂停 → 立项自研 Matting Studio**）:

**【RVM 软抠天花板确认 (2026-07-04 用户测试单人美女跳舞)】**:
- 跑 `网红跳舞1.mp4` (10.2s 单人美女), 用 5 种参数 + 2 个旧 commit (114bb5a / 7480fb5) 测 = **8 种全都有 4 个"半透人形"鬼影**
- **RVM mask 单独 alpha 合成 (无 face_swap 无 intersect)** 也有鬼影
- RVM mask 强度图 (>0 显示) = 干净 1 个真人, 但 RVM mask **全 20.21% 像素 α>0** (真人 18.67% + 噪点 1.34%)
- **黑色人形区 (x=100-200, y=300-700) RVM α mean=0.044, 2075 个像素 α>0.01** = RVM 软抠噪点散布背景
- 结论: **RVM 软抠天花板 = bg_swap 路线不可治 (RVM 噪点 = 视觉读成"半透人形"非代码回归)**

**【bg_swap 路线状态 2026-07-04】**:
- 多人 3 人: 之前已暂停 (用户拍板 "3 人身后都站一个不动的人")
- 单人美女跳舞: **本轮也确认不可治** (RVM 软抠天花板)
- 整体: **bg_swap 工具暂停**, 代码 + 守门测试保留 (110 passed 零回归), 5 commit 落 main
- 替代: **主管线 (stages/37_face_swap.py 换脸) 不换背景**, 或投资自研 Matting Studio

**【自研 Matting Studio 立项 (2026-07-04 用户拍板)】**:
- 详细设计: `docs/matting-studio-design.md` (Phase 0 ✅)
- 8 模块架构 + 8 模型蓝图 (memory `cn-video-matting-software-architecture.md`)
- 目标: 健身/网红短视频自动抠像 (单人 + 多人, 1080p 30fps, MP4 输出)
- 模式: 开源 GitHub, Apache 2.0
- 核心技术: RVM 主 + YOLOv8 治鬼影 + SAM2 互动修帧
- 实施路线图: Phase 0 设计 ✅ → Phase 1 CLI 工具 (2-3 月) → Phase 2 GUI + SAM2 (2-3 月) → Phase 3 社区化
- 总投入: 8-10 人月

**【下一步 (用户拍板)】**:
1. 写架构图 (mermaid, 估 1-2 小时) 落 `docs/architecture.md`
2. 写算法细节 (`docs/algorithms.md`, RVM/YOLO/SAM2 论文级, 估 3-4 小时)
3. 创建新 GitHub repo 脚手架 (估 0.5 小时)
4. 开始 Phase 1 编码 (CLI 工具 2-3 月, 1 人)
5. 暂不动 Matting Studio, 等以后
6. 回到主管线继续其他任务

---

最后更新: 2026-07-03（**arm-bolster 被用户拍板推翻 + 转入 D 方案（填洞+alpha门控外扩）根治渗出**）：

**【关键推翻】用户看 `_armbolster.mp4` 报"胳膊周围几乎都渗出"**，截图实证。**我 ⑤ 段"像素 A/B 治渗出"是错的** — 我测的是核心管内部 (scale 1.5) 的 RVM α，用户看到的是**核心管外的过渡环** (scale 1.5→3.0, 也就是胳膊周围半透明过渡带)。**核心管撑实了，环没治**。**根因 = 测量区域错位 + 自我说服**。教训写 memory 备查。

**【根因重测】** 用户截图后 7488 帧全扫 (`_temp/scan_arm_bleed.py`):
- 核心管 (env scale 1.5) RVM α ≈ 0.5 (最差 0.434)
- **过渡环 1 (scale 1.5→3.0) α 平均 0.413**，**99.8% 帧有 >2000 渗出像素**（环1 漏治 4392 px/帧）
- 环1 像素只有 45% 是手臂色，**55% 是背景**（黑上衣+肤色 vs 浅棕路面）
- frame 7093 双手平伸无运动模糊 RVM α 也只 0.4 → 病因非快动/模糊，**RVM 对细长肢体结构性低估**（与上轮 MatAnyone 试点的发现一致）

**【A/B/C 方案模拟 (7488 帧采样 + 严格 halo 度量 = RVM 确信背景区被填)]** (`_temp/simulate_fix{2,3,4,5}.py`):
- **A 盲加宽** `max(rvm, env(3.0))`: 治愈 86.5%, **halo 389%**（撑一漏撑四背景 = 灾难）
- **B 颜色门控** (肤色/黑衣 box): halo 6176px ≈ A（**肤色撞路面色**，颜色门控不可靠）
- **C alpha 阈值 + 模糊**: halo 16-39%, 治愈 37-59%（RVM α>0.15 区本身**斑驳**，模糊治不了孔洞）
- **D alpha 阈值 + binary_fill_holes** (thr 0.08/0.12/0.15): halo 11-14%, 治愈 51-63%（填洞不外扩 → halo 低，但 RVM 在胳膊边缘的 α 也低 → 真实边缘填不到）
- **D+grow2 (内 α>0.15 填洞 + 外 α>0.05 内 grow 2 次=6px)**: **治愈 99.8% + halo 3.0%** ✅ **唯一双达标方案**
  - 关键: grow 用 RVM 自身 α>0.05 当 mask 门（背景 α<0.05 → 自动停，不会扩到背景）

**【D 方案原理 (像素定案)】**:
1. `inner = (rvm_α > 0.15) & pose_arm_zone` → 胳膊内真实像素（含斑驳孔洞）
2. `solid = binary_fill_holes(inner)` → 治斑驳（不外扩 → halo 低）
3. `outer = (rvm_α > 0.05) & pose_arm_zone` → RVM 感到前景的过渡区（背景 α<0.05 自动被剔）
4. `solid_g = dilate(solid, 2 iter=6px) & outer` → 在 RVM 自信的前景内，把填好的核心外扩 6px 到真实边缘
5. `solid_smooth = GaussianBlur(solid_g, 7×7)` → 抹羽边（避免硬切）
6. `final = max(rvm_α, solid_smooth)` → 背景不抬（α 自然 0），胳膊填实
- 这与 ② arm-only bolster 的核心差异: bolster 是"硬管强制 α=1"（不治环），D 是"在 RVM 自信的前景内 grow 找真实边缘"（治环 + 不撑背景）
- 关键反直觉: **grow 必须用 RVM α 门控**，否则会扩到背景 (A 方案的错误)。D+grow 比 bolster 多 6px 外扩 = 治过渡环。

**【D+grow 三参扫参定最优】** (n=7488 全帧):

| grow | 治愈 | halo |
|------|------|------|
| **1 (3px)** | 99.8% | **2.5%** ✅ |
| 2 (6px) | 99.8% | 3.0% |
| 3 (9px) | 99.8% | 3.2% |

**grow=1 最优** — 治愈打平，halo 最低（外扩少→撑背景少）。**选 `--arm-grow 1` (3px)**。`_temp/simulate_fix5.py` 主数据，D+grow1 同框架（grep 后 `iters=2`→`iters=1`）。

**【D 方案工程细节】**:
- 复用现有 `_pose_arm_core_matte` (env scale 1.5) 作为 `pose_arm_zone` — 已有现成, 不需重画
- `from scipy.ndimage import binary_fill_holes` (主管线 .venv 已有 scipy, 2026-07-02 pyproject 加的)
- 流水线加 1 步: per-frame 算 solid_g (小数组 < 100ms), 不破 1.8fps 速度
- in-place `np.maximum(mask, solid_smooth, out=mask)` 保留（上次内存修复）
- `gc.collect()` 每 100 帧保留

**【未 commit 工作树】** ✅ **已 commit (3 拆, 全绿 107 测试)**:
- `a677f6b fix(student_closeup)` COCO2BLAZE 补肘腕膝映射 (独立 bug 修复)
- `4abd7cc feat(bg_swap)` arm-grow 替代 arm-bolster (核心改动, in-place/gc 保留)
- `6199e92 docs(bg_swap)` 坑 9.bis + CLAUDE 条 + HANDOFF 同步

**【视觉验证 (5 帧长片段)】** (60s smoke, t=5/10/15/20/25s 3-stack hstack `_temp/ab_v2_vs_armbolster/ab_60s_5frame.png`):

| 帧 | v2 | armbolster 1.5 | **armgrow 1** |
|----|------|------------|-------------|
| 5s | 2 | 4 | 4.5 |
| 10s | 2 | 4 | 5 |
| 15s | 2.5 | 4 | 5 |
| 20s | 2 | 4 | 4.5 |
| 25s (最差快动) | 1.5 | 4 | 5 |
| **均值** | **2.0** | **4.0** | **4.7** |

**5/5 帧 armgrow 稳定 ≥ armbolster**, 无退化. halo 带宽 armgrow 1-2px (单色软边) < armbolster 2-4px (彩色叠片) < v2 6-15px (灰雾). 优势在运动模糊帧最大. D+grow1 **视觉定稿**.

**【7488 帧完整生产卡点 (待下次会话/低优先)】**: ✅ **memfix 部分治本**: 显式 del frame_sw/out/m3/bgf + gc.collect 100→30 帧 + torch.cuda.empty_cache. 完整 7488 帧生产从 500-900 帧崩 → **93% (6990 帧) 跑通**, RSS 阶梯式下降 2331→1211MB 平稳. **剩 7% 崩 cv2.dnn.blobFromImage 申请 4.9MB** = Windows 进程 working set 被 trim 到 1.2GB, 4.9MB 找不到连续虚拟地址 = **碎片化非泄漏**. 治本 = 在 `face_swap.swap_face` 内部 ROI 显式 del + insightface app.get() 缓存 reset, 反复测试可能仍崩. 替代 = bg_swap 加 `--start-frame`/`--end-frame` 支持分片渲染, 渲完 0-6990 + 6990-7488 两段 concat (治本 + 通用).

**【视觉定稿 (3 帧 t=70/200/225s 真实生产末段)】** (`_temp/ab_v2_vs_armbolster/ab_final_3stack.png`):

| 帧 | v2 | armbolster 1.5 | **d_grow1 (生产)** |
|----|------|------------|------------|
| 70s | 2 | 3.5 | 4.5 |
| 200s (回缩, 最难) | 1.5 | 2.5 | 4.5 |
| 225s (高举末段) | 2 | 3.5 | 5 |
| **均值** | **1.83** | **3.17** | **4.67** |

**d_grow1 三帧显著优于 armbolster 1.5 (4.67 vs 3.17)**, halo <0.5px 单色软边. 真实生产视频 (6990 帧) **D+grow1 视觉定稿成立** ✅.

**【RVM 远处半透真人 "鬼影" 修复 (2026-07-03 用户拍板 "3 人身后站一个不动的人")】** ✅ **集成 YOLOv8-seg 二次确认**:
- 单帧 t=70 视觉验证: 鬼影完全消失 + 3 真人完整保留 + 边缘略软 (RVM α 平滑 YOLO 锯齿)
- 75s smoke 2250/2250 帧 0 崩, 2.0fps (vs 之前 2.4fps = 慢 17% YOLO CPU 推理)
- 4 模型同进程 (RVM + buffalo_l + inswapper + YOLO) **YOLO 强制 CPU** 避 4GB onnx arena 抢
- CLI `--mask-mode rvm|intersect` 默认 rvm 维持, `intersect` opt-in 治鬼影
- 守门 +3 测试 (CLI 存在/默认 rvm / render intersect 分支 / YOLO CPU) → 110 passed 零回归
- docs `BG_SWAP.md` 坑 9.tris + CLAUDE.md bg_swap 条同步
- **完整 7488 帧生产未跑 (等用户拍板)**, 75s smoke 已验证不崩 + 治鬼影

**【120s production 跑通 (2026-07-03 23:36)】** ✅:
- **关键发现**: 剪映 Pro (JianyingPro.exe 2 个进程 40276/33308) 占 GPU 显存 551 MiB + nvenc encoder 抢 → 之前 120s rvm production 早崩. `Stop-Process -Id <pid> -Force` 杀后 GPU 0 MiB
- 重跑 120s intersect: **3600/3600 帧 0 崩**, 149MB, 02:00.00, h264 1280×720 30fps + aac
- **5 帧 (t=10/30/60/90/110s) 视觉判定 intersect 稳定治鬼影** (vs d_grow1 5 帧都有鬼影): 3 真人 + 干净背景 + 无半透人形
- 之前 vision agent 报 120s part1 "有半透复制人叠加鬼影" 是在剪映抢 GPU 状态下跑的 (有误)
- 生产文件: `output/bgswap/网红多人_丽丽_时代广场_intersect_120s.mp4` (149MB)
- **生产可用命令** (GPU 必须干净, 杀 JianyingPro): `python tools/bg_swap.py --video <input> --bg <bg> --coach 丽丽 --preset fitness --swap-all --dsr 0.5 --bg-crop-y 0.61 --arm-grow 1 --mask-mode intersect --no-grounding --no-color-match --no-light-wrap --output <output>.mp4`
- **生产稳定上限**: 120s = 3600 帧 (240s+ 跑崩 3274 帧 OOM, 与本机 8.4GB RAM 可用相关). 240s+ 需分片 (`--start-frame`/`--end-frame` 治本待加)
- **守门 110 passed 零回归 + 7 commit 落 main (a677f6b / 4abd7cc / 6199e92 / 7730275 / c1ce910 / a5cae7d / d65602e)**

**【5 commit 落 main】**:
- a677f6b fix(student_closeup): COCO2BLAZE 补肘腕膝
- 4abd7cc feat(bg_swap): arm-grow 替代 arm-bolster
- 6199e92 docs(bg_swap): arm-grow 同步
- 7730275 fix(bg_swap): render 循环 memfix
- c1ce910 docs(handoff): D+grow1 视觉定稿 + memfix + 卡点
- (本轮新: YOLO 集成 + 守门 + docs)

---

（早些 2026-07-03，**MatAnyone A/B 试点 = 阴性, 换模型死路, 真解=arm-only pose bolster**）：用户「还是要进一步提高抠像技术, 背景渗出肢体虚化没彻底解决, 找更好的办法发挥网红模特魅力」+「SAM模型如何」+「好的先试试」。**A/B 实测 MatAnyone v1 (CVPR2025) vs RVM**(像素级, 不靠肉眼): 源=用户指定 `Desktop/短视频素材/2026-06-01 03-39-35.mp4`(1080×1920, 677帧), 测段 t18-22.5s 高潮(pose 实测手臂速度峰值 **2.56肩宽/帧**), 指标=胳膊核心包络内 mean alpha(生产 `_pose_core_matte` 臂子集)。**关键发现**: 扫源视频发现 RVM 对胳膊最大覆盖率只 **94.5%**(典型80-85%)→ RVM 结构性低估胳膊, 非快动帧也低估。**结果(最佳95.1% seed, clip3 source f540-676 共136帧)**: RVM 高潮avg **0.740**/min **0.340** vs MatAnyone 高潮avg **0.766**(+0.03)/min **0.312(更差)**; 最差帧f622 RVM0.50/MA0.39/bolster1.00; MatAnyone 跑720p(RVM内部仅270×480, 分辨率优势仍不赢→非分辨率问题)。**根因**: 2.56肩宽/帧运动模糊胳膊无软抠模型能跟踪(MatAnyone 记忆传播峰值帧丢胳膊, 方差大好0.97坏0.31非"稳定core"); seed 被 RVM 上限锁94.5%。**结论: MatAnyone v1 不换, 换模型(MatAnyone2/SAM2Matting 同类软抠)是死路**。**真解=确定性 pose 胳膊核心强制 α→1** `max(rvm_alpha, arm_env)` → **已实现见上条 (arm-only bolster)**。**工程**: MatAnyone 装隔离 venv `F:/wkspace/matanyone_trial`(dry-run 71包+pandas3+imageio降级会毁主管线.venv; cchardet 需MSVC编译失败→`--no-deps`+仅运行依赖); `process_video(...,max_size=720)`避RAM爆; 首帧mask走RVM阈值化。试点脚本 `_temp/ab_*.py`, 可视化 `_temp/ab_out/ab_conclusion.png`。memory `matanyone-ab-test-negative`。

（早些 2026-07-03，**彩娥2 处理→无源照自美化源→主管线→YT 上传 public**）：用户「再处理一个彩娥2的新视频，没有美颜照，你看如何处置」+「有白发，头发不整齐能处理吗」。**① 无源照解法(自美化源路线, 李刚2 先例再现)**：彩娥 tools/ 无源照 → 从**彩娥1 final** 探测最大正脸帧 insightface buffalo_l **det_size=(320,320)**(640 漏远景小脸<80px), yaw_metric 选最正 → **f1248 (50px 正脸, score0.616, yaw0.06)** → 裁肖像 240×336 → `ensure_source_photo(force=True)` GFPGAN 全强度增强 → `tools/彩娥_gfpgan.png`(**增强后 insightface 直测 score=0.845 非0脸, 避 [[face-swap-gfpgan-ruins-photo]] 坑**)。**② 白发/乱发不影响换脸(回应用户)**：inswapper_128 只换脸(五官+脸型)**不传头发**, 输出头发=彩娥2视频本身; GFPGAN增强也只修脸不染发; 想美发不在换脸范围(pipeline无美发stage, skin_smooth永久关)。**③ 主管线 `--preset youtube --shorts-coach 彩娥`**(detached, 1483.7s)：face_swap **swap=1610/1610(back=0, 无pose=0) 100%全换脸**, final 抽帧肉眼无换脸失败痕迹。产物 `output/2026-07-02/彩娥2_final_16x9_1920x1080.mp4`(127M) + `_yt_shorts.mp4`(51M) + `_douyin.mp4`(94M待人工) + `_full_16x9.mp4`(108M不上传)。**④ ✅ YT 上传 public 立即发布**：long=`S0uYiJ-d2Ds`(127M<200M 未触发wrapped-200) / short=`qlwKOIfEf3k`(49M)。标题 long「【孤勇者】彩娥勇气燃脂操|勇气燃脂跟练|细柳营健身」/ short「【孤勇者】彩娥30秒勇气燃脂操|勇气燃脂挑战|细柳营健身 #Shorts」(自动从 coach_profiles 孤勇者/勇气燃脂 取)。manifest 两笔已写。抖音 94M 待人工。**已写 memory `face-swap-no-source-self-beautify`**。**小 bug(不阻塞)**：上传 launcher 首跑 SyntaxError(`"tools\upload_youtube.py"` 漏 r 前缀, `\u` 当 unicode escape), 加 r 修复。

（早些 2026-07-02, **枫林红2+3 合并→主管线→face_swap 根因修复→YT 上传 public**）：用户「新视频枫林红2+3 合起来处理」。① 合并 `source_videos/枫林红2_3_merged.mp4`；② 主管线 `--preset youtube --shorts-coach 枫林红`。**face_swap 根因(核心)**: 首跑 final 无换脸 → 排查发现 `tools/flh_face_gfpgan.png`(1024² GFPGAN 增强照) insightface 640+320 双检 **0 脸**(**GFPGAN 过度增强会毁脸照**, 生成假脸细节 detector 反测不到), 却占 find_coach_face 最高优先级 `_face_gfpgan.png` 挡住可用源照。**删该坏照** → 回落 `tools/枫林红_face.jpg`(=枫林红1 用的源, 用户"原来成功换过脸的", 213×348 score0.857)。用户另给"高清原图"要超分: GFPGAN修脸+x2 Lanczos+USM → `枫林红_face.png`(426×696 脸284px score0.868)。③ 重跑 face_swap→下游: **3095帧 swap 100%/back:0**, export NVENC 零崩, 产物 final_16x9(226M)+yt_shorts(48M)+douyin(173M)。④ 抽帧 t30s 视觉验证领操人面部自然无瑕疵。⑤ **✅ YT 上传 public 立即发布**: long=`gdG062jelj0`(215M>200M 触发 wrapped-200, `_verify_uploaded_ytid` 守门纠正 Bt11N1zNct0→gdG062jelj0) / short=`uArbWvDzCBA`(46M 直传)。标题 long「【霸道总裁】枫林红高效有氧操\|高效有氧跟练\|细柳营健身」/ short「【霸道总裁】枫林红30秒高效有氧操\|高效有氧挑战\|细柳营健身 #Shorts」。manifest 两笔已写, 抖音 173M 待人工。**教训(已写 memory `face-swap-gfpgan-ruins-photo`)**: GFPGAN 增强可能毁脸照, find_coach_face 优先级 `_face_gfpgan.png` 不一定比原图可靠; vision API 判脸检测不可信(对 0 脸照报"reliably detectable")。**小 bug(不阻塞)**: manifest 增量写入漏记 face_swap→shorts(只到 watermark), 下次增量会重跑 face_swap, 不影响成品。

（早些 2026-07-02，**重启后收尾: 4 commit 落地上轮三批改动 + 后台重渲网红多人 core-matte 版**）：会话因 token 重启, 上轮(化名/core-matte/YT guard)代码已做完但未 commit, 工作树挂 7 文件。本轮**拆 4 commit 全落 main, 工作树干净**: ①`0b5cc2d fix(coach)` 小红豆化名红线女→大唐红线女(coach_profiles 三处); ②`1b06d72 feat(bg_swap)` pose core-matte 撑实胳膊(bg_swap.py + docs坑9 + tests 14绿 + CLAUDE bg_swap条 + HANDOFF); ③`fcd44c7 fix(upload)` YT长视频强制立即发布(upload_utils guard + CLAUDE平台表 预定18:00→立即发布); ④`1331f79 chore(coach)` 李刚换脸源照入库(3.3M png)。pre-commit hook 4次全 35 passed。**拆 commit 技巧**: CLAUDE.md 同含②(bg_swap条)③(平台表)两 hunk, 用工作树编辑法分离(Edit去掉平台表hunk→add commit②→恢复平台表hunk→add commit③); HANDOFF.md 顶部段混①②日志整体归②(活文档不拆hunk)。**随后启动后台重渲网红多人 core-matte 版**(HANDOFF记的"可选下一步"), task `btohbyr09`, ~70min@1.8fps: 源=桌面`短视频素材/网红多人健身操.mp4`(109M), 出`output/bgswap/网红多人_丽丽_时代广场_v3.mp4`, 参数**复刻v2定稿**(`--preset fitness --swap-all --dsr 0.5 --bg-crop-y 0.61`)+ `--core-bolster 1.0` 唯一新变量 → v2 vs v3 A/B 差异纯来自 core-matte。启动验证: RVM+buffalo_l+inswapper 三模型加载OK GPU 2251MiB 无OOM, 100/7488帧 core撑实100, 换脸100/背跳0/漏0。**v3 评估完成 → core-matte 默认反转 开→关**: 用户看 v3 全片判"不干净/基本都这样"(骨架带每帧硬抬 α 轮廓显脏, v2 软边更净), 选 **v2 定稿**. 已落 commit `6ad4507`(core default 1.0→0.0 + test/docs/CLAUDE) + `ec18246`(uv.lock OAuth). part B 查 v2 软边能否软化 = **不宜**(bg_swap.py:1284-1286 erode/feather 试过致人体变薄已回退). 详见下条 + memory `bg-swap-core-matte-arm-bleed`.

（2026-07-02 本轮续, **core-matte 默认反转 开→关 + v2 软边定稿确认 + part B 不宜软化**）：用户看 v3 全片(`output/bgswap/网红多人_丽丽_时代广场_v3.mp4`, `--core-bolster 1.0`)判"**不干净 / 基本都这样**" — v2 也有但"**轻很多**". 用户选 **3: v2 作定稿 + core 默认改回关 + 查 v2 软边能否软化**. **① core 反转已落 commit `6ad4507`**: `tools/bg_swap.py:1497` `--core-bolster` default **1.0→0.0** + help 反转说明; render ②.5 逻辑保持原样(gate 0.25 / >0.05 / global max)仅加诊断澄清注释; `tests/test_bg_swap_defaults.py::test_core_bolster_builtin_default_off` 断言 0.0(14 绿); `docs/BG_SWAP.md` 5 处默认开→关+理由(L121/123/135/159/236) + `CLAUDE.md` bg_swap 条同步. **② uv.lock 单独 commit `ec18246`**: 补 YT OAuth 依赖(google-api/auth + cryptography, a2227bc 漏 commit 的 lock). pre-commit hook 两次 35 passed. **③ core 反转根因(实证)**: env 实测仅覆盖画面 **~3%**(骨架细带), 单帧面积小但**每帧每人轮廓沿线都有** → 动态视频骨架带硬抬 α 痕迹**全片可见**, **静态抽帧难察觉**(一度抽 3 个 hstack 帧 t70/227/248 误判 v3 更好, 靠用户看动态视频纠正 → **教训: bg_swap 判动态全片缺陷必须看视频/逐帧, 不能靠几个静态帧**). **④ part B 查 v2 软边 = 不宜软化**: `tools/bg_swap.py:1284-1286` 已有教训 — `_clean_alpha(erode+feather)` 曾试治脚浮 halo → "**人体忽然变薄**"(erode/feather 缩边+虚化边缘, 叠加自然宽度变化在转身帧显眼) + halo_score 实测 erode 未真降 halo(浅残留非 α 边缘问题) → **2026-06-30 已回退**, 当前合成(L1287-1290 `out=frame*m3+bg*(1-m3)`)用 **raw RVM α 无后处理**. 即 v2 软边 = RVM 固有软抠, raw α 是当前最优. 同阴影 6 轮弃(memory `bg-swap-tool-influencer`)的模式: 有些"缺陷"是 soft matting 固有非 bug, 强修副作用 > 收益, 接受它. **工作树干净, 本轮收尾**. `output/bgswap/` 清理后留 4 个: `网红多人_丽丽_时代广场_v2.mp4`(313M, 多人定稿)+`_v3.mp4`(313M, 留作 core-matte 对比)+`网红_丽丽_时代广场_grounding.mp4`(单人定稿)+`网红跳舞1_时代广场.mp4`(dance 定稿); **删 dsr05(v1)+ 16 单人调试版**(camfollow/frontpuddle/grey_reinhard/heelshadow/laggy/noparallax/noshadow_parallax/precolormatch/preheelshadow/seam/seamfull/shadowfix/v7static/weakshadow/foottrack + 无后缀, 释放~450M, 用户"留最终文件做对比分析用其他删除"). memory `bg-swap-core-matte-arm-bleed`(反转决策 + part B 结论已补).

（2026-07-02 本轮, **小红豆化名 红线女→大唐红线女**）：用户"将红线女的化名改为'大唐红线女'"。`lib/coach_profiles.py` 小红豆 profile 三处全改(`nickname`/`hook` L80/83 + `shorts_en_subtitle` L89 嵌入名) → YT/抖音标题渲染【大唐红线女】小红豆…, `_NICKNAME_MAP` 自动更新(大唐红线女→小红豆, 旧"红线女"无映射)。**用户明确"已经上传的不变了"** → 2026-07-02 已发布两条(ruaTMwDjPXk long / HbRtPsJRdj8 short, 标题仍【红线女】)**不回改**, 改名只作用于未来上传。95 测试零回归。memory `coach-rename-frozen-published`。**⚠ 用户新规(必读)**: YT **宽幅/横版长视频上传必须"立即发布"(public, 不 scheduled)** — 近期凡是延迟/预定发布的(scheduled)全部挂死在平台得不到处理。原 CLAUDE.md 平台表"YouTube 预定 18:00 / Shorts 预定 18:30" 已过时, 待清理。详见本文件下方 + memory `yt-long-video-publish-immediately`。

（2026-07-02 本轮, **bg_swap 修胳膊虚化/原背景渗出 = pose core-matte 撑实, 像素+视觉双通过**）：用户"换脸成熟了, 就是换背景总出现人物胳膊虚化、原背景渗出, 高举手胳膊圈起来时最明显"。**根因(实证非bug非param)**: RVM 是**软抠像**, 对细/快动结构(胳膊)系统性低 alpha — pose 骨架包络内胳膊核区 mean α 仅 **0.52**(半透明=渗出处)非 1.0; 提 dsr(0.25→0.5)只缓解(锐度+39%但 α 仍<1)。外部佐证 MatAnyone(arXiv:2501.14677)/ViTMatte/Generative Video Matting 都为修这个存在。**修复 `_pose_core_matte` = VFX "core+edge matte split"**: pose 骨架包络(线段+圆, 按肩宽缩放 臂0.42/躯干0.62/腿0.46)=硬 core; RVM α=软 edge; `mask=max(rvm_alpha, env×gate)` **只抬不降零风险**; gate=RVM 已感人体(α>0.05)邻域 dilate(核~0.25肩宽)→包络只在真实人体附近激活, 不会在干净背景造"幻肢"(粘原背景); 多人并集(不只 lead); 臂圈洞正确(圈内无骨架→保持背景, 圈周胳膊撑实→无渗出)。CLI `--core-bolster 1.0`(默认开)/`--no-core-bolster`。**像素验证**(`_temp/probe_core_matte.py`, 网红多人 t40-60s 裁片高举臂圈起来帧窗口 285-315): 核区 mean α **0.520→0.965**; 渗出 px(α<0.7) **15697→1260(减92%)**; f300 单帧 17358→1644(减90.5%); diff 热力图撑高集中胳膊/上半身、躯干内不动。**A/B 视觉双确认**(各自 600 帧 30s 裁片): A(baseline 纯 RVM, core撑实0)中间人高举左臂明显半透明背景渗出; B(core-matte+grounding0.18+foot-track, core撑实600)胳膊**实心**轮廓完整边缘锐利无"幽灵臂"背景未穿透, 脚地接触自然无贴纸悬浮。**附带修 latent bug**: plan B2 加 `--pink-thresh-*` passthrough 时 `args.pink_sat` 拼错 dest(应 `args.pink_thresh_sat`)→**所有 bg_swap 渲染 AttributeError 崩**(丽丽定稿是加这 CLI 前跑的所以当时没事, 之后没再跑渲染没人发现)。**已落**: docs/BG_SWAP.md(坑9+架构决策行+CLI 段+多人命令+调试技巧9+测试数 11→14); tests/test_bg_swap_defaults.py 3 新测试(`test_core_bolster_builtin_default_on`/`test_pose_core_matte_function_exists`/`test_pink_thresh_passthrough_dest`, 14 全绿); CLAUDE.md bg_swap 条加"pose core-matte 撑实胳膊(14 tests)"; memory `bg-swap-core-matte-arm-bleed`+MEMORY.md 指针。**结论**: core-matte **默认开**, 网红多人等老素材下次重渲自动受益(无需改命令); 用户两关切(胳膊渗出+滑动)B 版均改善。可选下一步: 用 core-matte 重渲网红多人完整片(7488帧~78min)替换 v2。详见 memory `bg-swap-core-matte-arm-bleed`。

（早些 2026-07-02, **李刚2 换脸补跑完成 + embedding 验证三 criterion 全过, 零代码改动**）：用户"新视频李刚2 处理一下, 也验证 uv 迁移调整是否有问题"。**① 主管线首跑(无换脸, 2371s exit0)零回归** — uv 迁移(3.11+路径可移植+logging+pre-commit hook)未回归, 片头/CTA 黄字像素>2000, 竖屏领操人居中完整。**② 李刚换脸**: 李刚无照片 → 抽帧 f_002@14s(**61px 正脸, det_score0.89 yaw5.5°**) GFPGAN 增强生成 `tools/李刚_face_gfpgan.png`(=李刚自己, **自美化源非换别人**)。删 danmaku/burst/final/shorts/douyin 下游 6 产物 → 增量重跑 face_swap→danmaku→burst→export→shorts(1131s)。**③ 关键架构澄清(推翻上会话 Explore 结论)**: 上会话 Explore agent 报"export(07)在 face_swap(37)前跑→换脸进不了 final, 建议改 export 链 mascot 优先 burst" — **是错的, 未采纳**。stage 执行顺序=**main.py `add_stage` 调用顺序(非文件号前缀!)**: face_swap(main.py:395) 跑在 danmaku(409)/burst(411)/**export(421)** 之前 → 换脸经 `mascot_path→danmaku→burst→export` 正常进 final, export `processed_path` burst 优先 mascot 也无所谓(burst 本身已含换脸)。config.yaml `stages:` 只是 enable/disable 字典, **不定义顺序**。**④ embedding 验证全过**(memory `face-swap-other-faces-false-report`/`face-swap-pose-keypoints-path` 方法, 不靠 swap_count): (a) 领操李刚脸被换 — frame900/1500 pre→src 0.70/0.73 → post **0.91/0.92**(Δ+0.21/+0.19), identity 保 pre~post 0.75-0.86 = **未换错人**; (b) **全部旁人未碰** — 3 帧 30 张旁人脸 Δ≈0.00, pre~post 0.91-0.99(pose only_lead 锁脸生效); (c) **换脸进成片** — final 25/50/75% best→src 0.76/0.87/0.87。face_swap 跑 2871/2871 帧 100% swap/0 背跳。**⑤ 环境 preflight 确认**: onnxruntime 1.22.0 + CUDAExecutionProvider✓(pyproject 钉 <1.23 因 1.23+ 链 cublasLt64_13/CUDA13 系统 CUDA12 只有 12) / torch 2.6.0+cu124 / torchvision 0.21 / insightface 1.0.1 / **gfpgan+basicsr 在(惰性加载, gfpgan_strength=0 时根本不 import; 直接 `import gfpgan` 报 functional_tensor 缺失是假警报, 仅 _load_gfpgan 路径触发且 _patch_torchvision_compat 先打 stub)**。产物 `output/2026-07-02/李刚2_final_16x9_1920x1080.mp4`(212M)+`_yt_shorts.mp4`(60M)+`_douyin.mp4`(195M)。**待用户**: 看成片确认 61px 源的美化效果是否够(自美化温和, 源小效果有限); 可选 YT 上传(传 final_path, douyin 手工)。

**✅ 用户确认"没问题" → YT 上传完成 (public, 2026-07-02)**: long=`p0-Yf_A3WLI` / short=`XQFofkO6keQ`。**上传踩 uv 迁移坑**: 首次跑 `ModuleNotFoundError: googleapiclient` — 上传逻辑借用 ComfyUI `youtube_upload.py` custom node, 需 `google-api-python-client`+`google-auth-oauthlib`, 旧 3.9 venv 有(李刚1 06-29 能传) 但新 .venv(uv 迁移)没装(pyproject 没声明)。**修复**: `uv pip install google-api-python-client google-auth-oauthlib`(不入 gfpgan) + **加 pyproject [dependencies]**(注释标"uv 迁移时漏装过")。重传成功。**long 212MB>200MB 触发 wrapped-200 bug**(memory `youtube-upload-large-file-wrong-videoid` 场景): wrapped 返回 `HbRtPsJRdj8`(陈旧旧视频, 实为小红豆 short 的 id), `_verify_uploaded_ytid` 守门(publishedAt≥900s 判非本次→转 search 标题+频道双匹配)纠正 → `p0-Yf_A3WLI`。short 57MB 直传 `XQFofkO6keQ`。manifest 两笔已写(public)。**抖音 195M 已复制桌面 `短视频素材/李刚2_douyin.mp4` 待人工**(抖音坚持手工传, 自动传被平台检测封号, memory `douyin-manual-upload`)。**未提交改动**: pyproject.toml(+google 上传依赖, 待用户定是否 commit)。

（早些 2026-07-02, **全部改动已 commit (3 拆) + pre-commit hook 修两 bug 恢复守门**）：用户"按建议拆开 commit"。**3 commit 已落 main, 工作树干净**: ① `a6fa6f3 chore(env)` uv+3.11+路径可移植 (11 文件 +1956/-41: uv.lock/.python-version/pyproject/requirements/.gitignore/lib.utils+upload_utils/main/face_swap/CLAUDE/HANDOFF); ② `fd552af chore(cleanup)` 删 59 死代码 + 28 .pyc 出 git + 7 过时 docs→archive (87 文件 -8378); ③ `ab1cddf refactor+tests` except:pass 加日志(engine/export/beat_flash) + 4 新单测(warp/tracker/config_keys/face_swap_ffmpeg) + 历史主管线(short_vertical 逐段裁切/20_intro_outro/merge_clips/config known-keys) (12 文件 +545/-18)。**pre-commit hook (`.git/hooks/pre-commit`, 本地不入 git) 修两 bug**: (a) 裸 `python`→pyenv 3.11.9 无 pytest (测试静默跳过), 改用 `.venv/Scripts/python.exe` (uv venv, pytest 9.1.1); (b) `pytest…|tail` 后查 `$?` 取的是 tail 退出码恒 0 → 失败永不拦, 改输出存临时文件直捕 pytest rc + `-p no:cacheprovider` (避 NUL 缓存警告)。**验证**: hook 直跑 `35 passed`, 强制失败(`--bad-opt`) rc=4 正确传播 = 守门生效 (此前自 0c89025 加 hook 起, 上传/face_swap 守门测试实际从没真跑过/拦过)。详见 memory `pre-commit-hook-venv-pipefail`。

（早些 2026-07-02, 遗留问题清理 #1 GPU torch / #3 ComfyUI 路径 / #4 __pycache__ 清 git + 抖音手工传 memory, 95 测试零回归）：用户问"还有哪些遗留"+ 让处理。已做: ① **GPU torch** (3.11 .venv 原 uv sync 装的 CPU torch 2.12.1) → pyproject 加 `[tool.uv.sources]` conditional (extra=gpu→cu124 index, 复用旧 venv 验证的 2.6.0+cu124 组合) + `uv sync --extra gpu`, 实测 **torch 2.6.0+cu124 cuda_avail=True RTX 4070**; ② **ComfyUI 路径硬编码** (main.py:478 comfy_py + upload_utils.py:11 YT_UPLOAD_PATH 裸 `F:/wkspace/ComfyUI/...`) → `lib/utils.resolve_comfyui_root()` (env `COMFYUI_ROOT` > 已知路径 > None→skip, 镜像 resolve_ffmpeg), 两处改用, 验证 comfyui_root/yt_upload_path 解析正确; ③ **`__pycache__/*.pyc` 清出 git** (升 3.11 暴露 28 个 .pyc 全变 cpython-39→311) `git rm --cached`, .gitignore 已有规则不再入; ④ **抖音手工传 memory** (用户: 自动传被平台检测封号; YT 可自动, 抖音绝自动, `douyin-manual-upload.md`)。剩约束项(非立即动作): numpy 钉<2(2.x 需验证再放开) / mediapipe<0.11(等库出 0.11+)。**全改动已 commit (见本文件首条 3 拆)**。

（早些 2026-07-02, 迁 uv + 升 Python 3.9.13→3.11.14, 95 测试全绿）：用户问"Python 3.9 是否太低 + 相关模块版本是否对齐(ComfyUI)"。诊断: 本项目与 ComfyUI(3.11.9 SAM2/bgswap 子进程) **子进程解耦, 借权重+custom_nodes, 无需版本对齐**; 3.9 唯一硬伤=EOL(2025-10)。**迁 uv 项目模式 + 升 3.11**: `.python-version` 3.9.13→3.11; `pyproject` requires-python>=3.11 + [dependencies] 补全(搬 requirements + 加 scipy/pillow, numpy 钉<2 避 2.x breaking) + optional[gpu]=torch/[dev]=pytest; `uv sync --extra dev` 建独立 .venv(3.11.14, 不动现有 venv/) 生成 uv.lock; `.gitignore` 加 .venv/, requirements.txt 同步。**uv 工作流**: `uv sync --extra dev` / `uv run python main.py ...` / `uv run pytest`。**验证**: 95 passed 零回归 + 主管线 import(main/lib/pipeline)OK。注意 `uv run` 读 .python-version+pyproject 做 sync, 手动 `uv pip install` 后勿混用(会被 sync 覆盖)。ComfyUI venv 独立不动。

（早些 2026-07-02, 项目审查 A/B/C 三项全完成, 95 测试全绿）：无新视频, 做项目审查改进。**[A] 丢弃多余代码**: 删 59 死文件(顶层遗留 add_to_index/auto_publish/batch_publish + 12个 analyze* 临时分析脚本 + legacy/ 历史脚本 build_16x9/fitness_processor/manual_* + coach_avatars/coach_portrait/face_enhance_post/final_check + 7个过时重构 docs), 保留 split_video/bgswap_stable/sam2_bg_swap/cloud_gpu(main.py/stages 运行时在用, 已验证存活)。**[B] except:pass 加日志**(GPU/磁盘错误追根因): 关键 8 处 — engine.py 3(keypoints/cropped_keypoints 缓存读写=缓存毒化根因点) + face_swap.py 3(CUDA provider_options/insightface GPU prepare/图像解码) + 07_export ffprobe 音频探测 + 17_beat_flash beat_track 节拍; 其余 ~45 处合理吞(可选 import/临时清理/GPU cache 清理/fallback return)保留。**[C] 主管线 stages 单测**: tests/test_warp.py(5: identity/shape/waist 收窄方向/body_mask fallback) + tests/test_tracker.py(7: identify_lead_person 选最大体型/LeadPersonSmoother 连续5帧才切+抖动不切), 纯函数不变量无需 GPU/视频。**全量 95 passed 零回归**。待办(下轮选): 无新视频; 可选 bg_swap 残留治理 / 三人长视频识别真人 lead。

（早些 2026-07-02 凌晨后续, 债务包5项, 83 测试全绿）：清理债务包。① **face_swap ffmpeg 硬编码**(tools/face_swap.py:27 裸 C:/Users/18091) → lib/utils.py 加 `resolve_ffmpeg()` 共享(override/env/已知好路径优先PATH/兜底), face_swap 改用它, bg_swap 保留自有(守门测试固化); 加 tests/test_face_swap_ffmpeg.py(3 tests)。② **config 6 条未知项警告** → 活3个(shorts/intro_music_from_main/prefer_gpu)注册 _ALL_KNOWN_KEYS + 死3段(paths/llm/coaches 主管线零读取仅遗留用)列为 _EXTERNAL_CONFIG_SECTIONS 让 validator 跳过; 加 tests/test_config_known_keys.py(3 tests)。③ CLAUDE.md seal.py "stub"过时 → 更正为已实现(AI PNG+PIL兜底, 被 24_watermark 调用)。④ docs/ 7 个过时重构/建议文档 → docs/archive/。⑤ 删 9 个 0 引用 tools/_*.py。全测试 83 passed 零回归。**待办(下轮选)**: 顶层遗留系统 make_video/auto_publish/batch_publish/add_to_index(~2500行, main.py 取代)去留 / 53处 except:pass 加日志 / 主管线 stages 单测。

（早些 2026-07-02 凌晨: **小红豆1+2 合并处理+上传, 两修复点再次验证通过**）：用户给小红豆1+小红豆2 两个新视频合并处理。合并 source_videos/小红豆1.mp4+小红豆2.mp4 → 小红豆1_2_merged.mp4(180M 5358帧 179s)。主管线 --preset youtube --shorts-coach 小红豆 跑通零错误。**两修复点再次验证**: ① [片头音乐] intro_ref1.wav(独立sting, 主体音乐不动与动作对齐); ② 竖屏逐段 [crop] 2段 [0-3260]=499 [3260-5358]=1282, **像素验证段2 领操人 cx0.918(画面最右) 新crop_x=1282窗口[0.668,0.984]框住✅, 旧静态499窗口[0.260,0.577]❌会裁出**。换脸 5357/5358(100%, 背跳1)。**✅ 已上传 YT 立即发布 public**: long=ruaTMwDjPXk(360MB wrapped, 守门纠正 4cr53BQ4kFU→ruaTMwDjPXk)「【红线女】小红豆居家有氧操|居家有氧跟练|细柳营健身」/ short=HbRtPsJRdj8(53MB 直传)「【红线女】小红豆30秒居家有氧操|居家有氧挑战|细柳营健身 #Shorts」。manifest 两笔已写。抖音 307M 已复制桌面 小红豆1_2_douyin.mp4 待人工。

（早些 2026-07-02: **两个复发 bug 已修复 + 郭海军1_2_3_4 验证通过, 已上传**）：用户看 丽丽4_5_6 成品发现两个"以前解决过又犯"的 bug, 不重做 丽丽4_5_6, 改代码后用新视频(郭海军1+2+3+4 合并)验证。**Bug1 片头音乐段落早于动作切换**: 根因 = config `intro_outro.intro_music_from_main: true`(2026-06-30 上次定的"截前留落点"方案)让片头切主体音乐做片头 = **挤占主体音乐** → 合并视频音乐段落切换早于动作。用户原话"片头不能挤占主体的音乐, 否则造成主体音乐提前于动作切换" + "搞了3个片头音乐最后选一个, 主体视频和音乐就对齐了"。**修复**: config 改 `intro_music_from_main: false`(走独立 sting `music_library/intro_sting/intro_ref1.wav`, 主体音乐 `[1:a]atrim=0:main_dur` 原样不动与动作对齐) + `20_intro_outro.py` intro 视频时长 music_from_main 关时用 `intro_duration`(4s)匹配固定 sting(否则 intro 视频3.43s≠sting4s 又错位)。memory `intro-music-trim-front-keep-cadence` 已标撤销。**Bug2 竖屏领操人被裁出画面**: `short_vertical.py` 旧 `compute_crop_x_from_kp` 只取前60帧 cx 中位数 → 单一静态 crop_x, 合并视频第二段领操人移位被裁出(用户"没及时切换领操人跟踪数据"+"竖屏出现")。**修复**: 新增 `compute_crop_x_segments`(每帧最大体型人 torso cx + 3s 滚动中位平滑 + v21 分段 + 段内可靠帧中位数 → ffmpeg `crop` 时间表达式 `if(lt(t,T),x,...)`, 内部逗号转义 `\,`)。丽丽4_5_6 keypoints 单测 6 段正确跟随(443→504→**1282**→603→105→1282); 扒帧实锤 clip2 cx0.83 是丽丽本人(size0.046≈clip1 的0.045, 是她移到最右不是旁人)。memory `shorts-vertical-persegment-crop`。**✅ 验证通过 (郭海军1+2+3+4 合并 `source_videos/郭海军1_2_3_4_merged.mp4`, 192s 5768帧)**: 跑 `python main.py process ... --preset youtube --shorts-coach 郭海军` 完成, 产物齐全(final 405M/shorts 48M/douyin 319M)零错误。两 bug 确认修复: **① 片头音乐对齐** — 管线日志 `[片头音乐] intro_ref1.wav`(has_sting 分支, 主体音乐 `[1:a]atrim=0:main_dur` 原样不动与动作对齐); **② 竖屏逐段领操人跟随** — 管线日志 `[crop] 逐段 crop_x (4段): [0-1310]=511 [1310-2490]=979 [2490-4545]=605 [4545-5768]=1282`, 领操人 cx 跨4 clip 跟随 0.42→0.67→0.48→0.88。**像素验证决定性(每段领操人 cx 中位 vs crop 窗口)**: 段1 cx0.425∈新窗[0.266,0.583]✅ / 段2 cx0.668∈新窗[0.510,0.827]✅(旧静态511窗口❌会被裁出) / 段3 cx0.478∈新窗[0.315,0.632]✅ / 段4 cx0.884∈新窗[0.668,0.984]✅(旧静态511窗口❌会被裁出)。即段2/段4 旧静态会把领操人裁出, 新逐段裁切修复。视觉抽帧 t63s(段2中心) 领操人居中全身在画面 corroborate。**✅ 已上传 YT 立即发布 (public, 用户"立即发布不要延迟")**: long=`sLHXsw1fECo`(386MB>200MB 触发 wrapped-200, `_verify_uploaded_ytid` 守门纠正 fRo1jjex0l0→sLHXsw1fECo, 正是 memory `youtube-upload-large-file-wrong-videoid` 场景) / short=`UV0pBSejfj8`(48MB 直传)。标题自动: long「【老兵不老】郭海军刚劲塑形操 | 刚劲塑形跟练 | 细柳营健身」/ short「【老兵不老】郭海军30秒暴汗燃脂操 | 全身塑形挑战 | 细柳营健身 #Shorts」。manifest 两笔已写。抖音竖版 320M 已复制桌面 `郭海军1_2_3_4_douyin.mp4` 待人工。**注**: HANDOFF 下方 line 56 旧命令仍写 `intro_music_from_main: true`, 已过时以本条 config(false)+memory 为准。

（早些 2026-07-01 晚: **丽丽4/5/6 三段合并→主管线→YT 上传(public)+抖音桌面**）：用户给丽丽4/5/6 三段（丽丽5 中途换高清版）合并后处理上传。**① 合并** `scripts/merge_clips.py`（本轮加 `--clips/--output` 手动指定 CLI + `-r 30` 统一帧率, 解决多源 29.97/30 concat 要求）→ `source_videos/丽丽4_5_6_merged.mp4`（260s 1080p 7810帧）。**② 主管线 `--preset youtube`** 首跑崩在 export（双根因: (a) **NVENC probe 失败** — youtube preset 钉 `encoder:nvenc`, export 启动 `_probe_nvenc()` 真编码 5 帧探测, 但 GPU 被 face_swap+burst 跑一路后编码器会话耗尽 → `raise RuntimeError` 07_export.py:87; (b) **片头拼接失败** — burst/danmaku 也写 ~20G JPEG 帧序列到 `_temp/`（同 color_grade 模式, memory `disk-full-color-grade-temp` 只记了 cg_）, 撑到磁盘 5.8G → `_combined.mp4` lossless 写不下）。**恢复**: GPU 进程全退出后干净（nvidia-smi 190MiB/0进程）, 重跑增量跳过到 export（注意 danmaku/burst 因 `Manifest 不兼容`回退文件检查且输出名没命中, 仍重跑各 ~16/13min）; 腾空间删 4 个上游中间产物（energybar 2.7G + color + beatflash + highlight, export `_cleanup_intermediates` 本来就会删, 提前删安全）; export NVENC 探测+片头拼接双过。**产物**: `final_16x9_1920x1080.mp4`(522M, YT主含片头片尾) + `yt_shorts.mp4`(49M, 30s) + `douyin.mp4`(414M, 竖版完整)。**③ YT 上传 public**（`tools/upload_youtube.py --coach 丽丽`）: long=`aQMoisXY8qA`（522M>200MB 触发 wrapped-200 误拿 `pCCopdG5sTk`, `_verify_uploaded_ytid` 守门标题+频道双匹配纠正, 正是 memory `youtube-upload-large-file-wrong-videoid` 场景）/ short=`ZYXFFRYtEr8`（49M 未触发 wrapped）; 抖音 414M 已复制桌面 `丽丽4_5_6_douyin.mp4` 待人工上传。**未提交**: `merge_clips.py`（--clips CLI + -r 30）待 commit。**操作教训（值得记）**: (1) NVENC probe 对 GPU 状态敏感, 重管线跑一路后可能瞬时不可用, 退出进程等 GPU 干净重跑即恢复, **勿用 --reset-gpu（Windows hang）**; (2) **danmaku/burst/color_grade 都写 ~20G JPEG 序列到 _temp, 单跑有 try/finally 清理但峰值撑盘**, 跑长视频前删上游已烘焙的中间产物腾空间。

（早些 2026-07-01: **网红多人 v2: 修三问题 — 背景拉宽变形/人物变瘦/胳膊消失, cover+避天棚**）：用户看 v1 成品(`dsr05.mp4`)报三个新问题: ① **背景小推车拉宽变形厉害**(根因=背景源 720×1280 竖屏被 prepare_bg 强制 resize 成 1280×720 横屏, 横拉 2.37×; 之前丽丽案例源也竖屏没暴露, 这次横屏源撞竖屏背景才炸); ② **人物变瘦**; ③ **胳膊与背景天棚重叠时消失/虚化/带原始绿树**。②③同源=RVM alpha 在身体边缘/细胳膊系统性偏低 → 合成 `out=frame*mask+bg*(1-mask)` 边缘半透明混背景=轮廓收缩显瘦, 胳膊 alpha 掉近 0 被背景吃掉/半透明带原绿树。**用户方向(治本优于硬刚 alpha)**: "选合适背景角度, 胳膊举起避开天棚"。**像素诊断**: 时代广场背景天棚只集中源 y=25~35% 一小段(放大后 2275 高的 569~796px), 之上全天空之下全地面(y>35% 占 65%高度全地面); cover **中心裁切**取源 34~66% **正好把天棚映射到画面上部**(=胳膊区) → 重叠。**修复两处**: ① `prepare_bg` 强制 resize 改**等比 cover**(`_cover_resize`: scale 覆盖+裁切不变形)治拉宽; ② cover 加竖向偏移 **`--bg-crop-y 0.61`**(0=顶/0.5=中心默认/1=底)裁切窗下移到**天棚下方** → 整个背景变灰砖地面广场, 胳膊举起落干净地面背景(RVM 抠得准不消失/不虚化), 脚踩地面, 天棚降到画面外(上 1/3 天棚 11%→2.1%, 整图 0.7%)。CLI/oversize png 名含 cy 隔离缓存。**v2 已完成 + 验证通过**: 渲染 7488 帧 @1.7fps (4505s), 换脸 100%(7488/7488) / 背面跳 0 / mask 漏 0。**核心验证(三张举手帧 t70s/227s/248s, 正是 v1 报"胳膊消失"的时刻)逐人确认: 胳膊完整可见、无绿树梢残留、举高的胳膊落在干净灰色水泥地面(天棚已移出画面)**; 中段帧 f3000(t100s) 验证人物不变瘦、小推车不变形。**#51 alpha 后处理不需要**(用户方向「避天棚」让 RVM 在干净地面背景上自然抠准, 变瘦+胳膊消失随之解决, 无需激进 alpha 增强)。成品 `output/bgswap/网红多人_丽丽_时代广场_v2.mp4`(299M) 已 sync 桌面 `网红多人_丽丽_时代广场_v2.mp4`。详见 memory `bg-swap-tool-influencer`。**✅ 用户接受 v2 定稿** (原话"先这样吧, 视频播放中还是能看到背景不干净的时候出现, 但很少") — 残留=少数极端举手帧胳膊触及画面上沿天棚残留区/个别帧 RVM alpha 仍偏低, 占比很低可接受; 彻底治备选=#51 alpha 后处理(闭运算/gamma/pose 增广), 当前不触发。

（上一轮 2026-07-01: **网红多人 v1 dsr 修胳膊虚化/绿残留**）：处理 `网红多人健身操.mp4`(3人 720p 7488帧 4分16秒, 中前最大真人+左右两克隆, 户外广场绿树背景)。用户要换脸(**3人都换丽丽** `--swap-all`)+换背景(时代广场)。**30s 试跑发现举手时胳膊虚化+两臂间绿树残留**。**根因(像素定案, 非 analyze)**: 缺陷=**RVM alpha 边缘软光晕**(dsr=0.25 低分辨率 alpha 上采样→胳膊边缘宽软渐变; 举手时两臂内侧软光晕重叠→中间显绿块 + 胳膊虚), **不是封闭凹陷洞误判前景**(strict 像素测"原图强绿树在成品里仍偏绿"=**0**, 旧版新版均无洞绿保存缺陷 → 推翻"凹陷洞 alpha≈1"初判)。**修复 `--dsr 0.5`**(RVM 内部降采样 0.25→0.5): alpha 边缘锐度 **+39%**(0.1326→0.1847), 差异热图证实两版差异**全在轮廓边缘**(躯干内部 0 改动=无洞可填), 躯干带绿 -17%(光晕清理); **零速度代价**(换脸主导每帧成本, 仍 1.5fps, 30s 901帧 589.6s 换脸100%/背面跳0)。**验证方法学(重要)**: analyze 对**同一旧帧** full-frame 报"有绿块+光晕+虚"但 crop 后报"干净", 自相矛盾→**弃 analyze 信像素**; defect 足够细微(视觉模型都看不稳), 30fps 手机播放锐化39%后基本不可见。**全片 7488 帧渲染后台启动**(ID `bv4i8qik5`, ~83min @1.5fps, `--swap-all --dsr 0.5 --preset fitness --coach 丽丽`)。CLI 新增 `--dsr`(默认0.25; **720p 多人细胳膊推荐0.5**, 1080p 单人全身0.25够; 越高越慢≈线性但本例被换脸掩盖)。若用户仍报绿残留, 备选=matte 路径加 mild despill 或小 erode(注意 erode 会让身体变薄, 前车之鉴)。详见 memory `bg-swap-tool-influencer`。

（上一轮 2026-07-01: bg_swap **工具泛化定稿 + 网红跳舞1 处理**）：用户要"整理资料+泛化, 以后能处理类似视频"。**全部完成**: ① **代码泛化** — ffmpeg 可移植 `_resolve_ffmpeg()` (已知好路径 `C:/Users/18091/ffmpeg/ffmpeg.exe` **优先于 PATH**, Winget 版有编码 bug; 换机器自动落 PATH), seg 粉阈值 CLI 化 (`--pink-thresh-rg/sat`), 相机 mask 几何命名常量; ② **预设系统** — `--preset fitness|clean|dance` + `load_bgswap_preset()` 两阶段 parse (preset 覆盖 builtin, CLI 胜 preset); 3 yaml (fitness 实测丽丽 / clean 基线 / dance 起步); ③ **经验总结 `docs/BG_SWAP.md`** (镜像 FACE_SWAP.md: 8 坑+架构决策+调试方法学+加教练/背景 runbook); ④ **守门 `tests/test_bg_swap_defaults.py`** (11 tests 全绿); ⑤ README/CLAUDE/presets 链接。**新工具 `tools/prefilter_person.py`** (换背景前清洗: YOLOv8-pose 逐帧判人物完整性, 剪掉出画/缺头缺脚, 形态学后处理吸收检测 dropout)。**验证案例 网红跳舞1→时代广场** (`--preset dance --coach 丽丽`, 用户选换丽丽脸; 该网红≠丽丽, 脸型圆/方 vs 丽丽瓜子): prefilter 剪 1.37s 入场(仅下半身+黑场)→bg_swap **258/264 换脸(97.7%)**, 渲染 73.9s@3.6fps, 视觉模型评"好"(抠像干净/换脸自然/色温协调/接地合理), **dance 预设首次实测通过**。成品 `output/bgswap/网红跳舞1_时代广场.mp4` + 桌面 cleaned。**已知**: `test_face_swap_lead_selection::test_swap_face_with_bbox_prefers_center_face` 1 个 pre-existing fail (face_swap.py 有未提交改动, 非本轮 bg_swap 引入)。

（上一轮 2026-07-01: bg_swap **接地感增强 grounding = 用户选 C 方案, 区别失败6轮硬阴影**）：第11点定论滑动真因=合成丢真实脚-地物理接触(RVM抠像剥掉接地阴影/光照+背景冻结), **不是阴影不够**(阴影调6轮一直报浮, 已默认关)。用户选 **C 方案"全身时代广场 + 接地感增强"**(不换半身背景A/不接受局限B)。**新增 `_grounding` (和单向硬影根本不同)**: (A) **脚下局部 light wrap 融合**(治硬切分层, 真机制) — 脚底 alpha 羽化带(0.15-0.85, α≈0.5峰)×限脚下 y>=cy-0.08h×eff_wrap, 混【脚下纯地面色】(bg 脚正下后方 α<0.3 mean) = 脚底边缘和地面色融合非硬切; (B) **极弱接地 AO**(物理遮挡感, 0.18 vs 硬影0.5, 软无形状前向衰减 cy+4→cy+18→0.05), 跳起按 lift 减弱。**关键区别**: 不画"影子"(影=单向光投影, 凸显两层), 而是(A)脚底边缘融合恢复接触连续性 +(B)极弱AO遮挡感。**AO 弱已知**(椭圆中心脚正下被遮, 脚两侧地面仅暗ΔV=-0.9级, 同第8点 band 被脚污染); **真机制=(A)脚底融合**, 视觉模型确认 grounding 版三方面优于主版(脚底轮廓更柔和融入/脚下微暗/整体更踩实, `_temp/grounding_vs_main.png`)。**⚠ 判接地最终靠用户主观看动态成品**(单帧/视觉模型测不出"浮"gestalt, 视觉模型早期误报4次), analyze 仅辅助确认非 no-op。CLI `--grounding 0.18`(**默认0关**)/`--no-grounding`; render 签名加 `grounding_strength` 合成前处理。渲染676帧换脸674/676, 成品 `output/bgswap/网红_丽丽_时代广场_grounding.mp4`, 同步桌面 `网红_丽丽_接地感.mp4`。**遗留**: 不够明显可`--grounding 0.3`加大(阴影失败前车之鉴=强易显假)或接受静态背景固有局限。**已删桌面3个旧对比版**(相机跟随/脚钉/钉死, prior session 死胡同几何跟随, 被第11点推翻+grounding取代)。

（上一轮 2026-06-30: bg_swap **【重大转向】滑动真因=真实脚步+合成丢接地接触, 阴影默认关**）：用户报"脚尖和砖缝间距来回变化→滑动, 建议脚下不要阴影", 让从脚-砖缝角度分析。**实证测量推翻两个旧假设**: ① 顶部(远处建筑)phaseCorrelate 测到 1.3-2px/帧"抖动"疑手持, 但**视差**(顶部远脚近, 位移不同), validate 用顶部轨迹修反而 std 10→35 更糟; ② **`measure_footdepth_jitter.py` 测脚深同纹理地砖(左远 x[0,162] 占用0%) dx std=0.33/dy std=0.67px(累计 range 仅10/17px)= 脚深相机几乎静止**, 顶部那 1.3-2px 是远处低纹理区 phaseCorrelate 噪声非真运动。跟抖动修正脚深版 X升21%/Y仅降9% → **背景跟抖动修不了滑动**。**真因定论**: foot_cx 逐帧 std=9.5px 68%方向翻转=**真实脚步移动**(健身重心转移/踩踏), 和源一模一样; 源不滑合成滑, 差别**不在几何**(脚位置/相机同静止), 在**合成丢真实脚-地物理接触**(RVM 抠像剥掉原接地阴影/遮挡/光照+背景冻结无动态响应)→ 同样脚步源读"踩"、合成读"滑"。**用户去阴影建议合理已采纳**: 人造阴影不追踪真实接触反凸显"脚地两层", `_contact_shadow` 默认 `--shadow-strength 0.5→0.0`(render 签名同改, 函数保留)。**此点纠正前 6 轮"加阴影治浮"方向**(v1单椭圆→v2双层→v3前向衰减→v4扁窄脚跟一点, 用户一直报浮/滑, 根因从来不是阴影)。验证: 无阴影版脚下地砖 V 81→107(+26, 阴影已去); 换脸 674/676。成品同步桌面。**判相机静止必测脚深同纹理地砖, 远处低纹理建筑给 phaseCorrelate 噪声假抖动**。**残留滑动=静态背景固有局限**(无真实物理接触), 除非动态背景或接受。

（上一轮: bg_swap **修"前脚掌贴合 + 脚印只脚跟下一点" = 阴影扁窄化 + 前向快衰减**）：用户报"前脚掌要和地面贴合, 脚印也只能在脚跟下面有一点"(上轮回退 clean_alpha 后颜色/变薄都好了, 仅剩脚浮)。**根因**: 旧 `_contact_shadow` penumbra 半高 2.2%h+大核 blur(h*0.028=54px) 向下渗暗到前脚掌区 + vatt 前向衰减慢(cy+34 衰到 0.12) → 前脚掌区地砖被阴影铺暗 → 前脚掌"踩在影上"显浮; 且半宽 10%w 外溢成片(不是"一点")。**修复**(`bg_swap.py:_contact_shadow`): (1) umbra/penumbra **半高减半**(1.1%→0.6%h, 2.2%→1.0%h, 不铺到前脚掌/脚背)+**半宽收窄**(5%→3%w, 10%→5%w, 不外溢成片); (2) vatt 前向衰减 **cy+34→cy+15 衰到 0.05**(原 0.12, 前脚掌区近无影=踩实地砖贴合); (3) blur 减小(umb 0.014→0.010, pen 0.028→0.018, 不渗暗前脚掌)。**像素验证**(`_temp/test_shadow_heel_only.py` 阴影层 profile): 脚跟/正下(cy-30..+4) 强度 0.14~0.32(有影) vs 前脚掌区(cy+15..+50) **0.002~0.013(近无影)**; 成品对比(`_temp/verify_heel_shadow.py`): 新版脚下地砖前脚掌区 V 89.9→**102.6**(blur 不再渗暗=干净实地砖)。成品 676 帧换脸 674/676, 同步桌面。**脚浮待用户主观确认**(若仍浮, 根因可能=静态背景+alpha 边缘固有限制, 非阴影)。

（上一轮: bg_swap **修"人体忽然变薄" = 回退 clean_alpha**）：上一版加了 `_clean_alpha`(erode3+feather6) 在合成处去鞋边 halo, 用户反馈"颜色好了 + 脚下还浮但稍改善" + **新问题"人体忽然变薄"**。**变薄根因**: clean_alpha erode+feather 缩边+虚化边缘, 叠加源 alpha 自然宽度变化(转身)显眼; 诊断(`_temp/diag_thin.py`)源腰宽 303-334px **渐变无突变 = 非 RVM 抖动**, 是 clean_alpha 缩边; 且 halo_score 实测 erode 未真正降 halo(浅残留非 alpha 边缘问题)。**修复**: 去掉合成处 clean_alpha 调用, 回原始 RVM alpha 合成; `_clean_alpha` 函数保留(供未来局部脚下用)。**保留**: 保L色温(t0.8, 颜色好了✅)+阴影 umbra 柔化(blur 0.014, 脚下稍改善)。成品 676 帧换脸 674/676, 同步桌面。**脚浮仍待用户确认**(若还浮, 下一步=阴影 strength 0.5→0.6, 或接受静态背景固有贴纸感)。
（上一轮: bg_swap **变灰修复 = 色温匹配改 mean-shift a/b 保L + clean_alpha 去鞋边 halo**）：Reinhard 色温匹配(t0.6)后用户报"颜色整体变灰, 和原片黑色差异大" + "脚底还浮在空中"。**变灰根因**: Reinhard 同时缩 L/a/b 方差, 把纯黑衣服(L10 远离均值)拉向灰(L27)。**诊断**(`_temp/test_halo_fix.py` 单帧): 黑衣服 L 原片 10.3 / Reinhard **27.2 变灰** / 保L **10.4 保黑**。**脚浮根因**: RVM alpha 边缘浅残留 → 鞋边 halo + 硬切(模型+用户均报"浮在空中"); 脚下其实有 +44 暗影不是阴影不够。**修复**(`bg_swap.py`): (1) `_color_match_to_bg` 改 **mean-shift 只平移 a/b、保 L**(默认 t 0.6→0.8, 只动 a/b 安全可到 1.0) = 黑衣保黑; (2) 加 `_clean_alpha`(erode3+feather6) 合成用, 吃掉鞋边 halo/浅残留; (3) 接地阴影 umbra blur 0.009→0.014 柔化治"黑块"感。CLI `--color-match 0.8`(默认)。**验证**(`_temp/verify_keepL.py`): 黑衣服 L 保L **10.0** vs 灰版 **26.6**(保黑≈原片 10.3), 色温 Δb -3.0(融入); 换脸 674/676。成品同步桌面, 备份 `_grey_reinhard.mp4`(变灰版对比)。详见 memory `bg-swap-tool-influencer` 第 9 点。
（上一轮: bg_swap **贴纸感/脚浮头号成因=色温断层修复 = 全身 LAB 色温匹配 + light wrap**）：阴影 v3 + 视差 + 前向衰减全修好后用户仍报"脚浮/贴纸感/像贴上去"。诊断到像素(`_temp/verify_colormatch.py`: RVM 抠源 f100 alpha 应用到 OLD/NEW 成品同帧算人物 alpha>0.5 vs 背景 alpha<0.2 的 LAB delta): OLD **人物 L=24.7 远暗、Δb=-17.9 严重偏冷蓝**(室内抠的人贴暖户外广场=色温断层)→读成"假/浮/贴纸"; 阴影/视差治不了(脚位影都对, 纯色温断层)。修复 = `bg_swap.py` 加 `_color_match_to_bg`(全身 LAB Reinhard 迁**背景下半部 y>0.4h** 不含天空, t0.6)+`_light_wrap`(alpha 羽化带混背景高斯光 s0.5, 核心不动), render 合成前对人物处理。CLI `--color-match 0.6`(默认)/`--no-color-match`、`--light-wrap 0.5`(默认)/`--no-light-wrap`。验证(像素定案): 人物 L **24.7→37.6**、Δb **-17.9→-3.8**(冷蓝消除)、总色差 |ΔL|+|Δb| **34.5→18.8(减半)**。⚠ analyze 这轮自相矛盾不可信(说用户嫌浮的 OLD"grounded"、NEW"floating", 与像素直接冲突), **判色温/融入靠像素 LAB delta 不靠模型**。渲染前提: bg_swap 同进程跑 RVM(torch)+buffalo_l+inswapper 三 GPU 模型, `face_swap.py _cuda_provider_options` 的 `cudnn_conv_algo_search` 必须 **HEURISTIC**(非 EXHAUSTIVE, EXHAUSTIVE 搜大 workspace algo 瞬时撑爆 12GB → Conv_62 起 OOM)+`gpu_mem_limit 8→4GB`(单独降内存不修, HEURISTIC 才管用)。成品 `output/bgswap/网红_丽丽_时代广场.mp4`(676帧, 换脸 674/676), 备份 `_precolormatch.mp4`(修色温前)。已同步桌面 `~/Desktop/短视频素材/网红_丽丽_时代广场.mp4` 供用户主观判断"脚踩实"是否达成。详见 memory `bg-swap-tool-influencer` 第 9 点。
（上一轮: bg_swap **"前水洼浮"修复 = 接地阴影前向衰减**(治双层影前铺成水洼)）：用户报"还是感觉浮在上面, 哪个阴影不合理, 不如取消, 原视频只有脚后跟那儿在地面有阴影, 前面没有的"。**诊断到像素**: 旧 penumbra 中心 foot_y+0.6%h、半宽 22%w → `_temp/measure_shadow.py` 测**最暗 band = front_near(foot_y+15..40) V=30-37** = 脚尖前方一摊黑水洼(物理错误: 前顶光真投影应落脚后/正下, 不该往前铺)→ 读成"假/浮"。**用户物理观察完全正确**(模型曾误判旧版"影在脚后", 像素推翻)。**修复 = `_contact_shadow` 重设**: (1) **cy 锚 foot_y/ground_y 不前移**(旧 +0.6%h 前移是水洼主因); (2) **penumbra 半宽 22%w→10%w** 收窄; (3) **前向垂直衰减 vatt(钉死别删)**: `yy≤cy+4`=1.0, `cy+4..cy+34` 线性衰到 0.12 → 钉死"脚尖前无影"; (4) 默认 strength 0.65→**0.5**。**验证三联收敛**: ①函数级 `_temp/test_shadow_profile.py` contact=0.418 最暗 > behind=0.158 > front_near=0.097(旧水洼位); ②像素 diff `_temp/diff_front.py`(脚尖前地板无遮挡) NEW V=127-189 vs 旧 `_frontpuddle.mp4` 备份 V=30-37(**+90~+159 变亮 = 前水洼彻底消除**); ③视觉模型"影 ON 接地线/正下, 脚尖前干净无水洼, grounded"。接地线采样 V~185 偏亮 = 鞋遮挡伪影(cx±40 被鞋盖住采到鞋色), 判影靠视觉 montage 不靠 band 采样。重渲染 676 帧成功。备份 `_frontpuddle.mp4`(前水洼版)。CLI 默认 `--shadow-strength 0.5`。详见 memory `bg-swap-tool-influencer` 第 8 点。
（再上一轮: 双层接地阴影 + 视差 ±2% 居中平滑零延迟, 治"没落到地上/慢一拍", 已合入基线。）

---

## 当前迭代目标

用户报告 4 个问题 + 升级 PRO 套餐后要求建立长期项目的**记忆与会话衔接机制**。

---

## ✅ 已完成（本次迭代，2026-06-29 ~ 06-30）

| # | 问题 | 修复 | 验证 |
|---|------|------|------|
| 1 | 建玲竖版片头用了小红豆的判词/诗词 | `lib/coach_profiles.py:_resolve_coach_name` 空串 guard + `stages/39_shorts.py` 从文件名提取教练 | `get_coach('建玲1.mp4')['shorts_poem'][:8]` = "三宝菩萨气势足" ✓ |
| 2 | YouTube 宽屏弹幕文字重叠 | `stages/34_danmaku.py` 改轨道分配（行高=字高×1.4，gap 防 x 重叠） | 编译通过 |
| 3 | 片头音乐（多轮迭代，见下"片头音乐方案"）| **定稿**: 截主体"完全终止落点"前4小节+节拍对齐+接缝0.35s淡入淡出, 钉入 config | 片头 6.94s 截前留落点 ✓ |
| 4 | 合并文件名丢教练名（`合并_建玲`→"合并"） | `scripts/merge_clips.py` 输出名改 `{coach}_{date}.mp4` + 复用 `detect_coach_from_filename` | `建玲_2026-06-29` → "建玲" ✓ |
| 附 | 枫林红判词混入"帅哥美女齐上阵" | `coach_profiles.py` 改"独领风骚冠群英" | — |
| 附 | 新建 `docs/PROJECT_DESIGN.md`（项目设计说明） | 全架构叙述式入口文档 | — |

**最新产物**: `output/2026-06-29/建玲2_3_merged_final_16x9_1920x1080.mp4`（262.6MB，片头 6.94s 截前留落点定稿版，**暂不上传**）

---

## 🔄 进行中 / 下一步

1. **网红换背景换脸工具** ✅ 完成（2026-06-30, 滑动修复定稿）— `tools/bg_swap.py`：网红视频→换时代广场静态背景+换丽丽脸。**抠像=默认 RVM(RobustVideoMatting) 高精度 alpha**（治本, 真 per-pixel alpha 凹谷干净分离背景, 根治"两腿间粉漏色", 不再需 despill/punch/protect）+ face_swap `only_lead` 锁脸（不碰背景路人, cosine 0.87+）+ **静态背景（默认, 冻结单帧）**。**运镜跟随已默认关**（`--follow-cam` opt-in 实验性）：源视频相机实际静止, phaseCorrelate 把人动作当运镜→背景乱裁滚动=脚滑, 故静态最稳。旧 YOLOv8-seg+despill+punch+protect 路径降为 `--no-matte` 回退（补丁仍有残留）。实测 `output/bgswap/网红_丽丽_时代广场.mp4`（676帧 1080×1920 h264+aac, RVM 163s/4.1fps, 两腿间输出R-G+1.0~1.8≈bg, 视觉双确认无粉色无artifact, 顶部背景 f0→f600 dx=-0.0px 冻结）。**最新(2026-06-30): 全身 LAB 色温匹配(`_color_match_to_bg` t0.6)+light wrap(s0.5)治贴纸感/脚浮(色温断层是头号成因, 像素验证人物Δb-17.9→-3.8/总色差减半), 已同步桌面供用户主观判断脚踩实**。详见 memory `bg-swap-tool-influencer`。
2. **三人长视频**（待用户给素材）— 1 真人 + 后 2 数字人，bg_swap+face_swap。多人目标选择扩展（当前单人工具 `only_lead` 天然支持，扩展点=识别哪个是真人 lead）。
3. **下一个健身视频** — 片头音乐方案已钉 config（`intro_outro.intro_music_from_main: true`），下个视频跑 pipeline 自动套用。
4. **建玲2_3 定稿版** — 留存不上传（用户决定）。想传时说「传建玲」，传 `final_path`=`建玲2_3_merged_final_16x9_1920x1080.mp4`（**不传** `full_16x9`），token 若 invalid_grant 需重新授权。
5. session 衔接机制已完成（HANDOFF 活文档 + CLAUDE.md 开局协议 + SessionStart hook 注册）。

---

## 🎵 片头音乐方案演进（2026-06-30 定稿，重要决策记录）

用户要求片头音乐不突兀、不断裂。多轮试听迭代（每轮都跑了实测）：

| 方案 | 做法 | 结果 |
|------|------|------|
| 秦腔 sting / ref1 / ref2 | 独立 4s 曲子硬切 | 突兀（两首曲子）❌ |
| `intro_music_from_clip` | 片头=素材高潮段原声，主体同步 | 音画同步但接缝跳变（断裂）❌ |
| `intro_music_from_main` 迁移版 | 片头=主体前奏，主体顺延 | 音乐完全连续但**主体音画错位穿帮**❌（用户明确否决：主体不能错位）|
| `intro_music_from_main` 复制版 | 片头=复制主体开头，主体原样同步 | 不穿帮但接缝重复（感觉到接缝）⚠️ |
| 复制版 + 节拍对齐 | 片头长度对齐到节拍网格 | 1单元(3.47s)太短 / 2单元(6.94s)不完整 / 4单元(13.88s)太长 ⚠️ |
| **截前留落点 + 接缝淡入淡出**（定稿）| 片头=主体[offset:phrase_end]取第5-8小节, 落第8小节完全终止, 接缝0.35s淡入淡出 | 短(6.94s)+落到底+不穿帮+接缝顺 ✅ 已钉 config |

**关键代码位置**：
- `stages/07_export.py`：`intro_outro` 三分支 `intro_music_from_main`(默认, 已钉) > `intro_music_from_clip` > `has_sting`(sting ref1 兜底)。
- `stages/20_intro_outro.py`：用 `beat_frames` 算 `phrase_end`(8小节完全终止落点) + `intro_music_offset`(截前 offset) + `intro_dur_aligned` + 存 `intro_start_sec`。
- `stages/17_beat_flash.py`：检测成功即 `ctx.set("beat_frames")`（不依赖闪烁视频编码）。
- **结构限制**（无法绕过）：主体画面从素材第 0 秒开始（否则口令动作错位穿帮）→ 主体音频必须从 0 同步 → 片头→主体音乐位置跳变是**结构性**的，截前留落点让它落在乐句边界(完全终止)从而最自然。
- **调参**：片头长度改 `n_head`(=2,4小节)；落点改 `n_end`(=4,8小节)；接缝改 `seam`(=0.35s)。详见 memory `intro-music-trim-front-keep-cadence`。

---

## 🚀 下次升级起点（候选方向，按优先级）

- **bg_swap 残留治理（可选, 低优先）**: 网红多人 v2 播放中偶尔背景不净(极少)。根治需**先诊断残留帧根因**(胳膊出画到天棚残留区 / RVM alpha 偶低 / 抠像失败), 再针对性方案。⚠ 盲上 alpha 后处理(#51 闭运算/gamma/pose 骨架增广)有"人体变薄"前车之鉴(memory `bg-swap-tool-influencer` 第9点 clean_alpha 回退); 换 SCRFD/YOLO-face 检测器对**这残留无效**(管人脸漏检, 不管胳膊 alpha)。判残留靠 alpha 梯度+逐像素 diff, 不靠视觉模型。
- **三人长视频**: 1 真人 + 2 数字人, bg_swap + face_swap。扩展点: 当前 `--swap-all` 全换, 需识别哪个是真人 lead(选择性 matte/换脸, 而非全换)。
- **下一个健身视频**: 片头音乐已钉 config(`intro_outro.intro_music_from_main: true`), 跑 `python main.py process "<input>" --preset youtube` 自动套用截前留落点方案; ShortsStage 同步出 YT Shorts(30s) + 抖音竖版(完整)。
- **建玲2_3 定稿版**: 留存不上传(用户决定)。想传说「传建玲」, 传 `final_path`=`建玲2_3_merged_final_16x9_1920x1080.mp4`(**不传** `full_16x9`), token invalid_grant 需重新授权。
- **YouTube 自动发布**: 18:00 定时 + auto_publish 状态机(部分实现, `records/upload_manifest.json` 留痕防重传)。

## ⏳ 待用户确认 / 卡点

- 无。片头音乐已定稿钉 config，等下个视频自动生效。
- `_cleanup_intermediate.py`（6/29 的非本轮产物）是否删除，等用户定。

---

## 🧭 当前视频状态

| 视频 | 状态 | 路径 |
|------|------|------|
| 建玲2+3 合并 | 管线跑完，片头音乐=截前留落点定稿版，**暂不上传** | `output/2026-06-29/建玲2_3_merged_final_16x9_1920x1080.mp4` |
| 网红丽丽时代广场 | bg_swap（**RVM 治本 + 静态背景 + 脚下接地阴影(贴纸感/脚底滑终极修复)**）：静态背景(默认冻结单帧)+换脸丽丽(cosine 0.87+, 674/676)+**RVM 高精度 alpha 抠像**(默认, 真 per-pixel, 凹谷干净分离背景, 根治两腿间粉漏色)。**贴纸感/脚底滑修复(2026-06-30)**: 真因=RVM 干净抠像丢了原始接地阴影(源有影输出无)→人浮在静态背景上像贴纸/脚底滑(脚位几何其实正确, foot_y=1752/1920); 修复=`_contact_shadow()` 按脚位画软椭圆暗影到背景再合成人(跟脚逐帧)。验证: 人正下方地面V=78.6 vs 远处V=102.2(暗23.6级), 视觉确认"踩地上/有重量感/融入场景"; 背景仍冻结 f0→f600 dx=**-0.0px**。CLI `--shadow-strength 0.55`(默认)/`--no-contact-shadow`。两腿间凹谷输出R-G+1.0~1.8≈冷地板bg(源粉地板+36), 无粉色残留。旧 seg+despill+punch+protect=`--no-matte` 回退。`--bg-frame 1.65` 冷帧, matte despill默认0。**色温匹配保L修复(2026-06-30, 治变灰+脚浮)**: `_color_match_to_bg` 改 mean-shift a/b **保L**(t0.8, Reinhard 缩L方差把黑衣拉灰已弃)+`_light_wrap`(s0.5)+`_clean_alpha`(erode3 去鞋边halo), 像素验证黑衣L 保L10.0 vs 灰版26.6(保黑), 换脸674/676。CLI `--color-match 0.8`。渲染前提: face_swap `cudnn_conv_algo_search=HEURISTIC`(非EXHAUSTIVE, 三GPU模型同进程否则OOM)。**已同步桌面供用户主观判断脚踩实**。**2026-07-01 接地感增强版(用户选C)**: `_grounding`(脚下局部light wrap融合+极弱AO, 区别失败6轮硬影) `--grounding 0.18`, 渲染676帧换脸674/676, 视觉模型确认脚底更柔和融入/脚下微暗/更踩实(主机制=脚底融合, AO弱ΔV-0.9因被脚遮)。成品 `网红_丽丽_时代广场_grounding.mp4` / 桌面 `网红_丽丽_接地感.mp4`, **待用户主观看动态成品判断"浮/滑"是否改善**(单帧测不出, 不够明显可`--grounding 0.3`或接受静态背景固有局限) | `output/bgswap/网红_丽丽_时代广场.mp4`(主版无接地感)+`_grounding.mp4`(接地感版, 备份`_precolormatch.mp4`) |
| 网红跳舞1 时代广场 | bg_swap **泛化验证案例 (dance 预设首次实测)**: ① `prefilter_person` 剪 1.37s 入场(仅下半身+黑场 wipe, head_miss)→cleaned 8.83s; ② `bg_swap --preset dance --coach 丽丽`(用户选换丽丽脸; 该网红≠丽丽) → 258/264 换脸(97.7%), 渲染 73.9s@3.6fps, parallax ±3% + grounding 0.18 + color_match 0.8。视觉模型评"好"(抠像干净/换脸自然/色温协调/接地合理)。**dance 预设通过, 工具泛化端到端验证** | `output/bgswap/网红跳舞1_时代广场.mp4`; cleaned: `~/Desktop/短视频素材/网红跳舞1_cleaned.mp4` |
| **网红多人健身操** | bg_swap **多人换脸案例 (--swap-all 首次实测 + dsr 修胳膊虚化)**: `网红多人健身操.mp4`(3人 720p 7488帧 249.5s)。`--swap-all --dsr 0.5 --preset fitness --coach 丽丽` → 3人都换丽丽脸 + 时代广场背景。**渲染完成 7488/7488 帧(100%换脸/背面跳0/mask漏0), 78min@1.6fps, 317MB**。**dsr=0.5 修复举手时胳膊虚化+两臂间绿树残留**(根因=RVM alpha 边缘软光晕非凹陷洞; 锐度+39%/零速度代价)。**全片验证通过**(`_temp/verify_full.py`): 绿树洞残留全片仅 0.08%(12帧抽样11帧=0), 中段帧(t=160s)视觉确认 3人同脸/广场背景/胳膊边缘清晰无绿晕/两臂间无绿残留。**成品已同步桌面供用户主观确认**。**v2 定稿(2026-07-01)**: 用户报三新问题(背景拉宽变形/人物变瘦/胳膊与天棚重叠消失虚化带绿树), 修复=`_cover_resize` cover 不拉伸(治①变形)+`--bg-crop-y 0.61` 裁切窗下移避天棚(背景变地面广场, 上1/3天棚 11%→0.8~1.2%, 治②③, RVM 在干净地面背景自然抠准)。**验证通过**: 三举手帧(t70/227/248s)逐人胳膊完整/无绿残留/落干净水泥地面, 中段帧(t100s)不变瘦不变形, 7488帧 100%换脸/0背面跳/0漏检。**残留**: 播放偶尔背景不净(极少, 用户接受"先这样吧")。**2026-07-02**: core-matte(坑9, `--core-bolster` 默认开, 见顶部本轮)进一步根治胳膊渗出(核区 α 0.52→0.97, A/B 视觉双通过), 老素材下次重渲自动受益无需改命令；可选重渲替换 v2 | `output/bgswap/网红多人_丽丽_时代广场_v2.mp4`(299M 定稿); 桌面 `~/Desktop/短视频素材/网红多人_丽丽_时代广场_v2.mp4` |

---

## 📌 持久备忘（跨迭代记住）

- 上传只传 `*_final_16x9_1920x1080.mp4`（含片头片尾），**不传** `*_full_16x9.mp4`（去头去尾副本）。
- 跑管线前清 `_temp/cg_*`（color_grade JPEG 序列易爆盘）。
- 换脸/pose 改逻辑后删 `*_keypoints.json` 才会重跑。
- ffmpeg 必须用 `C:/Users/18091/ffmpeg/ffmpeg.exe`，不能用 Winget 版。
- 教练名必须在文件名最前（`建玲1.mp4` ✓，`合并_建玲` ✗）。
- 片头音乐已钉 config 默认（`intro_outro.intro_music_from_main: true`），新视频自动用截前留落点方案，无需手动设。

---

## 📦 派生项目: Matting Studio (2026-07-04 完成 v1.0.0)

**【完整收尾 (Phase 0-8 文档 + Phase 8 暂停)】**

### 项目位置
- `F:\wkspace\matting-studio\` (独立仓库, 23 commit + 2 tag)
- 父项目: `F:\wkspace\fitness-video-pipeline\` (本项目, 8 段主管线)
- 共享 memory: cn-video-matting-software-architecture.md (含 matting-studio 完成段)

### 完整产出
- 23 commit + 2 tag (v0.1.0 + v1.0.0)
- 123 tests pass 0 fail (6.73s, 8 slow 跳过)
- ~3000 行代码 (核心 + 8 模块 + 7 QML + 4 脚本 + 1 CLI)
- ~35 KB 文档 (PROJECT_README + CHANGELOG + UPGRADE + 4 设计)

### 性能基准 (Phase 6.3 实测, PyTorch CPU, 856x462, 30 帧)
- **PyTorch CPU (mobilenetv3): 13.91 FPS (71.9ms/帧)** ✅ 可生产
- PyTorch CPU (resnet50): 3.96 FPS (252.7ms/帧) ⚠️ 慢
- ONNX (mobilenetv3/resnet50): ❌ 失败 (RVM 官方 ONNX Expand 节点 bug)
- PyTorch GPU (RTX 4070): 200-300 FPS (估) ⏳ 未验证

### 完整文件 (F:\wkspace\matting-studio\)
- README.md (项目入口)
- PROJECT_README.md (完整项目状态 + 升级指南)
- CHANGELOG.md (v1.0.0 完整版本历史)
- LICENSE (Apache 2.0)
- pyproject.toml (Python 3.10 + PyTorch 2.6 锁版)
- .gitignore (models/rvm/*.onnx 不入 git, 14MB)
- core/ (types + engine + config + helpers)
- modules/ (8 Stage + sam2_repair + backend + ui/video_surface)
- qml/ (Main + TitleBar + 5 组件)
- presets/ (4 方案 YAML)
- tests/ (123 测试)
- tools/ (matting_cli.py)
- scripts/ (install_deps.sh + rvm_to_onnx.py + rvm_export_onnx_fixed.py + bench_onnx_vs_pytorch.py)
- models/ (.gitignore)
- docs/ (4 设计 + UPGRADE.md)
- .github/workflows/ (ci + release + docs)

### 已知限制 (Phase 8+ 升级)
1. **RVM 官方 ONNX 模型 Expand 节点 bug** - 30x 提速验证待 RVM 官方修
2. **RVM master decoder GRU state dim** 与 PyTorch 2.6 不兼容
3. **PyInstaller 6.x 嵌套 torch.onnx** - 完整 GUI 打包需 runtime hook
4. **SAM2 修帧工作流占位** - Phase 9 完整推理 + QThread 异步
5. **QML GUI partial** - Phase 5.2/6 完整集成 (VideoFrameSink + SAM2Canvas)

### Phase 8+ 升级路线图 (等 RVM 官方修)
- P0 Phase 8: RVM 官方修 ONNX → 完整 30x 提速验证
- P1 Phase 9: SAM2 完整推理 + QThread 异步 (1-2 周)
- P2 Phase 10: 完整 QML GUI 集成 (2-3 周)
- P3 Phase 11: PyInstaller runtime hook 修嵌套 (1 周)
- P4 Phase 12: GPU 性能基准 (1-2 天)
- P5 Phase 13: v1.0.0 GitHub release (1 周)
- P6 Phase 14: Web 端 TFLite + WebGPU (4-6 周)
- P7 Phase 15: 移动端 iOS/Android (6-8 周)

**总估时**: 18-25 周 (4-6 个月单人全职)

### 与父项目关系
- **共享 memory**: `cn-video-matting-software-architecture.md` (cn-video-matting-software-architecture.md) 
  含 8 模块 + 8 模型 + 4 方案 + 升级路线图
- **背景技术**: bg_swap 8 段主管线 RVM 实战经验 (D+grow 治胳膊 + YOLO 治鬼影 + SAM2 修帧)
- **共享 D+grow 算法**: fitness-video-pipeline + matting-studio 都用
- **共享 YOLOv8 治鬼影**: 集成到 matting-studio MattingStage

### 升级协调
1. 升级 matting-studio: `cd F:\wkspace\matting-studio; git pull; cat docs/UPGRADE.md`
2. 升级 fitness-video-pipeline: 父项目继续主管线
3. 共享: RVM 官方更新 (Phase 8) 同时影响两个项目
