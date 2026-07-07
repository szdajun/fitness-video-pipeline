# EXIT HANDOVER — 2026-07-07 00:01 (本次会话结束)

> **会话结束专用摘要**。下次会话只要读本文件 + `git log --oneline -10` 就能立刻继续。
> 详细进度看 `HANDOFF.md` (活文档),架构看 `docs/PROJECT_DESIGN.md`,历史坑看 `memory/`。
> 权威规则始终是 `CLAUDE.md` (含清理产物白名单/YT 立即发布/产物只在 output/)。

---

## 🎯 本次会话总览 (2026-07-06 21:25 ~ 2026-07-07 00:01)

**用户指令链**:
1. "跑主管线，处理枫林红1，枫林红2两个视频，合并为一个文件进行处理"
2. "如果找到了就算了，以前应该处理过" (用户发了原大头美颜照,放弃,走自美化源)
3. "这是枫林红的美颜照，处理一下就可以用了" (图 #3, 用户重发大头美颜照)
4. "处理好的用来换脸的照片入库，以后直接用即可，不用每次都折腾找照片"
5. "继续处理李刚1视频"
6. "退出一下，做好工作移交文档" ← 当前

**会话产出 (4 套三件套 + 1 个 commit 链)**:
- 枫林红1+2 合并主管线 (65min, 6 commits 含源码入库)
- 李刚1 主管线 (50min)
- 彩娥3 + 郭海军1_2 (本会话前已存在,保留使用)

---

## 📦 待用户拍板上传 (4 套三件套全在 `output/2026-07-06/`)

| 教练 | YT long (MB) | YT short (MB) | 抖音 (MB) | 备注 |
|------|------|------|------|------|
| 郭海军1_2_merged | 312 | 50 | 225 | 04:50-04:52 出 (旧, 用户未传) |
| 彩娥3 | 182 | 58 | 160 | 06:33-06:35 出 (旧) |
| **枫林红1_2_merged** | **322** | **54** | **246** | 22:34-22:36 出 (新) |
| **李刚1** | **259** | **52** | **214** | 23:57-23:59 出 (新) |

**总大小**: 7.1G (output/2026-07-06/)。全部按 CLAUDE.md `--output` 子目录按源文件 mtime 落 `2026-07-06/`。

---

## 🎯 YT 上传命令 (待用户拍板后跑)

**一次性 4 套 long+short public 立即发布**:
```bash
cd F:/wkspace/fitness-video-pipeline
uv run python tools/upload_youtube.py --coach 郭海军
uv run python tools/upload_youtube.py --coach 彩娥
uv run python tools/upload_youtube.py --coach 枫林红
uv run python tools/upload_youtube.py --coach 李刚
```

或用 `lib/upload_utils.upload_video` 单条传 (manifest 自动写)。
抖音坚持手工传 (memory `douyin-manual-upload`, 自动被封号)。

**YT 标题模板 (CLAUDE 钉死 + 自动从 coach_profiles 取)**:
- long: `【{nickname}】{coach}{workout}操 | {focus}跟练 | 细柳营健身`
- short: `【{nickname}】{coach}30秒{focus}操 | {focus}挑战 | 细柳营健身 #Shorts`

**已确认的标题值 (本轮)**:
- 郭海军: 【老兵不老】郭海军刚劲塑形操 | 刚劲塑形跟练 | 细柳营健身
- 彩娥: 【孤勇者】彩娥勇气燃脂操 | 勇气燃脂跟练 | 细柳营健身
- 枫林红: 【霸道总裁】枫林红高效有氧操 | 高效有氧跟练 | 细柳营健身
- 李刚: 【胭脂虎】李刚力量塑形操 | 力量塑形跟练 | 细柳营健身

**`--publish-at` 禁用**: 长视频必须立即发布 (public, 0 延迟),延期挂死 (memory `yt-long-video-publish-immediately`)。

---

## 💾 本次 commit 链 (2026-07-06 21:00 ~ 23:59)

```
f26da7e docs(handoff): 李刚1 三件套 + 视觉验证 + 待用户拍板上传
b9c19d1 docs(handoff): 枫林红1+2 三件套 + 换脸源照入库 + 后续待用户拍板上传
0ebb2e4 chore(coach): 枫林红 换脸源照入库 (大头美颜照 0.857 + GFPGAN 增强 0.864)
... (前面是 06-06 旧 commits)
```

**新代码/数据**:
- `tools/枫林红_face.jpg` 11.3KB (213×348 原大头美颜照, score=0.857, fallback)
- `tools/枫林红_gfpgan.png` 119.5KB (213×348 GFPGAN 增强, score=0.864, **find_coach_face 自动首选**)

**已 commit 上轮重要原则**:
- `d2e9e7a` docs(claude): 清理产物原则 (白名单) — 2026-07-06 误删彩娥3 三件套教训
- `09478d5` fix(claude+tests): 钉死产物只在项目 output/, 禁止落 C 盘桌面

---

## 🔧 环境状态

- **C 盘**: 7.0G free (98% 用), 跑 pipeline cg_/eb_ 临时峰值 ~20G 风险 — **长视频前先 `uv run python scripts/cleanup_intermediate.py` 腾空间**
- **F 盘 (项目)**: 8.4G output/, 25M _temp/ (干净)
- **GPU**: face_swap 验证 GPU 0 MiB 干净, CUDNN 已钉 HEURISTIC + kSameAsRequested (memory `face-swap-cudnn-fix`)
- **uv 环境**: .venv 3.11.14 (Python >=3.11 钉死, numpy <2, torch 2.6.0+cu124 optional[gpu])

---

## 🔄 进行中 / 待办

### 用户待拍板 (本会话阻断项)
1. **YT 上传** (4 套三件套 long+short public 立即发布; 抖音手工传)
2. **下个健身视频** (source_videos/ 当前就绪: 无新视频待用户传)

### 下次会话候选 (低优先,非立即)
- **Matting Studio Phase 2** (QML 完整 + SAM2 修帧, `F:\wkspace\matting-studio\` 已 v1.0.0)
- **bg_swap 残留治理** (极罕举手帧天棚残留)
- **三人长视频识别真人 lead** (1 真 2 数字人场景)

### 重叠 session 备忘
- 抖音坚持手工传 (`memory/douyin-manual-upload`)
- 大文件 (>200MB) YT 上传 wrapped-200 误拿旧 id (memory `youtube-upload-large-file-wrong-videoid`,`upload_utils._verify_uploaded_ytid` 守门)
- 看板 Y 通道高占用 → git pull (本会话只是本地 main, 没 push)

---

## 🚨 下次开会首句提示

1. 读本文件 + `git log --oneline -10` + `git status` (working tree 干净)
2. 跑 `uv run python scripts/cleanup_intermediate.py` (清理 5.4G 已释放, 跑前可再清)
3. **首要问用户**: "YT 4 套三件套现在上传吗? 跑 tools/upload_youtube.py --coach {彩娥/郭海军/枫林红/李刚}"
4. 然后 `ls source_videos/` 问下个视频

---

## 📌 关键钉死规则 (CLAUDE 永久)

1. **清理产物白名单** (memory `cleanup-output-safelist`): 只删 `! -name "*_final_16x9_1920x1080.mp4" ! -name "*_final_16x9_1920x1080_yt_shorts.mp4" ! -name "*_final_16x9_1920x1080_douyin.mp4"` 之外的东西; 禁用前缀删除 (彩娥3 误删教训)
2. **产物只在项目 output/** (memory `no-desktop-output`): 不许 cp 到 C:/Users/18091/Desktop/短视频素材/
3. **YT 长视频立即发布** (memory `yt-long-video-publish-immediately`): public, publish_at=None, 禁用 scheduled
4. **不自动重跑 fix 后代码** (memory `no-auto-rerun-after-fix`): 修 commit 落就等未来视频自动生效,不删旧产物触发重跑
5. **片头音乐**: 钉 `intro_music_from_main: false`, 走独立 sting (`music_library/intro_sting/intro_ref1.wav`), 主体音乐不变
6. **face_swap 路径**: `lili/sun/etc_face_gfpgan.png` 优先 > `_face.{png,jpg}` > 原始, find_coach_face 自动命中 (新加教练丢一张照到 tools/)
7. **抖音手工传**: 抖音绝自动 (memory `douyin-manual-upload`)
8. **换脸源照入库后后续自动命中**: 用户说"以后直接用" — commit 后下次跑自动找优先命中的 gfpgan
9. **路径 portable**: `lib/utils.resolve_ffmpeg()` (`C:/Users/18091/ffmpeg/ffmpeg.exe` 优先于 PATH) + `lib/utils.resolve_comfyui_root()` (env override)
10. **uv 跑**: `uv run python main.py ...` / `uv run pytest`, 不用裸 python (pyenv shim 可能错)

---

## 👋 退出

本次会话产出完整,4 套三件套全在,换脸源照入库完毕,git 工作树干净,所有 commit 已落,所有 35 个 pre-commit 守门测试通过。

下次开会只要 1) 读本文件 + HANDOFF.md 2) `git status` 确认干净 3) 问用户"上传 YT 吗?" 就能立刻接续。
