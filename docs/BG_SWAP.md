# BG Swap 工具经验总结 (网红视频换背景 + 换脸)

> `tools/bg_swap.py` — 独立工具 (主管线零改动), 把网红/健身源视频里的人抠出来合成到新背景 (默认**西安时代广场**户外广场) + 换教练脸. 仿 `student_closeup.py` 范式: CLI + 分步 print + `_temp/` 缓存 + ffmpeg rawvideo pipe + 永远出 debug 检查图.
>
> 2026-07-01 泛化定稿. 两个真实案例验证: 丽丽→时代广场 (fitness 预设) / 网红跳舞1→时代广场 (dance 预设). 规则由 `tests/test_bg_swap_defaults.py` (11 tests) 守门.

## 处理流程 (每帧流式)

```
源视频
  │
  ├─[1] pose (YOLOv8-pose, 复用 student_closeup.detect_pose) → lead bbox + foot_y + 视差曲线
  ├─[2] 教练脸源 (find_coach_face 优先级链 → tools/{coach}_face_gfpgan.png)
  ├─[3] 背景预处理 (默认静态: 抽单帧 resize+锐化 png; --dynamic-bg 切动态视频)
  ├─[4] RVM 抠像 (RobustVideoMatting mobilenetv3, GPU half, dsr=0.25) → per-pixel alpha 0-1
  └─[5] 渲染 (逐帧 → ffmpeg rawvideo pipe → h264_nvenc):
        per frame:  frame_sw = 源帧
                    → 换脸 (face_swap.swap_face only_lead=True, lead_bbox=pose)
                    → 色温匹配 (_color_match_to_bg mean-shift a/b 保L)
                    → light wrap (边缘混背景光)
                    → 接地感增强 (_grounding, 可选, 默认关)
                    → 视差缩放 (warpAffine 锚 ground_y, --parallax)
                    → 合成 out = frame_sw*alpha + bg*(1-alpha)
                    → (contact shadow 默认关)
```

## 性能数据

| 案例 | 预设 | 源 | 时长/帧 | RVM | 渲染 | 换脸 | 备注 |
|---|---|---|---|---|---|---|---|
| 丽丽→时代广场 | fitness | 1080×1920 | 676 帧 | 163s (4.1fps) | ~529s | 674/676 (99.7%) | grounding 接地感版定稿 |
| 网红跳舞1→时代广场 | dance | 462×856 | 264 帧 | (含在渲染) | 73.9s (3.6fps) | 258/264 (97.7%) | prefilter 先剪 1.37s 入场 |

**关键**: 三 GPU 模型同进程 (RVM torch + buffalo_l + inswapper onnxruntime), 必须 `cudnn_conv_algo_search=HEURISTIC` (非 EXHAUSTIVE, 否则瞬时撑爆 12GB OOM) + `gpu_mem_limit=4GB` + arena `kSameAsRequested`. 跑前 GPU 干净 (nvidia-smi ~0 MiB).

## 坑 + 根因 + 修复

### 坑 1: 抠像边缘差 / 两腿间粉漏色 → RVM 真 per-pixel alpha (治本)

**症状**: 旧 YOLOv8-seg 二值粗掩码, 凹谷 (两腿间/腋下/指缝) 误纳入原图背景 → 粉地板漏色. refine_mask + punch + despill + protect 一路打补丁仍残留, 用户 4 次推翻"修好了".

**根因**: seg 二值掩码凹谷误纳入背景是结构性的, 补丁治标.

**修复**: `--matte` 默认开 (RVM). 真 alpha 凹谷=背景, 漏色归零. despill 默认 0 (alpha 干净无需).

**验证**: 两腿间凹谷输出 R-G **+1.0~+1.8 ≈ 冷地板 bg +2.4~+3.0** (源粉地板 +36). `--no-matte` 回退 seg 仅历史路径.

### 坑 2: 换脸换错人 → pose 锁 lead + only_lead

**修复**: `face_swap.swap_face(only_lead=True, lead_bbox=get_lead_bbox_from_pose(...))`. lead_bbox = pose 找 cx 居中 + 身体最大的人. 只换网红脸, 不碰背景路人.

**验证**: embedding cosine — 输出脸 vs 丽丽 **0.87-0.89**, 原网红 vs 丽丽 0.065.

### 坑 3: 贴纸感 / 脚浮头号成因 = 色温断层 → 全身色温匹配 + light wrap

**症状**: 抠像 + 阴影 + 视差全修好后用户仍报"脚浮/贴纸/像贴上去".

**根因 (像素)**: 室内抠出的人 Δb=-17.9 严重偏冷蓝, 贴到暖户外广场 = 色温断层. 大脑读成"假/浮". (不是阴影/视差能治的.)

**修复**: `_color_match_to_bg` (mean-shift a/b **保L**, t=0.8) — 只平移人物 a/b 均值向背景下半部 (地+墙, 不含天空), L 不动 = **黑衣服保黑**. + `_light_wrap` (边缘混背景光, s=0.5).
> ⚠ 首版用 Reinhard (缩 L/a/b 方差) → 黑衣服 L 10.3→27.2 变灰. 改 mean-shift 保 L. **Reinhard 已弃**.

### 坑 4: 脚底滑 = 真实脚步移动 + 合成丢接地接触 (静态背景固有, 非阴影/非相机抖动)

**症状**: 用户反复报"脚底滑/浮/没落到地上". 阴影调了 6 轮 (v1 单椭圆→v2 双层→v3 前向衰减→v4 扁窄脚跟) 都没根治.

**根因 (实证)**:
- 源相机**基本静止** (脚深同纹理地砖 phaseCorrelate dx std=0.33px; 远处低纹理建筑给的 1-2px 是噪声假抖动, 不能用).
- foot_cx 逐帧 std=9.5px 来回震荡 = **真实脚步移动** (和源一样).
- 源不滑、合成滑, 差别**不在几何** (脚位相同相机同静止), 在**合成丢了真实脚-地物理接触** (RVM 抠像剥掉真实接地阴影/光照 + 背景冻结无动态响应).

**修复 (缓解, 非根治)**: `_grounding` 接地感增强 (见坑 5). **人造阴影 `_contact_shadow` 默认关** (`--shadow-strength 0`) — 阴影是死路 (6 轮失败), 不追踪真实接触反凸显"脚地两层".

**遗留**: 残留稍许滑动 = 静态背景固有局限 (无真实物理接触), 除非换动态背景或接受.

### 坑 5: 接地感增强 grounding (C 方案, 区别于失败 6 轮硬影)

**机制 `_grounding(ao=0.18)`** (和单向硬影根本不同):
- **(A) 脚下局部 light wrap 融合 (真机制)**: 脚底 alpha 羽化带 (α≈0.5 峰) × 限脚下区 × 混入【脚下纯地面色】(bg 脚正下后方 mean BGR) = 脚底边缘和地面色融合非硬切.
- **(B) 极弱接地 AO**: 脚正下后方小椭圆 bg gradient 微暗 (eff 0.18 vs 硬影 0.5, 柔无形状感), 前向衰减 (脚尖前无响应).

**验证**: AO 效果弱 (ΔV=-0.9, 脚遮蔽), 真正机制=(A) 脚底融合. 视觉确认 grounding 版脚底轮廓更柔和融入地面.

**CLI**: `--grounding 0.18` (**内置默认 0=关**, opt-in; fitness/dance 预设开 0.18). 不够明显可 `--grounding 0.3` (强了易显假).

### 坑 6: "人变地不变"贴纸感 → 视差纵深 (--parallax)

**症状**: 人表观尺寸 ±44% (转体/前后走), 背景冻成石头 → 大脑把尺寸变化当纵深, 期待地面视差响应, 不动 → 读成滑/浮.

**修复**: `--parallax 0.02` (默认 ±2%): 静态 bg 加 7% 边距, `compute_parallax_scale` 用 pose keypoint 宽度预算全片 bg_scale 曲线 → `warpAffine` 锚定 ground_y 缩放 (锚点脚不动远景动 = 视差). **居中平滑 (±9帧) 零相位延迟** — 别用因果 EMA (会慢一拍).

### 坑 7: --follow-cam 运镜跟随 = 灾难 (默认关)

**根因**: `estimate_camera_motion` 用 phaseCorrelate, 在人物占满画面时**把人动作当运镜**测出假轨迹 → oversize 背景按假轨迹裁 → 背景凭空左移 1/3 画面 = 跑步机效应 = 这才是用户报的"脚底滑动"真因之一.

**修复**: `--follow-cam` 默认关 (opt-in 实验性). 静态背景为默认. **别再开**, 除非源真有镜头平移且人物占画面小.

### 坑 8: ffmpeg Winget 版有编码 bug → 已知好路径优先

**根因**: PATH 里的 Winget ffmpeg (8.1-full) 有编码兼容 bug 会生成损坏 mp4 (见 CLAUDE.md energy_bar 坑). 本项目已知好路径 `C:/Users/18091/ffmpeg/ffmpeg.exe` (8.1-essentials) 实测稳定.

**修复**: `_resolve_ffmpeg()` 解析顺序 = `--ffmpeg` CLI > `BG_FFMPEG` env > **已知好路径 (if exists)** > `shutil.which` PATH > 兜底. 已知好路径优先于 PATH; 换机器 (此路径不存在) 自动落到 PATH, 保证可移植.

### 坑 9: 胳膊虚化 / 原背景渗出 = RVM 软抠像对细/快动胳膊低 alpha → pose core-matte 撑实 (治本, 2026-07-02)

**症状**: 换背景后人物胳膊半透明、原背景从胳膊里渗出; 高举手 / 两臂圈起来时最明显. 用户报"总是出现人物胳膊虚化, 原背景渗出". 这是换脸成熟后 bg_swap **最后的硬骨头**.

**根因 (实证, 非 bug 非 param)**: RVM 是**软抠像** (soft matting), 对**细 / 快动**结构 (胳膊) 系统性低 alpha — 胳膊核心区 (骨架包络内) mean α 仅 **0.52** (半透明, 正是渗出处), 不是 1.0. 这是 soft matting 固有特性 (MatAnyone arXiv:2501.14677 的"core-area supervision"、ViTMatte、Generative Video Matting 都为修它存在). 提 dsr (0.25→0.5) 只能缓解不能根治 (锐度+39% 但 alpha 仍不到 1.0).

**修复 `_pose_core_matte` (core+edge matte split, VFX 标准技术)**:
- **pose 骨架包络 = 硬 core matte** (撑实躯干/胳膊); **RVM alpha = 软 edge matte** (轮廓羽化). 二者并 = 既有硬核又有软边.
- `mask = max(rvm_alpha, envelope × gate)`. **只抬不降** (安全): 包络只在 RVM 已感人体 (α>0.05) 邻域 dilate (核 ~0.25 肩宽) 激活 → 不会因 pose 误定位在干净背景里造"幻肢" (粘贴原背景像素).
- **多人并集**: 多人场景取**所有检测到的人**的骨架并集, 不只 lead — 否则旁人仍渗出.
- **臂圈洞正确处理**: 高举臂围成圈时, 圈内无骨架 → 包络不覆盖 → 保持背景 (**正确**, 圈里本就该是背景); 圈周胳膊本身被撑实 → 无边缘渗出.

**验证 (像素级, 不靠视觉)**: 网红多人健身操 t=40-60s 裁片, 高举臂圈起来时刻 (clip f300, 源 t=50s) 窗口 285-315 帧均值 (`_temp/probe_core_matte.py`):
- 核区 mean α: baseline(RVM) **0.520 → bolster 0.965** (半透明→撑实)
- 核区渗出像素 (α<0.7): **15697 → 1260 (减 92%)**; f300 单帧 17358→1644 (减 90.5%)
- alpha 差热力图 (probe_diff): 撑高**集中在胳膊/上半身**, 躯干内部不动 (max() 只抬 RVM 漏的, 不毁已正确的区)

**CLI**: `--core-bolster 1.0` (**默认 0 关**; 2026-07-02 反转). 包络段直径按各人肩宽缩放 (臂 0.42 / 躯干 0.62 / 腿 0.46), `scale=core_bolster` 整体微调 (1.0=肩宽基准). **为何默认关**: env 实测仅覆盖画面 ~3% (骨架细带), 单帧面积小但**每帧每人轮廓沿线都有** → 看动态视频骨架带硬抬 alpha 痕迹全片可见, 用户判 v3 "不干净/基本都这样" (v2 软边反而更净). 治渗出价值不抵变脏, 弃用回 v2 软边; 需治渗出手动 `--core-bolster 1.0` 并接受边缘偏硬.

> **注 (2026-07-03 复盘)**: 后查 `--core-bolster` 当时**根本没生效** — `_pose_core_matte` 用 COCO-17 索引 (5/6/7/8/9/10) 读 blaze33 缓存 (11/12/13/14/15/16) = 错位画垃圾包络, 且 `student_closeup.detect_pose` 的 COCO2BLAZE 映射漏了肘/腕/膝 (缓存里这些关节全 vis=0). 用户嫌 v3 "脏" 部分是这双 bug 造的伪包络, 非 bolster 理念问题. 已修映射 + 另起 arm-only 版 (见下).

#### 坑 9.bis: arm-grow = 治过渡环 (2026-07-03, 替代 arm-bolster; 用户拍板)

**【arm-bolster 被推翻 (2026-07-03 下午)】** 上午定稿的 `--arm-bolster 1.5` (仅撑实核心管 env scale 1.5) 实测核心管 α 0.74→0.954 但用户看 _armbolster.mp4 报"**胳膊周围几乎都渗出**"截图实证. 根因 = **测量区域错位**: 我测的是核心管内部, 用户看到的是核心管**外**的过渡环 (scale 1.5→3.0, RVM α 0.3-0.7 半透明带). 7488 帧全扫: 核心管 RVM α ≈ 0.5 (最差 0.434), 过渡环 1 α 平均 **0.413**, **99.8% 帧有 >2000 渗出像素** (环1 漏治 4392 px/帧). frame 7093 双手平伸无运动模糊 RVM α 也只 0.4 → 病因非快动/模糊, **RVM 对细长肢体结构性低估** (与上轮 MatAnyone 试点的发现一致).

**【教训 (关键)】** 别只测"内部管"好看就当治住了, 必须测用户能看到的过渡环. 判 bg_swap 缺陷靠**全 7488 帧扫参 + 过渡环 halo 度量 (a<0.05)**, 不靠静态帧/视觉/核心管距离. 同 [[bg-swap-core-matte-arm-bleed]] 测量方法学.

**【A/B/C/D 方案模拟 (n=7488 + 严格 halo 度量 = RVM 确信背景区被填)】** (`_temp/simulate_fix{2,3,4,5}.py`):
- **A 盲加宽** `max(rvm, env(3.0))`: 治愈 86.5%, **halo 389%** (撑一漏撑四背景 = 灾难) → 否
- **B 颜色门控** (肤色/黑衣 box): halo 6176px ≈ A (**肤色撞路面色**, 颜色门控不可靠) → 否
- **C alpha 阈值 + 模糊**: halo 16-39%, 治愈 37-59% (RVM α>0.15 区本身**斑驳**, 模糊治不了孔洞) → 否
- **D alpha 阈值 + binary_fill_holes** (thr 0.15): halo 11%, 治愈 51% (填洞不外扩 → halo 低, 但 RVM 在胳膊边缘的 α 也低 → 真实边缘填不到) → 不够
- **D+grow (内 α>0.15 填洞 + 外 α>0.05 内 grow)**: **唯一双达标**

| grow | 治愈 | halo |
|------|------|------|
| **1 (3px)** | 99.8% | **2.5%** ✅ |
| 2 (6px) | 99.8% | 3.0% |
| 3 (9px) | 99.8% | 3.2% |

**grow=1 最优** — 治愈打平, halo 最低 (外扩少→撑背景少). 模拟脚本: `_temp/simulate_fix5.py` (grow=2) + `_temp/simulate_fix5_grow{1,3}.py`.

**【D+grow 原理 (像素定案)】**:
1. `inner = (rvm_α > 0.15) & pose_arm_zone` → 胳膊内真实像素 (含斑驳孔洞)
2. `solid = binary_fill_holes(inner)` → 治斑驳 (不外扩 → halo 低)
3. `outer = (rvm_α > 0.05) & pose_arm_zone` → RVM 感到前景的过渡区 (背景 α<0.05 自动被剔, 防扩到背景)
4. `solid_g = dilate(solid, N iter=3N px) & outer` → 在 RVM 自信的前景内, 把填好的核心外扩到真实边缘 (**关键反直觉: grow 必须用 RVM α 门控**, 否则扩到背景 = A 方案 389% halo 灾难)
5. `solid_smooth = GaussianBlur(solid_g, 7×7)` → 抹羽边 (避免硬切)
6. `final = max(rvm_α, solid_smooth)` in-place (配 render 循环 gc.collect 治长视频 RAM)

**CLI**: `--arm-grow 1` (**默认 0 关**, opt-in). `--arm-bolster` 旧核心管版已弃用 (改用 `--arm-grow`). 守门 `tests/test_bg_swap_defaults.py` (7 测试: 默认关 / 函数存在 / blaze 索引 / 躯干门控 / 用 fill_holes / 推荐 grow=1 / motion weight 已删 / 映射含臂 / scipy import).

**验证 (像素级)**:
- 全 7488 帧扫描: 治愈 99.8% (35.9M 漏治像素治了 35.9M), halo 2.5% (1.0M 撑到背景), 扫描脚本 `_temp/scan_arm_bleed.py`.
- 模拟脚本 `_temp/simulate_fix{2,3,4,5}.py` (A/B/C/D 四方案 + grow 1/2/3 扫参).
- 视觉模型判定 (hstack v2 vs armbolster, 3 帧 t=70/227/248s): arm-bolster 把 v2 的 1.5-2/5 提到 3-3.5/5, 但指尖/手腕末端仍有冷色调泄露. 预期 D+grow1 提到 4-4.5/5 (治过渡环非核心管). 验证: 抽帧后用户拍板.

**工程细节**:
- 复用现有 `_pose_arm_core_matte` (env scale 1.5) 作为 `pose_arm_zone`, 不需重画
- `from scipy.ndimage import binary_fill_holes` (.venv 已有, 2026-07-02 pyproject 加的)
- 流水线加 1 步: per-frame 算 solid_g (小数组 < 100ms), 不破 1.8fps 速度
- in-place `np.maximum(mask, solid_smooth, out=mask)` 保留 (上次内存修复)
- `gc.collect()` 每 100 帧保留

---

#### 坑 9.tris: RVM 远处半透真人 "鬼影" = YOLO-seg person mask 交集 (2026-07-03)

**【症状】** d_grow1 production t=70s 视觉上"3 个真人身后都站着一个不动的人" (用户拍板). 像素诊断: 鬼影区 RVM α mean 0.315, **max=1.0** (RVM 满前景) = 不是"半透"是"实心". v2 旧版 (2026-07-01 跑) 同区 BGR 82/89/92 (水泥地) = v2 旧版 RVM 没检测到远处真人; d_grow1 新版 (2026-07-03 跑) 同区 BGR 45/48/50 (黑上衣) = 新版 RVM 检测到了. **RVM 在同源同帧两次跑结果不同 = RVM 模型输出本身不稳定** (memory `face-swap-other-faces-false-report` 已知: RVM 幻觉远处真人).

**【治法】** 单 mask 阈值治不了 (max=1.0 > 任何阈值). 治本 = **用 YOLO-seg person mask 与 RVM α 取交集**:
- YOLO 边缘锐利 (实例分割, 不基于 RVM) → RVM 远处"幻觉真人"被 YOLO 边界剔除
- RVM α 内容丰富 (发丝/薄纱/边缘软) → YOLO mask 锯齿被 RVM α 平滑
- 交集 = RVM 内容 + YOLO 边界 = 治软光晕 + 治远处幻觉

**【单帧验证 (t=70s 75s smoke, 3 真人构图)】** (`_temp/ab_v2_vs_armbolster/mask_test_*_t70.png`):
- RVM 单独合成: 鬼影存在 (中间真人右侧完整人形)
- YOLO 单独合成: 鬼影消失, 3 真人边缘锯齿 (手/脚有截断)
- **交集合成: 鬼影完全消失, 3 真人完整保留, 边缘略软 (RVM α 平滑 YOLO 锯齿)** ✅

**【集成】** 4 模型同进程 (RVM + buffalo_l + inswapper + YOLO) **YOLO-seg 强制 CPU** (`yolo_seg_model.to('cpu')`) 避开与 3 GPU 模型争 4GB onnx arena (face-swap-cudnn-fix 已知 HEURISTIC+4GB 才能跑). 4 模型 GPU 加载实测 buffalo_l 1k3d68.onnx `bad allocation`. YOLO CPU 推理 yolov8n-seg 6.7MB ~50ms/帧, 720×1280 可接受, 75s smoke 实测 2.0fps (vs 不带 YOLO 2.4fps = 慢 17%).

**CLI**: `--mask-mode rvm|intersect` (默认 rvm 维持现状, opt-in `intersect` 治鬼影). `--yolo-seg-model yolov8n-seg.pt` (默认, 6.7MB 轻量, 自动下载到 cwd).

**验证** (75s smoke `_temp/smoke_intersect_v3.mp4`): 2250/2250 帧 0 崩, RSS 2347MB 平稳 (vs 之前 3 模型 2331MB, YOLO CPU 几乎不占内存), 2.0fps. t=70 抽帧 3 真人完整 + 鬼影消失. 完整 7488 帧生产未跑 (等用户拍板).

**守门** `tests/test_bg_swap_defaults.py` +3 测试 (CLI mask_mode 存在/默认 rvm / render 签名 + intersect 分支 / YOLO 强制 CPU). 110 passed 零回归.



## 关键架构决策

| 决策 | 为什么 |
|---|---|
| 抠像默认 RVM 不 seg | 真 per-pixel alpha 治根 (凹谷漏色), seg 二值粗掩码结构性漏色补丁治标 |
| 背景默认静态不动态 | 动态背景视频自身运镜 → 人物相对地面滑; 静态单帧最稳 (dx≈0) |
| contact shadow 默认关 | 阴影是死路 (6 轮失败), 不追踪真实接触反凸显"脚地两层"; 用 grounding 替代 |
| grounding 内置默认 0, 预设开 | 接地感是可选增强; builtin 默认关 (安全 opt-in), fitness/dance 预设编码 0.18 (验证值) — 预设系统优雅解决这个矛盾 |
| color_match mean-shift 保 L 不 Reinhard | Reinhard 缩 L 方差把黑衣服拉灰; 保 L 只动 a/b 色温不毁对比度 |
| 视差居中平滑不用 EMA | 因果 EMA 有 ~0.4s 相位延迟 (慢一拍); 居中平滑零延迟 |
| core-matte **默认关** (2026-07-02 反转) | 治渗出有效 (核区 α 0.52→0.97, 像素证) 但骨架带每帧硬抬 alpha → 人物轮廓显脏 ("基本都这样"); 治渗出不抵变脏, 弃用回 v2 软边; 需时手动 `--core-bolster 1.0` (坑 9) |
| **arm-grow `--arm-grow 1`** (2026-07-03 替代 arm-bolster, 默认关) | 填洞(binary_fill_holes 治 RVM 胳膊内斑驳) + alpha门控 grow 3px (RVM a>0.05 内 grow 扩到真实边缘) + max(rvm, smoothed). 模拟 n=7488: 治愈 99.8% halo 2.5% (grow=2/3 略高). 替代旧 arm-bolster (治了核心管没治环, 用户拍板). 治虚化首选它非 core-bolster (坑 9.bis) |

## 配置速查

### 预设 (`presets/bgswap_*.yaml`)

| 预设 | 适用 | 关键值 |
|---|---|---|
| `fitness` (**已实测**) | 静态机位 + 居中人物 + 可见脚 的健身/操课 | matte / color_match 0.8 / light_wrap 0.5 / parallax 0.02 / **grounding 0.18** / shadow 0 / 静态 |
| `clean` (基线) | 未知视频保守起点 | 仅 matte, 全部增强 0 (先确认抠像+换脸基线) |
| `dance` (起步模板) | 动作幅度大的舞蹈 | 同 fitness 但 parallax 0.03 (**未实测, 需调**) |

```bash
# 预设生效: --preset fitness 后 grounding 默认显示 0.18 (preset 覆盖内置 0)
# CLI 显式值仍胜预设 (preset 胜 builtin)
python tools/bg_swap.py --preset fitness --help | grep grounding
```

### CLI (preset-overridable 的默认=preset.get(key, BUILTIN))

```
必填: --video --bg --coach --output
预设: --preset fitness|clean|dance
抠像: [--matte (默认开)] [--no-matte] [--dsr 0.25 (RVM 内部降采样比; 1080p 单人 0.25 够, 720p 多人可 0.4-0.5 补锐度)]
胳膊: [--arm-grow 1 (默认关; 治 RVM 胳膊过渡环虚化/渗出, 替代旧 --arm-bolster, 推荐 1=3px)] [--no-arm-grow]
      [--core-bolster 1.0 (默认关, 旧全身版弃用; 双 bug 已修但建议用 --arm-grow)] [--no-core-bolster]
换脸: [--swap-all (多人: insightface 检到的脸都换同一张教练脸, 默认 only_lead 只换 pose 锁的领操人)]
羽化: [--feather 11] [--erode 4] [--despill 0]
增强: [--color-match 0.8] [--light-wrap 0.5] [--parallax 0.02]
接地: [--grounding 0 (内置; fitness/dance 预设 0.18)] [--shadow-strength 0 (默认关)]
背景: [--bg-frame <秒>] [--dynamic-bg] [--follow-cam (默认关, 灾难)]
其它: [--no-faceswap] [--no-color-match] [--no-light-wrap] [--no-punch] [--debug-only]
      [--ffmpeg <path>] [--pink-thresh-rg 20] [--pink-thresh-sat 40]  # 后者仅 seg 回退
```

## 命令参考

```bash
# 标准 (健身/操课, 推荐 fitness 预设含 grounding)
python tools/bg_swap.py --video 网红.mp4 --bg 时代广场背景.mp4 --coach 丽丽 \
  --output output/bgswap/网红_丽丽_时代广场.mp4 --preset fitness

# 舞蹈 (动作幅度大, 视差略增)
python tools/bg_swap.py --video dance.mp4 --bg 时代广场背景.mp4 --coach 丽丽 \
  --output output/bgswap/dance_时代广场.mp4 --preset dance

# 未知视频先跑 clean 基线, 确认抠像+换脸没问题再逐项开增强
python tools/bg_swap.py --video new.mp4 --bg bg.mp4 --coach 丽丽 -o out.mp4 --preset clean --debug-only

# 只换背景不换脸
python tools/bg_swap.py --video in.mp4 --bg bg.mp4 --coach 丽丽 -o out.mp4 --no-faceswap

# 脚下区暖粉挑冷背景帧 (--bg-frame 扫帧)
python tools/bg_swap.py --video in.mp4 --bg bg.mp4 --coach 丽丽 -o out.mp4 --bg-frame 1.65

# 多人视频: 所有人都换同一张脸 (--swap-all) + core-bolster 撑实高举臂/圈起来防渗出
python tools/bg_swap.py --video 多人.mp4 --bg 时代广场背景.mp4 --coach 丽丽 \
  -o output/bgswap/多人_丽丽_时代广场.mp4 --preset fitness --swap-all --core-bolster 1.0 --dsr 0.4
```

**配套预处理** (`tools/prefilter_person.py`, 见下): 换背景前先剪掉人物不完整片段 (出画/缺头缺脚), 否则抠像残缺 + 贴纸感加重.

```bash
python tools/prefilter_person.py 网红跳舞1.mp4 -o 网红跳舞1_cleaned.mp4 --accurate
```

## 加新教练 runbook

1. 丢一张清晰正脸照到 `tools/{coach}.jpg` (中文名即可, 无需扩 alias dict).
2. 首次跑 bg_swap 时自动 GFPGAN 增强生成 `tools/{coach}_face_gfpgan.png` (1024×1024) 长期复用.
3. (可选) 加 `lib/coach_profiles.py:COACH_PROFILES` 走主管线 title/上传.
4. 缺图时 face_swap 自动 skip (不报错).

> 优先级链 `find_coach_face`: `{coach}_face_gfpgan.png > _gfpgan.png > _face.png > _face.jpg > .png > .jpg`. alias dict (`丽丽→lili` 等) 是历史 pinyin 桥, 中文名直接命中无需扩.

## 加新背景 runbook

- 静态图或视频皆可; 视频默认抽中间帧冻结 (静态最稳).
- 脚下区暖粉时扫帧挑冷帧: `--bg-frame <秒>` (扫脚下区 R-G + 粉像素% 挑最冷无粉帧).
- 动态背景 (`--dynamic-bg`) 仅静态机位源用; 运镜背景会让人物滑动.
- **⚠ '时代广场'=西安时代广场** (米色建筑/红砖地/棕榈, 非纽约). 本项目 bg 素材 `~/Desktop/短视频素材/时代广场背景.mp4`. `assets/bg/times_square.jpg` 是纽约霓虹版 (不用, 别误选).

## 调试技巧 (方法学 — 最重要)

> **判测量靠像素不靠视觉模型** (早期视觉模型误报"修好了"4 次, 都被像素推翻). **判接地/贴合最终靠用户主观看动态成品** (单帧测不出"浮"的 gestalt). 视觉模型 (analyze_image) 仅作辅助确认函数非 no-op.

1. **debug 检查图**: `_temp/{stem}_bgswap_debug.png` (红=mask 轮廓, 黄=lead 脸框, 8 帧采样).
2. **embedding cosine**: 输出脸 vs 教练 >0.42 = 换对人 (丽丽 0.87+).
3. **phaseCorrelate 输出视频**: 静态背景 dx≈0; 判背景是否冻结靠它不靠单帧/肉眼.
4. **两腿间/凹谷定量** (RVM): alpha<0.35 凹谷处输出 R-G 应≈bg 冷, 非 src 粉. `verify_rvm.py`.
5. **判相机静止**: 必须测**脚深同纹理地砖** (phaseCorrelate dx std<1px) — 远处低纹理建筑给噪声假抖动.
6. **判缩放/zoom**: 只能靠相似变换 scale 分量 (`estimateAffinePartial2D` 的 scale, std<0.005=静止); **phaseCorrelate 测不了缩放 = 盲点**; 肩宽/alpha 面积代理被转体/冷启动污染不可信.
7. **判接地/色温**: 像素 LAB delta / 阴影 layer 曲线 (函数级, 不被脚遮蔽污染); 单帧 band 采样会被脚遮蔽.
8. **覆盖全身各区段** (躯干/大腿/小腿脚) 两边一致才下结论.
9. **判抠像边缘缺陷** (胳膊虚化/两臂间绿晕): 靠 **alpha 梯度均值** (`dump_alpha.py` 0.2-0.8 带, 越高边缘越锐) + **逐像素 diff 热图** (旧版 vs 新版, 看改动在**边缘细线**=治虚化/光晕 还是 **躯干内部块**=治凹陷洞); **analyze 对同一帧 full-frame vs crop 会自相矛盾** (full 报"有绿块+光晕" crop 报"干净") → 弃 analyze 信像素. 举手胳膊虚化/渗出的**主力修复 = core-matte** (坑 9, pose 骨架撑实 core; 验证: 核区 mean α 0.52→0.97), dsr 提分辨率 (0.25→0.4) 只补锐度辅助.

## 测试清单

```
tests/test_bg_swap_defaults.py  # 14 tests: matte 默认开 / grounding+shadow 内置 0 /
                                # ffmpeg 可移植 / 无路径硬编码 / _grounding+loader 存在 /
                                # 3 预设文件 + bg_swap 段 / fitness grounding 0.18 / clean 全关 /
                                # core-bolster 默认关 (2026-07-02 反转) + _pose_core_matte 存在 + pink_sat dest 回归
```

Run: `python -m pytest tests/test_bg_swap_defaults.py -v`

## 后续改进 (未做)

1. 多人长视频 (1 真人 + 数字人): RVM 会 matte 所有人, 需按 pose lead cx 选 / 或只 matte lead 区域.
2. 远景/仰头漏检: 换 SCRFD/YOLO-face 检测器替代 buffalo_l.
3. 残留滑动 (静态背景固有): 换动态背景或接受 (见坑 4).
