# HANDOFF.md — 当前迭代状态（活文档）

> **新会话先读本文件**（见 `CLAUDE.md` 的"会话开局协议"）。
> 这里只记"现在在做什么 / 上次停在哪 / 下一步 / 待用户确认"，不重复架构（架构看 `docs/PROJECT_DESIGN.md`，规则看 `CLAUDE.md`，历史坑看 `memory/`）。
> **每次会话结束前更新本文件**——这是会话衔接的核心。

最后更新: 2026-07-01（**网红多人 v2: 修三问题 — 背景拉宽变形/人物变瘦/胳膊消失, cover+避天棚**）：用户看 v1 成品(`dsr05.mp4`)报三个新问题: ① **背景小推车拉宽变形厉害**(根因=背景源 720×1280 竖屏被 prepare_bg 强制 resize 成 1280×720 横屏, 横拉 2.37×; 之前丽丽案例源也竖屏没暴露, 这次横屏源撞竖屏背景才炸); ② **人物变瘦**; ③ **胳膊与背景天棚重叠时消失/虚化/带原始绿树**。②③同源=RVM alpha 在身体边缘/细胳膊系统性偏低 → 合成 `out=frame*mask+bg*(1-mask)` 边缘半透明混背景=轮廓收缩显瘦, 胳膊 alpha 掉近 0 被背景吃掉/半透明带原绿树。**用户方向(治本优于硬刚 alpha)**: "选合适背景角度, 胳膊举起避开天棚"。**像素诊断**: 时代广场背景天棚只集中源 y=25~35% 一小段(放大后 2275 高的 569~796px), 之上全天空之下全地面(y>35% 占 65%高度全地面); cover **中心裁切**取源 34~66% **正好把天棚映射到画面上部**(=胳膊区) → 重叠。**修复两处**: ① `prepare_bg` 强制 resize 改**等比 cover**(`_cover_resize`: scale 覆盖+裁切不变形)治拉宽; ② cover 加竖向偏移 **`--bg-crop-y 0.61`**(0=顶/0.5=中心默认/1=底)裁切窗下移到**天棚下方** → 整个背景变灰砖地面广场, 胳膊举起落干净地面背景(RVM 抠得准不消失/不虚化), 脚踩地面, 天棚降到画面外(上 1/3 天棚 11%→2.1%, 整图 0.7%)。CLI/oversize png 名含 cy 隔离缓存。**v2 已完成 + 验证通过**: 渲染 7488 帧 @1.7fps (4505s), 换脸 100%(7488/7488) / 背面跳 0 / mask 漏 0。**核心验证(三张举手帧 t70s/227s/248s, 正是 v1 报"胳膊消失"的时刻)逐人确认: 胳膊完整可见、无绿树梢残留、举高的胳膊落在干净灰色水泥地面(天棚已移出画面)**; 中段帧 f3000(t100s) 验证人物不变瘦、小推车不变形。**#51 alpha 后处理不需要**(用户方向「避天棚」让 RVM 在干净地面背景上自然抠准, 变瘦+胳膊消失随之解决, 无需激进 alpha 增强)。成品 `output/bgswap/网红多人_丽丽_时代广场_v2.mp4`(299M) 已 sync 桌面 `网红多人_丽丽_时代广场_v2.mp4`。详见 memory `bg-swap-tool-influencer`。**✅ 用户接受 v2 定稿** (原话"先这样吧, 视频播放中还是能看到背景不干净的时候出现, 但很少") — 残留=少数极端举手帧胳膊触及画面上沿天棚残留区/个别帧 RVM alpha 仍偏低, 占比很低可接受; 彻底治备选=#51 alpha 后处理(闭运算/gamma/pose 增广), 当前不触发。

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
| **网红多人健身操** | bg_swap **多人换脸案例 (--swap-all 首次实测 + dsr 修胳膊虚化)**: `网红多人健身操.mp4`(3人 720p 7488帧 249.5s)。`--swap-all --dsr 0.5 --preset fitness --coach 丽丽` → 3人都换丽丽脸 + 时代广场背景。**渲染完成 7488/7488 帧(100%换脸/背面跳0/mask漏0), 78min@1.6fps, 317MB**。**dsr=0.5 修复举手时胳膊虚化+两臂间绿树残留**(根因=RVM alpha 边缘软光晕非凹陷洞; 锐度+39%/零速度代价)。**全片验证通过**(`_temp/verify_full.py`): 绿树洞残留全片仅 0.08%(12帧抽样11帧=0), 中段帧(t=160s)视觉确认 3人同脸/广场背景/胳膊边缘清晰无绿晕/两臂间无绿残留。**成品已同步桌面供用户主观确认**。**v2 定稿(2026-07-01)**: 用户报三新问题(背景拉宽变形/人物变瘦/胳膊与天棚重叠消失虚化带绿树), 修复=`_cover_resize` cover 不拉伸(治①变形)+`--bg-crop-y 0.61` 裁切窗下移避天棚(背景变地面广场, 上1/3天棚 11%→0.8~1.2%, 治②③, RVM 在干净地面背景自然抠准)。**验证通过**: 三举手帧(t70/227/248s)逐人胳膊完整/无绿残留/落干净水泥地面, 中段帧(t100s)不变瘦不变形, 7488帧 100%换脸/0背面跳/0漏检。**残留**: 播放偶尔背景不净(极少, 用户接受"先这样吧") | `output/bgswap/网红多人_丽丽_时代广场_v2.mp4`(299M 定稿); 桌面 `~/Desktop/短视频素材/网红多人_丽丽_时代广场_v2.mp4` |

---

## 📌 持久备忘（跨迭代记住）

- 上传只传 `*_final_16x9_1920x1080.mp4`（含片头片尾），**不传** `*_full_16x9.mp4`（去头去尾副本）。
- 跑管线前清 `_temp/cg_*`（color_grade JPEG 序列易爆盘）。
- 换脸/pose 改逻辑后删 `*_keypoints.json` 才会重跑。
- ffmpeg 必须用 `C:/Users/18091/ffmpeg/ffmpeg.exe`，不能用 Winget 版。
- 教练名必须在文件名最前（`建玲1.mp4` ✓，`合并_建玲` ✗）。
- 片头音乐已钉 config 默认（`intro_outro.intro_music_from_main: true`），新视频自动用截前留落点方案，无需手动设。
