# 预设风格参考

## 预设速查表

| 预设 | 分辨率 | 竖/横 | 体型美化 | 稳定化 | 运镜 | 能量条 | 典型场景 |
|------|--------|-------|----------|--------|------|--------|----------|
| `shorts` | 1080x1920 | 竖 | 中度 | ❌ 关闭 | ❌ 关闭 | ✅ | 抖音/快手/Shorts |
| `youtube` | 1920x1080 | 横 | 无 | ❌ 关闭 | ❌ 关闭 | ✅ | YouTube 横屏 |
| `sexy` | 1080x1920 | 竖 | 强效 | ❌ 关闭 | smooth | ❌ | 身材展示 |
| `natural` | 1080x1920 | 竖 | 轻度 | ❌ 关闭 | smooth | ❌ | 自然记录 |
| `dramatic` | 1080x1920 | 竖 | 中度 | ❌ 关闭 | dual | ❌ | 电影感 |
| `gimbal` | 1080x1920 | 竖 | 中度 | ✅ 开启 | smooth | ❌ | 云台稳定拍摄 |
| `beauty` | 1080x1920 | 竖 | 中度 | ❌ 关闭 | smooth | ❌ | 多人场景 |
| `night_gym` | 1080x1920 | 竖 | 中度 | ❌ 关闭 | ❌ 关闭 | ❌ | 晚间低光环境 |
| `clean` | 1080x1920 | 竖 | 无 | ❌ 关闭 | ❌ 关闭 | ❌ | 最小处理 |

---

## 详细说明

### shorts — 竖版短视频（推荐默认）

适合：手机横拍、晚间手持拍摄、低分辨率素材

```yaml
stages:
  stabilize: false    # 低分辨率手持拍摄，stabilization 会放大抖动
  ken_burns: false    # 竖版裁切放大残余晃动，禁用运镜
  body_warp: true
  beat_flash: true
  highlight: true
  energy_bar: true
```

**为什么禁用 stabilize 和 ken_burns？**
- 原始素材分辨率低（960x544 / 1280x720）
- 9:16 裁切将高度放大 2x+，任何残余抖动同步放大
- 这类素材保留原始晃动感反而更自然

### youtube — YouTube 横屏版

适合：直接上传 YouTube，保持原始 16:9 构图

```yaml
stages:
  h2v_convert: false   # 不裁切，保持原始比例
  stabilize: false
  beat_flash: true
  highlight: true
  energy_bar: true
```

**为什么 16:9 不抖？**
- 像素 1:1 或近 1:1 映射，无放大
- 544px 高度 → 1080px，几乎原始精度

### gimbal — 云台/固定机位

适合：高清拍摄（1080p+）、有稳定云台的相机

```yaml
stages:
  stabilize: true     # 画面本身稳定，vid.stab 增强效果
  ken_burns: true     # 轻微 zoom，让画面更动感
  body_warp: true
```

**stabilize 参数：**
```yaml
stabilize:
  shakiness: 10
  accuracy: 15
  zoom: 3
  smoothing: 20
```

### dramatic — 电影感

适合：有视觉冲击力的调色 + 运镜

```yaml
stages:
  stabilize: false
  ken_burns: true     # dual 景别切换模式
  body_warp: true
  color_grade:
    brightness: 5
    clahe: true

ken_burns:
  mode: dual
  dual_close_zoom: 1.1
  dual_cycle_seconds: 8
  dual_pan_amplitude: 8
```

### night_gym — 晚间健身房

适合：光线不足、有色温偏差（黄灯）的环境

```yaml
stages:
  stabilize: false
  ken_burns: false
  color_grade:
    brightness: 10
    warmth: -10        # 抵消黄灯暖色
    clahe: true
```

---

## 体型参数参考

| 美化方向 | 参数 | 推荐值 | 说明 |
|----------|------|--------|------|
| 瘦腿 | `leg_slim` | 0.8~0.9 | 横向收紧 |
| 拉长腿 | `leg_lengthen` | 1.15~1.25 | 纵向拉伸（更显腿长）|
| 瘦腰 | `waist_slim` | 0.75~0.85 | 腰部收紧 |
| 丰满 | `chest_enlarge` | 1.15~1.35 | 胸部放大 |
| 整体瘦 | `overall_slim` | 0.85~0.95 | 全身均匀收缩 |
| 头身比 | `head_ratio` | 1.0~1.1 | 头大更显萌 |

**组合示例（sexy 预设）：**
```yaml
body_warp:
  leg_lengthen: 1.25
  waist_slim: 0.75
  chest_enlarge: 1.35
  head_ratio: 1.05
```

---

## 能量条参数

```yaml
energy_bar:
  width: 16              # 能量条宽度（像素）
  margin_right: 20      # 距右边框距离
  margin_bottom: 120    # 距底部距离
  height: 400           # 能量条最大高度
  smoothing: 0.85        # EMA 平滑系数（越大越平滑）
  min_fill_ratio: 0.15   # 最低填充比例
```

**参数调优建议：**
- 运动幅度大的操课 → `smoothing: 0.7`（响应更快）
- 运动幅度小的操课 → `smoothing: 0.9`（更平滑）
- 竖版靠右显示，`margin_bottom` 控制垂直位置

---

## Ken Burns dual 模式参数

```yaml
ken_burns:
  mode: dual            # smooth=微幅, dual=景别切换
  dual_close_zoom: 1.08  # 特写时放大比例
  dual_close_zoom_v: 1.04
  dual_cycle_seconds: 10  # 完整周期秒数
  dual_pan_amplitude: 5   # 微平移幅度（像素）
  dual_dwell: 0.5        # 特写停留系数（越大峰值越长）
  dual_motion_response: 0.3  # 运动响应系数
  dual_motion_zoom_response: 0.15  # 运动对 zoom 的影响
```

**参数调优：**
- 想要明显景别切换 → `dual_close_zoom: 1.15`，`dual_dwell: 0.3`
- 想要平滑过渡 → `dual_close_zoom: 1.05`，`dual_dwell: 0.7`
- 运动响应影响 pan 幅度和 zoom 调制，避免在低分辨率素材上放大抖动
