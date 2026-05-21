# 云端 GPU 人脸修复

用 AutoDL 云 GPU 跑 GFPGAN 人脸修复，提升视频脸部清晰度。

## 操作步骤

### 1. 注册 AutoDL (5分钟)
1. 打开 https://www.autodl.com 注册 + 实名认证
2. 充值 20-50 元（一次视频大概 2-8 元）

### 2. 创建实例 (3分钟)
1. 控制台 → 租用新实例
2. 地区选 **A区** 或 **北京**（便宜）
3. GPU 选 **RTX 4090**（2.48元/时）或 **RTX 3060**（1.18元/时）
4. 镜像选 **PyTorch**（如 `PyTorch 2.0.0 Python 3.9`）
5. 创建后开机

### 3. 上传文件 (1分钟)
1. 实例详情页 → **JupyterLab**
2. 把 `cloud_gpu/` 目录里的 `enhance_face.py` + `requirements.txt` 上传
3. 把要处理的视频（如 `艳青4_final_16x9.mp4`）上传

### 4. 安装依赖 + 运行 (3-5分钟)
在 JupyterLab 的 Terminal 里执行：

```bash
pip install -r requirements.txt
python enhance_face.py 艳青4_final_16x9.mp4 output_enhanced.mp4 --strength 0.8
```

### 5. 下载结果
在 JupyterLab 里右键 `output_enhanced.mp4` → 下载

### 6. 关机
下载完后记得**关机**（控制台→关机），否则继续计费。

## 参数说明

| 参数 | 默认 | 说明 |
|------|------|------|
| `--strength` | 0.8 | 修复强度 0-1，越大脸越清晰但可能失真 |
| `--interval` | 5 | 每 N 帧检测一次人脸，越小越准确但慢 |

## 预估费用

| GPU | 价格 | 2000帧耗时 | 费用 |
|-----|------|-----------|------|
| RTX 4090 | 2.48元/h | ~3分钟 | ~0.2元 |
| RTX 3060 | 1.18元/h | ~8分钟 | ~0.2元 |
