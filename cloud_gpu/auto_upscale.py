"""自动云超分：上传视频到 AutoDL → Real-ESRGAN 4K → 下载结果

用法:
    python auto_upscale.py input.mp4 [output.mp4]

依赖: pip install paramiko
"""
import argparse, os, sys, time, json
from pathlib import Path

try:
    import paramiko
except ImportError:
    paramiko = None

# AutoDL 连接信息（也可通过 config.yaml 配置）
DEFAULT_HOST = "connect.westd.seetacloud.com"
DEFAULT_PORT = 19088
DEFAULT_USER = "root"
DEFAULT_PASS = "ptIknGQbjTrL"


def _load_cloud_config():
    """从项目 config.yaml 加载云配置"""
    config = {}
    try:
        import yaml
        cfg_path = Path(__file__).parent.parent / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path, encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
            config = cfg.get("cloud_enhance", {})
    except Exception:
        pass
    return config


def auto_upscale(input_path, output_path, keep_remote=False):
    """上传 → 超分 → 下载"""
    cfg = _load_cloud_config()

    host = cfg.get("host") or DEFAULT_HOST
    port = cfg.get("port") or DEFAULT_PORT
    user = cfg.get("user") or DEFAULT_USER
    password = cfg.get("password") or os.environ.get("CLOUD_SSH_PASSWORD", DEFAULT_PASS)
    remote_dir = "/root/auto_upscale"
    model_remote = cfg.get("model_path_esrgan", "/root/upscale_weights/RealESRGAN_x4plus.pth")

    if not password:
        print("错误: 未配置 SSH 密码")
        return False

    if paramiko is None:
        print("错误: 需要 paramiko，请运行 pip install paramiko")
        return False

    input_path = Path(input_path)
    if not input_path.exists():
        print(f"错误: 文件不存在: {input_path}")
        return False

    file_size = input_path.stat().st_size
    print(f"输入: {input_path.name} ({file_size/1024/1024:.1f} MB)")
    print(f"超分目标: 720p → 4K (3840x2160)")

    # 连接
    print(f"连接 {host}:{port} ...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(host, port=port, username=user, password=password, timeout=30)
    except Exception as e:
        print(f"连接失败: {e}")
        print("提示: 确保 AutoDL 实例已开机")
        return False
    print("已连接")

    sftp = ssh.open_sftp()

    # 创建远程目录
    try:
        sftp.stat(remote_dir)
    except FileNotFoundError:
        sftp.mkdir(remote_dir)

    # 上传视频
    remote_video = f"{remote_dir}/{input_path.name}"
    print(f"上传视频 ({file_size/1024/1024:.1f} MB)...")
    t0 = time.time()
    sftp.put(str(input_path), remote_video,
             callback=lambda x, t: print(f"\r  {x/1024/1024:.1f}/{t/1024/1024:.1f} MB", end=''))
    print(f"\n上传完成 ({time.time()-t0:.0f}s)")

    # 检查模型文件
    try:
        sftp.stat(model_remote)
        print(f"模型已存在: {model_remote}")
    except FileNotFoundError:
        print(f"模型不存在: {model_remote}")
        print("请先上传 RealESRGAN_x4plus.pth 到 AutoDL")
        sftp.close()
        ssh.close()
        return False

    # 运行超分
    remote_output = f"{remote_dir}/{input_path.stem}_4K.mp4"
    print("运行 Real-ESRGAN 4K 超分...")
    cmd = (f"cd /root && python upscale_video.py '{remote_video}' '{remote_output}' "
           f"--model_path {model_remote}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=7200)  # 2h 超时
    exit_code = stdout.channel.recv_exit_status()
    output_text = stdout.read().decode('utf-8', errors='replace')
    error_text = stderr.read().decode('utf-8', errors='replace')
    print(output_text)
    if exit_code != 0:
        print(f"远程超分失败 (exit={exit_code}):")
        print(error_text[-500:])
        sftp.close()
        ssh.close()
        return False

    # 检查远程输出
    try:
        remote_size = sftp.stat(remote_output).st_size
    except FileNotFoundError:
        print("错误: 远程输出文件不存在")
        sftp.close()
        ssh.close()
        return False
    print(f"远程文件: {remote_size/1024/1024:.1f} MB")

    # 下载结果
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"下载结果...")
    t0 = time.time()
    sftp.get(remote_output, str(output_path),
             callback=lambda x, t: print(f"\r  {x/1024/1024:.1f}/{t/1024/1024:.1f} MB", end=''))
    print(f"\n下载完成 ({time.time()-t0:.0f}s)")

    # 清理远程
    if not keep_remote:
        for f in [remote_video, remote_output]:
            try:
                sftp.remove(f)
            except Exception:
                pass
        print("远程临时文件已清理")

    sftp.close()
    ssh.close()
    print(f"完成! 输出: {output_path}")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoDL 自动 4K 超分')
    parser.add_argument('input', help='输入视频路径')
    parser.add_argument('output', nargs='?', default=None, help='输出视频路径')
    parser.add_argument('--keep-remote', action='store_true', help='保留远程文件')
    args = parser.parse_args()

    out = args.output or str(Path(args.input).with_stem(Path(args.input).stem + '_4K'))
    success = auto_upscale(args.input, out, args.keep_remote)
    sys.exit(0 if success else 1)
