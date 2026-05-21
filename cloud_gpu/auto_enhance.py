"""自动云增强：上传视频到 AutoDL → GFPGAN 修复 → 下载结果

用法:
    python auto_enhance.py input.mp4 output.mp4 [--strength 0.8] [--keep-remote]

首次使用需在 config 中配置 cloud_enhance 或通过环境变量:
    CLOUD_SSH_HOST, CLOUD_SSH_PORT, CLOUD_SSH_USER, CLOUD_SSH_PASSWORD
"""
import argparse, os, sys, time, json
from pathlib import Path

# 可选依赖：无 paramiko 时打印详细指引
try:
    import paramiko
except ImportError:
    paramiko = None

# 加载本地配置
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def _load_cloud_config():
    """从 config.yaml 加载云配置"""
    config = {}
    try:
        import yaml
        if CONFIG_PATH.exists():
            with open(CONFIG_PATH, encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
            config = cfg.get("cloud_enhance", {})
    except Exception:
        pass
    return config


def auto_enhance(input_path, output_path, strength=0.8, keep_remote=False):
    """自动执行：上传 → 处理 → 下载"""
    cfg = _load_cloud_config()

    host = cfg.get("host") or os.environ.get("CLOUD_SSH_HOST", "connect.westd.seetacloud.com")
    port = cfg.get("port") or int(os.environ.get("CLOUD_SSH_PORT", "19088"))
    user = cfg.get("user") or os.environ.get("CLOUD_SSH_USER", "root")
    password = cfg.get("password") or os.environ.get("CLOUD_SSH_PASSWORD", "")
    remote_dir = cfg.get("remote_dir", "/root/auto_enhance")
    model_path_remote = cfg.get("model_path", "/root/gfpgan_weights/GFPGANv1.4.pth")

    if not password:
        print("错误: 未配置 SSH 密码")
        print("请在 config.yaml 中添加:")
        print("""  cloud_enhance:
    host: connect.westd.seetacloud.com
    port: 19088
    user: root
    password: 你的密码
    model_path: /root/gfpgan_weights/GFPGANv1.4.pth""")
        print("或设置环境变量 CLOUD_SSH_PASSWORD")
        return False

    if paramiko is None:
        print("错误: 需要 paramiko 库")
        print("请运行: pip install paramiko")
        return False

    input_path = Path(input_path)
    if not input_path.exists():
        print(f"错误: 输入文件不存在: {input_path}")
        return False

    file_size = input_path.stat().st_size
    print(f"输入: {input_path.name} ({file_size/1024/1024:.1f} MB)")
    print(f"连接 {host}:{port} ...")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(host, port=port, username=user, password=password, timeout=30)
    except Exception as e:
        print(f"SSH 连接失败: {e}")
        print("提示: 确保 AutoDL 实例已开机")
        return False
    print("已连接")

    sftp = ssh.open_sftp()

    # 确保远程目录存在
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

    # 运行 GFPGAN
    remote_output = f"{remote_dir}/{input_path.stem}_enhanced.mp4"
    print(f"运行 GFPGAN 增强...")
    cmd = (f"cd /root && python enhance_face.py '{remote_video}' '{remote_output}' "
           f"--strength {strength} --model_path {model_path_remote}")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=1800)
    exit_code = stdout.channel.recv_exit_status()
    output_text = stdout.read().decode('utf-8', errors='replace')
    error_text = stderr.read().decode('utf-8', errors='replace')
    print(output_text)
    if exit_code != 0:
        print(f"远程处理失败 (exit={exit_code}):")
        print(error_text[-500:])
        sftp.close()
        ssh.close()
        return False

    # 检查远程输出
    try:
        remote_size = sftp.stat(remote_output).st_size
    except FileNotFoundError:
        print(f"错误: 远程输出文件不存在")
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

    # 清理远程文件
    if not keep_remote:
        try:
            sftp.remove(remote_video)
            sftp.remove(remote_output)
            print("远程临时文件已清理")
        except Exception:
            pass

    sftp.close()
    ssh.close()
    print(f"完成! 输出: {output_path}")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoDL 自动云增强')
    parser.add_argument('input', help='输入视频路径')
    parser.add_argument('output', nargs='?', default=None, help='输出视频路径')
    parser.add_argument('--strength', type=float, default=0.8, help='修复强度 0-1')
    parser.add_argument('--keep-remote', action='store_true', help='保留远程文件')
    args = parser.parse_args()

    out = args.output or str(Path(args.input).with_stem(Path(args.input).stem + '_enhanced'))
    success = auto_enhance(args.input, out, args.strength, args.keep_remote)
    sys.exit(0 if success else 1)
