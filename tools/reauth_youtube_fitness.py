"""重新授权 YouTube 'fitness'(胭脂虎)频道 token.

2026-07-07: youtube_token_yanzhi.pickle 过期/被撤销 (invalid_grant), 刷新失败.
ComfyUI 的 get_authenticated_service 在 TOKEN_YANZHI 缺失时会 fallback 到 TOKEN_FILE
(可能是不通的别的账号), 不安全. 本脚本显式跑 InstalledAppFlow 并把新 token 存到
TOKEN_YANZHI, 保证 channel='fitness' 上传走胭脂虎账号.

用法 (会弹浏览器登录 Google, 选胭脂虎健身频道):
    uv run python tools/reauth_youtube_fitness.py
"""
import os
import pickle
import sys

YT_DIR = r"F:\wkspace\ComfyUI\custom_nodes"
sys.path.insert(0, YT_DIR)

from youtube_upload import TOKEN_YANZHI, CLIENT_SECRET, SCOPES  # noqa: E402
from google_auth_oauthlib.flow import InstalledAppFlow  # noqa: E402


def main():
    if not os.path.exists(CLIENT_SECRET):
        raise FileNotFoundError(f"缺 client_secret.json: {CLIENT_SECRET}")
    print(f"[reauth] 目标 token 文件: {TOKEN_YANZHI}")
    print("[reauth] 即将打开浏览器, 请登录 **胭脂虎健身** 频道对应的 Google 账号并授权...")
    flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET, SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent")
    os.makedirs(os.path.dirname(TOKEN_YANZHI), exist_ok=True)
    with open(TOKEN_YANZHI, "wb") as f:
        pickle.dump(creds, f)
    print(f"[reauth] OK, 新 token 已存: {TOKEN_YANZHI}")
    print("[reauth] 现在可以重跑上传 (channel='fitness' 会命中这个新 token).")


if __name__ == "__main__":
    main()
