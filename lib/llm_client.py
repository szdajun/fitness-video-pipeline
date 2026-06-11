"""
lib/llm_client.py - 统一的开源/免费 LLM 客户端

设计:
  - 所有 provider 都用 OpenAI 兼容协议 (chat/completions endpoint)
  - 按 config.yaml 的 llm.providers 顺序找第一个 enabled=true 的
  - api_key 支持 ${ENV_VAR} 语法, 从环境变量读 (避免明文写 yaml)
  - 失败自动重试; 全部 provider 都跑不通就返回 None, 调用方可降级用静态模板

用法:
    from lib.llm_client import LLMClient
    client = LLMClient.from_config()  # 读 config.yaml
    if client and client.enabled:
        text = client.chat("给胭脂虎健身视频写一句弹幕")
        if text:
            print(text)
        else:
            print("(LLM 调用失败, 用静态模板)")
"""

import json
import os
import re
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from urllib import request, error

CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def _resolve_env(value: str) -> str:
    """${VAR} → os.environ['VAR']; 找不到返回空串."""
    if not isinstance(value, str):
        return value
    match = re.match(r"^\$\{([A-Z_][A-Z0-9_]*)\}$", value.strip())
    if match:
        return os.environ.get(match.group(1), "")
    return value


class LLMClient:
    """统一 LLM 客户端 (OpenAI 兼容)"""

    def __init__(self, llm_config: Dict[str, Any]):
        self.cfg = llm_config or {}
        self.enabled = bool(self.cfg.get("enabled", False))
        self.providers = self.cfg.get("providers", [])
        self.temperature = self.cfg.get("temperature", 0.7)
        self.max_tokens = self.cfg.get("max_tokens", 1024)
        self.retry = int(self.cfg.get("retry", 2))
        self.uses = self.cfg.get("uses", {})
        # 解析第一个 enabled 且 api_key 有值的 provider
        self.active = self._select_active()

    def _select_active(self) -> Optional[Dict[str, Any]]:
        for p in self.providers:
            if not p.get("enabled"):
                continue
            api_key = _resolve_env(p.get("api_key", ""))
            if not api_key:
                continue  # 环境变量没设, 跳过
            return {
                "name": p.get("name"),
                "base_url": p.get("base_url", "").rstrip("/"),
                "api_key": api_key,
                "model": p.get("model"),
                "timeout": p.get("timeout", 30),
            }
        return None

    @classmethod
    def from_config(cls, config_path: Optional[Path] = None) -> Optional["LLMClient"]:
        """从 config.yaml 加载. 找不到 config 或 llm 块返回 None."""
        path = Path(config_path) if config_path else CONFIG_PATH
        if not path.exists():
            return None
        try:
            import yaml
            with open(path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            print(f"[LLM] config 读失败: {e}")
            return None
        return cls(cfg.get("llm", {}))

    def is_use_allowed(self, use_key: str) -> bool:
        """检查特定用途是否启用 (例: 'danmaku_lines')"""
        return self.enabled and self.active is not None and bool(self.uses.get(use_key, False))

    def chat(self, prompt: str, system: Optional[str] = None,
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> Optional[str]:
        """单轮对话. 失败返回 None, 调用方应降级到静态模板."""
        if not self.enabled or not self.active:
            return None

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.active["model"],
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        url = f"{self.active['base_url']}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.active['api_key']}",
            "Content-Type": "application/json",
        }

        for attempt in range(self.retry + 1):
            try:
                req = request.Request(
                    url,
                    data=json.dumps(payload).encode("utf-8"),
                    headers=headers,
                    method="POST",
                )
                with request.urlopen(req, timeout=self.active["timeout"]) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    return data["choices"][0]["message"]["content"].strip()
            except error.HTTPError as e:
                body = e.read().decode("utf-8", errors="ignore")[:300]
                print(f"[LLM:{self.active['name']}] HTTP {e.code} (attempt {attempt+1}): {body}")
            except Exception as e:
                print(f"[LLM:{self.active['name']}] {type(e).__name__} (attempt {attempt+1}): {e}")
            if attempt < self.retry:
                time.sleep(1.5 * (attempt + 1))

        return None

    def chat_with_fallback(self, prompt: str, fallback: str, **kwargs) -> str:
        """调 LLM, 失败返回 fallback. 调用方不必判 None."""
        result = self.chat(prompt, **kwargs)
        return result if result else fallback

    def __repr__(self):
        if not self.enabled:
            return "<LLMClient disabled>"
        if not self.active:
            return f"<LLMClient enabled, no active provider (check api_key env vars)>"
        return f"<LLMClient provider={self.active['name']} model={self.active['model']}>"


# 便捷函数: 全局单例
_GLOBAL_CLIENT: Optional[LLMClient] = None


def get_llm() -> Optional[LLMClient]:
    """全局单例 (懒加载). 调用方使用: get_llm().chat(...)"""
    global _GLOBAL_CLIENT
    if _GLOBAL_CLIENT is None:
        _GLOBAL_CLIENT = LLMClient.from_config()
    return _GLOBAL_CLIENT


if __name__ == "__main__":
    # 自检
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    c = LLMClient.from_config()
    print(f"Client: {c}")
    if c and c.active:
        print(f"用途开关: {c.uses}")
        text = c.chat("给胭脂虎健身打卡视频写一句 15 字以内的弹幕", system="你是中文弹幕大师, 简洁有力")
        print(f"测试响应: {text}")
    elif c:
        print("LLM 已启用但没有可用 provider — 检查 config.yaml 的 llm.providers[*].enabled 和环境变量")
    else:
        print("LLM 未启用 — 在 config.yaml 把 llm.enabled 设为 true 并配置 provider")
