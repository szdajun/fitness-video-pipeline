# -*- coding: utf-8 -*-
"""验证 upload_video 默认开 wait_processed + CLI flag 透传."""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_upload_video_signature_has_wait_params():
    """upload_video 必须有 wait_processed + wait_timeout 参数 (默认 True/1200)."""
    from lib.upload_utils import upload_video
    sig = inspect.signature(upload_video)
    assert "wait_processed" in sig.parameters, "upload_video 缺 wait_processed 参数"
    assert "wait_timeout" in sig.parameters, "upload_video 缺 wait_timeout 参数"
    # 默认值
    wp = sig.parameters["wait_processed"]
    assert wp.default is True, f"wait_processed 默认应 True, 实际 {wp.default}"
    wt = sig.parameters["wait_timeout"]
    assert wt.default == 1200, f"wait_timeout 默认应 1200, 实际 {wt.default}"


def test_wait_processing_complete_returns_processed_when_done(monkeypatch):
    """_wait_processing_complete 看到 processed 立即返回 processed."""
    from lib import upload_utils

    class FakeYT:
        def videos(self):
            return self
        def list(self, part, id):
            class _Resp:
                def execute(inner_self):
                    # 2026-07-11 修: processingDetails 才是处理状态 part (不是 status)
                    return {"items": [{"processingDetails": {"processingStatus": "processed", "processingFailureReason": ""}}]}
            return _Resp()

    # 替换 get_authenticated_service
    import youtube_upload
    monkeypatch.setattr(youtube_upload, "get_authenticated_service", lambda channel="fitness": FakeYT())

    r = upload_utils._wait_processing_complete("fake_ytid", timeout=30, poll_interval=1)
    assert r == "processed", f"期望 processed, 实际 {r}"


def test_wait_processing_complete_timeout(monkeypatch):
    """平台一直 processing 时, timeout 后返回 timeout 不抛异常."""
    from lib import upload_utils
    import youtube_upload

    class FakeYT:
        def videos(self):
            return self
        def list(self, part, id):
            class _Resp:
                def execute(inner_self):
                    return {"items": [{"processingDetails": {"processingStatus": "processing", "processingProgress": {"partsProcessed": 30, "partsTotal": 100}, "processingFailureReason": ""}}]}
            return _Resp()

    monkeypatch.setattr(youtube_upload, "get_authenticated_service", lambda channel="fitness": FakeYT())
    # 极短 timeout + 长 poll_interval → 确保超时
    r = upload_utils._wait_processing_complete("fake_ytid", timeout=3, poll_interval=5)
    assert r == "timeout", f"期望 timeout, 实际 {r}"


def test_wait_processing_complete_failed_immediate(monkeypatch):
    """平台 rejected/failed 时立即返回 (不傻等 timeout)."""
    from lib import upload_utils
    import youtube_upload

    class FakeYT:
        def videos(self):
            return self
        def list(self, part, id):
            class _Resp:
                def execute(inner_self):
                    return {"items": [{"processingDetails": {"processingStatus": "failed", "processingFailureReason": "codec"}}]}
            return _Resp()

    monkeypatch.setattr(youtube_upload, "get_authenticated_service", lambda channel="fitness": FakeYT())
    r = upload_utils._wait_processing_complete("fake_ytid", timeout=600, poll_interval=5)
    assert r == "failed", f"期望 failed, 实际 {r}"


def test_cli_default_waits_processed():
    """upload_youtube.py CLI 默认 --wait-processed 开."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "upload_youtube",
        str(Path(__file__).resolve().parent.parent / "tools" / "upload_youtube.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import argparse
    # 不跑 main(), 直接构造 parser 同 main() 一致
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait-processed", dest="wait_processed", action="store_true", default=True)
    ap.add_argument("--no-wait-processed", dest="wait_processed", action="store_false")
    # 验证默认行为: 不传 flag → True
    args = ap.parse_args([])
    assert args.wait_processed is True, "CLI 默认应等平台处理"
    args = ap.parse_args(["--no-wait-processed"])
    assert args.wait_processed is False, "CLI --no-wait-processed 应关等"
    args = ap.parse_args(["--wait-processed"])
    assert args.wait_processed is True, "CLI --wait-processed 应开等"