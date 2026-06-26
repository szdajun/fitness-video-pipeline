"""测试 _write_manifest — 防止下次再忘记写 manifest"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.upload_utils import _write_manifest

RECORDS_DIR = Path(__file__).parent.parent / "records"
MANIFEST = RECORDS_DIR / "upload_manifest.json"


def test_manifest_writes_entry():
    """_write_manifest 必须写一条记录"""
    if not MANIFEST.exists():
        # 首次跑: 创建空 manifest
        RECORDS_DIR.mkdir(parents=True, exist_ok=True)
        MANIFEST.write_text("[]", encoding="utf-8")

    _write_manifest(
        file_path=r"F:\test\video_ytid_check.mp4",
        coach="测试教练",
        video_type="long",
        ytid="test_ytid_check",
        title="测试标题",
        privacy="private",
        publish_at="2026-06-26T20:00:00+08:00",
    )

    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ytids = [e.get("ytid") for e in entries]
    assert "test_ytid_check" in ytids, f"manifest 缺 test_ytid_check: {ytids}"


def test_manifest_appends_not_overwrites():
    """_write_manifest 必须追加, 不能覆盖旧记录"""
    if not MANIFEST.exists():
        pytest_skip = True
        return
    before = json.loads(MANIFEST.read_text(encoding="utf-8"))
    before_ids = [e.get("ytid") for e in before if e.get("ytid")]

    _write_manifest(
        file_path=r"F:\test\dup.mp4",
        coach="测试",
        video_type="long",
        ytid="dup_ytid_check",
        title="重复测试",
        privacy="private",
    )

    after = json.loads(MANIFEST.read_text(encoding="utf-8"))
    after_ids = [e.get("ytid") for e in after if e.get("ytid")]

    # 旧的应该还在 (新增的不算)
    new_ids = {"test_ytid_check", "dup_ytid_check"}
    for old_id in before_ids:
        assert old_id in after_ids, f"旧记录 {old_id} 被覆盖!"
    # 新的加上
    assert "dup_ytid_check" in after_ids


def test_manifest_entry_has_required_fields():
    """每条记录必须有 ytid / coach / title / privacy"""
    if not MANIFEST.exists():
        return
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for e in entries:
        assert e.get("ytid"), f"缺 ytid: {e}"
        assert e.get("coach"), f"缺 coach: {e}"
        assert e.get("title"), f"缺 title: {e}"
        assert e.get("privacy"), f"缺 privacy: {e}"


def test_cleanup_test_entries():
    """清理测试数据 (避免污染真实 manifest)"""
    if not MANIFEST.exists():
        return
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cleaned = [e for e in entries
               if e.get("ytid") not in ("test_ytid_check", "dup_ytid_check")]
    if len(cleaned) != len(entries):
        MANIFEST.write_text(json.dumps(cleaned, ensure_ascii=False, indent=2),
                            encoding="utf-8")