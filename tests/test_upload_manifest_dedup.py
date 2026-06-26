"""测试 manifest 防重复: 同一文件不重复 ytid, 同一 ytid 不绑多个 file"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

MANIFEST = Path(__file__).parent.parent / "records" / "upload_manifest.json"


def test_no_duplicate_ytid_in_manifest():
    """同一 ytid 不能出现在多条记录"""
    if not MANIFEST.exists():
        return
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    ytids = [e.get("ytid") for e in entries if e.get("ytid") and e["ytid"] != "PENDING_UPLOAD"]
    dups = {y for y in ytids if ytids.count(y) > 1}
    assert not dups, f"manifest 含重复 ytid: {dups}"


def test_no_duplicate_file_in_manifest():
    """同一 file 不能上传两次"""
    if not MANIFEST.exists():
        return
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    files = [e.get("file") for e in entries if e.get("file")]
    dups = {f for f in files if files.count(f) > 1}
    assert not dups, f"manifest 含重复 file: {dups}"


def test_each_entry_has_required_fields():
    """每条记录必须有 ytid/coach/title/file"""
    if not MANIFEST.exists():
        return
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for e in entries:
        if e.get("ytid") == "PENDING_UPLOAD":
            continue  # 待上传的临时条目
        for field in ("ytid", "coach", "title", "file"):
            assert e.get(field), f"manifest 条目缺字段 {field}: {e}"