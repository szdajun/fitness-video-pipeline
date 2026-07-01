"""清理中间文件, 只留最终交付 (一次性).
保留: *_final_16x9_1920x1080.mp4 / *_douyin.mp4 / *_yt_shorts.mp4
清空: _temp/ 全部
删除: output/*/ 下其他所有文件 (各 stage 中间, keypoints, _combined, _full_16x9, intro/outro...)
不动: records/ tools/ source_videos/"""
import shutil
from pathlib import Path

ROOT = Path("F:/wkspace/fitness-video-pipeline")
KEEP_END = ("_final_16x9_1920x1080.mp4",
            "_final_16x9_1920x1080_douyin.mp4",
            "_final_16x9_1920x1080_yt_shorts.mp4")


def dsize(p):
    if p.is_dir():
        return sum(f.stat().st_size for f in p.rglob('*') if f.is_file())
    return p.stat().st_size if p.is_file() else 0


freed = 0; ndel = 0; kept = []; odd = []

# 1. _temp 全清
temp = ROOT / "_temp"
if temp.exists():
    for it in temp.iterdir():
        sz = dsize(it)
        try:
            if it.is_dir(): shutil.rmtree(it, ignore_errors=True)
            else: it.unlink()
            freed += sz; ndel += 1
        except Exception as e:
            print(f"[跳过] {it}: {e}")

# 2. output/ 删中间留最终
for d in (ROOT / "output").iterdir():
    if not d.is_dir():
        continue
    for f in list(d.iterdir()):
        if f.is_dir():
            sz = dsize(f); shutil.rmtree(f, ignore_errors=True)
            freed += sz; ndel += 1; continue
        if f.name.endswith(KEEP_END):
            kept.append(str(f.relative_to(ROOT))); continue
        if not f.suffix:
            odd.append(f"{f.name} ({dsize(f)/1048576:.0f}MB)")
        try:
            sz = f.stat().st_size; f.unlink()
            freed += sz; ndel += 1
        except Exception as e:
            print(f"[跳过] {f}: {e}")

print(f"\n=== 删除 {ndel} 项, 释放 {freed/1073741824:.1f} GB ===")
print(f"\n保留最终文件 ({len(kept)}):")
for k in sorted(kept):
    print(f"  {k}")
if odd:
    print(f"\n无扩展名文件(已删): {odd}")
