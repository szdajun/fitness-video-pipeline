"""健身视频管理工具 — 索引/查询/统计/管理

用法:
    python add_to_index.py scan                         # 扫描并更新索引
    python add_to_index.py scan --input DIR --output DIR # 指定目录扫描
    python add_to_index.py list                          # 列出所有视频
    python add_to_index.py list --status raw             # 按状态筛选
    python add_to_index.py list --coach 艳青             # 按教练筛选
    python add_to_index.py list --source input           # 仅素材/仅输出
    python add_to_index.py status 艳青和丽丽.mp4          # 视频详情
    python add_to_index.py stats                         # 统计信息
    python add_to_index.py search 关键词                 # 搜索
    python add_to_index.py tag 艳青和丽丽.mp4 双人       # 添加标签
    python add_to_index.py tag 艳青和丽丽.mp4 --remove 双人  # 删除标签
    python add_to_index.py note 艳青和丽丽.mp4 已处理完  # 添加备注
    python add_to_index.py reindex                       # 重建索引
"""

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Optional

# ── 路径配置 ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent
DEFAULT_INPUT_DIR = "C:/Users/18091/Desktop/短视频素材"
DEFAULT_OUTPUT_BASE = "C:/Users/18091/Desktop/shorts_output"
INDEX_DB = PROJECT_ROOT / "video_index.db"
COACHES_YAML = PROJECT_ROOT / "coaches.yaml"
FFPROBE = "C:/Users/18091/ffmpeg/ffprobe.exe"

# 已知教练关键词（用于从文件名自动识别）
COACH_KEYWORDS = ["艳青", "丽丽", "小红豆", "建玲", "郭海军", "枫林红", "笑笑", "秀秀"]


# ── 数据库 ─────────────────────────────────────────────────────

def _get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(INDEX_DB))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _init_db():
    conn = _get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS videos (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            filename    TEXT    NOT NULL,
            filepath    TEXT    NOT NULL,
            filesize    INTEGER DEFAULT 0,
            created_at  TEXT,
            modified_at TEXT,
            duration    REAL    DEFAULT 0,
            width       INTEGER DEFAULT 0,
            height      INTEGER DEFAULT 0,
            fps         REAL    DEFAULT 0,
            status      TEXT    DEFAULT 'source',
            source_dir  TEXT    DEFAULT 'input',
            coach_name  TEXT    DEFAULT '',
            video_date  TEXT,
            notes       TEXT    DEFAULT '',
            last_indexed TEXT   DEFAULT (datetime('now','localtime'))
        );

        CREATE TABLE IF NOT EXISTS processing_records (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id          INTEGER NOT NULL REFERENCES videos(id),
            output_path       TEXT,
            output_filename   TEXT,
            preset_used       TEXT,
            stages_completed  TEXT,
            processed_at      TEXT DEFAULT (datetime('now','localtime')),
            manifest_path     TEXT
        );

        CREATE TABLE IF NOT EXISTS tags (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL REFERENCES videos(id),
            tag      TEXT    NOT NULL,
            UNIQUE(video_id, tag)
        );

        CREATE INDEX IF NOT EXISTS idx_videos_status ON videos(status);
        CREATE INDEX IF NOT EXISTS idx_videos_coach  ON videos(coach_name);
        CREATE INDEX IF NOT EXISTS idx_videos_source ON videos(source_dir);
        CREATE INDEX IF NOT EXISTS idx_tags_tag      ON tags(tag);
    """)
    conn.commit()
    conn.close()


# ── FFprobe 读取视频元数据 ────────────────────────────────────

def probe_video(path: Path) -> dict:
    """用 ffprobe 读取视频时长、分辨率、fps"""
    meta = {"duration": 0, "width": 0, "height": 0, "fps": 0}
    if not FFPROBE or not Path(FFPROBE).exists():
        return meta
    try:
        cmd = [
            FFPROBE, "-v", "quiet",
            "-print_format", "json",
            "-show_format", "-show_streams",
            str(path)
        ]
        r = subprocess.run(cmd, capture_output=True, timeout=30)
        r.stdout = r.stdout.decode("utf-8", errors="replace")
        data = json.loads(r.stdout)
        if "format" in data:
            meta["duration"] = float(data["format"].get("duration", 0))
        for s in data.get("streams", []):
            if s.get("codec_type") == "video":
                meta["width"] = int(s.get("width", 0))
                meta["height"] = int(s.get("height", 0))
                tag_fps = s.get("r_frame_rate", "0/1").split("/")
                meta["fps"] = float(tag_fps[0]) / float(tag_fps[1]) if len(tag_fps) == 2 and float(tag_fps[1]) > 0 else 0
                break
    except Exception:
        pass
    return meta


# ── 文件名解析 ─────────────────────────────────────────────────

def guess_coach(filename: str) -> str:
    """从文件名猜测教练名称"""
    for coach in COACH_KEYWORDS:
        if coach in filename:
            return coach
    return ""


def guess_video_date(path: Path) -> Optional[str]:
    """从文件修改时间推断录制日期"""
    try:
        mtime = os.path.getmtime(path)
        return date.fromtimestamp(mtime).isoformat()
    except Exception:
        return None


# ── 扫描引擎 ───────────────────────────────────────────────────

def scan_directory(input_dir: Path, output_base: Path, show_progress: bool = True):
    """扫描素材目录和输出目录，更新索引"""
    conn = _get_db()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = 0
    new_count = 0
    updated_count = 0

    # 收集所有需要索引的文件
    files_to_index = []

    # 1. 扫描素材目录
    if input_dir.exists():
        for f in sorted(input_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
                if f.name.startswith("."):
                    continue
                files_to_index.append((f, "input"))
    else:
        print(f"  素材目录不存在: {input_dir}")

    # 2. 扫描输出目录
    if output_base.exists():
        for date_dir in sorted(output_base.iterdir()):
            if not date_dir.is_dir():
                continue
            for f in sorted(date_dir.iterdir()):
                if f.is_file() and f.suffix.lower() in (".mp4", ".mov", ".avi", ".mkv", ".webm"):
                    if f.name.startswith("."):
                        continue
                    files_to_index.append((f, "output"))
    else:
        print(f"  输出目录不存在: {output_base}")

    if not files_to_index:
        print("  未找到可索引的视频文件")
        conn.close()
        return

    total = len(files_to_index)
    idx = 0

    for fpath, source in files_to_index:
        idx += 1
        fname = fpath.name
        fsize = fpath.stat().st_size
        mtime = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M:%S")
        ctime = datetime.fromtimestamp(os.path.getctime(fpath)).strftime("%Y-%m-%d %H:%M:%S")

        if show_progress:
            print(f"  [{idx}/{total}] {fname}", end="", flush=True)

        # 查重：相同 filepath 且 filesize 没变则跳过
        row = conn.execute(
            "SELECT id, filesize, status FROM videos WHERE filepath = ?",
            (str(fpath.resolve()),)
        ).fetchone()

        if row and row["filesize"] == fsize:
            # 更新 last_indexed 即可
            conn.execute("UPDATE videos SET last_indexed=? WHERE id=?", (now, row["id"]))
            conn.commit()
            if show_progress:
                print("")
            continue

        # 需要探测元数据
        meta = probe_video(fpath)
        coach = guess_coach(fname)
        vdate = guess_video_date(fpath)
        status = "source" if source == "input" else "processed"

        if row:
            # 更新已有记录
            conn.execute("""
                UPDATE videos SET
                    filesize=?, modified_at=?, duration=?, width=?, height=?, fps=?,
                    status=?, source_dir=?, coach_name=?, video_date=?, last_indexed=?
                WHERE id=?
            """, (fsize, mtime, meta["duration"], meta["width"], meta["height"], meta["fps"],
                  status, source, coach, vdate, now, row["id"]))
            updated_count += 1
            vid = row["id"]
            if show_progress:
                print(" ↑ 已更新")
        else:
            conn.execute("""
                INSERT INTO videos
                    (filename, filepath, filesize, created_at, modified_at,
                     duration, width, height, fps, status, source_dir,
                     coach_name, video_date, last_indexed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (fname, str(fpath.resolve()), fsize, ctime, mtime,
                  meta["duration"], meta["width"], meta["height"], meta["fps"],
                  status, source, coach, vdate, now))
            vid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            new_count += 1
            if show_progress:
                print(" + 新索引")

        # 如果是输出文件，尝试关联 processing_records
        if source == "output":
            _try_link_processing(conn, vid, fpath)

    conn.commit()
    conn.close()
    print(f"\n  扫描完成: 共 {total} 个文件, 新增 {new_count}, 更新 {updated_count}")


def _try_link_processing(conn: sqlite3.Connection, video_id: int, output_path: Path):
    """尝试从文件名推断对应的源视频和 preset"""
    fname = output_path.stem
    preset = ""

    # 尝试匹配已知 preset 后缀模式
    for p in ["sexy", "natural", "dramatic", "clean", "night_gym", "gimbal",
              "beauty", "youtube", "shorts", "night_square_dance"]:
        if f"_{p}" in fname or f"-{p}" in fname:
            preset = p
            break

    # 检查是否已有处理记录
    existing = conn.execute(
        "SELECT id FROM processing_records WHERE output_path = ?",
        (str(output_path.resolve()),)
    ).fetchone()
    if existing:
        return

    conn.execute("""
        INSERT INTO processing_records
            (video_id, output_path, output_filename, preset_used, stages_completed)
        VALUES (?, ?, ?, ?, ?)
    """, (video_id, str(output_path.resolve()), output_path.name, preset, "[]"))


# ── 列表查询 ───────────────────────────────────────────────────

def list_videos(status: str = "", coach: str = "", source: str = "",
                tag: str = "", sort: str = "modified_at", limit: int = 0):
    """列出视频"""
    conn = _get_db()
    conditions = []
    params = []

    if status:
        conditions.append("v.status = ?")
        params.append(status)
    if coach:
        conditions.append("v.coach_name LIKE ?")
        params.append(f"%{coach}%")
    if source:
        conditions.append("v.source_dir = ?")
        params.append(source)
    if tag:
        conditions.append("v.id IN (SELECT video_id FROM tags WHERE tag = ?)")
        params.append(tag)

    where = " AND ".join(conditions) if conditions else "1=1"
    order_col = "v.modified_at" if sort == "modified_at" else "v.filename"
    query = f"""
        SELECT v.*, (
            SELECT GROUP_CONCAT(tag, ', ') FROM tags WHERE video_id = v.id
        ) AS tags
        FROM videos v
        WHERE {where}
        ORDER BY {order_col} DESC
    """
    if limit > 0:
        query += f" LIMIT {limit}"

    rows = conn.execute(query, params).fetchall()
    conn.close()

    if not rows:
        print("  (无匹配记录)")
        return

    # 表头
    print(f"{'文件名':<28} {'状态':<10} {'来源':<7} {'教练':<6} {'时长':<7} {'标签':<20}")
    print("-" * 90)
    for r in rows:
        dur = f"{r['duration']:.0f}s" if r["duration"] else "-"
        tags = r["tags"] or ""
        coach_display = r["coach_name"] or "-"
        print(f"{r['filename']:<28} {r['status']:<10} {r['source_dir']:<7} "
              f"{coach_display:<6} {dur:<7} {tags:<20}")


# ── 单视频详情 ────────────────────────────────────────────────

def show_status(filename: str):
    """显示单个视频详情"""
    conn = _get_db()
    rows = conn.execute(
        "SELECT * FROM videos WHERE filename LIKE ?", (f"%{filename}%",)
    ).fetchall()
    if not rows:
        # 也按 filepath 搜
        rows = conn.execute(
            "SELECT * FROM videos WHERE filepath LIKE ?", (f"%{filename}%",)
        ).fetchall()
    if not rows:
        print(f"  未找到匹配: {filename}")
        conn.close()
        return

    for r in rows:
        tags = conn.execute(
            "SELECT tag FROM tags WHERE video_id=?", (r["id"],)
        ).fetchall()
        records = conn.execute(
            "SELECT * FROM processing_records WHERE video_id=? ORDER BY processed_at DESC",
            (r["id"],)
        ).fetchall()

        print(f"\n{'='*60}")
        print(f"  文件名:     {r['filename']}")
        print(f"  路径:       {r['filepath']}")
        print(f"  状态:       {r['status']}")
        print(f"  来源:       {r['source_dir']}")
        print(f"  教练:       {r['coach_name'] or '-'}")
        print(f"  录制日期:   {r['video_date'] or '-'}")
        print(f"  大小:       {_fmt_size(r['filesize'])}")
        print(f"  时长:       {r['duration']:.1f}s" if r['duration'] else "  时长:       -")
        if r["width"] and r["height"]:
            print(f"  分辨率:     {r['width']}x{r['height']}")
        if r["fps"]:
            print(f"  FPS:        {r['fps']:.2f}")
        if r["notes"]:
            print(f"  备注:       {r['notes']}")
        if tags:
            print(f"  标签:       {', '.join(t['tag'] for t in tags)}")
        print(f"  索引时间:   {r['last_indexed']}")

        if records:
            print(f"\n  ── 处理记录 ──")
            for pr in records:
                print(f"    输出: {pr['output_filename'] or pr['output_path']}")
                if pr["preset_used"]:
                    print(f"    preset: {pr['preset_used']}")
                print(f"    时间: {pr['processed_at']}")
                print()
        else:
            print(f"\n  (无处理记录)")
    conn.close()


# ── 统计 ───────────────────────────────────────────────────────

def show_stats():
    """显示统计信息"""
    conn = _get_db()

    total = conn.execute("SELECT COUNT(*) FROM videos").fetchone()[0]
    source_count = conn.execute("SELECT COUNT(*) FROM videos WHERE source_dir='input'").fetchone()[0]
    output_count = conn.execute("SELECT COUNT(*) FROM videos WHERE source_dir='output'").fetchone()[0]
    raw_count = conn.execute("SELECT COUNT(*) FROM videos WHERE status='source'").fetchone()[0]
    processed_count = conn.execute("SELECT COUNT(*) FROM videos WHERE status='processed'").fetchone()[0]

    total_size = conn.execute("SELECT COALESCE(SUM(filesize),0) FROM videos").fetchone()[0]
    total_dur = conn.execute("SELECT COALESCE(SUM(duration),0) FROM videos").fetchone()[0]

    # 按教练分布
    coach_dist = conn.execute(
        "SELECT coach_name, COUNT(*) as cnt FROM videos WHERE coach_name != '' GROUP BY coach_name ORDER BY cnt DESC"
    ).fetchall()

    # 按状态分布
    status_dist = conn.execute(
        "SELECT status, COUNT(*) as cnt FROM videos GROUP BY status ORDER BY cnt DESC"
    ).fetchall()

    # 按日期分布
    date_dist = conn.execute(
        "SELECT video_date, COUNT(*) as cnt FROM videos WHERE video_date IS NOT NULL GROUP BY video_date ORDER BY video_date DESC LIMIT 10"
    ).fetchall()

    # 标签统计
    tag_count = conn.execute("SELECT COUNT(*) FROM tags").fetchone()[0]
    top_tags = conn.execute(
        "SELECT tag, COUNT(*) as cnt FROM tags GROUP BY tag ORDER BY cnt DESC LIMIT 10"
    ).fetchall()

    conn.close()

    print(f"\n  {'='*50}")
    print(f"  健身视频库统计")
    print(f"  {'='*50}")
    print(f"  总视频数:   {total}")
    print(f"  ├─ 素材:     {source_count}")
    print(f"  ├─ 输出:     {output_count}")
    print(f"  ├─ 未处理:   {raw_count}")
    print(f"  └─ 已处理:   {processed_count}")
    print(f"  总大小:     {_fmt_size(total_size)}")
    print(f"  总时长:     {total_dur:.0f}s ({total_dur/60:.1f}min)")
    print()

    if coach_dist:
        print(f"  ── 教练分布 ──")
        for r in coach_dist:
            print(f"    {r['coach_name']}: {r['cnt']} 个视频")
        print()

    if status_dist:
        print(f"  ── 状态分布 ──")
        for r in status_dist:
            print(f"    {r['status']}: {r['cnt']}")
        print()

    if date_dist:
        print(f"  ── 最近录制日期 ──")
        for r in date_dist:
            print(f"    {r['video_date']}: {r['cnt']} 个")
        print()

    if top_tags:
        print(f"  ── 热门标签（共 {tag_count} 个）──")
        for r in top_tags:
            print(f"    #{r['tag']}: {r['cnt']}")
        print()


# ── 搜索 ───────────────────────────────────────────────────────

def search_videos(query: str):
    """按关键词搜索"""
    conn = _get_db()
    like = f"%{query}%"
    rows = conn.execute("""
        SELECT v.*, (
            SELECT GROUP_CONCAT(tag, ', ') FROM tags WHERE video_id = v.id
        ) AS tags
        FROM videos v
        WHERE v.filename LIKE ?
           OR v.coach_name LIKE ?
           OR v.notes LIKE ?
           OR v.id IN (SELECT video_id FROM tags WHERE tag LIKE ?)
        ORDER BY v.modified_at DESC
    """, (like, like, like, like)).fetchall()
    conn.close()

    if not rows:
        print(f"  未找到匹配 \"{query}\"")
        return

    print(f"\n  找到 {len(rows)} 个匹配 \"{query}\":\n")
    for r in rows:
        dur = f"{r['duration']:.0f}s" if r["duration"] else "-"
        tags = f" [{r['tags']}]" if r["tags"] else ""
        print(f"    {r['filename']:<30} {r['status']:<10} {dur:<8}{tags}")


# ── 标签管理 ───────────────────────────────────────────────────

def add_tag(filename: str, tag: str):
    """给视频添加标签"""
    conn = _get_db()
    vid = _resolve_video_id(conn, filename)
    if not vid:
        print(f"  未找到: {filename}")
        conn.close()
        return
    try:
        conn.execute("INSERT OR IGNORE INTO tags (video_id, tag) VALUES (?, ?)", (vid, tag))
        conn.commit()
        print(f"  已添加标签 #{tag}")
    except Exception as e:
        print(f"  添加标签失败: {e}")
    conn.close()


def remove_tag(filename: str, tag: str):
    """删除视频标签"""
    conn = _get_db()
    vid = _resolve_video_id(conn, filename)
    if not vid:
        print(f"  未找到: {filename}")
        conn.close()
        return
    conn.execute("DELETE FROM tags WHERE video_id=? AND tag=?", (vid, tag))
    conn.commit()
    affected = conn.total_changes
    print(f"  已删除标签 #{tag}" if affected else f"  未找到标签 #{tag}")
    conn.close()


# ── 备注管理 ───────────────────────────────────────────────────

def set_note(filename: str, notes: str):
    """设置视频备注"""
    conn = _get_db()
    vid = _resolve_video_id(conn, filename)
    if not vid:
        print(f"  未找到: {filename}")
        conn.close()
        return
    conn.execute("UPDATE videos SET notes=? WHERE id=?", (notes, vid))
    conn.commit()
    print(f"  备注已更新")
    conn.close()


# ── 辅助函数 ───────────────────────────────────────────────────

def _resolve_video_id(conn: sqlite3.Connection, filename: str) -> Optional[int]:
    """根据文件名或路径查找视频 ID"""
    row = conn.execute(
        "SELECT id FROM videos WHERE filename LIKE ?", (f"%{filename}%",)
    ).fetchone()
    if row:
        return row["id"]
    row = conn.execute(
        "SELECT id FROM videos WHERE filepath LIKE ?", (f"%{filename}%",)
    ).fetchone()
    return row["id"] if row else None


def _fmt_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


# ── CLI ────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        description="健身视频管理工具 — 索引/查询/统计/管理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="command")

    # scan
    sp = sub.add_parser("scan", help="扫描并更新索引")
    sp.add_argument("-i", "--input", default=str(DEFAULT_INPUT_DIR), help="素材目录")
    sp.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT_BASE), help="输出目录")
    sp.add_argument("-q", "--quiet", action="store_true", help="静默模式")

    # list
    sp = sub.add_parser("list", help="列出视频")
    sp.add_argument("--status", choices=["source", "processed", "partial"], help="按状态筛选")
    sp.add_argument("--coach", type=str, help="按教练筛选")
    sp.add_argument("--source", choices=["input", "output"], help="按来源筛选")
    sp.add_argument("--tag", type=str, help="按标签筛选")
    sp.add_argument("--sort", choices=["filename", "modified_at"], default="modified_at")
    sp.add_argument("--limit", type=int, default=0, help="限制条数")

    # status
    sp = sub.add_parser("status", help="查看视频详情")
    sp.add_argument("filename", type=str, help="文件名或路径关键词")

    # stats
    sub.add_parser("stats", help="显示统计信息")

    # search
    sp = sub.add_parser("search", help="搜索视频")
    sp.add_argument("query", type=str, help="关键词")

    # tag
    sp = sub.add_parser("tag", help="管理标签")
    sp.add_argument("filename", type=str, help="文件名或路径关键词")
    sp.add_argument("tag", type=str, help="标签名")
    sp.add_argument("--remove", action="store_true", help="删除标签")

    # note
    sp = sub.add_parser("note", help="添加备注")
    sp.add_argument("filename", type=str, help="文件名或路径关键词")
    sp.add_argument("text", type=str, nargs="+", help="备注文字")
    sp.add_argument("--append", action="store_true", help="追加到现有备注")

    # reindex
    sub.add_parser("reindex", help="重建索引（删除旧库重新扫描）")

    return p


def main():
    _init_db()
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "scan":
        input_dir = Path(args.input)
        output_dir = Path(args.output)
        scan_directory(input_dir, output_dir, show_progress=not args.quiet)

    elif args.command == "list":
        list_videos(
            status=args.status or "",
            coach=args.coach or "",
            source=args.source or "",
            tag=args.tag or "",
            sort=args.sort,
            limit=args.limit,
        )

    elif args.command == "status":
        show_status(args.filename)

    elif args.command == "stats":
        show_stats()

    elif args.command == "search":
        search_videos(args.query)

    elif args.command == "tag":
        if args.remove:
            remove_tag(args.filename, args.tag)
        else:
            add_tag(args.filename, args.tag)

    elif args.command == "note":
        conn = _get_db()
        vid = _resolve_video_id(conn, args.filename)
        if not vid:
            print(f"  未找到: {args.filename}")
            conn.close()
            return
        text = " ".join(args.text)
        if args.append:
            existing = conn.execute("SELECT notes FROM videos WHERE id=?", (vid,)).fetchone()
            text = (existing["notes"] + "\n" + text) if existing["notes"] else text
        conn.execute("UPDATE videos SET notes=? WHERE id=?", (text, vid))
        conn.commit()
        print(f"  备注已{'追加' if args.append else '设置'}")
        conn.close()

    elif args.command == "reindex":
        if INDEX_DB.exists():
            INDEX_DB.unlink()
            print("  旧索引已删除")
        _init_db()
        scan_directory(
            Path(DEFAULT_INPUT_DIR),
            Path(DEFAULT_OUTPUT_BASE),
            show_progress=True,
        )


if __name__ == "__main__":
    main()
