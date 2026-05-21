"""
Coach Avatar Library Manager
Pick the best portrait per coach and organize for easy access
"""

import os, sys, glob, shutil

PORTRAITS_DIR = os.path.join(os.path.dirname(__file__), "coach_portraits")
AVATARS_DIR = os.path.join(os.path.dirname(__file__), "coach_avatars")

COACH_MAP = {
    "艳青": "胭脂虎",
    "丽丽": "腰女",
    "小红豆": "红娘子",
    "建玲": "三宝妈",
    "枫林红": "枫林红",
    "郭海军": "老教练",
}


def organize():
    """Group portraits by coach name, pick best per coach"""
    os.makedirs(AVATARS_DIR, exist_ok=True)

    # Group files by coach
    all_files = glob.glob(os.path.join(PORTRAITS_DIR, "*.jpg"))
    by_coach = {}

    for f in all_files:
        name = os.path.basename(f)
        # Try to match coach name from filename
        for key in COACH_MAP:
            if name.startswith(key):
                if key not in by_coach:
                    by_coach[key] = []
                by_coach[key].append(f)
                break

    print(f"Found {len(by_coach)} coaches with portraits")

    for coach, files in sorted(by_coach.items()):
        nickname = COACH_MAP[coach]
        print(f"\n{coach} ({nickname}): {len(files)} portraits")

        # Pick best by file size (largest = most detail)
        best = max(files, key=os.path.getsize)
        best_size = os.path.getsize(best) / 1024

        # Copy best to avatars dir with clean name
        out = os.path.join(AVATARS_DIR, f"{coach}_{nickname}.jpg")
        shutil.copy2(best, out)
        print(f"  Best: {os.path.basename(best)} ({best_size:.0f}KB) -> {out}")

    print(f"\nAvatar library: {AVATARS_DIR}")
    for f in sorted(glob.glob(os.path.join(AVATARS_DIR, "*.jpg"))):
        print(f"  {os.path.basename(f)}")


if __name__ == "__main__":
    organize()
