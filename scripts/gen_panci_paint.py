"""小红豆 4 句判词水墨插画生成 (走 ComfyUI HTTP API).

需要先启动 ComfyUI server: cd F:/wkspace/ComfyUI && python main.py --listen 0.0.0.0 --port 8188

输出: tools/panci_paint/xiaohongdou/1.png ... 4.png
"""
import json
import time
import urllib.request
import shutil
from pathlib import Path

COMFY_URL = "http://localhost:8188"
WORKFLOW = Path(r"F:\wkspace\ComfyUI\custom_nodes\workflows\text_to_image.json")
COMFY_OUTPUT = Path(r"F:\wkspace\ComfyUI\output")
OUT_DIR = Path(r"F:\wkspace\fitness-video-pipeline\tools\panci_paint\xiaohongdou")

WIDTH = 512
HEIGHT = 896
STEPS = 25
SEED_BASE = 42424242

PANCI = [
    {
        "id": 1,
        "text": "红豆生来俏模样",
        "prompt": (
            "a beautiful young Chinese woman in elegant red traditional hanfu dress, "
            "delicate smiling face, soft natural makeup, small and charming, "
            "standing among red bean flowers, mountain mist background, "
            "Chinese ink painting, sumi-e style, minimalist brush strokes, "
            "white space, vertical composition, masterpiece, high detail, "
            "muted color palette with red accent"
        ),
    },
    {
        "id": 2,
        "text": "香汗淋漓透红妆",
        "prompt": (
            "a young Chinese woman in modest red hanfu dress, "
            "gentle flowing motion, long sleeves drifting, "
            "soft mist and falling petals around, "
            "Chinese ink painting, sumi-e style, expressive brush strokes, "
            "red color emphasis, vertical composition, masterpiece, "
            "high detail, elegant subtle movement"
        ),
    },
    {
        "id": 3,
        "text": "娇喘微微惹人怜",
        "prompt": (
            "a delicate young Chinese woman in red hanfu, "
            "resting with hand on cheek, gentle breath, slightly flushed cheeks, "
            "peony flowers in background, "
            "Chinese gongbi painting style, meticulous fine brushwork, "
            "intimate close-up portrait, vertical composition, masterpiece, "
            "soft warm lighting, tender expression"
        ),
    },
    {
        "id": 4,
        "text": "花枝乱颤舞霓裳",
        "prompt": (
            "a young Chinese woman in flowing red hanfu with long trailing sleeves, "
            "twirling gracefully with sleeves swirling like flower petals, "
            "petals flying in wind, "
            "Chinese ink painting splashed-ink style, xieyi freehand, "
            "vivid red and black ink contrast, vertical composition, masterpiece, "
            "high detail, dramatic motion, full body graceful pose"
        ),
    },
]

NEGATIVE = (
    "blurry, low quality, ugly, deformed, bad anatomy, bad hands, extra fingers, "
    "watermark, text, signature, logo, modern clothing, jeans, sneakers, "
    "cartoon, 3d render, photo realistic, western face, "
    "frame, border, multiple views, "
    # 防性暗示/暴露 (per CLAUDE.md 弹幕/字幕内容审查基线)
    "nudity, lingerie, bikini, cleavage, bare shoulders, exposed chest, "
    "wedding dress, white dress, slit, high slit, thigh gap, navel, "
    "sexy, sensual, suggestive, revealing outfit, see-through, "
    # 重点防住 DreamShaper 8 偏好
    "short skirt, mini skirt, bare legs, bare thighs, high boots, "
    "knee high slit, side slit, leg visible, midriff, "
    "action pose, dynamic pose, dancing wildly"
)


def submit_workflow(prompt: str, neg: str, seed: int, prefix: str):
    wf = json.loads(WORKFLOW.read_text(encoding="utf-8"))
    wf["2"]["inputs"]["text"] = prompt
    wf["3"]["inputs"]["text"] = neg
    wf["4"]["inputs"]["width"] = WIDTH
    wf["4"]["inputs"]["height"] = HEIGHT
    wf["5"]["inputs"]["seed"] = seed
    wf["5"]["inputs"]["steps"] = STEPS
    wf["7"]["inputs"]["filename_prefix"] = prefix

    req = urllib.request.Request(
        f"{COMFY_URL}/prompt",
        data=json.dumps({"prompt": wf}).encode(),
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req)
    data = json.loads(resp.read())
    if data.get("node_errors"):
        raise RuntimeError(f"node_errors: {data['node_errors']}")
    return data["prompt_id"]


def wait_for_result(pid: str, timeout: int = 180) -> dict:
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(f"{COMFY_URL}/history/{pid}")
            data = json.loads(resp.read())
        except Exception:
            time.sleep(2)
            continue
        if pid in data:
            status = data[pid]["status"]["status_str"]
            if status == "success":
                return data[pid]
            elif status == "error":
                raise RuntimeError(f"ComfyUI error: {data[pid]}")
        time.sleep(2)
    raise TimeoutError(f"ComfyUI timeout ({timeout}s) for {pid}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    COMFY_OUTPUT.mkdir(exist_ok=True)

    print(f"[gen] ComfyUI: {COMFY_URL}, model=DreamShaper_8 ({WIDTH}x{HEIGHT}, {STEPS}steps)")

    for p in PANCI:
        seed = SEED_BASE + p["id"]
        prefix = f"panci_xhd_{p['id']}"
        print(f"[gen] {p['id']}/4 {p['text']} (seed={seed})")

        pid = submit_workflow(p["prompt"], NEGATIVE, seed, prefix)
        result = wait_for_result(pid)
        elapsed = result["status"].get("completed", True)
        outputs = result.get("outputs", {})
        # find image
        img_path = None
        for nid, out in outputs.items():
            images = out.get("images", [])
            if images:
                img = images[0]
                folder = img.get("subfolder", "")
                fname = img.get("filename", "")
                if folder:
                    img_path = COMFY_OUTPUT / folder / fname
                else:
                    img_path = COMFY_OUTPUT / fname
                break

        if not img_path or not img_path.exists():
            print(f"[gen] FAIL: no image output. result={result}")
            continue

        dst = OUT_DIR / f"{p['id']}.png"
        shutil.copy2(img_path, dst)
        print(f"[gen] saved {dst} ({dst.stat().st_size//1024}KB)")

    print("[gen] ALL DONE")


if __name__ == "__main__":
    main()