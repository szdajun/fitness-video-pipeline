"""通过本地 ComfyUI 生成「胭脂虎」汉印主图 (美女头+虎身, 不含文字)

构图: 1024x1024 方形, 上 60% 主体人物, 下 40% 留白给 PIL 后合成繁体字 + 红边.

依赖: 本机 ComfyUI 监听 http://127.0.0.1:8188, 已装 DreamShaper_8_pruned.safetensors.
出图后 PIL 在 compose_seal.py 加红边 + 繁体"胭脂虎"三字 + 轻微斑驳.

用法:
    python tools/gen_seal.py                  # 跑一次出 4 张候选
    python tools/gen_seal.py --seed 42       # 固定种子
    python tools/gen_seal.py --out tools/seal_main.png
"""
import argparse
import json
import os
import random
import sys
import time
import urllib.request

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8188")
COMFY_ROOT = r"F:\wkspace\ComfyUI"
OUTPUT_DIR = os.path.join(COMFY_ROOT, "output")

# workflow 模板 (与 custom_nodes/workflows/text_to_image.json 同构, 但 ckpt 用 SD1.5)
WORKFLOW = {
    "1": {"class_type": "CheckpointLoaderSimple",
          "inputs": {"ckpt_name": "DreamShaper_8_pruned.safetensors"}},
    "2": {"class_type": "CLIPTextEncode",
          "inputs": {"text": "", "clip": ["1", 1]}},
    "3": {"class_type": "CLIPTextEncode",
          "inputs": {"text": "", "clip": ["1", 1]}},
    "4": {"class_type": "EmptyLatentImage",
          "inputs": {"width": 1024, "height": 1024, "batch_size": 1}},
    "5": {"class_type": "KSampler",
          "inputs": {
              "model": ["1", 0], "seed": 0, "steps": 28, "cfg": 7.0,
              "sampler_name": "euler_ancestral", "scheduler": "karras",
              "positive": ["2", 0], "negative": ["3", 0],
              "latent_image": ["4", 0], "denoise": 1.0,
          }},
    "6": {"class_type": "VAEDecode",
          "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
    "7": {"class_type": "SaveImage",
          "inputs": {"filename_prefix": "seal_yanhu", "images": ["6", 0]}},
}

# 主体 prompt: 美女头(占上 60%) + 老虎身(占下 40% 不画, 因下 40% 留给字)
# 方形构图靠 prompt 强约束, 不用 ControlNet (本地没装)
PROMPT = (
    "a mythical Chinese chimera creature, "
    "upper body: beautiful Asian woman face, delicate features, "
    "ornate phoenix headdress, red lips, almond eyes, "
    "soft traditional Chinese painting style, "
    "framed by a square red border, "
    "centered composition, symmetrical, "
    "ink wash painting meets digital art, "
    "highly detailed, masterpiece, best quality"
)

NEGATIVE = (
    "blurry, low quality, deformed, bad anatomy, "
    "extra limbs, mutated hands, ugly, "
    "watermark, text, signature, "
    "western cartoon, anime, 3d render, photorealistic"
)


def submit(prompt_text: str, negative: str, seed: int,
           prefix: str, width: int = 1024, height: int = 1024,
           steps: int = 28, cfg: float = 7.0) -> str | None:
    wf = json.loads(json.dumps(WORKFLOW))  # deep copy
    wf["2"]["inputs"]["text"] = prompt_text
    wf["3"]["inputs"]["text"] = negative
    wf["4"]["inputs"]["width"] = width
    wf["4"]["inputs"]["height"] = height
    wf["5"]["inputs"]["seed"] = seed
    wf["5"]["inputs"]["steps"] = steps
    wf["5"]["inputs"]["cfg"] = cfg
    wf["7"]["inputs"]["filename_prefix"] = prefix

    body = json.dumps({"prompt": wf}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=body, headers={"Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        result = json.loads(resp.read())
    except Exception as e:
        print(f"[gen_seal] 提交失败: {e}", file=sys.stderr)
        return None
    if result.get("node_errors"):
        print(f"[gen_seal] 节点错误: {result['node_errors']}", file=sys.stderr)
        return None
    return result.get("prompt_id", "")


def wait_done(prompt_id: str, timeout_s: int = 240) -> list[dict]:
    """轮询 /history/{id}, 返回生成的 image 列表 [{filename, subfolder}, ...]"""
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            resp = urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
            data = json.loads(resp.read())
        except Exception:
            time.sleep(2)
            continue
        if prompt_id in data:
            status = data[prompt_id].get("status", {}).get("status_str")
            if status == "success":
                outs = data[prompt_id].get("outputs", {})
                imgs = []
                for nid, out in outs.items():
                    for img in out.get("images", []):
                        imgs.append(img)
                return imgs
            if status == "error":
                err = data[prompt_id].get("status", {}).get("messages", [])
                print(f"[gen_seal] 任务失败: {err}", file=sys.stderr)
                return []
        time.sleep(2)
    print(f"[gen_seal] 超时 ({timeout_s}s)", file=sys.stderr)
    return []


def fetch_image(img_meta: dict, dst: str) -> bool:
    """从 ComfyUI /view 拿图, 保存到 dst"""
    params = f"filename={img_meta['filename']}&type=output"
    if img_meta.get("subfolder"):
        params += f"&subfolder={img_meta['subfolder']}"
    url = f"{COMFYUI_URL}/view?{params}"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            with open(dst, "wb") as f:
                f.write(r.read())
        return True
    except Exception as e:
        print(f"[gen_seal] 拉图失败 {url}: {e}", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None, help="固定种子; 不传=随机")
    ap.add_argument("--n", type=int, default=4, help="生成候选数 (>=1)")
    ap.add_argument("--steps", type=int, default=28)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "_seal_candidates"))
    ap.add_argument("--prefix", default="seal_yanhu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 探活
    try:
        urllib.request.urlopen(COMFYUI_URL, timeout=5)
    except Exception as e:
        print(f"[gen_seal] ComfyUI 未响应 ({COMFYUI_URL}): {e}", file=sys.stderr)
        print("  请先启动: cd F:\\wkspace\\ComfyUI && python main.py", file=sys.stderr)
        sys.exit(2)

    saved = []
    for i in range(args.n):
        seed = args.seed if (args.seed is not None and args.n == 1) \
            else (args.seed + i if args.seed is not None else random.randint(0, 2**31))
        print(f"[gen_seal] #{i+1}/{args.n}  seed={seed}  steps={args.steps}")
        pid = submit(PROMPT, NEGATIVE, seed, args.prefix, 1024, 1024, args.steps, args.cfg)
        if not pid:
            continue
        imgs = wait_done(pid)
        for j, meta in enumerate(imgs):
            dst = os.path.join(args.out_dir, f"{args.prefix}_{i:02d}_seed{seed}.png")
            if fetch_image(meta, dst):
                print(f"[gen_seal]   -> {dst}")
                saved.append(dst)

    print(f"[gen_seal] 完成, 共 {len(saved)} 张候选")
    if not saved:
        sys.exit(1)


if __name__ == "__main__":
    main()
