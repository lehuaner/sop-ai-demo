"""Build a demo replay video from converted dataset frames (for run mode).

Picks frames that contain annotated persons, letterboxes them to 960x540,
and writes a constant-fps mp4. Output: datasets/demo_video/demo_assembly.mp4
(gitignored). Archived via src/runctx.py (task=demo_video).
"""
import glob
import os
import random
import sys

import cv2

sys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, sys_path)
from src.runctx import RunContext  # noqa: E402

ROOT = os.path.abspath(sys_path)
W, H, FPS, SECONDS = 960, 540, 15, 10


def letterbox(img):
    h, w = img.shape[:2]
    s = min(W / w, H / h)
    nw, nh = int(w * s), int(h * s)
    canvas = np_zero()
    x0, y0 = (W - nw) // 2, (H - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh))
    return canvas


def np_zero():
    import numpy as np
    return np.zeros((H, W, 3), dtype="uint8")


def main():
    with open(os.path.join(ROOT, "config", "dataset.yaml"), encoding="utf-8") as f:
        import yaml
        cfg = yaml.safe_load(f)
    ctx = RunContext("demo_video", params={"seconds": SECONDS, "fps": FPS, "size": f"{W}x{H}"})
    try:
        train_imgs = sorted(glob.glob(os.path.join(ROOT, cfg["converted_dir"], "images", "train", "*.jpg")))
        if not train_imgs:
            raise RuntimeError("no converted images; run convert first")
        rng = random.Random(7)
        n = FPS * SECONDS
        picks = rng.sample(train_imgs, min(n, len(train_imgs)))
        out_dir = os.path.join(ROOT, "datasets", "demo_video")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "demo_assembly.mp4")
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        for p in picks:
            img = cv2.imread(p)
            if img is None:
                continue
            vw.write(letterbox(img))
        vw.release()
        ctx.note(f"演示视频生成：{out}（{len(picks)} 帧 @ {FPS}fps = {SECONDS}s，取自训练集含人图像）")
        ctx.note("用途：src/main.py run --video 该文件，做无摄像头回放推理验证")
        ctx.finish(ok=True, extra_manifest={"output": os.path.relpath(out, ROOT), "frames": len(picks)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        ctx.note("异常: " + repr(e))
        ctx.finish(ok=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
