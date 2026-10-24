"""Dataset QC: stats + visual spot-check grid (docs/02 section 4).

Archived via src/runctx.py (task=check). Outputs to the run dir:
stats.json + samples/*.jpg (12 random train images with boxes/keypoints).
"""
import glob
import json
import os
import random

import cv2
import numpy as np
import yaml

sys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
import sys  # noqa: E402

sys.path.insert(0, sys_path)
from src.runctx import RunContext  # noqa: E402

ROOT = os.path.dirname(sys_path)

# COCO 17-kpt skeleton pairs for drawing (ignored for other kpt counts)
SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9),
            (6, 8), (8, 10), (5, 11), (6, 12), (11, 13), (13, 15), (12, 14),
            (14, 16), (5, 6), (11, 12)]


def load_label(path):
    objs = []
    if not os.path.exists(path):
        return objs
    with open(path) as f:
        for line in f:
            v = line.split()
            if len(v) < 5:
                continue
            vals = [float(x) for x in v]
            objs.append(vals)
    return objs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-yaml", default=None)
    args = ap.parse_args()
    with open(os.path.join(ROOT, "config", "dataset.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    data_yaml = args.data_yaml or os.path.join(ROOT, cfg["converted_dir"], "sop_pose.yaml")
    with open(data_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    root = data["path"]
    ctx = RunContext("check", params={"data_yaml": os.path.relpath(data_yaml, ROOT)})
    try:
        stats = {}
        for split in ("train", "val"):
            imgs = sorted(glob.glob(os.path.join(root, "images", split, "*")))
            n_person, kpt_vis, per_img = 0, [], []
            for ip in imgs:
                lp = os.path.join(root, "labels", split, os.path.splitext(os.path.basename(ip))[0] + ".txt")
                objs = load_label(lp)
                per_img.append(len(objs))
                nk = (len(objs[0]) - 5) // 3 if objs else 0
                for o in objs:
                    n_person += 1
                    k = o[5:]
                    kpt_vis.append(sum(1 for j in range(nk) if k[3 * j + 2] > 0))
            stats[split] = {
                "images": len(imgs),
                "persons": n_person,
                "avg_person_per_img": round(sum(per_img) / len(per_img), 3) if per_img else 0,
                "avg_visible_kpts": round(sum(kpt_vis) / len(kpt_vis), 2) if kpt_vis else 0,
                "empty_label_images": sum(1 for c in per_img if c == 0),
            }
            print(f"[{split}] {stats[split]}")
        with open(os.path.join(ctx.path, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        ctx.note(f"质检统计：train {stats['train']['images']} 张 / val {stats['val']['images']} 张")

        os.makedirs(os.path.join(ctx.path, "samples"), exist_ok=True)
        train_imgs = sorted(glob.glob(os.path.join(root, "images", "train", "*")))
        rng = random.Random(0)
        for ip in rng.sample(train_imgs, min(12, len(train_imgs))):
            img = cv2.imread(ip)
            if img is None:
                continue
            h, w = img.shape[:2]
            lp = os.path.join(root, "labels", "train", os.path.splitext(os.path.basename(ip))[0] + ".txt")
            for o in load_label(lp):
                cx, cy, bw, bh = o[1] * w, o[2] * h, o[3] * w, o[4] * h
                p1 = (int(cx - bw / 2), int(cy - bh / 2))
                cv2.rectangle(img, p1, (int(cx + bw / 2), int(cy + bh / 2)), (0, 255, 0), 2)
                nk = (len(o) - 5) // 3
                for j in range(nk):
                    x, y, v = o[5 + 3 * j] * w, o[6 + 3 * j] * h, o[8 + 3 * j]
                    if v > 0:
                        cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
                if nk == 17:
                    for a, b in SKELETON:
                        ka = o[5 + 3 * a:8 + 3 * a]
                        kb = o[5 + 3 * b:8 + 3 * b]
                        if ka[2] > 0 and kb[2] > 0:
                            cv2.line(img, (int(ka[0] * w), int(ka[1] * h)),
                                     (int(kb[0] * w), int(kb[1] * h)), (255, 128, 0), 1)
            out = os.path.join(ctx.path, "samples", os.path.basename(ip))
            cv2.imwrite(out, img)
        ctx.note(f"可视化抽查 12 张已写入 samples/（绿框=人体，红点=可见关键点）")
    except Exception as e:
        import traceback
        traceback.print_exc()
        ctx.note("异常: " + repr(e))
        ctx.finish(ok=False)
        sys.exit(1)
    ctx.finish(ok=True)


if __name__ == "__main__":
    main()
