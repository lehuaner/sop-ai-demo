"""Convert annotations to YOLO-pose format (docs/02 section 3).

coco_val2017 : COCO person keypoints (17) -> YOLO-pose txt, 80/20 split.
assembly101  : per-take poses@60fps JSON -> per-view 2D hand keypoints
               (42 = right 21 + left 21). NOTE: full conversion lands in
               M2 after verifying the JSON structure with granted access.

Archived via src/runctx.py (task=convert).
"""
import argparse
import glob
import json
import os
import random
import shutil

import yaml

sys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
import sys  # noqa: E402

sys.path.insert(0, sys_path)
from src.runctx import RunContext  # noqa: E402

ROOT = os.path.dirname(sys_path)


def coco_to_yolo_pose(cfg, ctx):
    s = cfg["sources"]["coco_val2017"]
    nk = s["num_keypoints"]
    min_k = s.get("min_kpts_per_person", 6)
    bg_ratio = s.get("background_ratio", 0.10)
    out = os.path.join(ROOT, cfg["converted_dir"])
    for sub in ("images/train", "images/val", "labels/train", "labels/val"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    with open(os.path.join(ROOT, s["keypoint_json"]), encoding="utf-8") as f:
        coco = json.load(f)
    imgs = {im["id"]: im for im in coco["images"]}
    per_img = {}
    for a in coco["annotations"]:
        if a.get("category_id") != 1 or a.get("num_keypoints", 0) < min_k:
            continue
        per_img.setdefault(a["image_id"], []).append(a)

    ids_with = sorted(per_img)
    empty_ids = [i for i in imgs if i not in per_img]
    rng = random.Random(cfg.get("seed", 42))
    keep_empty = rng.sample(empty_ids, int(len(empty_ids) * bg_ratio))
    all_ids = ids_with + keep_empty
    rng.shuffle(all_ids)
    n_train = int(len(all_ids) * cfg.get("train_ratio", 0.8))
    splits = {"train": all_ids[:n_train], "val": all_ids[n_train:]}

    stats = {"persons": 0, "images": 0}
    for split, ids in splits.items():
        for iid in ids:
            im = imgs[iid]
            w, h = im["width"], im["height"]
            anns = per_img.get(iid, [])
            lines = []
            for a in anns:
                x, y, bw, bh = a["bbox"]
                cx, cy = (x + bw / 2) / w, (y + bh / 2) / h
                nw, nh = bw / w, bh / h
                kpts = []
                kp = a["keypoints"]
                for i in range(nk):
                    px, py, pv = kp[3 * i], kp[3 * i + 1], kp[3 * i + 2]
                    vis = 0 if pv == 0 else (1 if pv == 1 else 2)
                    kpts += [px / w, py / h, vis]
                lines.append("0 " + " ".join(f"{v:.6f}" for v in (cx, cy, nw, nh, *kpts)))
                stats["persons"] += 1
            base = os.path.splitext(im["file_name"])[0]
            with open(os.path.join(out, "labels", split, base + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            src_img = os.path.join(ROOT, s["images_root"], im["file_name"])
            if os.path.exists(src_img):
                shutil.copy2(src_img, os.path.join(out, "images", split, im["file_name"]))
                stats["images"] += 1

    yaml_path = os.path.join(out, "sop_pose.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(
            f"path: {out}\ntrain: images/train\nval: images/val\n"
            f"kpt_shape: [{nk}, 3]\nnames:\n  0: person\n"
        )
    ctx.note(f"COCO 转换完成：{stats['images']} 张图 / {stats['persons']} 个人体实例"
             f"（每实例至少 {min_k} 个可见关键点，负样本比例 {bg_ratio:.0%}）")
    ctx.note("train/val 拆分：" + " / ".join(f"{k}={len(v)}" for k, v in splits.items()))
    ctx.note("数据集定义写入 " + os.path.relpath(yaml_path, ROOT))
    return yaml_path


def assembly101_to_yolo_pose(cfg, ctx):
    ctx.note("assembly101 标注转换在 M2 落地：需先在 HF 网站完成 gated 数据集授权并下载"
             " poses@60fps JSON，验证其内部结构后实现 42 关键点（双手）转换。")
    raise SystemExit("assembly101 conversion is planned for M2 (see runs/README)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None)
    args = ap.parse_args()
    with open(os.path.join(ROOT, "config", "dataset.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    src = args.source or cfg["active_source"]
    ctx = RunContext("convert", params={"source": src})
    try:
        if src == "coco_val2017":
            coco_to_yolo_pose(cfg, ctx)
        elif src == "assembly101":
            assembly101_to_yolo_pose(cfg, ctx)
        else:
            raise SystemExit("unknown source " + src)
    except SystemExit:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        ctx.note("异常: " + repr(e))
        ctx.finish(ok=False)
        sys.exit(1)
    ctx.finish(ok=True)


if __name__ == "__main__":
    main()
