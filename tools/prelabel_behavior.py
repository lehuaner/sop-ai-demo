"""Behavior detection dataset (user-directed route, agents.log 21:30):
person boxes x frame-stage labels -> YOLO detect labels (3 behavior classes).

Teacher chain: COCO person-pose model draws person boxes; official coarse
segments (via stage_labels.csv) name the behavior; operator = largest person
box. Human corrects boxes/classes in X-AnyLabeling (YOLO HBB), then trains
yolo11n detect. Same split.json. Archived task=behavior_prelabel.
"""
import csv
import json
import os
import shutil
import subprocess
import sys

import yaml

sys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, sys_path)
from src.runctx import RunContext  # noqa: E402

ROOT = os.path.abspath(sys_path)


def main():
    with open(os.path.join(ROOT, "config", "dataset.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    hands_dir = os.path.join(ROOT, cfg["converted_dir"].replace("sop_pose", "sop_hands"))
    out_dir = os.path.join(ROOT, "datasets/converted/sop_behavior")
    stage = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "datasets/converted/sop_stages/stage_labels.csv"), encoding="utf-8-sig")):
        stage[os.path.basename(r["image"])] = r["stage"]

    ctx = RunContext("behavior_prelabel", params={"classes": ["assemble", "disassemble", "idle"]})
    try:
        from ultralytics import YOLO
        y = YOLO(os.path.join(ROOT, "weights/yolo11n-pose.pt"))  # COCO person model
        for sub in ("images/train", "images/val", "images/test",
                    "labels/train", "labels/val", "labels/test"):
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
        # share hand-dataset images via directory junctions (no copy)
        for sub in ("train", "val", "test"):
            dst = os.path.join(out_dir, "images", sub)
            src = os.path.join(hands_dir, "images", sub)
            if not os.path.exists(dst):
                subprocess.run(["cmd", "/c", "mklink", "/J", dst, src], check=True,
                               capture_output=True)

        cls_id = {"assemble": 0, "disassemble": 1, "idle": 2}
        stats = {"frames": 0, "boxes": 0, "skipped_exclude": 0, "no_person": 0}
        for split_name, rels in json.load(open(os.path.join(hands_dir, "split.json"), encoding="utf-8"))["splits"].items():
            for img_rel in rels:
                name = os.path.basename(img_rel)
                st = stage.get(name, "")
                if st in ("", "EXCLUDE"):
                    stats["skipped_exclude"] += 1
                    continue
                img_path = os.path.join(hands_dir, img_rel)
                img = cv2_read(img_path)
                if img is None:
                    continue
                h, w = img.shape[:2]
                res = y.predict(img, verbose=False, device=0, conf=0.45)[0]
                if res.boxes is None or len(res.boxes) == 0:
                    stats["no_person"] += 1
                    continue
                xyxy = res.boxes.xyxy.cpu().numpy()
                areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
                i = int(areas.argmax())  # operator = largest person
                x1, y1, x2, y2 = xyxy[i]
                cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
                bw, bh = (x2 - x1) / w, (y2 - y1) / h
                with open(os.path.join(out_dir, "labels", split_name, name[:-4] + ".txt"), "w") as f:
                    f.write(f"{cls_id[st]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                shutil.copy2(img_path, os.path.join(out_dir, "images", split_name, name))
                stats["frames"] += 1
                stats["boxes"] += 1
        with open(os.path.join(out_dir, "behavior.yaml"), "w", encoding="utf-8") as f:
            f.write(f"path: {out_dir}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                    "names:\n  0: assemble\n  1: disassemble\n  2: idle\n")
        with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8") as f:
            f.write("assemble\ndisassemble\nidle\n")
        ctx.note(f"行为框伪标签：{stats['frames']} 帧（EXCLUDE 跳过 {stats['skipped_exclude']}，无人 {stats['no_person']}），"
                 f"每帧 1 框 = 最大人体，类别取帧级阶段")
        ctx.note("v1 训练: python src/main.py train --data datasets/converted/sop_behavior/behavior.yaml --model weights/yolo11n.pt(需下载 detect 权重)")
        ctx.note("人工修正: X-AnyLabeling 上传图片 images/<split> + YOLO HBB 导入 labels/<split> + classes.txt，画框/改类后导出")
        ctx.finish(ok=True, extra_manifest={"stats": stats})
    except Exception as e:
        import traceback
        traceback.print_exc()
        ctx.note("异常: " + repr(e))
        ctx.finish(ok=False)
        sys.exit(1)


def cv2_read(p):
    import cv2
    return cv2.imread(p)


if __name__ == "__main__":
    main()
