"""Behavior detection dataset v2 (user feedback: boxes too large, volume unrealistic):
- tight action-region boxes rebuilt from person KEYPOINTS (upper body + wrists,
  12% margin) instead of the model's full-body box;
- per-take cap 400 frames (evenly spaced) -> ~3200 frames total, feasible review.
"""
import csv
import json
import os
import subprocess
import sys

import yaml

sys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, sys_path)
from src.runctx import RunContext  # noqa: E402

ROOT = os.path.abspath(sys_path)
CAP_PER_TAKE = 400
MARGIN = 0.12
UPPER = list(range(0, 11))  # nose..eyes..ears..shoulders..elbows..wrists


def tight_box(kps_xy, kps_conf):
    pts = [kps_xy[i] for i in UPPER if kps_conf[i] > 0.3]
    if len(pts) < 4:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    mx, my = (x2 - x1) * MARGIN, (y2 - y1) * MARGIN
    return x1 - mx, y1 - my, x2 + mx, y2 + my


def main():
    with open(os.path.join(ROOT, "config", "dataset.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    hands_dir = os.path.join(ROOT, cfg["converted_dir"].replace("sop_pose", "sop_hands"))
    out_dir = os.path.join(ROOT, "datasets/converted/sop_behavior")
    stage = {}
    for r in csv.DictReader(open(os.path.join(ROOT, "datasets/converted/sop_stages/stage_labels.csv"), encoding="utf-8-sig")):
        stage[os.path.basename(r["image"])] = r["stage"]

    ctx = RunContext("behavior_prelabel", params={"classes": ["assemble", "disassemble", "idle"],
                                                  "cap_per_take": CAP_PER_TAKE,
                                                  "box": "upper-body keypoints +12%"})
    try:
        from ultralytics import YOLO
        y = YOLO(os.path.join(ROOT, "weights/yolo11n-pose.pt"))
        # rebuild label AND image dirs from scratch; images copied per kept frame
        import shutil as _sh
        for sub in ("images/train", "images/val", "images/test",
                    "labels/train", "labels/val", "labels/test"):
            p = os.path.join(out_dir, sub)
            if os.path.isdir(p):
                _sh.rmtree(p)
            os.makedirs(p, exist_ok=True)

        cls_id = {"assemble": 0, "disassemble": 1, "idle": 2}
        stats = {"frames": 0, "boxes": 0, "keypoint_fallback": 0, "skipped_exclude": 0, "no_person": 0}
        for split_name, rels in json.load(open(os.path.join(hands_dir, "split.json"), encoding="utf-8"))["splits"].items():
            by_take = {}
            for img_rel in rels:
                name = os.path.basename(img_rel)
                by_take.setdefault(name.split("_f")[0], []).append(name)
            for take, names in sorted(by_take.items()):
                if len(names) > CAP_PER_TAKE:
                    step = len(names) / CAP_PER_TAKE
                    names = [names[int(i * step)] for i in range(CAP_PER_TAKE)]
                for name in names:
                    st = stage.get(name, "")
                    if st not in cls_id:
                        stats["skipped_exclude"] += 1
                        continue
                    img_path = os.path.join(hands_dir, "images", split_name, name)
                    img = cv2_read(img_path)
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    res = y.predict(img, verbose=False, device=0, conf=0.45)[0]
                    if res.boxes is None or len(res.boxes) == 0:
                        stats["no_person"] += 1
                        continue
                    i = int(res.boxes.conf.argmax())
                    x1, y1, x2, y2 = res.boxes.xyxy.cpu().numpy()[i]
                    if res.keypoints is not None:
                        kxy = res.keypoints.xy[i].cpu().numpy()
                        kcf = res.keypoints.conf[i].cpu().numpy() if res.keypoints.conf is not None else [1] * len(kxy)
                        tb = tight_box(kxy, kcf)
                        if tb:
                            x1, y1, x2, y2 = tb
                        else:
                            stats["keypoint_fallback"] += 1
                    else:
                        stats["keypoint_fallback"] += 1
                    x1, x2 = max(0.0, x1 / w), min(1.0, x2 / w)
                    y1, y2 = max(0.0, y1 / h), min(1.0, y2 / h)
                    if x2 - x1 < 0.02 or y2 - y1 < 0.02:
                        stats["no_person"] += 1
                        continue
                if st not in cls_id:
                    continue
                with open(os.path.join(out_dir, "labels", split_name, name[:-4] + ".txt"), "w") as f:
                    f.write(f"{cls_id[st]} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}\n")
                _sh.copy2(img_path, os.path.join(out_dir, "images", split_name, name))
                stats["frames"] += 1
                stats["boxes"] += 1
        with open(os.path.join(out_dir, "behavior.yaml"), "w", encoding="utf-8") as f:
            f.write(f"path: {out_dir}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                    "names:\n  0: assemble\n  1: disassemble\n  2: idle\n")
        ctx.note(f"行为框伪标签 v2：{stats['frames']} 帧（紧框=上半身关键点包络+12%，退回全身框 {stats['keypoint_fallback']} 次，"
                 f"EXCLUDE 跳过 {stats['skipped_exclude']}，无人 {stats['no_person']}），每段上限 {CAP_PER_TAKE} 帧")
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
