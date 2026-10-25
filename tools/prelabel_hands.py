"""Route D core: MediaPipe Hands pre-labeling -> YOLO-pose hand dataset.

Per configured take (v1 view):
  sample frames at prelabel.fps_extract (default 10) -> HandLandmarker
  -> per hand: class right_hand(0)/left_hand(1) from handedness (SWAPPED for
     non-mirrored third-person input, per MediaPipe docs), bbox = landmark
     box + margin, 21 keypoints (x, y, vis)
  -> temporal contiguous 7/2/1 split per take -> datasets/converted/sop_hands
  -> split.json fixed (v2 human correction must NOT change the split)

Reports the labeled-frame threshold (>= prelabel.min_labeled_frames, default
3000); if below, take more videos (docs/02 section 3). Archived task=prelabel.
"""
import glob
import json
import os
import shutil
import sys

import cv2
import yaml

sys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, sys_path)
from src.runctx import RunContext  # noqa: E402

ROOT = os.path.abspath(sys_path)
MP_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                "hand_landmarker/float16/1/hand_landmarker.task")


def ensure_model():
    p = os.path.join(ROOT, "weights", "hand_landmarker.task")
    if not os.path.exists(p):
        import urllib.request
        os.makedirs(os.path.dirname(p), exist_ok=True)
        print("downloading hand_landmarker.task ...")
        urllib.request.urlretrieve(MP_MODEL_URL, p)
    return p


def build_detector(model_path):
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe.tasks.python import vision
    opts = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
    )
    return vision.HandLandmarker.create_from_options(opts)


def detect_frame(detector, bgr):
    import mediapipe as mp
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    out = []
    for lms, handed in zip(res.hand_landmarks, res.handedness):
        cat = handed[0].category_name
        # MediaPipe assumes mirrored (selfie) input; our third-person video is
        # NOT mirrored -> swap: reported 'Left' == actual right hand
        cls = 0 if cat == "Left" else 1
        pts = [(l.x, l.y, 2 if (getattr(l, "visibility", 0) or 0) >= 0.5 else 1) for l in lms]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        out.append((cls, xs, ys, pts))
    return out


def to_line(cls, xs, ys, pts, margin):
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    bx1, bx2 = max(0.0, x1 - mx), min(1.0, x2 + mx)
    by1, by2 = max(0.0, y1 - my), min(1.0, y2 + my)
    bw, bh = bx2 - bx1, by2 - by1
    if bw < 0.01 or bh < 0.01 or bh / max(bw, 1e-6) > 6:
        return None
    head = f"{cls} {(bx1 + bx2) / 2:.6f} {(by1 + by2) / 2:.6f} {bw:.6f} {bh:.6f}"
    kpts = " ".join(f"{x:.6f} {y:.6f} {v}" for x, y, v in pts)
    return head + " " + kpts


def main():
    with open(os.path.join(ROOT, "config", "dataset.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    pre = cfg.get("prelabel", {})
    fps_extract = int(pre.get("fps_extract", 10))
    margin = float(pre.get("margin", 0.25))
    neg_ratio = float(pre.get("neg_ratio", 0.10))
    min_frames = int(pre.get("min_labeled_frames", 3000))
    s = cfg["sources"][cfg["active_source"]]
    takes = s["takes"]
    view = s["view_map"][s["views"][0]]

    ctx = RunContext("prelabel", params={"takes": len(takes), "fps_extract": fps_extract,
                                         "split": "7/2/1 temporal", "margin": margin})
    detector = None
    try:
        out_dir = os.path.join(ROOT, cfg["converted_dir"].replace("sop_pose", "sop_hands"))
        if os.path.isdir(out_dir):
            shutil.rmtree(out_dir)  # rebuild clean (split must stay reproducible)
        for sub in ("images/train", "images/val", "images/test",
                    "labels/train", "labels/val", "labels/test"):
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

        import random
        rng = random.Random(cfg.get("seed", 42))
        split = {"train": [], "val": [], "test": []}
        stats = {"labeled": 0, "negatives": 0, "hands": 0, "per_class": {0: 0, 1: 0}}
        detector = build_detector(ensure_model())

        for take in takes:
            mp4 = os.path.join(ROOT, "datasets/raw/assembly101/recordings", take, view)
            if not os.path.exists(mp4):
                print("WARN missing video:", mp4)
                continue
            cap = cv2.VideoCapture(mp4)
            src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            step = max(int(round(src_fps / fps_extract)), 1)
            take_imgs = []  # (rel_img_path, labeled) in temporal order
            si = 0
            fi = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if fi % step == 0:
                    hands = detect_frame(detector, frame)
                    labeled = len(hands) > 0
                    keep = labeled or (rng.random() < neg_ratio)
                    if keep:
                        name = f"{take}_f{si:06d}.jpg"
                        rel_img = f"images/{{split}}/{name}"
                        lines = []
                        for cls, xs, ys, pts in hands:
                            ln = to_line(cls, xs, ys, pts, margin)
                            if ln:
                                lines.append(ln)
                                stats["hands"] += 1
                                stats["per_class"][cls] += 1
                        labeled = len(lines) > 0
                        take_imgs.append((name, labeled, frame, lines))
                        si += 1
                        if labeled:
                            stats["labeled"] += 1
                        else:
                            stats["negatives"] += 1
                fi += 1
            cap.release()

            # temporal contiguous 7/2/1 within this take
            n = len(take_imgs)
            n_train, n_val = int(n * 0.7), int(n * 0.2)
            bounds = {"train": (0, n_train), "val": (n_train, n_train + n_val), "test": (n_train + n_val, n)}
            for split_name, (a, b) in bounds.items():
                for name, labeled, frame, lines in take_imgs[a:b]:
                    img_rel = f"images/{split_name}/{name}"
                    cv2.imwrite(os.path.join(out_dir, img_rel), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 92])
                    with open(os.path.join(out_dir, f"labels/{split_name}/{name[:-4]}.txt"), "w") as f:
                        f.write("\n".join(lines) + ("\n" if lines else ""))
                    split[split_name].append(img_rel)
            print(f"[{take[:40]}...] frames kept={n} labeled={sum(1 for _, l, _, _ in take_imgs if l)}")

        with open(os.path.join(out_dir, "split.json"), "w", encoding="utf-8") as f:
            json.dump({"splits": split, "rule": "temporal 7/2/1 per take, contiguous",
                       "labeled": stats["labeled"], "negatives": stats["negatives"]},
                      f, ensure_ascii=False, indent=1)
        with open(os.path.join(out_dir, "sop_hands.yaml"), "w", encoding="utf-8") as f:
            f.write(f"path: {out_dir}\ntrain: images/train\nval: images/val\ntest: images/test\n"
                    "kpt_shape: [21, 3]\nnames:\n  0: right_hand\n  1: left_hand\n")

        ctx.note(f"预标注完成：带标注帧 {stats['labeled']}，负样本 {stats['negatives']}，"
                 f"手实例 {stats['hands']}（右手 {stats['per_class'][0]} / 左手 {stats['per_class'][1]}）")
        for k in ("train", "val", "test"):
            n_l = sum(1 for p in split[k] if os.path.exists(
                os.path.join(out_dir, p.replace("images/", "labels/")[:-4] + ".txt")
            ) and os.path.getsize(os.path.join(out_dir, p.replace("images/", "labels/")[:-4] + ".txt")) > 0)
            ctx.note(f"{k}: {len(split[k])} 帧（带标注 {n_l}）")
        if stats["labeled"] < min_frames:
            ctx.note(f"警告：带标注帧 {stats['labeled']} < 门槛 {min_frames}——按 docs/02 §3 继续取段扩数据")
            print(f"THRESHOLD NOT MET: {stats['labeled']} < {min_frames} -> take more videos")
        else:
            ctx.note(f"数据量达标（≥ {min_frames}）")
        ctx.note("v1 训练: python src/main.py train --data datasets/converted/sop_hands/sop_hands.yaml")
        ctx.note("人工修正: 用 X-AnyLabeling 打开 images/ + labels/ 修正后重训 v2（划分不动）")
        ctx.finish(ok=True, extra_manifest={"stats": stats})
    except Exception as e:
        import traceback
        traceback.print_exc()
        ctx.note("异常: " + repr(e))
        ctx.finish(ok=False)
        sys.exit(1)
    finally:
        if detector is not None:
            detector.close()


if __name__ == "__main__":
    main()
