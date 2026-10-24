"""Extract training frames from downloaded data (docs/02).

coco_val2017 : unzip already handled by download_subset; this step only
               records the extracted image count.
assembly101  : sample frames from downloaded view mp4s at fps_extract,
               skip near-duplicate and blurry frames.

Archived via src/runctx.py (task=extract).
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import yaml

sys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
import sys  # noqa: E402

sys.path.insert(0, sys_path)
from src.runctx import RunContext  # noqa: E402

ROOT = os.path.dirname(sys_path)


def blur_score(img):
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()


def extract_video(mp4, out_dir, fps, blur_thr=40.0, diff_thr=8.0):
    cap = cv2.VideoCapture(mp4)
    if not cap.isOpened():
        raise RuntimeError("cannot open " + mp4)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(src_fps / fps)), 1)
    os.makedirs(out_dir, exist_ok=True)
    index, prev_small, kept = [], None, 0
    fid = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        fno = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if (fno - 1) % step:
            continue
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (64, 36))
        if prev_small is not None and float(np.abs(small.astype(int) - prev_small).mean()) < diff_thr:
            prev_small = small
            continue
        if blur_score(frame) < blur_thr:
            prev_small = small
            continue
        fid += 1
        name = f"f{fid:06d}.jpg"
        cv2.imwrite(os.path.join(out_dir, name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        index.append({"file": name, "video_frame": fno})
        prev_small = small
        kept += 1
    cap.release()
    with open(os.path.join(out_dir, "frames_index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=1)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=None, help="override active_source")
    args = ap.parse_args()
    with open(os.path.join(ROOT, "config", "dataset.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    src = args.source or cfg["active_source"]
    ctx = RunContext("extract", params={"source": src})
    ok = True
    try:
        if src == "coco_val2017":
            s = cfg["sources"]["coco_val2017"]
            imgs = glob.glob(os.path.join(ROOT, s["images_root"], "*.jpg"))
            ctx.note(f"COCO val2017 图像已解压：{len(imgs)} 张（解压在 download 步骤完成）")
            if not imgs:
                raise RuntimeError("no images found, run download first")
        elif src == "assembly101":
            s = cfg["sources"]["assembly101"]
            total = 0
            for take in s["takes"]:
                for v in s["views"]:
                    mp4 = os.path.join(ROOT, "datasets/raw/assembly101/recordings", take, s["view_map"][v])
                    if not os.path.exists(mp4):
                        print("WARN missing", mp4)
                        continue
                    out = os.path.join(ROOT, "datasets/raw/assembly101/frames", f"{take}_{v}")
                    n = extract_video(mp4, out, s["fps_extract"])
                    total += n
                    ctx.note(f"抽帧 {take}/{v}：保留 {n} 帧 @ {s['fps_extract']}fps（去重+模糊过滤）")
            if total == 0:
                raise RuntimeError("no video frames extracted, run download first")
        else:
            raise SystemExit("unknown source " + src)
    except Exception as e:
        import traceback
        traceback.print_exc()
        ctx.note("异常: " + repr(e))
        ctx.finish(ok=False)
        sys.exit(1)
    ctx.finish(ok=ok)


if __name__ == "__main__":
    main()
