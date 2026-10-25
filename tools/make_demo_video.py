"""Build a demo replay video for run mode (v2, keypoint-aware).

v1 bug: every image lasted a single frame while SOP rules need
stable_frames=20 consecutive frames, and the 10s video was shorter than
the 30s timeout -> the SOP engine could mathematically never emit an event.

v2: uses the converted labels' wrist keypoints to SELECT images whose
hands actually fall inside the SOP ROIs (compensating the letterbox
transform), and holds each image ~2s (30 frames @15fps > stable_frames=20)
so steps genuinely complete. Sections: work_area (step1) -> object_tray
(step2) -> work_area (step3). Archived via src/runctx.py (task=demo_video).
"""
import glob
import os
import random
import sys

import cv2
import yaml
from PIL import Image

sys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, sys_path)
from src.runctx import RunContext  # noqa: E402

ROOT = os.path.abspath(sys_path)
W, H, FPS = 960, 540, 15
HOLD_SEC = 2.0
R = 9   # COCO right wrist, 10 = left wrist


def letterbox_params(w, h):
    s = min(W / w, H / h)
    nw, nh = int(w * s), int(h * s)
    return (W - nw) / 2 / W, (H - nh) / 2 / H, nw / W, nh / H  # x0, y0, sx, sy (frame-norm)


def to_frame_norm(x, y, w, h):
    x0, y0, sx, sy = letterbox_params(w, h)
    return x0 + x * sx, y0 + y * sy


def parse_wrists(label_path):
    """Return list of (right_wrist_xy, left_wrist_xy) normalized to ORIGINAL image."""
    out = []
    if not os.path.exists(label_path):
        return out
    with open(label_path) as f:
        for line in f:
            v = [float(t) for t in line.split()]
            if len(v) < 5 + 3 * 17:
                continue
            k = v[5:]
            wr = (k[3 * R], k[3 * R + 1], k[3 * R + 2])
            wl = (k[3 * 10], k[3 * 10 + 1], k[3 * 10 + 2])
            out.append((wr, wl))
    return out


def wrists_in_roi(person, roi, w, h):
    ok = []
    for x, y, v in person:
        if v <= 0:
            ok.append(False)
            continue
        fx, fy = to_frame_norm(x, y, w, h)
        ok.append(roi["x1"] <= fx <= roi["x2"] and roi["y1"] <= fy <= roi["y2"])
    return ok


def main():
    with open(os.path.join(ROOT, "config", "dataset.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with open(os.path.join(ROOT, "config", "sop_demo.yaml"), encoding="utf-8") as f:
        sop = yaml.safe_load(f)
    ctx = RunContext("demo_video", params={"seconds_per_image": HOLD_SEC, "fps": FPS, "size": f"{W}x{H}"})
    try:
        imgs = sorted(glob.glob(os.path.join(ROOT, cfg["converted_dir"], "images", "train", "*.jpg")))
        if not imgs:
            raise RuntimeError("no converted images; run convert first")
        roi_work = sop["roi"]["work_area"]
        roi_tray = sop["roi"]["object_tray"]

        pool_work, pool_tray = [], []
        for ip in imgs:
            lp = os.path.join(ROOT, cfg["converted_dir"], "labels", "train",
                              os.path.splitext(os.path.basename(ip))[0] + ".txt")
            persons = parse_wrists(lp)
            if not persons:
                continue
            with Image.open(ip) as im:
                w, h = im.size
            if any(all(wrists_in_roi(p, roi_work, w, h)) for p in persons):
                pool_work.append(ip)
            if any(any(wrists_in_roi(p, roi_tray, w, h)) for p in persons):
                pool_tray.append(ip)
        ctx.note(f"关键点选图：双腕入 work_area 的图 {len(pool_work)} 张，手腕入 object_tray 的图 {len(pool_tray)} 张")

        rng = random.Random(11)
        rng.shuffle(pool_work)
        rng.shuffle(pool_tray)
        need = 2
        seq = pool_work[:need] + pool_tray[:need] + pool_work[need:2 * need]
        if len(seq) < 3 * need:  # fall back: pad with work pool / any person images
            rest = [i for i in pool_work if i not in seq]
            seq += rest[:3 * need - len(seq)]
        if not seq:
            raise RuntimeError("no usable frames with visible wrists")
        ctx.note(f"编排段落：work_area({need}张, 步骤1) → object_tray({need}张, 步骤2) → work_area({need}张, 步骤3)，"
                 f"每张停留 {HOLD_SEC}s（{int(HOLD_SEC * FPS)} 帧 > stable_frames 阈值）")

        out_dir = os.path.join(ROOT, "datasets", "demo_video")
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, "demo_assembly.mp4")
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        frames = 0
        for ip in seq:
            img = cv2.imread(ip)
            if img is None:
                continue
            h, w = img.shape[:2]
            s = min(W / w, H / h)
            nw, nh = int(w * s), int(h * s)
            canvas = cv2.copyMakeBorder(
                cv2.resize(img, (nw, nh)),
                (H - nh) // 2, H - nh - (H - nh) // 2,
                (W - nw) // 2, W - nw - (W - nw) // 2,
                cv2.BORDER_CONSTANT, value=(0, 0, 0))
            for _ in range(int(HOLD_SEC * FPS)):
                vw.write(canvas)
                frames += 1
        vw.release()
        ctx.note(f"演示视频生成：{os.path.relpath(out, ROOT)}（{frames} 帧 @ {FPS}fps = {frames / FPS:.0f}s）")
        ctx.note("预期：三段各自满足对应 ROI 规则 → events.jsonl 应出现 step_done；若某段选图不足则相应步骤超时告警（同为有效演示）")
        ctx.finish(ok=True, extra_manifest={"output": os.path.relpath(out, ROOT), "frames": frames,
                                            "pool_work": len(pool_work), "pool_tray": len(pool_tray)})
    except Exception as e:
        import traceback
        traceback.print_exc()
        ctx.note("异常: " + repr(e))
        ctx.finish(ok=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
