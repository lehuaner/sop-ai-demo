"""SOP-AI demo main entry (docs/01 section 4): train / run modes.

  python src/main.py train  [--epochs N] [--data PATH] [--model PATH]
  python src/main.py run    --video PATH [--weights PATH]

All runs are archived under runs/<timestamp>-<task>/ (docs/01 section 6).
"""
import argparse
import csv
import glob
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.runctx import RunContext  # noqa: E402

DEFAULT_DATA = os.path.join(ROOT, "datasets/converted/sop_pose/sop_pose.yaml")
DEFAULT_MODEL = os.path.join(ROOT, "weights/yolo11n-pose.pt")


def find_weight(name):
    """Locate newest trained weight: runs/*/weights/best.pt preferred."""
    cands = sorted(glob.glob(os.path.join(ROOT, "runs", "*", "weights", name)))
    return cands[-1] if cands else None


def cmd_train(args):
    from ultralytics import YOLO

    data = args.data or DEFAULT_DATA
    model = args.model or DEFAULT_MODEL
    ctx = RunContext("train", params={
        "data": os.path.relpath(data, ROOT), "model": os.path.relpath(model, ROOT),
        "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
        "device": 0, "note": "RTX 3050 4GB: yolo11n-pose + AMP",
    })
    try:
        y = YOLO(model)
        y.train(data=data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                patience=args.patience, workers=args.workers, device=0,
                project=os.path.join(ROOT, "runs"), name=ctx.name, exist_ok=True)
        run_dir = os.path.join(ROOT, "runs", ctx.name)
        last = {}
        csv_path = os.path.join(run_dir, "results.csv")
        if os.path.exists(csv_path):
            with open(csv_path) as f:
                rows = list(csv.DictReader(f))
            if rows:
                last = rows[-1]
                for k, v in last.items():
                    for key in ("mAP50(B)", "mAP50(P)", "mAP50-95(B)", "mAP50-95(P)"):
                        if k.endswith(key):
                            ctx.note(f"{key} = {float(v):.4f}")
        ctx.note("训练完成，best 权重: runs/" + ctx.name + "/weights/best.pt")
        ctx.note("结果曲线: results.png | 混淆: 无(姿态任务) | 逐轮指标: results.csv")
        ctx.finish(ok=True, extra_manifest={"final_metrics": last})
    except Exception as e:
        import traceback
        traceback.print_exc()
        ctx.note("异常: " + repr(e))
        ctx.finish(ok=False)
        sys.exit(1)


def cmd_run(args):
    import cv2
    from ultralytics import YOLO

    weights = args.weights or find_weight("best.pt") or DEFAULT_MODEL
    ctx = RunContext("run", params={"video": args.video, "weights": os.path.relpath(weights, ROOT)})
    try:
        y = YOLO(weights)
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError("cannot open video " + args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_mp4 = os.path.join(ctx.path, "samples", "annotated.mp4")
        os.makedirs(os.path.dirname(out_mp4), exist_ok=True)
        vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        events_path = os.path.join(ctx.path, "events.jsonl")
        ev = open(events_path, "w", encoding="utf-8")
        fid, t0 = 0, cv2.getTickCount()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            res = y.predict(frame, verbose=False, device=0)[0]
            plot = res.plot()
            fid += 1
            if fid % int(fps) == 0:
                now = cv2.getTickCount()
                fps_now = fid / ((now - t0) / cv2.getTickFrequency())
                ev.write(json.dumps({"frame": fid, "fps": round(fps_now, 1),
                                     "persons": len(res.keypoints)}) + "\n")
                print(f"frame {fid} | {fps_now:.1f} fps | persons {len(res.keypoints)}")
            vw.write(plot)
        cap.release()
        vw.release()
        ev.close()
        ctx.note(f"回放推理完成：{fid} 帧，输出视频 samples/annotated.mp4，事件 events.jsonl")
        ctx.note("当前为 M2 起步版：仅姿态叠加；SOP 规则引擎/报警随后接入")
        ctx.finish(ok=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        ctx.note("异常: " + repr(e))
        ctx.finish(ok=False)
        sys.exit(1)


import json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=150)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--batch", type=int, default=4)
    t.add_argument("--patience", type=int, default=30)
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--data", default=None)
    t.add_argument("--model", default=None)
    r = sub.add_parser("run")
    r.add_argument("--video", required=True)
    r.add_argument("--weights", default=None)
    args = ap.parse_args()
    if args.cmd == "train":
        cmd_train(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
