"""SOP-AI demo main entry (docs/01 section 4): train / run modes.

  python src/main.py train  [--epochs N] [--data PATH] [--model PATH]
  python src/main.py run    --video PATH [--weights PATH] [--sop PATH]

All runs are archived under runs/<timestamp>-<task>/ (docs/01 section 6):
README.md (human summary) + manifest.json + log.txt + artifacts.
"""
import argparse
import csv
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.runctx import RunContext  # noqa: E402

DEFAULT_DATA = os.path.join(ROOT, "datasets/converted/sop_pose/sop_pose.yaml")
DEFAULT_MODEL = os.path.join(ROOT, "weights/yolo11n-pose.pt")
DEFAULT_SOP = os.path.join(ROOT, "config/sop_demo.yaml")


def find_weight(name):
    """Trained weight preference: weights/<name> (survives rotation) then runs/."""
    p = os.path.join(ROOT, "weights", name)
    if os.path.exists(p):
        return p
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
        best_src = os.path.join(ROOT, "runs", ctx.name, "weights", "best.pt")
        if os.path.exists(best_src):
            shutil.copy2(best_src, os.path.join(ROOT, "weights", "best.pt"))
            ctx.note("已复制部署权重到 weights/best.pt（轮转归档不影响 run 模式取用）")
        ctx.note("结果曲线 results.png | 逐轮指标 results.csv（大屏 /api/train_metrics 可读）")
        ctx.finish(ok=True, extra_manifest={"final_metrics": last})
    except Exception as e:
        import traceback
        traceback.print_exc()
        ctx.note("异常: " + repr(e))
        ctx.finish(ok=False)
        sys.exit(1)


def cmd_run(args):
    import cv2
    import numpy as np
    from ultralytics import YOLO

    from src.sop import SOPEngine

    weights = args.weights or find_weight("best.pt") or DEFAULT_MODEL
    sop_path = args.sop or DEFAULT_SOP
    engine = SOPEngine(sop_path)
    ctx = RunContext("run", params={
        "video": args.video, "weights": os.path.relpath(weights, ROOT),
        "sop": os.path.relpath(sop_path, ROOT),
    })
    try:
        y = YOLO(weights)
        cap = cv2.VideoCapture(args.video)
        if not cap.isOpened():
            raise RuntimeError("cannot open video " + args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        os.makedirs(os.path.join(ctx.path, "samples"), exist_ok=True)
        out_mp4 = os.path.join(ctx.path, "samples", "annotated.mp4")
        vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        ev = open(os.path.join(ctx.path, "events.jsonl"), "w", encoding="utf-8")
        fid, n_person, n_ev, t0 = 0, 0, 0, cv2.getTickCount()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if args.max_seconds and fid >= int(fps * args.max_seconds):
                break
            res = y.predict(frame, verbose=False, device=args.device)[0]
            kpts_px = None
            if res.keypoints is not None and len(res.keypoints) > 0:
                i = int(res.boxes.conf.argmax())
                kpts_px = res.keypoints.xy[i].cpu().numpy().astype(float)
                n_person = max(n_person, len(res.keypoints))
            # SOP ROIs are normalized (0-1) -> feed normalized keypoints
            kpts_norm = kpts_px / np.array([w, h], dtype=float) if kpts_px is not None else None
            events = engine.update(kpts_norm, fid)
            for e in events:
                ev.write(json.dumps(e, ensure_ascii=False) + "\n")
                n_ev += 1
                print(f"[alarm] {e['type']} step={e.get('step')} {e.get('message', '')}")
                if e["type"] in ("step_timeout", "out_of_order"):
                    try:
                        import winsound
                        winsound.Beep(1200, 350)
                    except Exception:
                        pass
            plot = engine.draw_overlay(res.plot(), fid)
            vw.write(plot)
            fid += 1
            if fid % int(fps) == 0:
                now = cv2.getTickCount()
                fps_now = fid / ((now - t0) / cv2.getTickFrequency())
                print(f"frame {fid} | {fps_now:.1f} fps | persons {len(res.keypoints) if res.keypoints is not None else 0}")
        cap.release()
        vw.release()
        ev.close()
        ctx.note(f"回放推理完成：{fid} 帧，画面中出现最多 {n_person} 人，触发事件 {n_ev} 条")
        ctx.note(f"SOP 步骤推进到：{engine.current}/{len(engine.steps)}")
        ctx.note(f"输出：samples/annotated.mp4 + events.jsonl（大屏 events 面板读取）")
        ctx.finish(ok=True, extra_manifest={"frames": fid, "events": n_ev,
                                            "sop_step_reached": engine.current})
    except Exception as e:
        import traceback
        traceback.print_exc()
        ctx.note("异常: " + repr(e))
        ctx.finish(ok=False)
        sys.exit(1)


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
    r.add_argument("--sop", default=None)
    r.add_argument("--device", default="0", help="0 = GPU, cpu = CPU smoke test")
    r.add_argument("--max-seconds", type=float, default=0, help="0 = full video")
    args = ap.parse_args()
    if args.cmd == "train":
        cmd_train(args)
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
