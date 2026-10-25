"""Compare two trained hand-pose models on the SAME test split (docs/02 §1).

  python tools/compare_models.py --a <run_dir_a> --b <run_dir_b>

Each run dir must contain weights/best.pt. Evaluates both on the `test`
split of datasets/converted/sop_hands/sop_hands.yaml and writes a delta
report to runs/<timestamp>-compare/. Archived task=compare.
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.runctx import RunContext  # noqa: E402


def resolve_run(name_or_path):
    p = name_or_path
    if not os.path.isabs(p):
        p = os.path.join(ROOT, "runs", name_or_path)
    w = os.path.join(p, "weights", "best.pt")
    if not os.path.exists(w):
        cands = sorted(glob.glob(os.path.join(ROOT, "runs", "*", "weights", "best.pt")))
        raise SystemExit(f"weights not found under {p}; available: {[c.split(os.sep)[-3] for c in cands]}")
    return os.path.basename(p), w


def evaluate(weight, data_yaml):
    from ultralytics import YOLO
    r = YOLO(weight).val(data=data_yaml, split="test", imgsz=640, device=0, verbose=False)
    return {
        "pose_mAP50": round(float(r.pose.map50), 4),
        "pose_mAP50_95": round(float(r.pose.map), 4),
        "box_mAP50": round(float(r.box.map50), 4),
        "speed_ms_per_img": round(float(r.speed.get("inference", 0)), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="run dir name of model v1")
    ap.add_argument("--b", required=True, help="run dir name of model v2")
    ap.add_argument("--data", default=os.path.join(ROOT, "datasets/converted/sop_hands/sop_hands.yaml"))
    args = ap.parse_args()

    name_a, w_a = resolve_run(args.a)
    name_b, w_b = resolve_run(args.b)
    ctx = RunContext("compare", params={"v1": name_a, "v2": name_b,
                                        "data": os.path.relpath(args.data, ROOT),
                                        "split": "test"})
    try:
        m_a = evaluate(w_a, args.data)
        m_b = evaluate(w_b, args.data)
        delta = {k: round(m_b[k] - m_a[k], 4) for k in m_a}
        lines = [
            f"- v1（伪标签，{name_a}）: {json.dumps(m_a)}",
            f"- v2（人工修正，{name_b}）: {json.dumps(m_b)}",
            f"- 差值（v2 - v1）: {json.dumps(delta)}",
            "- 结论: " + ("v2 综合更优" if delta["pose_mAP50"] >= 0 else "v1 综合更优（需检查修正标签质量/划分一致性）"),
        ]
        for l in lines:
            ctx.note(l)
            print(l)
        ctx.finish(ok=True, extra_manifest={"v1": m_a, "v2": m_b, "delta": delta})
    except Exception as e:
        import traceback
        traceback.print_exc()
        ctx.note("异常: " + repr(e))
        ctx.finish(ok=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
