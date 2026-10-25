"""Stage prelabel (docs/02 §7, decision #17): expand official coarse action
segments into frame-level stage labels over the sop_hands dataset frames.

Joins: split.json (frame -> split) x frames_manifest.json (frame -> source
60fps frame index fi) x coarse_labels (30fps segments, verb -> stage via
config stage_map). Output: datasets/converted/sop_stages/stage_labels.csv
(a copy lands in datasets/stage_review/ for human comparison).
"""
import csv
import json
import os
import sys

import yaml

sys_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, sys_path)
from src.runctx import RunContext  # noqa: E402

ROOT = os.path.abspath(sys_path)


def main():
    with open(os.path.join(ROOT, "config", "dataset.yaml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    stage_map = cfg["stage_map"]
    verb2stage = {v: s for s, vs in stage_map.items() for v in vs}
    lmap = cfg["coarse_label_map"]

    actions = {r["action_cls"]: r["verb_cls"]
               for r in csv.DictReader(open(
                   os.path.join(ROOT, "datasets/raw/assembly101/annotations/actions.csv"),
                   encoding="utf-8"))}

    hands_dir = os.path.join(ROOT, cfg["converted_dir"].replace("sop_pose", "sop_hands"))
    split = json.load(open(os.path.join(hands_dir, "split.json"), encoding="utf-8"))["splits"]
    manifest = {(m["take"], m["si"]): m["fi"]
                for m in json.load(open(os.path.join(hands_dir, "frames_manifest.json"), encoding="utf-8"))}

    ctx = RunContext("prelabel_stages", params={"stages": list(stage_map.keys())})
    try:
        rows = []
        for split_name, img_list in split.items():
            for img_rel in img_list:
                name = os.path.basename(img_rel)[:-4]          # <take>_f%06d
                take, si = name.rsplit("_f", 1)
                fi = manifest.get((take, int(si)))
                if fi is None:
                    continue
                label_stems = lmap.get(take, [])
                if isinstance(label_stems, str):
                    label_stems = [label_stems]
                f30 = fi // 2                                   # 60fps -> 30fps annotation clock
                stage, act = ("", "") if not label_stems else ("idle", "")
                for stem in label_stems:
                    seg_path = os.path.join(ROOT, "datasets/raw/assembly101/annotations", stem + ".txt")
                    for line in open(seg_path, encoding="utf-8"):
                        p = line.split()
                        if len(p) >= 3 and int(p[0]) <= f30 < int(p[1]):
                            act = " ".join(p[2:])          # 段内含空格的完整动作名（verb noun）
                            stage = verb2stage.get(actions.get(act, ""), "idle")
                            break
                rows.append((split_name, img_rel, take, stage, act))

        out_dir = os.path.join(ROOT, "datasets/converted/sop_stages")
        os.makedirs(out_dir, exist_ok=True)
        csv_path = os.path.join(out_dir, "stage_labels.csv")
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["split", "image", "take", "stage", "action"])
            w.writerows(rows)
        shutil_copy = os.path.join(ROOT, "datasets/stage_review/00_机器伪标签.csv")
        os.makedirs(os.path.dirname(shutil_copy), exist_ok=True)
        with open(shutil_copy, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["split", "image", "take", "stage", "action"])
            w.writerows(rows)

        dist = {}
        for _, _, _, st, _ in rows:
            dist[st] = dist.get(st, 0) + 1
        ctx.note(f"帧级阶段伪标签 {len(rows)} 条 -> {os.path.relpath(csv_path, ROOT)}（工作台副本 00_机器伪标签.csv）")
        for k in sorted(dist):
            ctx.note(f"  {k}: {dist[k]} 帧 ({dist[k]*100//len(rows)}%)")
        ctx.note("阶段定义见 config stage_map / docs/02 §7；人工分段结果回来后以此 CSV 为底进行合并")
        ctx.finish(ok=True, extra_manifest={"distribution": dist})
    except Exception as e:
        import traceback
        traceback.print_exc()
        ctx.note("异常: " + repr(e))
        ctx.finish(ok=False)
        sys.exit(1)


if __name__ == "__main__":
    main()
