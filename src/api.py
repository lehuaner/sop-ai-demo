"""SOP-AI demo backend API.

Dev stub + dashboard read API over local run archives (docs/01 section 6).
Runtime/SOP logic lands in src/main.py; the frontend (web/) talks to these.
"""
import csv
import glob
import json
import os

from fastapi import FastAPI, HTTPException

app = FastAPI(title="SOP-AI Demo API", version="0.2.0")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/summary")
def summary():
    runs_dir = os.path.join(ROOT, "runs")
    runs = sorted(d for d in os.listdir(runs_dir) if os.path.isdir(os.path.join(runs_dir, d))) \
        if os.path.isdir(runs_dir) else []

    latest_train = None
    for d in reversed(runs):
        if not d.endswith("-train"):
            continue
        mpath = os.path.join(runs_dir, d, "manifest.json")
        if os.path.exists(mpath):
            with open(mpath, encoding="utf-8") as f:
                latest_train = json.load(f)
            break

    events = []
    for d in reversed(runs):
        p = os.path.join(runs_dir, d, "events.jsonl")
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                events = [json.loads(l) for l in f if l.strip()][-30:]
            break

    return {"runs_total": len(runs), "runs": runs[-10:],
            "latest_train": latest_train, "events": events}


@app.get("/api/train_metrics")
def train_metrics(run: str):
    safe = os.path.basename(run)
    path = os.path.join(ROOT, "runs", safe, "results.csv")
    if not os.path.exists(path):
        raise HTTPException(404, "results.csv not found for " + safe)
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {"epochs": [], "series": []}

    def col(suffix):
        key = next((k for k in rows[0] if k.endswith(suffix)), None)
        return [float(r[key]) for r in rows] if key else []

    series = []
    for suffix, name, color in (
        ("mAP50(P)", "pose mAP@50", "#4dd0a6"),
        ("mAP50(B)", "box mAP@50", "#4da3ff"),
        ("train/loss", "train loss", "#e0a458"),
    ):
        vals = col(suffix)
        if vals:
            series.append({"name": name, "type": "line", "showSymbol": False,
                           "data": vals, "lineStyle": {"width": 2, "color": color}})
    return {"epochs": list(range(1, len(rows) + 1)), "series": series}
