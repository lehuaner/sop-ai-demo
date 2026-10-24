"""SOP-AI demo backend API (dev stub).

Real train/run logic lands in src/ per docs/01; this app exposes the
HTTP surface the Vue 3 frontend talks to.
"""
from fastapi import FastAPI

app = FastAPI(title="SOP-AI Demo API", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}
