"""Download the configured dataset subset (docs/02).

Usage:
  python tools/download_subset.py --dry-run
  python tools/download_subset.py --yes

Supports:
  coco_val2017 : val2017.zip + annotations zip from images.cocodataset.org
  assembly101  : poses@60fps per-take JSON + per-view mp4 from HF mirror
                 (gated dataset: export HF_TOKEN after access grant)

Every real (non dry-run) execution is archived via src/runctx.py.
Total pending download size is capped by max_download_gb in config.
"""
import argparse
import json
import os
import sys
import zipfile

import requests
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.runctx import RunContext  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def load_config():
    with open(os.path.join(ROOT, "config", "dataset.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def head_size(url, token=None):
    headers = dict(UA)
    if token:
        headers["Authorization"] = "Bearer " + token
    try:
        r = requests.head(url, headers=headers, allow_redirects=True, timeout=30)
        if r.status_code == 200 and "Content-Length" in r.headers:
            return int(r.headers["Content-Length"])
    except Exception:
        pass
    return None


def human(n):
    if n is None:
        return "unknown"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}"
        n /= 1024.0


def download_file(url, dest, token=None, ctx=None, attempts=3):
    import time
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    part = dest + ".part"
    for attempt in range(attempts):
        headers = dict(UA)
        if token:
            headers["Authorization"] = "Bearer " + token
        done = os.path.getsize(part) if os.path.exists(part) else 0
        if done:
            headers["Range"] = f"bytes={done}-"
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 180),
                              allow_redirects=True) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code} for {url}")
                mode = "ab" if r.status_code == 206 else "wb"
                with open(part, mode) as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            os.replace(part, dest)
            return os.path.getsize(dest)
        except requests.RequestException as e:
            if attempt == attempts - 1:
                raise
            print(f"  retry {attempt + 1}/{attempts - 1} after: {type(e).__name__}")
            time.sleep(3)
    raise RuntimeError("unreachable")


def plan_and_check(items, cap_bytes, dry, ctx=None):
    total = 0
    print("== download plan ==")
    for it in items:
        local = os.path.join(ROOT, it["dest"])
        have = os.path.getsize(local) if os.path.exists(local) else 0
        remote = it.get("size")
        if remote is None:
            remote = head_size(it["url"], it.get("token"))
        pending = max((remote or 0) - have, 0)
        total += pending
        state = "have" if have and not pending else ("resume" if have else "new")
        print(f"  [{state:>6}] {it['dest']}  remote={human(remote)} local={human(have)} pending={human(pending)}")
    print(f"  total pending = {human(total)}  cap = {human(cap_bytes)}")
    if total > cap_bytes:
        print(f"ABORT: pending {human(total)} exceeds cap {human(cap_bytes)}. "
              "Reduce takes/views in config/dataset.yaml. Cap is NOT raised.")
        return False
    if dry:
        print("dry-run: nothing downloaded.")
        return False
    return True


def fetch_coco(cfg, cap, dry, ctx):
    s = cfg["sources"]["coco_val2017"]
    items = [
        {"url": d["url"], "dest": d["dest"]} for d in s["download"]
    ]
    for it in items:
        it["size"] = head_size(it["url"])
    if not plan_and_check(items, cap, dry, ctx):
        return False
    for it in items:
        local = os.path.join(ROOT, it["dest"])
        print("downloading", it["dest"], "...")
        size = download_file(it["url"], local, ctx=ctx)
        ctx.note(f"下载 {it['dest']}（{human(size)}）")
        if it["dest"].endswith(".zip"):
            out = os.path.join(ROOT, os.path.dirname(it["dest"]))
            print("unzipping", local, "->", out)
            with zipfile.ZipFile(local) as z:
                z.extractall(out)
            ctx.note(f"解压 {os.path.basename(local)} 到 {out}")
    return True


def load_token(s):
    """HF token: env var first, then .venv/hf_token.txt (gitignored, never committed)."""
    tok = os.environ.get(s.get("token_env", "HF_TOKEN"), "")
    if tok:
        return tok
    p = os.path.join(ROOT, ".venv", "hf_token.txt")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return f.read().strip()
    return ""


def fetch_assembly101(cfg, cap, dry, ctx):
    s = cfg["sources"]["assembly101"]
    token = load_token(s)
    items = []
    if s.get("with_poses", True):
        for take in s["takes"]:
            items.append({
                "url": f"{s['resolve']}/poses%4060fps/{take}.json",
                "dest": f"datasets/raw/assembly101/poses@60fps/{take}.json",
                "token": token,
            })
    for take in s["takes"]:
        for v in s["views"]:
            fn = s["view_map"][v]
            items.append({
                "url": f"{s['resolve']}/recordings/{take}/{fn}",
                "dest": f"datasets/raw/assembly101/recordings/{take}/{fn}",
                "token": token,
            })
    for it in items:
        it["size"] = head_size(it["url"], token)
        if it["size"] is None:
            print(f"WARN: cannot HEAD size for {it['dest']} "
                  "(gated? set HF_TOKEN after granting access on huggingface.co)")
    if not plan_and_check(items, cap, dry, ctx):
        return False
    for it in items:
        local = os.path.join(ROOT, it["dest"])
        print("downloading", it["dest"], "...")
        size = download_file(it["url"], local, token, ctx)
        ctx.note(f"下载 {it['dest']}（{human(size)}）")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--yes", action="store_true", help="actually download")
    args = ap.parse_args()
    dry = args.dry_run or not args.yes
    cfg = load_config()
    src_name = cfg["active_source"]
    cap = int(float(cfg["max_download_gb"]) * (1 << 30))
    ctx = RunContext("download", params={"source": src_name, "dry_run": dry,
                                         "cap_gb": cfg["max_download_gb"]})
    ok = False
    try:
        if src_name == "coco_val2017":
            ok = fetch_coco(cfg, cap, dry, ctx)
        elif src_name == "assembly101":
            ok = fetch_assembly101(cfg, cap, dry, ctx)
        else:
            raise SystemExit(f"unknown active_source: {src_name}")
        ctx.note(f"数据源 {src_name} 下载计划{'已执行' if ok else '未执行（dry-run 或超限）'}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        ctx.note("异常: " + repr(e))
        ctx.finish(ok=False)
        sys.exit(1)
    ctx.finish(ok=True if dry else ok)
    sys.exit(0 if ok or dry else 1)


if __name__ == "__main__":
    main()
