"""Rotate old run archives: keep the newest N runs, move the rest into
archive/runs/, append one row to archive/ARCHIVE_LOG.md, then commit with
the archive: prefix (docs/00 section 4-5: an archive action == one commit).

  python tools/archive_runs.py [--keep 5]
"""
import argparse
import datetime
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
ARCHIVE = os.path.join(ROOT, "archive", "runs")
LOG = os.path.join(ROOT, "archive", "ARCHIVE_LOG.md")


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=5)
    args = ap.parse_args()
    if not os.path.isdir(RUNS):
        print("no runs/ directory, nothing to archive")
        return
    names = sorted(n for n in os.listdir(RUNS) if os.path.isdir(os.path.join(RUNS, n)))
    stale = names[:-args.keep] if len(names) > args.keep else []
    if not stale:
        print(f"{len(names)} runs, all within retention {args.keep}, nothing to archive")
        return
    os.makedirs(ARCHIVE, exist_ok=True)
    moved = []
    for n in stale:
        dst = os.path.join(ARCHIVE, n)
        if os.path.exists(dst):
            print("skip (already archived):", n)
            continue
        shutil.move(os.path.join(RUNS, n), dst)
        moved.append(n)
        print("archived:", n)
    with open(LOG, "a", encoding="utf-8") as f:
        for n in moved:
            f.write(f"| {datetime.date.today().isoformat()} | {n} | runs/{n} | 超过保留数({args.keep})自动轮转 |\n")
    git("add", "archive/ARCHIVE_LOG.md")
    r = git("commit", "-m", f"archive: rotate runs (kept latest {args.keep}, moved {len(moved)})")
    if r.returncode == 0:
        print("git commit (archive:) created")
    else:
        print("git commit skipped:", r.stderr.strip()[:200])


if __name__ == "__main__":
    main()
