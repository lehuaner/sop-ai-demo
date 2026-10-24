"""Run archiving helper (docs/01 section 6).

Every train/run/convert/download/check execution archives into
runs/<YYYYMMDD-HHMMSS>-<task>/ with: README.md (human-readable Chinese
summary), manifest.json, log.txt (tee of console output).
"""
import json
import os
import shutil
import sys
import time


class _Tee:
    def __init__(self, path):
        self.file = open(path, "a", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, s):
        self.stdout.write(s)
        try:
            self.file.write(s)
            self.file.flush()
        except Exception:
            pass

    def flush(self):
        try:
            self.stdout.flush()
            self.file.flush()
        except Exception:
            pass


class RunContext:
    def __init__(self, task, params=None, root="."):
        self.root = os.path.abspath(root)
        self.name = time.strftime("%Y%m%d-%H%M%S") + "-" + task
        self.path = os.path.join(self.root, "runs", self.name)
        os.makedirs(self.path, exist_ok=True)
        self.params = params or {}
        self._summary = []
        self._tee = _Tee(os.path.join(self.path, "log.txt"))
        sys.stdout = self._tee
        sys.stderr = self._tee

    def note(self, *lines):
        self._summary.extend(str(l) for l in lines)

    def copy_in(self, src, sub=""):
        dst_dir = os.path.join(self.path, sub) if sub else self.path
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, os.path.basename(src.rstrip("/\\")))
        if os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        return dst

    def finish(self, ok=True, extra_manifest=None):
        sys.stdout = self._tee.stdout
        sys.stderr = self._tee.stdout
        manifest = {
            "run": self.name,
            "ok": bool(ok),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "argv": sys.argv,
            "params": self.params,
            "summary": self._summary,
        }
        if extra_manifest:
            manifest.update(extra_manifest)
        with open(os.path.join(self.path, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2, default=str)
        lines = [
            "# 运行归档：" + self.name,
            "",
            "- 结果：" + ("成功" if ok else "失败"),
            "- 时间：" + manifest["time"],
            "",
            "## 本次运行做了什么",
            "",
        ]
        lines += [f"- {s}" for s in self._summary] or ["-（无记录）"]
        lines += [
            "",
            "## 运行参数",
            "",
            "```json",
            json.dumps(self.params, ensure_ascii=False, indent=2, default=str),
            "```",
            "",
            "## 文件说明",
            "",
            "- README.md：本摘要（人读）",
            "- manifest.json：机器可读参数与结果",
            "- log.txt：全量控制台输出",
        ]
        with open(os.path.join(self.path, "README.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"[runctx] archived -> {self.path} (ok={ok})")
