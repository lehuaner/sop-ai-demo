"""SOP rule engine v1 (docs/01 section 5.3).

Detects step progress from person keypoints (COCO 17): rules operate on
wrist keypoints (9=right, 10=left) inside ROI rectangles. Emits events:
  step_done / step_timeout / out_of_order
Full sequence state machine for missing/wrong parts is phase 2.
"""
import os
import time

import cv2
import yaml

WRISTS = (9, 10)  # COCO right/left wrist


class SOPEngine:
    def __init__(self, yaml_path):
        with open(yaml_path, encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)
        self.source = os.path.basename(yaml_path)
        self.steps = self.cfg["steps"]
        self.rois = self.cfg.get("roi", {})
        self.alarm_cfg = self.cfg.get("alarm", {})
        self.current = 0        # index of the step currently being executed
        self.stable = 0         # consecutive frames the current rule held
        self.t_step_start = time.time()
        self.last_alarm_frame = -10**9
        self._ooo_emitted = False   # rising-edge flag: emit out_of_order once per intrusion

    # ---- rules -------------------------------------------------------
    @staticmethod
    def _in_roi(pt, r):
        if pt is None or r is None:
            return False
        x, y = pt
        return r["x1"] <= x <= r["x2"] and r["y1"] <= y <= r["y2"]

    def _rule_ok(self, expect, kpts_px):
        if kpts_px is None:
            return False
        r = self.rois.get(expect.get("roi"))
        if r is None:
            return False
        wrists = [kpts_px[i] for i in WRISTS]
        rule = expect.get("rule")
        if rule == "hands_in_roi":
            return all(self._in_roi(p, r) for p in wrists)
        if rule == "hand_near_object":
            return any(self._in_roi(p, r) for p in wrists)
        return False

    # ---- per-frame update -------------------------------------------
    def update(self, kpts_px, frame_idx):
        evs = []
        if self.current < len(self.steps):
            step = self.steps[self.current]
            if self._rule_ok(step.get("expect"), kpts_px):
                self.stable += 1
                need = step.get("expect", {}).get("stable_frames", 15)
                if self.stable >= need:
                    evs.append({"type": "step_done", "step": step["id"],
                                "name": step["name"], "frame": frame_idx, "ts": time.time()})
                    self.current += 1
                    self.stable = 0
                    self._ooo_emitted = False
                    self.t_step_start = time.time()
                    self.last_alarm_frame = frame_idx
            else:
                self.stable = 0
                timeout = self.alarm_cfg.get("step_timeout_sec", 60)
                if time.time() - self.t_step_start > timeout:
                    evs.append({"type": "step_timeout", "step": step["id"],
                                "name": step["name"], "frame": frame_idx, "ts": time.time(),
                                "message": f"step {step['id']} '{step['name']}' not done in {timeout}s"})
                    self.t_step_start = time.time()
                    self.last_alarm_frame = frame_idx
            # out-of-order probe: next step's rule firing early (rising-edge only)
            nxt = self.current + 1
            if nxt < len(self.steps) and self.alarm_cfg.get("out_of_order") == "warn":
                later = self.steps[nxt]
                if self._rule_ok(later.get("expect"), kpts_px):
                    if not self._ooo_emitted:
                        evs.append({"type": "out_of_order", "step": later["id"],
                                    "name": later["name"], "frame": frame_idx, "ts": time.time(),
                                    "message": f"step {later['id']} active before {self.steps[self.current]['id']} done"})
                        self.last_alarm_frame = frame_idx
                    self._ooo_emitted = True
                else:
                    self._ooo_emitted = False
        return evs

    # ---- overlay -----------------------------------------------------
    def draw_overlay(self, img, frame_idx):
        h, w = img.shape[:2]
        for name, r in self.rois.items():
            p1 = (int(r["x1"] * w), int(r["y1"] * h))
            p2 = (int(r["x2"] * w), int(r["y2"] * h))
            cv2.rectangle(img, p1, p2, (255, 200, 0), 1)
            cv2.putText(img, name, (p1[0], max(14, p1[1] - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
        x0, y0, lh = 12, 16, 22
        cv2.putText(img, f"SOP: {self.cfg.get('sop_id', 'demo')}",
                    (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 255, 255), 2)
        for i, s in enumerate(self.steps):
            if i < self.current:
                mark, color = "[done]", (120, 220, 120)
            elif i == self.current:
                mark, color = "[ >>> ]", (80, 200, 255)
            else:
                mark, color = "[    ]", (170, 170, 170)
            cv2.putText(img, f"{mark} step {s['id']}: {s['name']}",
                        (x0, y0 + lh * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        if frame_idx - self.last_alarm_frame < 45:
            cv2.rectangle(img, (0, h - 42), (w, h), (0, 0, 220), -1)
            cv2.putText(img, "ALARM: check SOP", (12, h - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        return img
