"""Demo mode: camera picture with the RTMPose points on top, a live score while the try runs,
and a short summary after Enter. Used on testing day via `python -m app.ui --show-score`
(decided by Daryn, 2026-11-19); without that flag app.ui shows no score.

Same keys and pipeline as app.ui (Runner, Controller, Session, app.main.score_try), so saved tries
and log.csv are identical; only the picture differs. app/ui.py is not changed.

    python -m app.demo --camera 0 --device cuda            # live webcam, manual Space start/stop
    python -m app.demo --fake                               # fake tries, no webcam or GPU
    python -m app.demo --video clip.mp4 --device cpu

Flow: setup (child, age, consent) -> readiness Enter -> press 1/2/3 (task) and L/R (leg)
-> Space starts -> points + live score -> Space stops -> Enter saves -> summary on screen.

The live score is the same model applied to the frames so far, so it changes as the try goes on.
The official score is the one written at Enter. Screening aid only, never a diagnosis.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

from app.ui import (
    SCREEN_H,
    SCREEN_W,
    STATE_COLOURS,
    STATUS_H,
    Runner,
    big_lines,
    key_name,
    status_lines,
)

SCORE_NAMES = {0: "Bad", 1: "Good", 2: "Perfect"}
SCORE_COLOURS = {0: (60, 60, 240), 1: (0, 200, 255), 2: (80, 220, 80)}  # BGR
LIVE_EVERY_S = 0.5  # re-score the try so far this often
LIVE_MIN_S = 1.0  # need at least this much of the try before a live score


class DemoRunner(Runner):
    """Runner plus a live score during recording and the result of the last saved try."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.live: dict | None = None  # {"score", "confidence", "reasons", "task"}
        self.result: dict | None = None  # last saved try, read back from its JSON
        self._last_live = -1e9

    def feed(self, item) -> None:
        super().feed(item)
        c = self.ctrl
        now = item[0]
        if now - self._last_live < LIVE_EVERY_S:
            return
        if c.state == "recording" and self.buf is not None and len(self.buf) >= 2:
            frames_src = [(b[0], b[1]) for b in self.buf]
        elif len(self.hist) >= 2:
            frames_src = list(self.hist)
        else:
            return
        if frames_src[-1][0] - frames_src[0][0] < LIVE_MIN_S:
            return
        self._last_live = now
        from app.main import score_try
        from common.format import Try
        from model.task_classifier import predict_task_from_clues

        t = np.array([f[0] for f in frames_src], dtype=np.float64)
        pts = np.stack([f[1] for f in frames_src]).astype(np.float32)
        tr = Try(points=pts, t=t - t[0], meta={})
        try:
            task = c.preset_task or c.choice_task
            if task is None:
                task, _leg, _tconf = predict_task_from_clues(tr.points, tr.t)
            _clues, score, conf, reasons = score_try(tr, task)
            self.live = {"score": score, "confidence": conf, "reasons": reasons, "task": task}
        except ValueError as e:  # e.g. no torso found yet
            self.live = {"error": f"no live score yet: {e}"}

    def _do_start(self, how):
        super()._do_start(how)
        self.live, self.result, self._last_live = None, None, -1e9

    def _do_save(self, task, leg, corrected):
        n_before = len(self.saved)
        super()._do_save(task, leg, corrected)
        self.live = None
        if len(self.saved) > n_before:
            try:
                d = json.loads(Path(self.saved[-1]).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return
            self.result = {
                "task": task,
                "leg": leg,
                "score": d.get("score"),
                "confidence": float(d.get("confidence") or 0.0),
                "check": bool(d.get("check_by_teacher")),
                "reasons": d.get("reasons", ""),
            }


def _text(img, line, xy, size, colour, thick=2):
    import cv2

    cv2.putText(img, line, xy, cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), thick + 4, cv2.LINE_AA)
    cv2.putText(img, line, xy, cv2.FONT_HERSHEY_SIMPLEX, size, colour, thick, cv2.LINE_AA)


def _panel(img, x, y, w, h):
    sub = img[y : y + h, x : x + w]
    sub[:] = (sub * 0.35).astype(np.uint8)


def summary_lines(r: dict) -> tuple[list[str], tuple[int, int, int]]:
    """Plain words for the last saved try. Reference only, never a diagnosis."""
    from common.body25 import TASKS

    head = f"Task {r['task']} ({TASKS[r['task']]}), leg {r['leg']}"
    if r["check"] or r["score"] is None:
        return (
            [head, "Model is unsure: check by teacher", r["reasons"]],
            (200, 200, 200),
        )
    s = int(r["score"])
    words = {
        0: "did not complete the task this time",
        1: "completed the task with some difficulty",
        2: "completed the task smoothly",
    }[s]
    return (
        [
            head,
            f"Score {s} {SCORE_NAMES[s]} (sure {r['confidence']:.0%}): {words}",
            r["reasons"],
            "Reference only, not a diagnosis.",
        ],
        SCORE_COLOURS[s],
    )


def draw_demo(runner: DemoRunner):
    """Mirrored camera picture with the points drawn on it, live score, status bar."""
    import cv2

    from viz.replay import draw_figure

    c = runner.ctrl
    img = np.full((SCREEN_H + STATUS_H, SCREEN_W, 3), 25, dtype=np.uint8)
    area = img[:SCREEN_H]
    fw, fh = runner.frame_size
    s = min(SCREEN_W / fw, SCREEN_H / fh)
    dw, dh = int(round(fw * s)), int(round(fh * s))
    ox, oy = (SCREEN_W - dw) // 2, (SCREEN_H - dh) // 2
    if runner.last_frame is not None:
        area[oy : oy + dh, ox : ox + dw] = cv2.resize(cv2.flip(runner.last_frame, 1), (dw, dh))
    if runner.last_points is not None:
        pts = runner.last_points.copy()
        pts[:, 0] = np.where(pts[:, 2] > 0, fw - pts[:, 0], 0)  # mirror like the picture
        draw_figure(area, pts, s, ox, oy, labels=False, thick=3)

    colour = STATE_COLOURS.get(c.state, (200, 200, 200))
    cv2.rectangle(img, (0, 0), (SCREEN_W - 1, SCREEN_H - 1), colour, 6)
    _text(
        img,
        "Screening aid, not a diagnosis",
        (SCREEN_W - 470, SCREEN_H - 20),
        0.7,
        (255, 255, 255),
    )

    y = 80
    for i, line in enumerate(big_lines(c)):
        _text(img, line, (40, y), 1.3 if i == 0 else 0.9, colour)
        y += 55 if i == 0 else 42

    live = runner.live
    if live:
        _panel(img, SCREEN_W - 580, 20, 560, 130)
        if "error" in live:
            _text(img, live["error"][:50], (SCREEN_W - 565, 70), 0.6, (230, 230, 230), 1)
        else:
            sc = int(live["score"])
            tk = int(live.get("task") or 1)
            conf = float(live.get("confidence") or 0.0)
            label = f"Task {tk} LIVE: {sc} {SCORE_NAMES[sc]} ({conf:.0%})"
            _text(img, label, (SCREEN_W - 565, 70), 1.05, SCORE_COLOURS[sc], 3)
            _text(img, live["reasons"][:58], (SCREEN_W - 565, 115), 0.6, (230, 230, 230), 1)

    if runner.result and c.state in ("idle", "ready"):
        lines, col = summary_lines(runner.result)
        _panel(img, 20, SCREEN_H - 210, SCREEN_W - 40, 180)
        yy = SCREEN_H - 170
        for i, line in enumerate(lines):
            _text(
                img,
                line[:90],
                (40, yy),
                1.0 if i == 1 else 0.7,
                col if i == 1 else (230, 230, 230),
                2 if i == 1 else 1,
            )
            yy += 42

    y = SCREEN_H + 26
    for i, line in enumerate(status_lines(c)):
        cv2.putText(
            img,
            line[:140],
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            colour if i == 0 else (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        y += 27
    return img


def run_demo(stream, out_dir, mode="manual", frame_size=(1280, 720), source="fake"):
    import cv2

    from app.ui import PoseWorker

    runner = DemoRunner(out_dir, mode=mode, frame_size=frame_size, source=source)
    name = "Gross motor screening (screening aid, not a diagnosis)"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    worker = PoseWorker(stream, paced=source == "fake")
    try:
        while not runner.quit:
            for item in worker.take(wait_s=0.005):
                runner.feed(item)
            if worker.error is not None:
                runner.ctrl.message = (
                    f"Camera stopped ({worker.error}). Earlier tries are saved. Backspace if a "
                    "try is open, then Q; check the camera and restart with the same child code."
                )
            elif worker.ended and worker.items.empty():
                runner.ctrl.message = "The video ended. Press Q to quit."
            cv2.imshow(name, draw_demo(runner))
            runner.press(key_name(cv2.waitKeyEx(1)))
            if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        stopped = worker.stop()
        runner.close()  # an unconfirmed video clip is discarded
        stream.close()
        if not stopped:
            worker.stop(timeout=2.0)
        cv2.destroyAllWindows()
    return runner.saved


def main(argv=None) -> int:
    default_device = "cpu" if sys.platform == "darwin" else "cuda"
    default_pose_mode = "lightweight" if sys.platform == "darwin" else "balanced"
    ap = argparse.ArgumentParser(
        description="Window with live score and summary (screening aid, not a diagnosis)"
    )
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--fake", action="store_true")
    src.add_argument("--video", type=Path)
    src.add_argument("--camera", type=int)
    ap.add_argument("--mode", choices=["auto", "manual"], default="auto")
    ap.add_argument("--device", default=default_device)
    ap.add_argument(
        "--pose-mode",
        choices=["balanced", "lightweight", "performance"],
        default=default_pose_mode,
    )
    ap.add_argument("--out", type=Path, default=Path("runs") / f"demo_{date.today().isoformat()}")
    args = ap.parse_args(argv)
    if not args.fake and args.video is None and args.camera is None:
        args.camera = 0

    if args.fake:
        from app.ui import FakeLoopStream

        stream, size, source = FakeLoopStream(), (1280, 720), "fake"
    else:
        import cv2

        from live.pose import PoseStream

        source_arg = str(args.video) if args.video else args.camera
        stream = PoseStream(source=source_arg, mode=args.pose_mode, device=args.device)
        w = int(stream.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        h = int(stream.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        size, source = (w, h), ("video" if args.video else "webcam")
    saved = run_demo(stream, args.out, mode=args.mode, frame_size=size, source=source)
    print(f"{len(saved)} tries saved in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
