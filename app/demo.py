"""Demo mode: camera picture with the RTMPose points on top, a live score while the try runs,
and a short summary after Enter. Used on testing day via `python -m app.ui --show-score`
(decided by Daryn, 2026-11-19); without that flag app.ui shows no score.

Same keys, pipeline and window as app.ui (Runner, Controller, Session, app.main.score_try,
app.screen), so saved tries and log.csv are identical; only the score cards are added. Every key
is also a button in the bottom bar.

    python -m app.demo --camera 0 --device cuda            # live webcam, manual Space start/stop
    python -m app.demo --fake                               # fake tries, no webcam or GPU
    python -m app.demo --video clip.mp4 --device cpu

Flow, manual mode: setup (child, age, consent) -> readiness Enter -> press 1/2/3 (task) and L/R
(leg) -> Space starts -> points + live score -> Space stops -> Enter saves -> summary on screen.

Flow, auto mode (the default): setup -> readiness Enter -> the child starts and ends the try
(Segmenter) -> "Detecting: <task> · NN%" and the live score while it runs -> when it ends, a
task guess with confidence >= AUTO_ACCEPT is saved at once (no Enter), otherwise the confirm card
asks as in manual mode. There are no task/leg buttons; the keys 1/2/3 and L/R still work as a
hidden backup and then win over the guess.

The live score is the same model applied to the frames so far, so it changes as the try goes on.
The official score is the one written at save. Screening aid only, never a diagnosis.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import deque
from datetime import date
from pathlib import Path

import numpy as np

from app.ui import Runner

LIVE_EVERY_S = 0.5  # re-score the try so far this often
LIVE_MIN_S = 1.0  # need at least this much of the try before a live score
PREVIEW_WINDOW_S = 3.5
AUTO_ACCEPT = 0.8  # group B out-of-fold: conf >= 0.8 covers 71% of tries at 95.6% accuracy (Marty, 2026-11-25)


class DemoRunner(Runner):
    """Runner plus a live score during recording and the result of the last saved try.

    In auto mode (auto_flow) it also shows the task detected so far and saves a sure try without
    the confirm card.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_flow = self.ctrl.mode == "auto"  # app.screen hides the task/leg buttons
        self.preview: deque = deque()
        self.live: dict | None = None
        self.detect: dict | None = None  # {"task", "leg", "confidence"} of the try so far
        self.result: dict | None = None
        self._last_live = -1e9

    def feed(self, item) -> None:
        now = item[0]
        if self.preview and now <= self.preview[-1][0]:
            self.preview.clear()
            self._last_live = -1e9
        super().feed(item)
        pts25 = (
            self.last_points
            if self.last_points is not None
            else np.zeros((25, 3), dtype=np.float32)
        )
        self.preview.append((now, pts25))
        while self.preview and now - self.preview[0][0] > PREVIEW_WINDOW_S:
            self.preview.popleft()

        c = self.ctrl
        if now - self._last_live < LIVE_EVERY_S:
            return
        recording = c.state == "recording" and self.buf is not None and len(self.buf) >= 2
        if recording:
            frames_src = [(b[0], b[1]) for b in self.buf]
        elif len(self.preview) >= 2:
            frames_src = list(self.preview)
        else:
            return
        if frames_src[-1][0] - frames_src[0][0] < LIVE_MIN_S:
            return
        self._last_live = now
        from app.main import score_try
        from common.format import Try
        from model.task_classifier import predict_task

        t = np.array([f[0] for f in frames_src], dtype=np.float64)
        pts = np.stack([f[1] for f in frames_src]).astype(np.float32)
        tr = Try(points=pts, t=t - t[0], meta={})
        detecting = recording and self.auto_flow
        try:
            # Auto mode, during a try: guess the task of the try so far (same 0.5 s rhythm) and
            # score that task, unless a hidden key 1/2/3 set it.
            task = c.preset_task or (None if detecting else c.choice_task)
            if task is None or detecting:
                g_task, g_leg, g_conf = predict_task(tr.points, tr.t)
                if detecting:
                    self.detect = {"task": g_task, "leg": g_leg, "confidence": float(g_conf)}
                task = task or g_task
            clues, score, conf, reasons = score_try(tr, task)
            if not recording and task == 3 and clues.get("air_s", 0.0) <= 0.0:
                self.live = {
                    "hint": "Press Space to start",
                    "detail": (
                        "Stand at ~50% height (both feet visible), press Space, "
                        "then jump forward once."
                    ),
                    "task": 3,
                }
            else:
                self.live = {"score": score, "confidence": conf, "reasons": reasons, "task": task}
        except ValueError as e:  # e.g. no torso found yet
            self.live = {"error": f"no live score yet: {e}"}

    def _do_child(self, code, age, consent):
        super()._do_child(code, age, consent)
        self.preview.clear()
        self.live, self.result, self._last_live = None, None, -1e9

    def _do_start(self, how):
        super()._do_start(how)
        self.live, self.result, self._last_live = None, None, -1e9
        self.detect = None

    def _do_stop(self):
        """Auto mode: a sure guess is saved at once; an unsure one waits on the confirm card."""
        super()._do_stop()  # guesses the task of the whole try -> confirm state
        c = self.ctrl
        if not self.auto_flow or c.state != "confirm" or c.guess is None:
            return
        gt, gl, gc = c.guess
        if gc < AUTO_ACCEPT:
            return
        # Same as Enter on the confirm card: the choice is the hidden-key preset or the guess.
        task, leg = c.choice_task, c.choice_leg
        corrected = task != gt or (task != 3 and leg != gl)
        c.state, c.guess = "idle", None
        self._do_save(task, leg, corrected)
        if self.result is not None:
            self.result["auto"] = {"task": gt, "confidence": float(gc)}

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
                "try": int(d.get("try") or len(self.saved)),
                "task": task,
                "leg": leg,
                "score": d.get("score"),
                "confidence": float(d.get("confidence") or 0.0),
                "check": bool(d.get("check_by_teacher")),
                "reasons": d.get("reasons", ""),
            }


def draw_demo(runner: DemoRunner, view: str = "camera"):
    """Camera picture with the points, live score card and last saved try (app.screen)."""
    from app.screen import render

    return render(runner, view, show_score=True).img


def run_demo(stream, out_dir, mode="manual", frame_size=(1280, 720), source="fake", view="camera"):
    """Same window loop as app.ui.run_gui (app.screen.run_window), with the live score."""
    from app.screen import run_window

    runner = DemoRunner(out_dir, mode=mode, frame_size=frame_size, source=source)
    return run_window(runner, stream, lambda r: draw_demo(r, view), paced=source == "fake")


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
    ap.add_argument("--view", choices=["camera", "stick", "neutral"], default="camera")
    ap.add_argument("--device", default=default_device)
    ap.add_argument(
        "--pose-mode",
        choices=["balanced", "lightweight", "performance"],
        default=default_pose_mode,
    )
    ap.add_argument("--out", type=Path, default=Path("runs") / f"demo_{date.today().isoformat()}")
    args = ap.parse_args(argv)
    if not args.fake and args.video is None and args.camera is None:
        from app.main import pick_camera

        args.camera = pick_camera()

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
    saved = run_demo(
        stream, args.out, mode=args.mode, frame_size=size, source=source, view=args.view
    )
    print(f"{len(saved)} tries saved in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
