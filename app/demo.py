"""Demo mode: camera picture with the RTMPose points on top, a live score while the try runs,
and a short summary after Enter. Used on testing day via `python -m app.ui --show-score`
(decided by Daryn, 2026-11-19); without that flag app.ui shows no score.

Same keys, pipeline and window as app.ui (Runner, Controller, Session, app.main.score_try,
app.screen), so saved tries and log.csv are identical; only the score cards are added. Every key
is also a button in the bottom bar.

    python -m app.demo --camera 0 --device cuda            # live webcam, gesture flow
    python -m app.demo --fake                               # fake tries, no webcam or GPU
    python -m app.demo --video clip.mp4 --device cpu
    python -m live.gesture --download                       # once: the open-palm hand model

Flow, gesture mode (the default, also for app.ui --show-score and Open GUI.command; Daryn
2026-11-25): setup -> readiness Enter -> the child (the tracked person) shows an open palm to the
camera and holds it -> big 3-2-1 in the middle -> recording. One recording can hold several tasks in
a row (stand on one leg, hop, jump): live.bouts cuts it into attempts, and after each attempt a card
shows the task, its score and its time ("Task 1 · 2 Perfect · 3.2 s on one leg"); attempts that are
not a task (walking back) are not shown. Crossed arms (an X), held, stop the recording: every
attempt is saved as its own try (same JSON and log.csv as before) and a summary lists them. Without
the palm nothing is recorded, but the attempts are still detected and shown ("not recorded").
Space starts and stops at once, Backspace cancels, P pauses the gesture start.

Flow, manual mode: setup (child, age, consent) -> readiness Enter -> press 1/2/3 (task) and L/R
(leg) -> Space starts -> points + live score -> Space stops -> Enter saves -> summary on screen.

Flow, auto mode (--mode auto, the old default): setup -> readiness Enter -> the child starts and ends the try
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
COUNTDOWN_S = 3.0  # gesture mode: 3-2-1 before the recording starts
INTERMEZZO_S = 2.5  # how long the card of a finished attempt stays
SUMMARY_S = 6.0  # how long the list of saved tries stays after a recording
PREVIEW_KEEP = 3  # attempts listed while not recording
# An attempt counts as the task only if its main clue reached this (GUESS, Daryn 2026-11-25; e.g.
# walking back to the start line is not a task). Tune on the live demo.
DONE_CLUE = {1: ("one_foot_s", 1.0), 2: ("hops", 2.0), 3: ("air_s", 0.0)}


class DemoRunner(Runner):
    """Runner plus a live score during recording and the result of the last saved try.

    In auto mode (auto_flow) it also shows the task detected so far and saves a sure try without
    the confirm card. In gesture mode (gesture_flow) a palm starts and an X stops one recording
    that can hold several tasks (module docstring).
    hand: a live.gesture.HandChecker for the open-palm check, or None (a raised hand is enough).
    """

    def __init__(self, *args, hand=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.gesture_flow = self.ctrl.mode == "gesture"
        # app.screen hides the task/leg buttons in both
        self.auto_flow = self.ctrl.mode in ("auto", "gesture")
        self.preview: deque = deque()
        self.live: dict | None = None
        self.detect: dict | None = None  # {"task", "leg", "confidence"} of the try so far
        self.result: dict | None = None
        self._last_live = -1e9
        # gesture flow (times are stream seconds, like the frames)
        self.now = 0.0
        self.hand = hand
        self.gesture = self.splitter = None
        if self.gesture_flow:
            from live.bouts import BoutSplitter
            from live.gesture import GestureDetector

            self.gesture = GestureDetector(hand_checker=hand)
            self.splitter = BoutSplitter(pre_roll_s=self.pre_roll_s)
        self.countdown_until: float | None = None
        self.intermezzo: dict | None = None  # card of the attempt that just ended (+ "until")
        self.rec_bouts: list[dict] = []  # attempts of the recording in progress
        self.preview_bouts: deque = deque(maxlen=PREVIEW_KEEP)  # attempts while not recording
        self.summary: dict | None = None  # {"rows", "until"} after a recording
        self._hand_note = False
        self._cut_s: float | None = None  # stream time the X began (cut off the recording)

    # ------------------------------------------------------------ gesture flow
    def countdown_left(self) -> float | None:
        if self.ctrl.state != "countdown" or self.countdown_until is None:
            return None
        return max(0.0, self.countdown_until - self.now)

    def card(self) -> dict | None:
        """The attempt card to show now, or None."""
        r = self.intermezzo
        return r if r is not None and self.now < r["until"] else None

    def summary_card(self) -> dict | None:
        r = self.summary
        if r is None or self.now >= r["until"] or self.ctrl.state not in ("idle", "ready"):
            return None
        return r

    def _gesture_feed(self, item) -> None:
        t, frame = item[0], item[3]
        c = self.ctrl
        pts = self.last_points
        want = None
        if c.state in ("readiness", "idle", "ready") and not c.paused:
            want = "palm"
        elif c.state == "recording":
            want = "cross"
        ev = self.gesture.update(pts, t, want, frame)
        if self.gesture.status == "no hand model" and not self._hand_note:
            self._hand_note = True
            c.message = "Hand model not available: raising one hand is enough to start."
        if ev == "cross":
            self._cut_s = self.gesture.cross_since
        if ev is not None:
            self._apply(c.gesture(ev))
        if (
            c.state == "countdown"
            and self.countdown_until is not None
            and t >= self.countdown_until
        ):
            self._apply(c.countdown_done())
        if c.state in ("idle", "ready", "countdown", "recording"):
            raw = self.pre[-1][2] if self.pre else None
            for bout in self.splitter.update(t, pts, raw):
                self._attempt_done(bout, recorded=c.state == "recording")
        self._live_attempt(t)

    def _judge(self, bout) -> dict | None:
        """Task, score and time of one attempt; None if it is not one of the three tasks."""
        from app.main import score_try
        from common.format import Try
        from model.predict import CHECK_BY_TEACHER_BELOW
        from model.task_classifier import predict_task

        c = self.ctrl
        t = bout.t
        tr = Try(points=bout.points, t=t - t[0], meta={})
        try:
            g_task, g_leg, g_conf = predict_task(tr.points, tr.t)
            task = c.preset_task or g_task
            leg = c.preset_leg or g_leg
            clues, score, conf, reasons = score_try(tr, task)
        except ValueError:  # e.g. no torso in the attempt
            return None
        key, least = DONE_CLUE[task]
        value = float(clues.get(key, 0.0))
        if value < least or (least == 0.0 and value <= 0.0):
            return None
        if task == 1:
            headline = f"{value:.1f} s on one leg"
        elif task == 2:
            headline = f"{int(value)} hops · {bout.duration_s:.1f} s"
        else:
            headline = f"Jump · {bout.duration_s:.1f} s"
        return {
            "task": task,
            "leg": leg,
            "guess": (g_task, g_leg, float(g_conf)),
            "pinned": c.preset_task is not None,
            "score": int(score),
            "confidence": float(conf),
            "check": conf < CHECK_BY_TEACHER_BELOW,
            "reasons": reasons,
            "duration_s": bout.duration_s,
            "headline": headline,
            "bout": bout,
        }

    def _attempt_done(self, bout, recorded: bool) -> None:
        r = self._judge(bout)
        if r is None:
            return
        r["recorded"] = recorded
        if recorded:
            self.rec_bouts.append(r)
        else:
            self.preview_bouts.append(r)
        self.intermezzo = {**r, "since": self.now, "until": self.now + INTERMEZZO_S}
        self.summary = None
        self.live = self.detect = None
        self._last_live = -1e9

    def _live_attempt(self, now: float) -> None:
        """Live score and task of the attempt in progress (every LIVE_EVERY_S)."""
        frames = self.splitter.current()
        if not frames or frames[-1][0] - frames[0][0] < LIVE_MIN_S:
            if not frames:
                self.live = self.detect = None
            return
        if now - self._last_live < LIVE_EVERY_S:
            return
        self._last_live = now
        from app.main import score_try
        from common.format import Try
        from model.predict import predict_proba
        from model.task_classifier import predict_task

        c = self.ctrl
        t = np.array([f[0] for f in frames], dtype=np.float64)
        pts = np.stack([f[1] for f in frames]).astype(np.float32)
        tr = Try(points=pts, t=t - t[0], meta={})
        try:
            if c.preset_task is not None:
                task, self.detect = c.preset_task, None
            else:
                g_task, g_leg, g_conf = predict_task(tr.points, tr.t)
                self.detect = {"task": g_task, "leg": g_leg, "confidence": float(g_conf)}
                task = g_task
            clues, score, conf, reasons = score_try(tr, task)
            self.live = {
                "score": score,
                "confidence": conf,
                "reasons": reasons,
                "task": task,
                "proba": predict_proba(task, clues),  # the evaluation bar (app.screen)
            }
        except ValueError as e:  # e.g. no torso found yet
            self.live = {"error": f"no live score yet: {e}"}

    def _do_countdown(self):
        self.countdown_until = self.now + COUNTDOWN_S
        self.summary = self.intermezzo = None

    def _do_cancel_countdown(self):
        self.countdown_until = None

    def _finish_recording(self) -> None:
        """X or Space: every attempt of the recording becomes its own saved try."""
        from common.format import Try

        c = self.ctrl
        cut, self._cut_s = self._cut_s, None
        self.buf = None
        for bout in self.splitter.flush(until_t=cut):
            r = self._judge(bout)
            if r is not None:
                r["recorded"] = True
                self.rec_bouts.append(r)
        results, self.rec_bouts = self.rec_bouts, []
        c.state = "idle"
        self.live = self.detect = self.intermezzo = None
        if not results or self.session is None:
            self._close_video(keep=False)
            c.message = "No task found in this recording (nothing saved)."
            self.summary = {"rows": [], "since": self.now, "until": self.now + SUMMARY_S}
            return
        video_path = self._close_video(keep=True)
        rows = []
        for k, r in enumerate(results):
            bout = r["bout"]
            t = bout.t
            meta = {
                "source": self.source,
                "mode": "gesture",
                "fps": 30.0,
                "width": self.frame_size[0],
                "height": self.frame_size[1],
                "t0_stream_s": float(t[0]),
                "attempt": k + 1,
                "attempts_in_recording": len(results),
            }
            tr = Try(points=bout.points, t=t - t[0], meta=meta)
            p = {"tr": tr, "raw26": bout.raw26, "guess": r["guess"]}
            gt, gl, _gc = r["guess"]
            task, leg = r["task"], r["leg"]
            corrected = task != gt or (task != 3 and leg != gl)
            path = self._write_try(p, task, leg, corrected, video_path)
            if path is None:
                continue
            try:
                d = json.loads(Path(path).read_text(encoding="utf-8"))
            except (OSError, ValueError):
                d = {}
            rows.append(
                {
                    **{k2: v for k2, v in r.items() if k2 != "bout"},
                    "try": int(d.get("try") or 0),
                    "score": d.get("score", r["score"]),
                    "confidence": float(d.get("confidence") or r["confidence"]),
                    "check": bool(d.get("check_by_teacher", r["check"])),
                }
            )
        self.summary = {"rows": rows, "since": self.now, "until": self.now + SUMMARY_S}
        c.message = f"Saved {len(rows)} {'try' if len(rows) == 1 else 'tries'} from this recording."
        if video_path is not None:
            c.message += " Video saved."
        elif self.video_error:
            c.message += f" {self.video_error} (points are saved)."
        self.seg.reset()

    def feed(self, item) -> None:
        now = item[0]
        if self.preview and now <= self.preview[-1][0]:
            self.preview.clear()
            self._last_live = -1e9
            if self.gesture_flow:
                self.splitter.reset()
                self.gesture.reset()
        super().feed(item)
        self.now = now
        if self.gesture_flow:
            self._gesture_feed(item)
            return
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
        from model.predict import predict_proba
        from model.task_classifier import predict_task

        t = np.array([f[0] for f in frames_src], dtype=np.float64)
        pts = np.stack([f[1] for f in frames_src]).astype(np.float32)
        tr = Try(points=pts, t=t - t[0], meta={})
        detecting = recording and self.auto_flow
        try:
            if c.preset_task is not None:
                task = c.preset_task
                if detecting:
                    self.detect = None
            else:
                g_task, g_leg, g_conf = predict_task(tr.points, tr.t)
                if detecting:
                    self.detect = {"task": g_task, "leg": g_leg, "confidence": float(g_conf)}
                task = g_task
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
                self.live = {
                    "score": score,
                    "confidence": conf,
                    "reasons": reasons,
                    "task": task,
                    "proba": predict_proba(task, clues),
                }
        except ValueError as e:  # e.g. no torso found yet
            self.live = {"error": f"no live score yet: {e}"}

    def press(self, k: str | None, now: float | None = None) -> None:
        prev_task = self.ctrl.preset_task
        super().press(k, now)
        if self.ctrl.preset_task != prev_task:
            self._last_live = -1e9
            if self.ctrl.preset_task is not None:
                self.detect = None

    def _do_child(self, code, age, consent):
        super()._do_child(code, age, consent)
        self.preview.clear()
        self.live, self.result, self._last_live = None, None, -1e9
        self.detect = None
        if self.gesture_flow:
            self.splitter.reset()
            self.gesture.reset()
            self.rec_bouts, self.summary, self.intermezzo = [], None, None
            self.preview_bouts.clear()
            self.countdown_until = None

    def _do_pause(self, paused: bool):
        super()._do_pause(paused)
        if self.gesture_flow:
            self.gesture.reset()

    def _do_start(self, how):
        super()._do_start(how)
        self.live, self.result, self._last_live = None, None, -1e9
        self.detect = None
        if self.gesture_flow:
            self.rec_bouts, self.summary, self.intermezzo = [], None, None
            self.countdown_until = None
            self.splitter.restart()
            self.gesture.reset()

    def _do_cancel(self):
        super()._do_cancel()
        if self.gesture_flow:
            self.rec_bouts, self.intermezzo = [], None
            self.live = self.detect = None
            self.splitter.restart()

    def close(self):
        super().close()
        if self.hand is not None:
            self.hand.close()

    def _do_stop(self):
        if self.gesture_flow:
            self._finish_recording()
            return
        super()._do_stop()
        c = self.ctrl
        if not self.auto_flow or c.state != "confirm" or c.guess is None:
            return
        gt, gl, gc = c.guess
        pinned = c.preset_task is not None
        if not pinned and gc < AUTO_ACCEPT:
            return
        task, leg = c.choice_task, c.choice_leg
        corrected = task != gt or (task != 3 and leg != gl)
        c.state, c.guess = "idle", None
        self._do_save(task, leg, corrected)
        if self.result is not None:
            if pinned:
                self.result["pinned"] = True
            else:
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


def run_demo(
    stream,
    out_dir,
    mode="manual",
    frame_size=(1280, 720),
    source="fake",
    view="camera",
    device="cpu",
):
    """Same window loop as app.ui.run_gui (app.screen.run_window), with the live score.

    Gesture mode on a camera or video also starts the hand model (open-palm check) in its thread.
    """
    from app.screen import run_window

    hand = None
    if mode == "gesture" and source != "fake":
        from live.gesture import HandChecker

        hand = HandChecker(device=device)
    runner = DemoRunner(out_dir, mode=mode, frame_size=frame_size, source=source, hand=hand)
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
    ap.add_argument("--mode", choices=["gesture", "auto", "manual"], default="gesture")
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
        stream,
        args.out,
        mode=args.mode,
        frame_size=size,
        source=source,
        view=args.view,
        device=args.device,
    )
    print(f"{len(saved)} tries saved in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
