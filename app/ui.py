"""The GUI window for testing day. Owner: Freya (#37 skeleton, #40 testing mode, #41 automatic
mode, #42 crash-safe).

One OpenCV window on the live laptop (no installer, web or phone app). Testing mode only: the
screen NEVER shows a score or the timeline (the screen faces the child; KEE scores on paper).

    python -m app.ui --fake                    # fake tries in a loop (no webcam, no GPU)
    python -m app.ui --fake --mode manual      # Space starts and stops each try
    python -m app.ui --video clip.mp4 --device cpu   # a recorded clip (needs rtmlib)
    python -m app.ui --camera 0 --device cuda        # live webcam (needs rtmlib)
    python -m app.ui --camera 0 --device cpu --pose-mode lightweight   # laptop without NVIDIA
    options: --view stick|camera|neutral  --out runs/2026-11-29

Screens (Controller.state):
    setup      type child code, Tab/Enter, age in months, Tab/Enter, consent level 1/2/3, Enter
    readiness  live.quality.check in plain words; Enter starts (also allowed with problems)
    idle       waiting for the child to stand still at the start line
    ready      (automatic mode) the child stands still: the next movement starts the try
    recording  the try is being recorded
    confirm    shows the guessed task and leg: Enter confirms, 1/2/3 and L/R correct,
               Backspace cancels the try

Keys: 1/2/3 task, L/R leg, Space manual start/stop (fallback, also in automatic mode),
Enter confirm, Backspace cancel the try, N next child, V video on/off (consent level 3 only),
Q quit (Esc in the setup screen).

Split: Controller = keys and states only (pure, tested without a window). Runner = the pipeline
for one frame (pose -> pick -> Segmenter -> task guess -> score -> Session.save_try), with no
drawing. draw_screen = the picture. app/main.py (Daryn) can import Runner when the parts are
joined on Nov 22; the scoring itself is app.main.score_try, so rescore.py gives the same scores.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from datetime import date
from pathlib import Path

import numpy as np

from app.session import AGE_RANGE_MONTHS, Session, age_in_range
from common.body25 import TASKS
from common.format import Try

STATES = ("setup", "readiness", "idle", "ready", "recording", "confirm")
DEBOUNCE_S = 0.35  # the same key again within this time is ignored (no double saves)
READINESS_WINDOW_S = 1.5  # points used for the readiness check
FIELDS = ("child code", "age in months", "consent level (1/2/3)")


def key_name(code: int) -> str | None:
    """cv2.waitKeyEx code -> 'ENTER', 'SPACE', 'BACKSPACE', 'TAB', 'ESC', or one upper-case char."""
    if code is None or code < 0:
        return None
    if code in (13, 10):
        return "ENTER"
    if code == 32:
        return "SPACE"
    if code in (8, 127):
        return "BACKSPACE"
    if code == 9:
        return "TAB"
    if code == 27:
        return "ESC"
    c = code & 0xFF
    if 33 <= c <= 126 and code < 0x100:
        return chr(c).upper()
    return None


class Controller:
    """Keys and states. Methods return a list of actions for the Runner:

    ("child", code, age, consent)   new child entered          ("start", how)  start recording
    ("stop",)                       stop recording, guess      ("cancel",)     drop this try
    ("save", task, leg, corrected)  operator confirmed         ("video", on)   V pressed
    ("ready_ok",)                   readiness accepted         ("quit",)
    It never holds a score, so nothing it shows can contain one.
    """

    def __init__(self, mode: str = "auto", debounce_s: float = DEBOUNCE_S):
        if mode not in ("auto", "manual"):
            raise ValueError("mode must be 'auto' or 'manual'")
        self.mode, self.debounce_s = mode, debounce_s
        self.state = "setup"
        self.form = ["", "", ""]
        self.field = 0
        self.child: str | None = None
        self.age: int | None = None
        self.consent: int | None = None
        self.video = False
        self.preset_task: int | None = None
        self.preset_leg: str | None = None
        self.manual = False  # the current recording was started with Space
        self.guess: tuple[int, str, float] | None = None
        self.choice_task: int | None = None
        self.choice_leg: str | None = None
        self.problems: list[str] = []
        self.message = "Type the child code (e.g. K03), then Enter."
        self.tries_saved = 0
        self._last = ("", -math.inf)

    # ------------------------------------------------------------------ keys
    def key(self, k: str | None, now: float | None = None) -> list[tuple]:
        if not k:
            return []
        now = time.monotonic() if now is None else now
        if self.state == "setup":
            return self._setup_key(k)
        if k == self._last[0] and now - self._last[1] < self.debounce_s:
            return []  # held or bounced key
        self._last = (k, now)
        if k == "Q":
            if self.state in ("recording", "confirm"):
                self.message = "Finish this try first: Enter confirms, Backspace cancels."
                return []
            return [("quit",)]
        if k == "V":
            return self._video()
        if k == "N":
            if self.state in ("recording", "confirm"):
                self.message = "Finish this try first: Enter confirms, Backspace cancels."
                return []
            self._new_form()
            return []
        handler = getattr(self, f"_{self.state}_key")
        return handler(k)

    def _new_form(self):
        self.state, self.form, self.field = "setup", ["", "", ""], 0
        self.child = self.age = self.consent = None
        self.video, self.preset_task, self.preset_leg = False, None, None
        self.tries_saved = 0
        self.message = "Next child: type the child code, then Enter."

    def _setup_key(self, k: str) -> list[tuple]:
        if k == "ESC":
            return [("quit",)]
        if k == "BACKSPACE":
            self.form[self.field] = self.form[self.field][:-1]
            return []
        if k in ("TAB", "ENTER"):
            err = self._field_error(self.field)
            if err:
                self.message = err
                return []
            if self.field < 2:
                self.field += 1
                self.message = f"Type the {FIELDS[self.field]}, then Enter."
                return []
            return self._finish_setup()
        if len(k) == 1:
            if self.field == 0 and k.isalnum() and len(self.form[0]) < 8:
                self.form[0] += k
            elif self.field == 1 and k.isdigit() and len(self.form[1]) < 3:
                self.form[1] += k
            elif self.field == 2 and k in "123":
                self.form[2] = k
        return []

    def _field_error(self, i: int) -> str | None:
        v = self.form[i]
        if i == 0 and not v:
            return "The child code is empty (e.g. K03)."
        if i == 1 and (not v or not 12 <= int(v) <= 120):
            return "Age in months must be a number between 12 and 120."
        if i == 2 and v not in ("1", "2", "3"):
            return "Consent level must be 1, 2 or 3 (from the signed form)."
        return None

    def _finish_setup(self) -> list[tuple]:
        self.child, self.age, self.consent = self.form[0], int(self.form[1]), int(self.form[2])
        self.state = "readiness"
        lo, hi = AGE_RANGE_MONTHS
        self.message = (
            ""
            if age_in_range(self.age)
            else f"Age {self.age} months is outside {lo}-{hi}: tries are flagged in the log."
        )
        return [("child", self.child, self.age, self.consent)]

    def _video(self) -> list[tuple]:
        if self.state == "setup":
            return []
        if self.consent != 3:
            self.message = "Video only for consent level 3."
            return []
        self.video = not self.video
        self.message = f"Video saving {'ON' if self.video else 'OFF'} for {self.child}."
        return [("video", self.video)]

    def _readiness_key(self, k: str) -> list[tuple]:
        if k == "ENTER":
            self.state = "idle"
            self.message = (
                "Started although the readiness check listed problems."
                if self.problems
                else "Ready. Waiting for the child at the start line."
            )
            return [("ready_ok",)]
        return []

    def _preset(self, k: str) -> bool:
        if k in ("1", "2", "3"):
            self.preset_task = int(k)
            return True
        if k in ("L", "R"):
            self.preset_leg = k
            return True
        return False

    def _start(self, how: str) -> list[tuple]:
        self.state, self.manual = "recording", how == "manual"
        self.message = "Recording." + (" Space stops." if self.manual else "")
        return [("start", how)]

    def _idle_key(self, k: str) -> list[tuple]:
        if self._preset(k):
            return []
        if k == "SPACE":
            return self._start("manual")
        return []

    _ready_key = _idle_key

    def _recording_key(self, k: str) -> list[tuple]:
        if self._preset(k):
            return []
        if k == "SPACE":
            self.state = "idle"  # becomes "confirm" when the Runner hands over the guess
            return [("stop",)]
        if k == "BACKSPACE":
            self.state = "idle"
            self.message = "Try cancelled (not saved)."
            return [("cancel",)]
        return []

    def _confirm_key(self, k: str) -> list[tuple]:
        if k in ("1", "2", "3"):
            self.choice_task = int(k)
        elif k in ("L", "R"):
            self.choice_leg = k
        elif k == "BACKSPACE":
            self.state, self.guess = "idle", None
            self.message = "Try cancelled (not saved)."
            return [("cancel",)]
        elif k == "ENTER":
            gt, gl, _ = self.guess
            task, leg = self.choice_task, self.choice_leg
            corrected = task != gt or (task != 3 and leg != gl)
            self.state, self.guess = "idle", None
            return [("save", task, leg, corrected)]
        return []

    # ------------------------------------------------- events from the Runner
    def segment(self, event: str) -> list[tuple]:
        """Automatic mode: one Segmenter event per frame (idle/ready/start/recording/end)."""
        if self.mode != "auto" or self.state not in ("idle", "ready", "recording"):
            return []
        if self.state == "recording":
            if event == "end" and not self.manual:
                self.state = "idle"
                return [("stop",)]
            return []
        if event == "start":
            return self._start("auto")
        if event in ("ready", "idle"):
            self.state = event
        return []

    def set_guess(self, task: int, leg: str, confidence: float):
        """The Runner guessed the task and leg of the try that just stopped."""
        self.guess = (task, leg, confidence)
        self.choice_task = self.preset_task or task
        self.choice_leg = self.preset_leg or leg
        self.state = "confirm"
        self.message = "Enter confirms; 1/2/3 and L/R correct; Backspace cancels."

    def saved(self, task: int, n: int):
        self.tries_saved += 1
        self.message = f"Saved: task {task}, try {n}."


class Runner:
    """One frame at a time: pose -> pick -> Segmenter -> buffers -> guess -> score -> save.

    No drawing here, so tests drive it without a window: feed(item) per stream item and
    press(key) per key.
    """

    def __init__(self, out_dir, mode="auto", frame_size=(1280, 720), source="fake", clock=None):
        from app.main import PRE_ROLL_S
        from live.segment import Segmenter

        self.out_dir = Path(out_dir)
        self.ctrl = Controller(mode)
        self.seg = Segmenter()
        self.frame_size = frame_size
        self.source = source
        self.pre_roll_s = PRE_ROLL_S
        self.clock = clock or time.monotonic
        self.session: Session | None = None
        self.pre: deque = deque()  # (t, points25, raw26) of the last PRE_ROLL_S
        self.buf: list | None = None  # frames of the try being recorded
        self.pending: dict | None = None  # stopped try waiting for Enter/Backspace
        self.hist: deque = deque()  # (t, points25) for the readiness check and fps
        self.last_points = None
        self.last_frame = None
        self.saved: list[Path] = []
        self.quit = False

    # ------------------------------------------------------------- per frame
    def fps(self) -> float:
        if len(self.hist) < 2:
            return 0.0
        dt = self.hist[-1][0] - self.hist[0][0]
        return (len(self.hist) - 1) / dt if dt > 0 else 0.0

    def feed(self, item) -> None:
        from common.pick import pick_child
        from live.convert import halpe_to_body25
        from live.quality import check

        t, kp26, sc26, frame = item
        if self.hist and t <= self.hist[-1][0]:  # the clock went back (a new clip): start over
            self.hist.clear()
            self.pre.clear()
            self.seg.reset()
            if self.buf is not None:
                self.buf = []
        people25 = halpe_to_body25(kp26, sc26)
        found = people25.shape[0] > 0
        child25 = pick_child([people25])[0]  # live: Revallo's tracked pick replaces this
        raw = (
            np.concatenate([kp26, sc26[..., None]], axis=-1)[0].astype(np.float32)
            if found
            else np.zeros((26, 3), np.float32)
        )
        self.last_points = child25 if found else None
        self.last_frame = frame

        self.hist.append((t, child25))
        while self.hist and t - self.hist[0][0] > READINESS_WINDOW_S:
            self.hist.popleft()
        self.pre.append((t, child25, raw))
        while self.pre and t - self.pre[0][0] > self.pre_roll_s:
            self.pre.popleft()

        c = self.ctrl
        if c.state == "readiness":
            pts = np.stack([p for _, p in self.hist])
            c.problems = check(pts if found else None, self.fps(), self.frame_size)
        if self.buf is not None:
            self.buf.append((t, child25, raw))
        if c.mode == "auto" and c.state in ("idle", "ready", "recording"):
            ev = self.seg.update(child25 if found else None, t)
            self._apply(c.segment(ev))

    def press(self, k: str | None, now: float | None = None) -> None:
        self._apply(self.ctrl.key(k, now))

    # ------------------------------------------------------------- actions
    def _apply(self, actions: list[tuple]) -> None:
        for a in actions:
            getattr(self, f"_do_{a[0]}")(*a[1:])

    def _do_quit(self):
        self.quit = True

    def _do_child(self, code, age, consent):
        self.session = Session(
            self.out_dir, child=code, age_months=age, video=False, consent_level=consent
        )
        self.hist.clear()

    def _do_video(self, on):
        # Saving the picture itself is Revallo's task (#30); the flag goes into JSON and log.
        if self.session is not None:
            self.session.video = bool(on)

    def _do_ready_ok(self):
        self.seg.reset()

    def _do_start(self, how):
        self.buf = list(self.pre)  # keep the pre-roll before the start
        self.pending = None

    def _do_cancel(self):
        self.buf, self.pending = None, None
        self.seg.reset()

    def _do_stop(self):
        from model.task_classifier import predict_task

        buf, self.buf = self.buf or [], None
        if len(buf) < 2:
            self.ctrl.state = "idle"
            self.ctrl.message = "Try too short (not saved)."
            self.seg.reset()
            return
        t = np.array([b[0] for b in buf], dtype=np.float64)
        tr = Try(
            points=np.stack([b[1] for b in buf]).astype(np.float32),
            t=t - t[0],
            meta={
                "source": self.source,
                "mode": "manual" if self.ctrl.manual else "auto",
                "fps": 30.0,
                "width": self.frame_size[0],
                "height": self.frame_size[1],
                "t0_stream_s": float(t[0]),
            },
        )
        raw26 = np.stack([b[2] for b in buf])
        try:
            task, leg, conf = predict_task(tr.points, tr.t)
        except ValueError:  # e.g. the child was never found
            task, leg, conf = self.ctrl.preset_task or 1, self.ctrl.preset_leg or "L", 0.0
        self.pending = {"tr": tr, "raw26": raw26, "guess": (task, leg, conf)}
        self.ctrl.set_guess(task, leg, conf)

    def _do_save(self, task, leg, corrected):
        from app.main import score_try
        from features.clues import timeline

        p, self.pending = self.pending, None
        if p is None or self.session is None:
            return
        tr = p["tr"]
        gt, gl, gc = p["guess"]
        try:
            clues, score, confidence, reasons = score_try(tr, task)
            events = timeline(task, tr.points, tr.t)
        except ValueError as e:  # e.g. no torso found in any frame
            clues, score, confidence, reasons, events = {}, 1, 0.0, f"not scored: {e}", None
        path = self.session.save_try(
            tr,
            raw26=p["raw26"],
            task=task,
            leg=leg,
            guessed_task=gt,
            guessed_leg=gl,
            task_confidence=gc,
            corrected=corrected,
            score=score,
            confidence=confidence,
            clues=clues,
            reasons=reasons,
            events=events,
        )
        self.saved.append(path)
        self.ctrl.saved(task, self.session._counts[task])  # never the score (testing mode)
        self.seg.reset()


# ---------------------------------------------------------------------- drawing

SCREEN_W, SCREEN_H, STATUS_H = 1280, 720, 150
STATE_COLOURS = {
    "setup": (160, 160, 160),
    "readiness": (0, 200, 255),
    "idle": (200, 200, 200),
    "ready": (80, 220, 80),
    "recording": (60, 60, 240),
    "confirm": (255, 180, 60),
}


def status_lines(c: Controller) -> list[str]:
    """Plain-language lines for the status bar. Built only from Controller, which holds no score."""
    lines = [f"[{c.state.upper()}]  mode {c.mode}"]
    if c.child:
        lo, hi = AGE_RANGE_MONTHS
        flag = "" if age_in_range(c.age) else f" (outside {lo}-{hi})"
        lines.append(
            f"child {c.child}  age {c.age} months{flag}  consent {c.consent}  "
            f"video {'ON' if c.video else 'off'}  tries saved {c.tries_saved}"
        )
    if c.state in ("idle", "ready", "recording"):
        task = f"{c.preset_task} ({TASKS[c.preset_task]})" if c.preset_task else "auto"
        lines.append(f"task {task}  leg {c.preset_leg or 'auto'}")
    if c.message:
        lines.append(c.message)
    lines.append(
        "keys: 1/2/3 task  L/R leg  Space start/stop  Enter confirm  Backspace cancel  "
        "N next child  V video  Q quit"
    )
    return lines


def big_lines(c: Controller) -> list[str]:
    """Text in the middle of the picture (setup form, readiness problems, the guess)."""
    if c.state == "setup":
        out = ["New child"]
        for i, name in enumerate(FIELDS):
            cursor = "_" if i == c.field else ""
            out.append(f"{'>' if i == c.field else ' '} {name}: {c.form[i]}{cursor}")
        return out
    if c.state == "readiness":
        if not c.problems:
            return ["Readiness check: OK", "Press Enter to start."]
        return ["Readiness check:"] + [f"- {p}" for p in c.problems] + ["Enter starts anyway."]
    if c.state == "confirm" and c.guess:
        gt, gl, gc = c.guess
        return [
            f"Guess: task {gt} ({TASKS[gt]}), leg {gl}  (sure {gc:.0%})",
            f"Saving as: task {c.choice_task} ({TASKS[c.choice_task]}), leg {c.choice_leg}",
            "Enter confirm   1/2/3 L/R correct   Backspace cancel",
        ]
    if c.state == "recording":
        return ["RECORDING"]
    if c.state == "ready":
        return ["READY"]
    return []


def draw_screen(runner: Runner, view: str = "stick"):
    """The window picture. view: stick (mirrored stick figure), camera (mirrored picture),
    neutral (no person shown). Pose detection always ran on the unmirrored picture."""
    import cv2

    from viz.replay import draw_figure

    c = runner.ctrl
    img = np.full((SCREEN_H + STATUS_H, SCREEN_W, 3), 25, dtype=np.uint8)
    area = img[:SCREEN_H]
    fw, fh = runner.frame_size
    if view == "camera" and runner.last_frame is not None:
        area[:] = cv2.resize(cv2.flip(runner.last_frame, 1), (SCREEN_W, SCREEN_H))
    elif view == "neutral":
        area[:] = (60, 50, 40)
    elif runner.last_points is not None:
        pts = runner.last_points.copy()
        pts[:, 0] = np.where(pts[:, 2] > 0, fw - pts[:, 0], 0)  # mirror for display only
        s = min(SCREEN_W / fw, SCREEN_H / fh)
        draw_figure(area, pts, s, (SCREEN_W - fw * s) / 2, (SCREEN_H - fh * s) / 2, labels=False)
    colour = STATE_COLOURS.get(c.state, (200, 200, 200))
    cv2.rectangle(img, (0, 0), (SCREEN_W - 1, SCREEN_H - 1), colour, 6)

    y = 80
    for i, line in enumerate(big_lines(c)):
        size = 1.3 if i == 0 else 0.9
        cv2.putText(img, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 6, cv2.LINE_AA)
        cv2.putText(img, line, (40, y), cv2.FONT_HERSHEY_SIMPLEX, size, colour, 2, cv2.LINE_AA)
        y += 55 if i == 0 else 42
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


# ---------------------------------------------------------------------- streams and window


class FakeLoopStream:
    """Fake tries one after another (task 1, 2, 3, legs alternating), with continuous time."""

    def __init__(self, seed: int = 0):
        self.k, self.seed, self.offset, self.cur, self.last_t = 0, seed, 0.0, None, 0.0
        self._next()

    def _next(self):
        from live.pose import FakePoseStream

        task, leg = self.k % 3 + 1, "LR"[(self.k // 3) % 2]
        self.cur = FakePoseStream(task=task, leg=leg, seed=self.seed + self.k)
        self.offset = self.last_t + 1 / 30
        self.k += 1

    def read(self):
        item = self.cur.read()
        if item is None:
            self._next()
            item = self.cur.read()
        t, kp, sc, frame = item
        self.last_t = t + self.offset
        return self.last_t, kp, sc, frame

    def close(self):
        pass


def run_gui(stream, out_dir, mode="auto", view="stick", frame_size=(1280, 720), source="fake"):
    """Open the window and run until Q. Every try is saved the moment Enter is pressed."""
    import cv2

    runner = Runner(out_dir, mode=mode, frame_size=frame_size, source=source)
    name = "Gross motor screening (screening aid, not a diagnosis)"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    paced = source == "fake"
    t_start, t_first = time.monotonic(), None
    try:
        while not runner.quit:
            item = stream.read()
            if item is None:
                runner.ctrl.message = "The video ended. Press Q to quit."
            else:
                runner.feed(item)
                if paced:  # fake frames come instantly: play them in real time
                    t_first = item[0] if t_first is None else t_first
                    wait = (item[0] - t_first) - (time.monotonic() - t_start)
                    if wait > 0:
                        time.sleep(wait)
            cv2.imshow(name, draw_screen(runner, view))
            runner.press(key_name(cv2.waitKeyEx(1)))
            if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        stream.close()
        cv2.destroyAllWindows()
    return runner.saved


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Testing-day window (screening aid, not a diagnosis)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fake", action="store_true", help="fake tries in a loop")
    src.add_argument("--video", type=Path, help="a recorded clip (needs rtmlib)")
    src.add_argument("--camera", type=int, help="webcam index (needs rtmlib)")
    ap.add_argument("--mode", choices=["auto", "manual"], default="auto")
    ap.add_argument("--view", choices=["stick", "camera", "neutral"], default="stick")
    ap.add_argument("--device", default="cuda", help="cuda or cpu (rtmlib)")
    ap.add_argument(
        "--pose-mode",
        choices=["balanced", "lightweight", "performance"],
        default="balanced",
        help="rtmlib model size; the plan uses balanced, lightweight is faster on a CPU",
    )
    ap.add_argument("--out", type=Path, default=Path("runs") / date.today().isoformat())
    args = ap.parse_args(argv)

    if args.fake:
        stream, size, source = FakeLoopStream(), (1280, 720), "fake"
    else:
        import cv2

        from live.pose import PoseStream

        source_arg = str(args.video) if args.video else args.camera
        stream = PoseStream(source=source_arg, mode=args.pose_mode, device=args.device)
        w = int(stream.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        h = int(stream.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        size, source = (w, h), ("video" if args.video else "webcam")
    saved = run_gui(
        stream, args.out, mode=args.mode, view=args.view, frame_size=size, source=source
    )
    print(f"{len(saved)} tries saved in {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
