"""Render every phase of the cyber HUD (app/screen.py) to PNGs, without a camera. Marty, 2026-11-26.

    python scripts/preview_cyber_ui.py                # -> runs/ui_preview/*.png (git-ignored)
    python scripts/preview_cyber_ui.py --size 1512x982

A fake child (common.format.make_fake_try, task 1) stands in the start area. The Runner fields are
set by hand (state, countdown, live score with probabilities, detected task, attempt card, rows),
and app.screen._clock is replaced so each picture is drawn at a chosen moment of its animation.
Files starting with t_ are testing mode (show_score=False: never a score). No child data is used.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import screen  # noqa: E402
from app.demo import DemoRunner  # noqa: E402
from app.ui import Runner  # noqa: E402
from common.format import make_fake_try  # noqa: E402
from live.start_area import default_start_area  # noqa: E402

OUT = Path("runs") / "ui_preview"


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _camera_frame(w: int, h: int) -> np.ndarray:
    """A dull fake room: wall gradient, floor, a little noise (no real picture)."""
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    wall = np.array([70, 64, 58], np.float32) * (1 - 0.3 * yy)
    img = np.broadcast_to(wall, (h, w, 3)).copy()
    img[int(h * 0.62) :] = np.array([52, 58, 66], np.float32)
    img += np.random.default_rng(0).normal(0, 3, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def _standing_points(frame_size, on_plate: bool) -> np.ndarray:
    """One frame of a fake child, moved so the feet stand in (or beside) the start area."""
    tr = make_fake_try(1, "R", seconds=2.0, idle_before=1.0)
    pts = tr.points[5].copy()
    x0, y0, x1, y1 = default_start_area(frame_size)
    seen = pts[:, 2] > 0
    feet_y = max(pts[seen, 1])
    body_h = feet_y - min(pts[seen, 1])
    k = 0.48 * frame_size[1] / body_h  # ~50 % of the picture height, as the guide asks
    cx = float(np.median(pts[seen, 0]))
    target_x = (x0 + x1) / 2 if on_plate else x1 + 0.12 * frame_size[0]
    target_y = (y0 + y1) / 2 + 0.25 * (y1 - y0)
    pts[seen, 0] = (pts[seen, 0] - cx) * k + target_x
    pts[seen, 1] = (pts[seen, 1] - feet_y) * k + target_y
    return pts


def _to_idle(r, child="K00") -> None:
    t = [0.0]

    def clk():
        t[0] += 10.0
        return t[0]

    for k in [*child, "ENTER", "4", "5", "ENTER", "3", "ENTER"]:
        r.press(k)
    r.press("ENTER", clk())


def _attempt(task: int, score: int, conf: float, recorded: bool, now: float) -> dict:
    heads = {1: "3.2 s on one leg", 2: "6 hops · 3.1 s", 3: "Jump · 1.4 s"}
    reasons = {
        1: "stood on one leg 3.2 s, foot down once at 2.0 s, arms steady",
        2: "6 hops, steady rhythm, landed on the same foot",
        3: "feet together at take-off, 0.31 s in the air, jump size 0.84",
    }
    return {
        "task": task,
        "leg": "R",
        "score": score,
        "confidence": conf,
        "check": False,
        "reasons": reasons[task],
        "duration_s": 3.2,
        "headline": heads[task],
        "recorded": recorded,
        "bout": object(),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--size", default="1280x720", help="window size WxH (default 1280x720)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    w, h = (int(v) for v in args.size.lower().split("x"))
    screen.set_size(w, h)
    args.out.mkdir(parents=True, exist_ok=True)
    clock = Clock()
    screen._clock = clock
    fs = (1280, 720)
    frame = _camera_frame(*fs)
    tmp = Path(tempfile.mkdtemp(prefix="ui_preview_"))
    written: list[str] = []

    def shot(name, r, show=True, view="camera"):
        r.last_frame = frame
        img = screen.render(r, view, show_score=show).img
        path = args.out / f"{name}.png"
        cv2.imwrite(str(path), img)
        written.append(path.name)

    for show, pre in ((True, ""), (False, "t_")):
        r = DemoRunner(tmp / f"g{pre}", mode="gesture", frame_size=fs)
        _to_idle(r)
        r.last_points = _standing_points(fs, on_plate=False)
        clock.t = 1000.0
        shot(f"{pre}01_floor_idle", r, show)
        r.last_points = _standing_points(fs, on_plate=True)
        for i, dt in enumerate((0.0, 0.33)):
            clock.t = 1001.0 + dt
            shot(f"{pre}02_ready_{'ab'[i]}", r, show)

        # countdown: each digit, and the glitch-in of the 3
        r.ctrl.state = "countdown"
        r.now = 50.0
        r.countdown_until = 53.0
        for i, dt in enumerate((0.0, 0.05, 0.10)):
            r.now = 50.0 + dt
            clock.t = 1002.0 + dt
            shot(f"{pre}03_countdown_3_glitch_{i}", r, show)
        for n in (3, 2, 1):
            r.now = 53.0 - n + 0.5
            clock.t = 1003.0 + (3 - n)
            shot(f"{pre}03_countdown_{n}", r, show)

        # GO: the state goes countdown -> recording
        r.ctrl.state = "recording"
        r.buf = [(53.0, r.last_points, None), (53.1, r.last_points, None)]
        for i, dt in enumerate((0.0, 0.2, 0.4)):
            clock.t = 1010.0 + dt
            shot(f"{pre}04_go_{i}", r, show)
        # detection: task panel and eval bar
        clock.t = 1011.0
        r.detect = {"task": 1, "leg": "R", "confidence": 0.91}
        r.live = {
            "score": 2,
            "confidence": 0.72,
            "reasons": "",
            "task": 1,
            "proba": (0.08, 0.20, 0.72),
        }
        shot(f"{pre}05_detect", r, show)
        clock.t = 1012.0
        shot(f"{pre}05_detect_settled", r, show)
        # task change 1 -> 2 (glitch out, flicker in) with the bar moving
        r.detect = {"task": 2, "leg": "R", "confidence": 0.84}
        r.live = {
            "score": 1,
            "confidence": 0.55,
            "reasons": "",
            "task": 2,
            "proba": (0.25, 0.55, 0.20),
        }
        for i, dt in enumerate((0.05, 0.15, 0.30)):
            clock.t = 1013.0 + dt
            shot(f"{pre}06_task_change_{i}", r, show)
        clock.t = 1014.0
        shot(f"{pre}06_task_change_done", r, show)

        # result box: in, hold, out (stream time drives the card window)
        first = _attempt(1, 2, 0.88, True, 60.0)
        r.rec_bouts = [first]
        card = {**_attempt(2, 1, 0.71, True, 60.0), "since": 60.0, "until": 62.5}
        r.rec_bouts.append(card)
        r.intermezzo = card
        r.detect, r.live = None, None
        for i, dt in enumerate((0.08, 0.25, 0.45)):
            r.now = 60.0 + dt
            clock.t = 1020.0 + dt
            shot(f"{pre}07_result_in_{i}", r, show)
        r.now = 61.2
        shot(f"{pre}07_result_hold", r, show)
        for i, dt in enumerate((0.42, 0.25, 0.08)):
            r.now = 62.5 - dt
            shot(f"{pre}08_result_out_{i}", r, show)
        r.now = 62.6
        shot(f"{pre}09_sidebar_flash", r, show)

        # summary after the X
        r.ctrl.state = "idle"
        r.buf = None
        r.intermezzo = None
        rows = [
            {**_attempt(1, 2, 0.88, True, 0), "try": 1},
            {**_attempt(2, 1, 0.71, True, 0), "try": 2},
            {**_attempt(3, 0, 0.66, True, 0), "try": 3},
        ]
        r.summary = {"rows": rows, "since": 70.0, "until": 76.0}
        r.now = 71.0
        shot(f"{pre}10_summary", r, show)

    # testing mode with the plain Runner (app.ui without --show-score): result box after a save
    r = Runner(tmp / "plain", mode="auto", frame_size=fs)
    _to_idle(r, "K01")
    r.last_points = _standing_points(fs, on_plate=True)
    clock.t = 2000.0
    r.ctrl.state = "recording"
    r.buf = [(0.0, r.last_points, None), (0.1, r.last_points, None)]
    shot("t_11_plain_recording", r, False)
    r.ctrl.state = "confirm"
    r.ctrl.set_guess(2, "R", 0.9)
    shot("t_11_plain_confirm", r, False)
    r.ctrl.state = "idle"
    r.ctrl.saved(2, 1)
    for i, dt in enumerate((0.0, 0.3, 1.2, 2.2)):
        clock.t = 2001.0 + dt
        shot(f"t_11_plain_saved_{i}", r, False)

    print(f"{len(written)} pictures in {args.out}:")
    for name in written:
        print("  " + name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
