"""app/screen.py (owner Daryn): the one window picture, its buttons and the shared window loop."""

import numpy as np

from app import screen
from app.demo import DemoRunner
from app.ui import Runner
from live.pose import FakePoseStream

SCORE_WORDS = ("score", "bad", "good", "perfect", "check by teacher", "timeline", "steady")


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 10.0
        return self.t


def to_idle(r, clock, consent="3"):
    for k in ["K", "0", "1", "ENTER", "4", "5", "ENTER", consent, "ENTER"]:
        r.press(k)
    r.press("ENTER", clock())
    assert r.ctrl.state == "idle"


def keys(r):
    return {b.key for b in screen.buttons(r) if b.key is not None}


def click(r, key, clock):
    """Press the middle of the button for key, the way the mouse handler does."""
    btn = next(b for b in screen.buttons(r) if b.key == key)
    assert screen.hit(screen.buttons(r), btn.x + btn.w / 2, btn.y + btn.h / 2) == key
    r.press(key, clock())


def test_buttons_follow_the_state_and_drive_the_same_keys(tmp_path):
    clock = Clock()
    r = Runner(tmp_path, mode="manual")
    assert keys(r) == {"ESC", "ENTER"}
    for k in ["K", "0", "1", "ENTER", "4", "5", "ENTER", "3"]:
        r.press(k)
    click(r, "ENTER", clock)
    assert r.ctrl.state == "readiness" and keys(r) == {"N", "Q", "ENTER"}
    click(r, "ENTER", clock)
    assert keys(r) == {"1", "2", "3", "L", "R", "V", "N", "Q", "SPACE"}
    click(r, "2", clock)
    assert r.ctrl.preset_task == 2
    assert next(b for b in screen.buttons(r) if b.key == "2").selected
    click(r, "SPACE", clock)
    assert r.ctrl.state == "recording" and {"SPACE", "BACKSPACE"} <= keys(r)
    stream = FakePoseStream(task=2)
    while (item := stream.read()) is not None:
        r.feed(item)
    click(r, "SPACE", clock)
    assert r.ctrl.state == "confirm" and {"ENTER", "BACKSPACE", "1", "L"} <= keys(r)
    click(r, "ENTER", clock)
    assert len(r.saved) == 1


def test_buttons_fit_in_the_bar_and_do_not_overlap(tmp_path):
    r = Runner(tmp_path, mode="manual")
    to_idle(r, Clock())
    btns = sorted(screen.buttons(r), key=lambda b: b.x)
    assert btns[0].x >= screen.VX and btns[-1].x + btns[-1].w <= screen.VX + screen.VW
    for a, b in zip(btns, btns[1:], strict=False):
        assert a.x + a.w <= b.x, (a.label, b.label)
    assert screen.hit(btns, 0, 0) is None


def test_no_video_button_without_consent_level_3(tmp_path):
    r = Runner(tmp_path, mode="manual")
    to_idle(r, Clock(), consent="2")
    assert "V" not in keys(r)


def test_every_state_and_view_renders(tmp_path):
    clock = Clock()
    r = DemoRunner(tmp_path, mode="manual", frame_size=(640, 480))
    s = FakePoseStream(task=1)
    for _ in range(5):
        r.feed(s.read())
    r.last_frame = np.full((480, 640, 3), 90, np.uint8)  # 4:3 camera: letterboxed
    seen = set()

    def check():
        for view in ("stick", "camera", "neutral"):
            for show in (False, True):
                img = screen.render(r, view, show_score=show).img
                assert img.shape == (screen.H, screen.W, 3) and img.dtype == np.uint8
        seen.add(r.ctrl.state)

    check()
    to_idle(r, clock)
    check()
    r.press("SPACE", clock())
    for _ in range(60):
        r.feed(s.read())
    check()
    while (item := s.read()) is not None:
        r.feed(item)
    r.press("SPACE", clock())
    check()
    r.press("ENTER", clock())
    check()
    assert seen == {"setup", "idle", "recording", "confirm"}


def test_testing_mode_draws_no_score_but_show_score_does(tmp_path):
    clock = Clock()
    r = DemoRunner(tmp_path, mode="manual")
    to_idle(r, clock)
    drawn = []
    for task in (1, 2, 3):
        s = FakePoseStream(task=task)
        r.press("SPACE", clock())
        for _ in range(60):
            r.feed(s.read())
        drawn += screen.render(r, "camera", show_score=False).texts
        while (item := s.read()) is not None:
            r.feed(item)
        r.press("SPACE", clock())
        r.press(str(task), clock())
        r.press("ENTER", clock())
        drawn += screen.render(r, "camera", show_score=False).texts
    joined = " ".join(drawn).lower()
    for word in SCORE_WORDS:
        assert word not in joined, word
    shown = " ".join(screen.render(r, "camera", show_score=True).texts)
    assert "LAST SAVED TRY" in shown and "Reference only, not a diagnosis." in shown


def test_text_wrap_and_fonts():
    lines = screen.wrap("one two three four five six seven eight nine ten", 15, "regular", 80, 2)
    assert len(lines) == 2 and lines[-1].endswith("…")
    for w in ("regular", "medium", "semibold", "bold"):
        assert screen.measure("Gross Motor Screening", 16, w) > 50


def test_window_is_drawn_at_the_screen_shape():
    """Mac 1512x982 points: drawn at that size; a 16:9 camera fills the height, sides cut a bit."""
    assert screen.canvas_size((1512, 982)) == (1512, 982)
    assert screen.canvas_size((2560, 1440)) == (1920, 1080)  # big screens: at most 1080 tall
    assert screen.canvas_size((1024, 600))[1] == 720
    old = (screen.W, screen.H)
    try:
        screen.set_size(1512, 982)
        s, dw, dh, ox, oy = screen._stage_transform((1280, 720))
        assert dh == 982 and oy == 0 and -0.08 * dw <= ox < 0
        s, dw, dh, ox, oy = screen._stage_transform((640, 480))  # 4:3: never cut, bars instead
        assert ox >= 0 and oy >= 0
        r = Runner("unused", mode="manual")
        r.ctrl.state = "idle"
        r.last_frame = np.full((720, 1280, 3), 90, np.uint8)
        img = screen.render(r, "camera").img
        assert img.shape == (982, 1512, 3)
        assert img[491, 2].mean() > 40  # the camera picture reaches the edge (no side bar)
    finally:
        screen.set_size(*old)
