"""Gesture flow of the score window (Daryn, 2026-11-25): live.bouts cuts one recording into task
attempts; app.demo starts on an open palm (3-2-1), shows a card per attempt, stops on an X and saves
each attempt as its own try. Synthetic points only (no child data)."""

import csv
import json

import numpy as np
import pytest

import model.task_classifier as tc
from app import screen
from app.demo import COUNTDOWN_S, DemoRunner
from app.ui import Controller
from app.ui import main as ui_main
from common import body25 as b
from common.format import make_fake_try
from live.bouts import BoutSplitter, split_stand_then_hop
from live.convert import HALPE_TO_BODY25

FPS = 30.0
SCORE_WORDS = ("score", "bad", "good", "perfect", "check by teacher")


def _stand():
    return make_fake_try(1, "R", seconds=0.1, idle_before=1.0).points[0].astype(np.float32)


def raised(pose):
    p = pose.copy()
    p[b.R_ELBOW, :2] = p[b.R_SHOULDER, :2] + (-25, -45)
    p[b.R_WRIST, :2] = p[b.R_SHOULDER, :2] + (-30, -100)
    return p


def crossed(pose):
    p = pose.copy()
    hip = p[b.MID_HIP, :2].copy()
    p[b.L_ELBOW, :2], p[b.L_WRIST, :2] = hip + (45, -60), hip + (-25, -85)
    p[b.R_ELBOW, :2], p[b.R_WRIST, :2] = hip + (-45, -60), hip + (25, -85)
    return p


def hop_points(seconds=3.0, lead_s=0.3):
    """Stand on the right leg (left foot up) for lead_s, then hop every 0.5 s, then both feet down.

    The fake try's own task 2 hops are too slow for features.clues.timeline(2), so these are
    sharper (50 px, 0.5 s)."""
    t1 = make_fake_try(1, "R", seconds=4.0)
    lifted = t1.points[int(2.0 * FPS)].copy()
    stand = _stand()
    n_lead, n_hop = int(lead_s * FPS), int(seconds * FPS)
    s = np.arange(n_hop) / FPS
    hop = np.repeat(lifted[None], n_hop, axis=0).copy()
    hop[:, :, 1] -= (50 * np.abs(np.sin(np.pi * s / 0.5)))[:, None]
    hop[:, :, 0] += (10 * s)[:, None]
    lead = np.repeat(lifted[None], n_lead, axis=0)
    return np.concatenate([lead, hop]), stand


class Script:
    """BODY_25 frames in a row; each piece is moved so it starts where the last one ended."""

    def __init__(self):
        self.frames: list[np.ndarray] = []
        self.marks: dict[str, float] = {}
        self.rng = np.random.default_rng(0)

    @property
    def t(self) -> float:
        return len(self.frames) / FPS

    def add(self, pts, mark=None):
        pts = np.asarray(pts, np.float32).copy()
        if self.frames:
            pts[:, :, :2] += self.frames[-1][b.MID_HIP, :2] - pts[0, b.MID_HIP, :2]
        if mark:
            self.marks[mark] = self.t
        self.frames.extend(pts)
        return self

    def hold(self, pose, seconds, mark=None):
        n = int(round(seconds * FPS))
        pts = np.repeat(pose[None], n, axis=0).copy()
        pts[:, :, :2] += self.rng.normal(0, 0.5, pts[:, :, :2].shape)
        return self.add(pts, mark)

    def last(self):
        return self.frames[-1].copy()

    def items(self):
        idx = np.array(HALPE_TO_BODY25)
        for i, p in enumerate(self.frames):
            kp = np.zeros((1, 26, 2), np.float32)
            sc = np.zeros((1, 26), np.float32)
            kp[0, idx], sc[0, idx] = p[:, :2], p[:, 2]
            kp[0, 17], sc[0, 17] = p[0, :2] + (0, -20), 0.9
            yield i / FPS, kp, sc, None


def three_tasks(sc: Script) -> Script:
    sc.add(make_fake_try(1, "R", seconds=4.0, idle_before=1.0, idle_after=2.0).points, "task1")
    hop, stand = hop_points()
    sc.hold(sc.last(), 1.0).add(hop, "task2").hold(stand, 2.0)
    sc.add(make_fake_try(3, "R", seconds=4.0, idle_before=1.0, idle_after=2.0).points, "task3")
    return sc


def fake_predict(points, t):
    """Task from the shape of the movement (the real classifier is trained on group B tries)."""
    hip = points[:, b.MID_HIP, :2]
    if np.ptp(hip[:, 0]) > 80:
        return 3, "R", 0.9
    if np.ptp(hip[:, 1]) > 30:
        return 2, "R", 0.9
    return 1, "R", 0.9


@pytest.fixture
def guess(monkeypatch):
    monkeypatch.setattr(tc, "predict_task", fake_predict)


def to_idle(r):
    now = [0.0]

    def clock():
        now[0] += 10.0
        return now[0]

    for k in ["K", "0", "1", "ENTER", "4", "5", "ENTER", "2", "ENTER"]:
        r.press(k)
    r.press("ENTER", clock())
    assert r.ctrl.state == "idle"
    return clock


def texts(r):
    return " ".join(screen.render(r, "camera", show_score=True).texts)


# ------------------------------------------------------------------ live.bouts


def test_splitter_finds_each_task_of_a_long_recording():
    sc = three_tasks(Script().hold(_stand(), 1.5))
    sp = BoutSplitter()
    bouts = []
    for t, kp, s_, _ in sc.items():
        p = np.zeros((25, 3), np.float32)
        p[:, :2], p[:, 2] = kp[0, HALPE_TO_BODY25], s_[0, HALPE_TO_BODY25]
        bouts += sp.update(t, p)
    assert len(bouts) == 3
    # first movement: the lift after idle_before; the hops; the jump 1 s into the fake task 3
    starts = [sc.marks["task1"] + 1.0, sc.marks["task2"], sc.marks["task3"] + 2.0]
    for bout, move in zip(bouts, starts, strict=True):
        assert bout.t[0] <= move + 0.5 and bout.t[-1] > move + 1.0


def test_stand_straight_into_hops_is_two_attempts():
    t1 = make_fake_try(1, "R", seconds=4.0)
    lift = t1.points[(t1.t > 0.5) & (t1.t < 3.0)]
    hop, _ = hop_points(lead_s=0.0)
    pts = np.concatenate([lift, hop]).astype(np.float32)
    t = np.arange(len(pts)) / FPS
    assert split_stand_then_hop(pts, t) == [(0, len(lift)), (len(lift), len(pts))]
    assert split_stand_then_hop(hop, np.arange(len(hop)) / FPS) == [(0, len(hop))]


def test_flush_cuts_the_frames_after_the_x():
    sp = BoutSplitter()
    sc = Script().hold(_stand(), 1.5).add(make_fake_try(1, "R", seconds=4.0).points)
    for t, kp, s_, _ in sc.items():
        p = np.zeros((25, 3), np.float32)
        p[:, :2], p[:, 2] = kp[0, HALPE_TO_BODY25], s_[0, HALPE_TO_BODY25]
        assert sp.update(t, p) == []
    assert sp.current() is not None
    (bout,) = sp.flush(until_t=4.0)
    assert bout.t[-1] < 4.0 and sp.current() is None


# ------------------------------------------------------------------ controller


def test_controller_gesture_events():
    c = Controller("gesture")
    c.state = "idle"
    assert c.gesture("cross") == []
    assert c.gesture("palm") == [("countdown",)] and c.state == "countdown"
    assert c.key("BACKSPACE", 1.0) == [("cancel_countdown",)] and c.state == "idle"
    c.gesture("palm")
    assert c.countdown_done() == [("start", "gesture")] and c.state == "recording"
    assert c.gesture("palm") == []
    assert c.gesture("cross") == [("stop",)] and c.state == "idle"
    auto = Controller("auto")
    auto.state = "idle"
    assert auto.gesture("palm") == [] and auto.state == "idle"


def test_gesture_mode_needs_show_score():
    with pytest.raises(SystemExit):
        ui_main(["--fake", "--mode", "gesture"])


# ------------------------------------------------------------------ app.demo


def test_palm_countdown_three_tasks_x_saves_three_tries(tmp_path, guess):
    r = DemoRunner(tmp_path, mode="gesture")
    to_idle(r)
    sc = Script().hold(_stand(), 1.5).hold(raised(_stand()), 1.0, "palm").hold(_stand(), 3.5)
    three_tasks(sc)
    sc.hold(crossed(sc.last()), 1.5, "x").hold(sc.last(), 1.0)

    states, cards, countdown_text = [], [], None
    for item in sc.items():
        r.feed(item)
        st = r.ctrl.state
        if not states or states[-1] != st:
            states.append(st)
            if st == "countdown":
                countdown_text = texts(r)
        card = r.card()
        if card is not None and (not cards or cards[-1] != (card["task"], card["recorded"])):
            cards.append((card["task"], card["recorded"]))
        if card is not None and abs(r.now - card["since"] - 0.5) < 0.02:  # faded in by now
            assert f"TASK {card['task']}" in texts(r)

    assert states[:4] == ["idle", "countdown", "recording", "idle"]
    assert countdown_text is not None and "3" in countdown_text and "Get ready" in countdown_text
    start = sc.marks["palm"] + 0.7 + COUNTDOWN_S
    assert cards == [(1, True), (2, True), (3, True)]
    assert len(r.saved) == 3
    docs = [json.loads(p.read_text(encoding="utf-8")) for p in r.saved]
    assert [d["meta"]["task"] for d in docs] == [1, 2, 3]
    assert all(d["meta"]["mode"] == "gesture" for d in docs)
    assert all(d["meta"]["t0_stream_s"] >= start - 0.6 for d in docs)
    assert docs[-1]["meta"]["t0_stream_s"] < sc.marks["x"]
    rows = list(csv.DictReader((tmp_path / "log.csv").open(encoding="utf-8")))
    assert len(rows) == 3
    assert [row["task"] for row in r.summary["rows"]] == [1, 2, 3]
    shown = texts(r)
    assert "Recording saved" in shown and "3 tries" in shown


def test_without_palm_attempts_are_shown_but_not_saved(tmp_path, guess):
    r = DemoRunner(tmp_path, mode="gesture")
    to_idle(r)
    sc = Script().hold(_stand(), 1.5)
    sc.add(make_fake_try(1, "R", seconds=4.0, idle_before=1.0, idle_after=2.0).points)
    seen = False
    for item in sc.items():
        r.feed(item)
        if r.card() is not None:
            seen = seen or "Not recorded" in texts(r)
    assert r.ctrl.state == "idle" and not r.saved
    assert seen and [x["task"] for x in r.preview_bouts] == [1]
    assert r.preview_bouts[0]["recorded"] is False
    assert "DETECTED · NOT RECORDED" in texts(r)


def test_backspace_cancels_the_countdown(tmp_path, guess):
    r = DemoRunner(tmp_path, mode="gesture")
    clock = to_idle(r)
    sc = Script().hold(_stand(), 1.0).hold(raised(_stand()), 1.0)
    for item in sc.items():
        r.feed(item)
    assert r.ctrl.state == "countdown"
    r.press("BACKSPACE", clock())
    assert r.ctrl.state == "idle"
    for item in Script().hold(_stand(), 4.0).items():
        r.feed((item[0] + sc.t, *item[1:]))
    assert r.ctrl.state == "idle" and r.buf is None


def test_space_records_at_once_and_an_empty_recording_saves_nothing(tmp_path, guess):
    r = DemoRunner(tmp_path, mode="gesture")
    clock = to_idle(r)
    r.press("SPACE", clock())
    assert r.ctrl.state == "recording"
    for item in Script().hold(_stand(), 2.0).items():
        r.feed(item)
    r.press("SPACE", clock())
    assert r.ctrl.state == "idle" and not r.saved
    assert "No task found" in r.ctrl.message
    for item in Script().hold(_stand(), 0.5).items():  # the summary fades in
        r.feed((item[0] + 2.0, *item[1:]))
    assert "No task found in this recording" in texts(r)


def test_testing_mode_screen_never_shows_a_score_in_gesture_states(tmp_path, guess):
    r = DemoRunner(tmp_path, mode="gesture")
    to_idle(r)
    sc = Script().hold(_stand(), 1.5).hold(raised(_stand()), 1.0).hold(_stand(), 3.5)
    sc.add(make_fake_try(1, "R", seconds=4.0, idle_before=1.0, idle_after=2.0).points)
    for item in sc.items():
        r.feed(item)
        shown = " ".join(screen.render(r, "camera", show_score=False).texts).lower()
        assert not any(w in shown for w in SCORE_WORDS)


def test_palm_on_the_readiness_screen_starts_the_countdown(tmp_path, guess):
    r = DemoRunner(tmp_path, mode="gesture")
    for k in ["K", "0", "1", "ENTER", "4", "5", "ENTER", "2", "ENTER"]:
        r.press(k)
    assert r.ctrl.state == "readiness"
    assert "Show an open hand" in texts(r)
    sc = Script().hold(_stand(), 1.0).hold(raised(_stand()), 1.0).hold(_stand(), 3.5)
    states = []
    for item in sc.items():
        r.feed(item)
        if not states or states[-1] != r.ctrl.state:
            states.append(r.ctrl.state)
    assert states == ["readiness", "countdown", "recording"]
