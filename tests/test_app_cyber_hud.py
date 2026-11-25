"""Cyber HUD of app/screen.py (Marty, 2026-11-26): floor plate, READY, task panel, eval bar,
result box, sidebar, and the no-score rule of testing mode. predict_proba of model/predict.py."""

import numpy as np
import pytest

from app import screen
from app.demo import DemoRunner
from app.ui import Runner
from common import body25 as b
from common.format import make_fake_try
from features.clues import compute
from live.start_area import default_start_area
from model.predict import predict, predict_proba

SCORE_WORDS = ("score", "bad", "good", "perfect", "check by teacher", "%", "observations")


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 10.0
        return self.t


@pytest.fixture
def fake_clock(monkeypatch):
    t = {"now": 100.0}
    monkeypatch.setattr(screen, "_clock", lambda: t["now"])
    return t


def to_idle(r):
    clock = Clock()
    for k in ["K", "0", "1", "ENTER", "4", "5", "ENTER", "3", "ENTER"]:
        r.press(k)
    r.press("ENTER", clock())
    assert r.ctrl.state == "idle"
    return clock


def feet_at(r, x, y):
    """A fake child with both ankles at frame pixel (x, y)."""
    pts = make_fake_try(1, "R").points[0].copy()
    for i in (b.L_ANKLE, b.R_ANKLE):
        pts[i] = (x, y, 0.9)
    r.last_points = pts


@pytest.mark.parametrize("task", [1, 2, 3])
def test_predict_proba_matches_predict(task):
    tr = make_fake_try(task, "R")
    clues = compute(task, tr.points, tr.t)
    p = predict_proba(task, clues)
    score, conf = predict(task, clues)
    assert len(p) == 3 and sum(p) == pytest.approx(1.0)
    assert int(np.argmax(p)) == score and max(p) == pytest.approx(conf)


def test_floor_plate_is_a_trapezoid_on_the_start_area(tmp_path):
    r = Runner(tmp_path, mode="manual")
    q = screen.plate_quad(r)
    x0, y0, x1, y1 = default_start_area(r.frame_size)
    s, _dw, _dh, _ox, oy = screen._stage_transform(r.frame_size)
    assert q[2, 1] == pytest.approx(oy + y1 * s) and q[3, 1] == pytest.approx(oy + y1 * s)
    assert q[2, 0] - q[3, 0] == pytest.approx((x1 - x0) * s)  # near edge: full width
    assert q[1, 0] - q[0, 0] < q[2, 0] - q[3, 0]  # far edge narrower
    assert q[0, 1] < q[3, 1]  # far edge higher in the picture


def test_ready_only_when_both_ankles_are_on_the_plate(tmp_path, fake_clock):
    r = Runner(tmp_path, mode="manual")
    to_idle(r)
    x0, y0, x1, y1 = default_start_area(r.frame_size)
    feet_at(r, (x0 + x1) / 2, y1 - 0.3 * (y1 - y0))
    assert screen.ankles_on_plate(r)
    assert "READY" in screen.render(r, "camera").texts
    r.last_points[b.L_ANKLE, 2] = 0.1  # one ankle not found: not ready
    assert not screen.ankles_on_plate(r)
    feet_at(r, x1 + 200, y1 - 10)
    assert not screen.ankles_on_plate(r)
    assert "READY" not in screen.render(r, "camera").texts
    assert r.ctrl.state == "idle"  # visual only: nothing starts


def test_go_after_the_countdown_then_the_task_panel(tmp_path, fake_clock):
    r = DemoRunner(tmp_path, mode="gesture")
    to_idle(r)
    r.ctrl.state, r.countdown_until, r.now = "countdown", 3.0, 0.5
    texts = screen.render(r, "camera", show_score=True).texts
    assert "3" in texts and "Get ready" in texts
    r.ctrl.state = "recording"
    fake_clock["now"] += 0.1
    texts = screen.render(r, "camera", show_score=True).texts
    assert "GO" in texts and "TASK ?" in texts
    fake_clock["now"] += 1.0
    assert "GO" not in screen.render(r, "camera", show_score=True).texts


def test_same_moment_draws_the_same_picture(tmp_path, fake_clock):
    r = DemoRunner(tmp_path, mode="gesture")
    to_idle(r)
    r.ctrl.state, r.countdown_until, r.now = "countdown", 3.0, 0.05  # glitch-in of the 3
    a = screen.render(r, "stick", show_score=True).img
    b2 = screen.render(r, "stick", show_score=True).img
    assert np.array_equal(a, b2)


def _attempt(task, recorded=True):
    return {
        "task": task,
        "leg": "R",
        "score": 2,
        "confidence": 0.88,
        "check": False,
        "reasons": "stood on one leg 3.2 s, arms steady",
        "duration_s": 3.2,
        "headline": "3.2 s on one leg",
        "recorded": recorded,
        "bout": object(),
    }


def test_result_box_and_sidebar_with_show_score(tmp_path, fake_clock):
    r = DemoRunner(tmp_path, mode="gesture")
    to_idle(r)
    r.ctrl.state = "recording"
    first, card = _attempt(1), _attempt(2)
    r.rec_bouts = [first, card]
    r.intermezzo = {**card, "since": 10.0, "until": 12.5}
    r.now = 11.0
    texts = screen.render(r, "camera", show_score=True).texts
    assert {"TASK 2", "Perfect", "Score 2 · 88% sure", "OBSERVATIONS", "preview"} <= set(texts)
    assert "THIS RECORDING" in texts and "T1 STAND" in texts and "T2 HOP" not in texts
    r.now = 12.6  # the box has landed: the new row is in the sidebar
    texts = screen.render(r, "camera", show_score=True).texts
    assert "T2 HOP" in texts and "T1 STAND" in texts and "OBSERVATIONS" not in texts
    assert "Reference only, not a diagnosis." in texts


def test_testing_mode_shows_task_but_never_a_score(tmp_path, fake_clock):
    r = DemoRunner(tmp_path, mode="gesture")
    to_idle(r)
    r.ctrl.state = "recording"
    r.detect = {"task": 2, "leg": "R", "confidence": 0.9}
    r.live = {"score": 2, "confidence": 0.9, "reasons": "x", "task": 2, "proba": (0, 0.1, 0.9)}
    card = _attempt(2)
    r.rec_bouts = [card]
    r.intermezzo = {**card, "since": 10.0, "until": 12.5}
    drawn = []
    for now in (10.1, 11.0, 12.2, 12.6):
        r.now = now
        fake_clock["now"] += 0.2
        drawn += screen.render(r, "camera", show_score=False).texts
    assert "TASK 2" in drawn and "TASK ?" in drawn  # box shows the task; panel never reads detect
    joined = " ".join(drawn).lower()
    for word in SCORE_WORDS:
        assert word not in joined, word
    assert "eval" not in joined and "this recording" not in joined


def test_testing_mode_box_after_a_saved_try(tmp_path, fake_clock):
    r = Runner(tmp_path, mode="manual")
    to_idle(r)
    screen.render(r, "camera")
    r.ctrl.set_guess(3, "L", 0.9)
    screen.render(r, "camera")  # confirm card
    r.ctrl.state = "idle"
    r.ctrl.saved(3, 1)
    fake_clock["now"] += 0.1
    texts = screen.render(r, "camera").texts
    assert "TASK 3" in texts and "SAVED" in texts
    fake_clock["now"] += screen.BOX_SHOW_S + 0.1
    assert "TASK 3" not in screen.render(r, "camera").texts
