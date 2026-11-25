"""Tests for live/start_area.py, framing/feet checks, and DemoRunner Task 3 preview UX."""

from __future__ import annotations

import numpy as np
import pytest

from app import screen
from app.demo import PREVIEW_WINDOW_S, DemoRunner
from app.ui import Runner
from common import body25 as b
from common.format import make_fake_try
from live import start_area
from live.pose import FakePoseStream
from live.quality import MAX_BODY_FRAC, check, feet_visible


def _scale_body(points: np.ndarray, frame_size: tuple[int, int], factor: float) -> np.ndarray:
    """Scale a pose around its vertical center while keeping points inside the frame."""
    q = points.copy()
    seen = q[..., 2] > 0.3
    cy = float(np.median(q[..., 1][seen]))
    cx = float(np.median(q[..., 0][seen]))
    q[..., 0] = np.where(seen, (q[..., 0] - cx) * factor + frame_size[0] * 0.5, 0)
    q[..., 1] = np.where(seen, (q[..., 1] - cy) * factor + frame_size[1] * 0.5, 0)
    return q


def test_start_area_guides_and_mirror():
    assert start_area.default_start_area((1280, 720)) == pytest.approx(
        (0.34 * 1280, 0.70 * 720, 0.66 * 1280, 0.90 * 720)
    )
    y_head, y_feet = start_area.standing_guide_lines((1280, 720))
    assert y_feet == pytest.approx(0.80 * 720)
    assert y_head == pytest.approx(0.30 * 720)
    box = (420.0, 500.0, 800.0, 620.0)
    mirrored = start_area.mirror_box_x(box, 1280.0)
    assert mirrored == (480.0, 500.0, 860.0, 620.0)
    assert start_area.mirror_box_x(mirrored, 1280.0) == box


def test_feet_visible_and_too_close_for_forward_jump():
    tr = make_fake_try(task=3)
    pts = tr.points[:30]
    assert feet_visible(pts, (1280, 720)) is True
    assert feet_visible(pts[0], (1280, 720)) is True

    no_ankle = pts[0].copy()
    no_ankle[b.L_ANKLE, 2] = 0.0
    assert feet_visible(no_ankle, (1280, 720)) is False

    clipped = pts[0].copy()
    clipped[b.R_ANKLE, 1] = 718.0
    assert feet_visible(clipped, (1280, 720)) is False

    tall = _scale_body(pts, (1280, 720), factor=1.95)
    assert check(tall, 30, (1280, 720)) == []
    msgs = check(tall, 30, (1280, 720), max_body_frac=MAX_BODY_FRAC)
    assert any("Too close for a forward jump" in m for m in msgs), msgs


def _enter_child(runner, code: str = "K01", *, idle: bool = True):
    t = 0.0

    def clk() -> float:
        nonlocal t
        t += 10.0
        return t

    for k in [*code, "ENTER", "4", "8", "ENTER", "2", "ENTER"]:
        runner.press(k)
    if idle:
        runner.press("ENTER", clk())
    return clk


def test_runner_readiness_uses_start_area_and_screen_shows_feet_indicator(tmp_path):
    runner = Runner(tmp_path, mode="manual", frame_size=(1280, 720), start_area=(0, 0, 50, 50))
    _enter_child(runner, "K01", idle=False)
    assert runner.ctrl.state == "readiness"

    stream = FakePoseStream(task=3)
    for _ in range(10):
        runner.feed(stream.read())
    stream.close()
    assert any("start area" in m for m in runner.ctrl.problems)

    rendered = screen.render(runner, view="camera", show_score=False)
    joined = " ".join(rendered.texts)
    assert "Both feet visible" in joined
    assert "Start area" in joined
    assert "Target ~50% height" in joined

    runner.last_points[b.R_ANKLE, 2] = 0.0
    assert "Feet not visible" in " ".join(
        screen.render(runner, view="camera", show_score=False).texts
    )


def test_demo_runner_task3_idle_hint_preview_window_and_result_persistence(tmp_path):
    runner = DemoRunner(tmp_path, mode="manual", frame_size=(1280, 720), source="fake")
    clk = _enter_child(runner, "K02")
    runner.press("3", clk())
    assert runner.ctrl.state == "idle" and runner.ctrl.preset_task == 3

    s_idle = FakePoseStream(task=3, idle_before=2.2, seconds=2.0, idle_after=1.0)
    for _ in range(55):
        runner.feed(s_idle.read())
    assert runner.live is not None and runner.live.get("hint") == "Press Space to start"
    assert "Press Space to start" in " ".join(
        screen.render(runner, view="camera", show_score=True).texts
    )

    assert PREVIEW_WINDOW_S >= 3.0
    while (item := s_idle.read()) is not None:
        runner.feed(item)
    s_idle.close()
    assert runner.live is not None and "score" in runner.live

    s_try = FakePoseStream(task=3)
    runner.press("SPACE", clk())
    while (item := s_try.read()) is not None:
        runner.feed(item)
    s_try.close()
    runner.press("SPACE", clk())
    runner.press("ENTER", clk())
    assert runner.result is not None
    assert "LAST SAVED TRY" in " ".join(screen.render(runner, view="camera", show_score=True).texts)
    runner.press("SPACE", clk())
    assert runner.result is None


def test_floating_shoulder_score_badge_follows_body_and_stays_off_in_testing_mode(tmp_path):
    runner = DemoRunner(tmp_path, mode="manual", frame_size=(1280, 720), source="fake")
    _enter_child(runner, "K03")

    s = FakePoseStream(task=1, leg="L", seed=0)
    for _ in range(45):
        runner.feed(s.read())
    s.close()
    runner.live = {"score": 2, "confidence": 0.85, "reasons": "steady", "task": 1}

    shown_text = " ".join(screen.render(runner, view="camera", show_score=True).texts)
    assert "Perfect" in shown_text and "Preview · 85% · Stand" in shown_text
    assert "LIVE SCORE" not in shown_text

    runner.ctrl.state = "recording"
    rec_text = " ".join(screen.render(runner, view="camera", show_score=True).texts)
    assert "Live · 85% · Stand" in rec_text
    runner.ctrl.state = "idle"

    hidden_text = " ".join(screen.render(runner, view="camera", show_score=False).texts).lower()
    for word in ("bad", "good", "perfect", "score"):
        assert word not in hidden_text

    assert "LIVE SCORE" in " ".join(screen.render(runner, view="neutral", show_score=True).texts)
    runner.last_points = None
    assert "LIVE SCORE" in " ".join(screen.render(runner, view="camera", show_score=True).texts)


def test_setup_back_navigation_and_readiness_single_focus(tmp_path):
    runner = DemoRunner(tmp_path, mode="manual", frame_size=(1280, 720), source="fake")
    setup_texts = " ".join(screen.render(runner, view="camera", show_score=True).texts)
    assert "1 points only · 2 stick figure · 3 + video" in setup_texts

    for k in ["K", "0", "4", "ENTER"]:
        runner.press(k)
    assert runner.ctrl.field == 1
    runner.press("BACKSPACE")
    assert runner.ctrl.field == 0 and runner.ctrl.form[0] == "K04"
    runner.press("FIELD_1")
    assert runner.ctrl.field == 1
    for k in ["4", "8", "ENTER", "2", "ENTER"]:
        runner.press(k)
    assert runner.ctrl.state == "readiness"

    readiness_texts = " ".join(screen.render(runner, view="camera", show_score=True).texts)
    assert "LIVE SCORE" not in readiness_texts
    assert "live.start_area" in readiness_texts
