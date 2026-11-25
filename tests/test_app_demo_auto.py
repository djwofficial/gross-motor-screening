"""app.demo auto flow (Marty, 2026-11-25): no task buttons, a Detecting chip, and auto-save of a
sure task guess (conf >= AUTO_ACCEPT) without the confirm card. Testing mode is unchanged."""

import json

import pytest

import model.task_classifier as tc
from app import screen
from app.demo import AUTO_ACCEPT, DemoRunner
from app.ui import Runner
from live.pose import FakePoseStream

TASK_KEYS = {"1", "2", "3", "L", "R"}
SCORE_WORDS = ("score", "bad", "good", "perfect", "check by teacher", "detecting", "detected")


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 10.0
        return self.t


def to_idle(r, clock):
    for k in ["K", "0", "1", "ENTER", "4", "5", "ENTER", "2", "ENTER"]:
        r.press(k)
    r.press("ENTER", clock())
    assert r.ctrl.state == "idle"


def keys(r):
    return {b.key for b in screen.buttons(r) if b.key is not None}


@pytest.fixture
def guess(monkeypatch):
    """Fix what model.task_classifier.predict_task returns: guess["value"] = (task, leg, conf)."""
    g = {"value": (2, "R", 0.9)}
    monkeypatch.setattr(tc, "predict_task", lambda _p, _t: g["value"])
    return g


def run_try(r, task=2, until=("idle", "confirm")):
    """Feed one fake try; stop once the Segmenter has started and then ended it."""
    stream = FakePoseStream(task=task, leg="R", seed=0)
    started = False
    while (item := stream.read()) is not None:
        r.feed(item)
        started = started or r.ctrl.state == "recording"
        if started and r.ctrl.state in until:
            return
    raise AssertionError(f"the try never ended (state {r.ctrl.state})")


def saved_json(r):
    return json.loads(r.saved[-1].read_text(encoding="utf-8"))


def test_auto_accept_value():
    assert AUTO_ACCEPT == 0.8


def test_task_buttons_hidden_in_demo_auto_mode_but_keys_still_work(tmp_path, guess):
    clock = Clock()
    r = DemoRunner(tmp_path, mode="auto")
    to_idle(r, clock)
    assert not keys(r) & TASK_KEYS and {"SPACE", "N", "Q"} <= keys(r)
    r.press("SPACE", clock())
    assert r.ctrl.state == "recording" and not keys(r) & TASK_KEYS
    assert {"SPACE", "BACKSPACE"} <= keys(r)
    r.press("3", clock())  # hidden backup key
    assert r.ctrl.preset_task == 3
    r.press("BACKSPACE", clock())

    r.press("0", clock())
    assert r.ctrl.preset_task is None and r.ctrl.preset_leg is None
    guess["value"] = (2, "R", 0.5)
    run_try(r)
    assert r.ctrl.state == "confirm" and TASK_KEYS <= keys(r)

    for other in (Runner(tmp_path / "a", mode="auto"), DemoRunner(tmp_path / "b", mode="manual")):
        to_idle(other, clock)
        assert TASK_KEYS <= keys(other)


def test_pinned_task_and_leg_prioritized_even_under_auto_accept(tmp_path, guess):
    clock = Clock()
    r = DemoRunner(tmp_path, mode="auto")
    to_idle(r, clock)
    r.press("1", clock())
    r.press("L", clock())
    guess["value"] = (2, "R", 0.45)
    run_try(r, task=1)
    assert r.ctrl.state == "idle" and len(r.saved) == 1
    d = saved_json(r)
    assert d["meta"]["task"] == 1 and d["meta"]["leg"] == "L"
    assert d["guessed_task"] == 2 and d["corrected"] is True
    assert r.result.get("pinned") is True and "auto" not in r.result
    shown = " ".join(screen.render(r, "camera", show_score=True).texts)
    assert "Task prioritized by key" in shown and "Task detected:" not in shown

    r.press("0", clock())
    assert r.ctrl.preset_task is None and r.ctrl.preset_leg is None
    run_try(r, task=2)
    assert r.ctrl.state == "confirm"


def test_pause_key_blocks_auto_start_until_resumed(tmp_path, guess):
    clock = Clock()
    r = DemoRunner(tmp_path, mode="auto")
    to_idle(r, clock)
    r.press("P", clock())
    assert r.ctrl.paused is True
    assert "Auto paused" in screen.render(r, "camera", show_score=True).texts
    stream = FakePoseStream(task=1, leg="L", seed=0)
    while (item := stream.read()) is not None:
        r.feed(item)
        assert r.ctrl.state == "idle"
    assert not r.saved
    r.press("SPACE", clock())
    assert r.ctrl.state == "recording"
    r.press("BACKSPACE", clock())
    r.press("P", clock())
    assert r.ctrl.paused is False
    run_try(r)
    assert len(r.saved) == 1


def test_sure_guess_is_saved_without_the_confirm_card(tmp_path, guess):
    clock = Clock()
    r = DemoRunner(tmp_path, mode="auto")
    to_idle(r, clock)
    guess["value"] = (2, "R", 0.9)
    run_try(r)
    assert r.ctrl.state == "idle" and len(r.saved) == 1
    d = saved_json(r)
    assert d["meta"]["task"] == 2 and d["meta"]["leg"] == "R"
    assert d["guessed_task"] == 2 and d["task_confidence"] == 0.9 and d["corrected"] is False
    assert r.result["auto"] == {"task": 2, "confidence": 0.9}
    shown = screen.render(r, "camera", show_score=True).texts
    assert "Task detected: hop on one leg · 90%" in shown and "LAST SAVED TRY" in shown


def test_space_stopped_try_is_also_auto_saved(tmp_path, guess):
    clock = Clock()
    r = DemoRunner(tmp_path, mode="auto")
    to_idle(r, clock)
    r.press("SPACE", clock())
    stream = FakePoseStream(task=2, leg="R", seed=0)
    for _ in range(60):
        r.feed(stream.read())
    r.press("SPACE", clock())
    assert r.ctrl.state == "idle" and len(r.saved) == 1


def test_hidden_key_wins_and_is_marked_corrected(tmp_path, guess):
    clock = Clock()
    r = DemoRunner(tmp_path, mode="auto")
    to_idle(r, clock)
    r.press("1", clock())
    r.press("L", clock())
    guess["value"] = (2, "R", 0.95)
    run_try(r)
    assert r.ctrl.state == "idle" and len(r.saved) == 1
    d = saved_json(r)
    assert d["meta"]["task"] == 1 and d["meta"]["leg"] == "L"
    assert d["guessed_task"] == 2 and d["corrected"] is True


def test_unsure_guess_shows_the_confirm_card(tmp_path, guess):
    clock = Clock()
    r = DemoRunner(tmp_path, mode="auto")
    to_idle(r, clock)
    guess["value"] = (2, "R", AUTO_ACCEPT - 0.01)
    run_try(r)
    assert r.ctrl.state == "confirm" and not r.saved
    assert "Check this try" in screen.render(r, "camera", show_score=True).texts
    r.press("ENTER", clock())
    assert len(r.saved) == 1 and saved_json(r)["corrected"] is False
    assert "auto" not in r.result


def test_detecting_chip_only_with_show_score(tmp_path, guess):
    clock = Clock()
    r = DemoRunner(tmp_path, mode="auto")
    to_idle(r, clock)
    guess["value"] = (3, "L", 0.87)
    r.press("SPACE", clock())
    assert "Detecting…" in screen.render(r, "camera", show_score=True).texts
    stream = FakePoseStream(task=3, seed=0)
    for _ in range(60):  # 2 s of the try: the guess runs every 0.5 s after the first 1 s
        r.feed(stream.read())
    assert r.ctrl.state == "recording"
    assert r.detect == {"task": 3, "leg": "L", "confidence": 0.87}
    shown = screen.render(r, "camera", show_score=True).texts
    assert "Detecting: standing long jump · 87%" in shown
    hidden = " ".join(screen.render(r, "camera", show_score=False).texts).lower()
    assert "detecting" not in hidden
    r.press("2", clock())
    assert "Task set by key: hop on one leg" in screen.render(r, "camera", show_score=True).texts


def test_testing_mode_unchanged(tmp_path, guess):
    """Runner (no --show-score) never auto-saves, keeps the buttons and never draws a score."""
    clock = Clock()
    r = Runner(tmp_path, mode="auto")
    to_idle(r, clock)
    guess["value"] = (2, "R", 0.99)
    run_try(r)
    assert r.ctrl.state == "confirm" and not r.saved and TASK_KEYS <= keys(r)
    texts = screen.render(r, "camera", show_score=False).texts
    r.press("ENTER", clock())
    texts += screen.render(r, "camera", show_score=False).texts
    assert len(r.saved) == 1
    joined = " ".join(texts).lower()
    for word in SCORE_WORDS:
        assert word not in joined, word
