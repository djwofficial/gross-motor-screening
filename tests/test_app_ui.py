"""app/ui.py (owner Freya): keys, states, testing mode, automatic/manual mode, crash safety.

Runs without a window: the Runner is driven with fake stream items and key names.
"""

import csv
import subprocess
import sys
import textwrap

import pytest

from app.session import LOG_FIELDS, load_record
from app.ui import (
    Controller,
    FakeLoopStream,
    Runner,
    big_lines,
    draw_screen,
    key_name,
    status_lines,
)
from live.pose import FakePoseStream


class Clock:
    """Key times far apart, so the debounce never hides a key unless a test wants it to."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 10.0
        return self.t


def setup_child(r, code="K01", age="47", consent="3", clock=None):
    for ch in code:
        r.press(ch)
    r.press("ENTER")
    for ch in age:
        r.press(ch)
    r.press("ENTER")
    r.press(consent)
    r.press("ENTER")
    assert r.ctrl.state == "readiness"
    r.press("ENTER", clock())
    assert r.ctrl.state == "idle"


def feed_all(r, stream):
    while (item := stream.read()) is not None:
        r.feed(item)


def rows(path):
    return list(csv.DictReader((path / "log.csv").open(encoding="utf-8")))


def test_key_names():
    assert key_name(13) == "ENTER" and key_name(32) == "SPACE" and key_name(8) == "BACKSPACE"
    assert key_name(ord("l")) == "L" and key_name(ord("2")) == "2" and key_name(-1) is None


def test_setup_validates_fields(tmp_path):
    r = Runner(tmp_path)
    r.press("ENTER")
    assert r.ctrl.state == "setup" and "empty" in r.ctrl.message
    for ch in "K07":
        r.press(ch)
    r.press("ENTER")
    r.press("ENTER")  # age empty
    assert r.ctrl.field == 1 and "Age" in r.ctrl.message
    r.press("4")
    r.press("X")  # ignored: not a digit
    r.press("2")
    r.press("ENTER")
    r.press("5")  # not a consent level
    r.press("ENTER")
    assert r.ctrl.state == "setup"
    r.press("2")
    r.press("ENTER")
    assert (r.ctrl.child, r.ctrl.age, r.ctrl.consent) == ("K07", 42, 2)
    assert r.ctrl.state == "readiness"


def test_manual_mode_all_keys(tmp_path):
    clock = Clock()
    r = Runner(tmp_path, mode="manual")
    setup_child(r, clock=clock)
    stream = FakePoseStream(task=2, leg="R")
    for _ in range(30):
        r.feed(stream.read())
    r.press("2", clock())  # preset task
    r.press("R", clock())  # preset leg
    r.press("SPACE", clock())
    assert r.ctrl.state == "recording"
    for _ in range(150):
        r.feed(stream.read())
    r.press("SPACE", clock())
    assert r.ctrl.state == "confirm"
    assert (r.ctrl.choice_task, r.ctrl.choice_leg) == (2, "R")  # preset wins over the stub guess
    r.press("1", clock())
    r.press("2", clock())
    r.press("L", clock())
    r.press("R", clock())
    r.press("ENTER", clock())
    assert r.ctrl.state == "idle" and len(r.saved) == 1
    assert r.saved[0].name == "K01_task2_try1_R.json"
    row = rows(tmp_path)[0]
    assert row["task"] == "2" and row["leg"] == "R" and row["corrected"] == "1"  # stub guesses 1
    assert row["consent_level"] == "3" and row["age_in_range"] == "1"


def test_automatic_mode_finds_start_and_end(tmp_path):
    clock = Clock()
    r = Runner(tmp_path, mode="auto")
    setup_child(r, clock=clock)
    seen = set()
    stream = FakePoseStream(task=3, leg="L")
    while (item := stream.read()) is not None:
        r.feed(item)
        seen.add(r.ctrl.state)
    assert {"ready", "recording", "confirm"} <= seen
    assert r.ctrl.state == "confirm"
    r.press("3", clock())
    r.press("ENTER", clock())
    rec = load_record(r.saved[0])
    assert rec["meta"]["mode"] == "auto" and rec["events"]
    assert rec["points26"] is not None and len(rec["points26"]) == rec["try"].n_frames


def test_both_modes_write_the_same_fields(tmp_path):
    clock = Clock()
    out = {}
    for mode in ("auto", "manual"):
        r = Runner(tmp_path / mode, mode=mode)
        setup_child(r, clock=clock)
        stream = FakePoseStream(task=1, leg="L")
        if mode == "manual":
            r.press("SPACE", clock())
        feed_all(r, stream)
        if mode == "manual":
            r.press("SPACE", clock())
        r.press("1", clock())
        r.press("ENTER", clock())
        out[mode] = (load_record(r.saved[0]), rows(tmp_path / mode))
    (ra, la), (rm, lm) = out["auto"], out["manual"]
    assert set(ra) == set(rm)
    assert list(la[0]) == list(lm[0]) == LOG_FIELDS
    assert len(la) == len(lm) == 1
    for k in ("child", "task", "leg", "consent_level", "guessed_task", "corrected"):
        assert la[0][k] == lm[0][k], k


def test_backspace_cancels_and_double_enter_saves_once(tmp_path):
    r = Runner(tmp_path, mode="manual")
    clock = Clock()
    setup_child(r, clock=clock)
    stream = FakeLoopStream()
    r.press("SPACE", clock())
    for _ in range(40):
        r.feed(stream.read())
    r.press("BACKSPACE", clock())  # cancel while recording
    assert r.ctrl.state == "idle" and r.buf is None
    r.press("SPACE", clock())
    for _ in range(40):
        r.feed(stream.read())
    r.press("SPACE", clock())
    r.press("BACKSPACE", clock())  # cancel at the confirm step
    assert r.ctrl.state == "idle" and r.pending is None
    r.press("SPACE", clock())
    for _ in range(40):
        r.feed(stream.read())
    r.press("SPACE", clock())
    t = clock()
    r.press("ENTER", t)
    r.press("ENTER", t + 0.05)  # bounce
    r.press("ENTER", t + 5.0)  # a later Enter in idle does nothing either
    assert len(r.saved) == 1 and len(rows(tmp_path)) == 1


def test_video_only_for_consent_level_3(tmp_path):
    clock = Clock()
    r = Runner(tmp_path)
    setup_child(r, code="K02", consent="1", clock=clock)
    r.press("V", clock())
    assert r.ctrl.video is False and "level 3" in r.ctrl.message
    r.press("N", clock())
    assert r.ctrl.state == "setup"
    setup_child(r, code="K03", consent="3", clock=clock)
    r.press("V", clock())
    assert r.ctrl.video is True and r.session.video is True
    r.press("V", clock())
    assert r.session.video is False


def test_next_child_and_quit_wait_for_the_open_try(tmp_path):
    clock = Clock()
    r = Runner(tmp_path, mode="manual")
    setup_child(r, clock=clock)
    r.press("SPACE", clock())
    r.press("N", clock())
    r.press("Q", clock())
    assert r.ctrl.state == "recording" and not r.quit
    r.press("BACKSPACE", clock())
    r.press("Q", clock())
    assert r.quit


def test_age_outside_range_is_flagged(tmp_path):
    clock = Clock()
    r = Runner(tmp_path, mode="manual")
    setup_child(r, code="K09", age="60", clock=clock)
    assert "outside 36-53" in status_lines(r.ctrl)[1]
    stream = FakePoseStream(task=1)
    r.press("SPACE", clock())
    feed_all(r, stream)
    r.press("SPACE", clock())
    r.press("ENTER", clock())
    assert rows(tmp_path)[0]["age_in_range"] == "0"


def test_screen_never_shows_a_score_or_timeline(tmp_path):
    clock = Clock()
    r = Runner(tmp_path, mode="manual")
    setup_child(r, clock=clock)
    texts = []
    for task in (1, 2, 3):
        stream = FakePoseStream(task=task)
        r.press("SPACE", clock())
        feed_all(r, stream)
        r.press("SPACE", clock())
        texts += status_lines(r.ctrl) + big_lines(r.ctrl)
        r.press(str(task), clock())
        r.press("ENTER", clock())
        texts += status_lines(r.ctrl) + big_lines(r.ctrl)
        img = draw_screen(r, "stick")
        assert img.shape[:2] == (720 + 150, 1280)
    joined = " ".join(texts).lower()
    for word in ("score", "bad", "good", "perfect", "check by teacher", "timeline", "steady"):
        assert word not in joined, word
    assert len(rows(tmp_path)) == 3


def test_readiness_problems_in_plain_words(tmp_path):
    r = Runner(tmp_path, mode="auto")
    for k in ["K", "1", "ENTER", "4", "0", "ENTER", "1", "ENTER"]:
        r.press(k)
    r.feed((0.0, FakePoseStream().kp26[:0], FakePoseStream().sc26[:0], None))  # nobody
    assert any("Nobody" in p for p in r.ctrl.problems)
    assert any("Nobody" in line for line in big_lines(r.ctrl))
    stream = FakePoseStream(task=1)
    for _ in range(45):
        r.feed(stream.read())
    assert r.ctrl.problems == [], r.ctrl.problems
    assert big_lines(r.ctrl)[0] == "Readiness check: OK"


def test_five_fake_tries_give_five_json_and_five_rows(tmp_path):
    clock = Clock()
    r = Runner(tmp_path, mode="manual")
    setup_child(r, code="K03", clock=clock)
    for k in range(5):
        stream = FakePoseStream(task=k % 3 + 1, leg="LR"[k % 2], seed=k)
        r.press("SPACE", clock())
        feed_all(r, stream)
        r.press("SPACE", clock())
        r.press(str(k % 3 + 1), clock())
        r.press("LR"[k % 2], clock())
        r.press("ENTER", clock())
    assert len(list(tmp_path.glob("K03_task*_try*_*.json"))) == 5
    assert len(rows(tmp_path)) == 5


def test_killed_mid_session_keeps_earlier_tries(tmp_path):
    script = textwrap.dedent(
        f"""
        import os
        from app.ui import Runner
        from live.pose import FakePoseStream
        r = Runner(r"{tmp_path}", mode="manual")
        for k in ["K", "0", "4", "ENTER", "4", "5", "ENTER", "2", "ENTER", "ENTER"]:
            r.press(k)
        now = 100.0
        for i in range(3):
            s = FakePoseStream(task=1, seed=i)
            now += 10; r.press("SPACE", now)
            while (item := s.read()) is not None:
                r.feed(item)
            if i == 2:
                os._exit(1)  # killed while the third try is being recorded
            now += 10; r.press("SPACE", now)
            now += 10; r.press("ENTER", now)
        """
    )
    proc = subprocess.run([sys.executable, "-c", script], cwd=".", capture_output=True, text=True)
    assert proc.returncode == 1, proc.stderr
    files = sorted(tmp_path.glob("K04_task1_try*_*.json"))
    assert [p.name for p in files] == ["K04_task1_try1_L.json", "K04_task1_try2_L.json"]
    for p in files:
        load_record(p)  # complete, valid files
    assert len(rows(tmp_path)) == 2
    assert not list(tmp_path.glob("*.tmp"))


def test_controller_rejects_unknown_mode():
    with pytest.raises(ValueError):
        Controller(mode="other")


def test_old_log_with_other_columns_is_refused(tmp_path):
    from app.session import Session

    (tmp_path / "log.csv").write_text("time,child,age_months\n2026-11-08,K00,48\n")
    with pytest.raises(ValueError, match="other columns"):
        Session(tmp_path, child="K01", age_months=48)
