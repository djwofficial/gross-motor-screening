"""app/ui.py + live/video_writer.py (owners Freya, Revallo): the V key really saves video.

Video is recorded only for consent level 3 with V on, published on Enter, discarded on
Backspace or when the window closes. A video failure never loses the points.
"""

import csv
import json

import numpy as np
import pytest

from app.ui import Runner
from live.pose import FakePoseStream

SIZE = (64, 48)  # small even frame size keeps the test fast


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 10.0
        return self.t


def items_with_frames(task=1, seed=0):
    """FakePoseStream items with a real BGR picture instead of None."""
    s = FakePoseStream(task=task, seed=seed)
    k = 0
    while (item := s.read()) is not None:
        t, kp, sc, _ = item
        kp = kp * np.array([SIZE[0] / 1280, SIZE[1] / 720], dtype=np.float32)
        frame = np.full((SIZE[1], SIZE[0], 3), k % 255, dtype=np.uint8)
        k += 1
        yield t, kp, sc, frame


def start(tmp_path, consent="3", video_on=True):
    clock = Clock()
    r = Runner(tmp_path, mode="manual", frame_size=SIZE, source="webcam")
    for k in ["K", "0", "1", "ENTER", "4", "5", "ENTER", consent, "ENTER"]:
        r.press(k)
    r.press("ENTER", clock())
    if video_on:
        r.press("V", clock())
    return r, clock


def one_try(r, clock, keep=True):
    r.press("SPACE", clock())
    for item in items_with_frames():
        r.feed(item)
    r.press("SPACE", clock())
    r.press("ENTER" if keep else "BACKSPACE", clock())


def rows(path):
    return list(csv.DictReader((path / "log.csv").open(encoding="utf-8")))


def videos(path):
    return sorted(path.glob("*.avi"))


def test_confirmed_try_saves_a_video_named_in_the_json(tmp_path):
    cv2 = pytest.importorskip("cv2")
    r, clock = start(tmp_path)
    one_try(r, clock)
    clips = videos(tmp_path)
    assert len(clips) == 1 and clips[0].name.startswith("K01_")
    rec = json.loads(r.saved[0].read_text(encoding="utf-8"))
    assert rec["meta"]["video_file"] == clips[0].name and rec["meta"]["video"] is True
    assert rows(tmp_path)[0]["video"] == "1"
    cap = cv2.VideoCapture(str(clips[0]))
    ok, img = cap.read()
    cap.release()
    assert ok and img.shape == (SIZE[1], SIZE[0], 3)
    assert "Video saved" in r.ctrl.message
    assert r.session.video is True  # V stays on for the next try


def test_cancelled_try_leaves_no_video(tmp_path):
    r, clock = start(tmp_path)
    one_try(r, clock, keep=False)
    assert videos(tmp_path) == [] and not list(tmp_path.glob(".*partial*"))
    assert not (tmp_path / "log.csv").exists()


@pytest.mark.parametrize(("consent", "video_on"), [("1", True), ("2", True), ("3", False)])
def test_no_video_without_level_3_and_v_on(tmp_path, consent, video_on):
    r, clock = start(tmp_path, consent=consent, video_on=video_on)
    one_try(r, clock)
    assert videos(tmp_path) == []
    assert rows(tmp_path)[0]["video"] == "0"
    assert "video_file" not in json.loads(r.saved[0].read_text(encoding="utf-8"))["meta"]


def test_video_failure_keeps_the_points(tmp_path, monkeypatch):
    from live.video_writer import TryVideoRecorder

    def broken_close(self, *, keep, timeout=2.0):
        raise OSError("disk full")

    monkeypatch.setattr(TryVideoRecorder, "close", broken_close)
    r, clock = start(tmp_path)
    one_try(r, clock)
    assert len(r.saved) == 1 and rows(tmp_path)[0]["video"] == "0"
    assert "Video not saved: disk full" in r.ctrl.message and "points are saved" in r.ctrl.message


def test_wrong_frame_size_stops_video_but_not_the_try(tmp_path):
    r, clock = start(tmp_path)
    r.press("SPACE", clock())
    for t, kp, sc, _frame in items_with_frames():
        r.feed((t, kp, sc, np.zeros((10, 10, 3), np.uint8)))
    r.press("SPACE", clock())
    r.press("ENTER", clock())
    assert videos(tmp_path) == [] and len(r.saved) == 1
    assert "Video not recorded" in r.ctrl.message


def test_closing_the_window_mid_try_discards_the_clip(tmp_path):
    r, clock = start(tmp_path)
    r.press("SPACE", clock())
    for item in items_with_frames():
        r.feed(item)
    r.close()
    assert videos(tmp_path) == [] and not list(tmp_path.glob(".*partial*"))
