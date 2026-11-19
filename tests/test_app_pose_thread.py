"""app/ui.py (owner Freya): pose detection runs in its own thread (PoseWorker).

Revallo measured 21.7 fps for pose alone (#56) but 13.6 fps in the app window (#65), because the
window drew and checked keys between two pose detections. With PoseWorker the window no longer
makes pose wait. These tests check that every frame still arrives, in order, that errors and the
end of a video reach the window, that stop() works, and that slow drawing no longer slows pose.
"""

import csv
import sys
import threading
import time

import numpy as np

import app.ui as ui
from app.ui import PoseWorker


class ListStream:
    def __init__(self, n, per_s=0.0, fail_at=None):
        self.n, self.per_s, self.fail_at, self.i = n, per_s, fail_at, 0
        self.closed = False

    def read(self):
        if self.per_s:
            time.sleep(self.per_s)
        if self.fail_at is not None and self.i == self.fail_at:
            raise OSError("Timed out waiting for a camera frame")
        if self.i >= self.n:
            return None
        self.i += 1
        return (self.i / 30.0, None, None, None)

    def close(self):
        self.closed = True


def take_all(w, timeout=5.0):
    got, end = [], time.monotonic() + timeout
    while time.monotonic() < end:
        got += w.take(wait_s=0.01)
        if w.ended and w.items.empty():
            break
    return got


def test_every_frame_arrives_in_order():
    w = PoseWorker(ListStream(500), max_items=16)  # small queue: the worker must wait, not drop
    got = take_all(w)
    assert [it[0] for it in got] == [(i + 1) / 30.0 for i in range(500)]
    assert w.ended and w.error is None and w.stop()


def test_camera_error_reaches_the_window():
    w = PoseWorker(ListStream(100, fail_at=40))
    got = take_all(w)
    assert len(got) == 40 and w.ended
    assert isinstance(w.error, OSError) and "Timed out" in str(w.error)


def test_stop_ends_the_thread_quickly():
    w = PoseWorker(ListStream(10_000, per_s=0.01))
    time.sleep(0.05)
    t0 = time.monotonic()
    assert w.stop(timeout=2.0) and time.monotonic() - t0 < 1.0


def test_stop_waits_for_a_blocked_read_then_gives_up():
    gate = threading.Event()

    class Blocked:
        def read(self):
            gate.wait(5)
            return None

    w = PoseWorker(Blocked())
    assert w.stop(timeout=0.1) is False  # still inside read(): run_gui then closes the stream
    gate.set()
    assert w.stop(timeout=2.0)


def test_fake_stream_is_played_at_its_own_speed():
    w = PoseWorker(ListStream(9), paced=True)  # 9 frames at 30 fps: about 0.27 s
    t0 = time.monotonic()
    got = take_all(w)
    assert len(got) == 9 and time.monotonic() - t0 >= 0.2


def test_slow_drawing_no_longer_slows_pose(tmp_path, monkeypatch):
    """Pose takes 20 ms per frame and drawing 20 ms: one after the other that is 25 fps at best.

    With the pose thread the saved try must keep well above that.
    """

    class PoseLike:
        def __init__(self):
            from live.pose import FakePoseStream

            self.fake = FakePoseStream(task=1, seconds=30, idle_before=0, idle_after=0)
            self.t0 = time.perf_counter()

        def read(self):
            time.sleep(0.02)  # like pose detection on the GPU
            item = self.fake.read()
            return time.perf_counter() - self.t0, item[1], item[2], None

        def close(self):
            pass

    plan = [ord("t"), ord("1"), 13, ord("4"), ord("5"), 13, ord("1"), 13, 13, 32]
    keys = iter(plan)
    t_space = []

    def wait_key(ms):
        k = next(keys, None)
        if k == 32:
            t_space.append(time.monotonic())
        if k is None and t_space and time.monotonic() - t_space[0] > 2.0 and len(t_space) == 1:
            t_space.append(time.monotonic())
            return 32  # stop after about 2 s
        if k is None and len(t_space) == 2:
            t_space.append(0)
            return 13  # save
        if k is None and len(t_space) == 3:
            return ord("q")
        return -1 if k is None else k

    fake_cv2 = type(
        "cv2",
        (),
        {
            "WINDOW_NORMAL": 0,
            "WND_PROP_VISIBLE": 0,
            "namedWindow": staticmethod(lambda *a: None),
            "imshow": staticmethod(lambda *a: time.sleep(0.02)),  # slow drawing
            "waitKeyEx": staticmethod(wait_key),
            "getWindowProperty": staticmethod(lambda *a: 1),
            "destroyAllWindows": staticmethod(lambda: None),
        },
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.setattr(ui, "draw_screen", lambda r, view: None)  # imshow above is the slow part
    monkeypatch.setattr(ui.Controller.__init__, "__defaults__", ("auto", 0.0))  # no key debounce
    saved = ui.run_gui(PoseLike(), tmp_path, mode="manual", source="webcam")
    assert len(saved) == 1
    row = next(csv.DictReader((tmp_path / "log.csv").open(encoding="utf-8")))
    fps = int(row["n_frames"]) / float(row["duration_s"])
    assert fps > 35, fps  # one after the other it would be about 25
    assert np.isfinite(fps)
