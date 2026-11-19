"""Pose stream behavior without a camera, RTMLib downloads, or a GPU."""

import queue
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from live import pose


class VideoCapture:
    def __init__(self, frames, timestamps=None, fps=30, opened=True):
        self.frames = frames
        self.timestamps = timestamps if timestamps is not None else [0] * len(frames)
        self.fps = fps
        self.opened = opened
        self.index = 0
        self.releases = 0
        self.settings = []

    def isOpened(self):
        return self.opened

    def get(self, prop):
        if prop == 5:
            return self.fps
        if prop == 0:
            return self.timestamps[self.index - 1]
        return 0

    def set(self, prop, value):
        self.settings.append((prop, value))
        return True

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        self.releases += 1


class CameraCapture(VideoCapture):
    def __init__(self):
        super().__init__([])
        self.queue = queue.Queue()
        self.acquired = threading.Event()

    def read(self):
        value = self.queue.get(timeout=2)
        if isinstance(value, BaseException):
            raise value
        if value is None:
            return False, None
        self.index += 1
        self.acquired.set()
        return True, value

    def release(self):
        super().release()
        self.queue.put(None)


def install_fakes(monkeypatch, cap, model_factory=None):
    seen, initializations = [], []

    def factory(**kwargs):
        initializations.append(kwargs)
        if model_factory is not None:
            return model_factory(**kwargs)

        def model(frame):
            seen.append(frame.copy())
            return np.ones((1, 26, 2)), np.ones((1, 26))

        return model

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(
            VideoCapture=lambda source: cap,
            CAP_PROP_FRAME_WIDTH=3,
            CAP_PROP_FRAME_HEIGHT=4,
            CAP_PROP_FPS=5,
            CAP_PROP_POS_MSEC=0,
        ),
    )
    monkeypatch.setitem(sys.modules, "rtmlib", SimpleNamespace(BodyWithFeet=factory))
    return seen, initializations


def test_video_preserves_frames_and_media_time_despite_slow_inference(monkeypatch):
    frames = [np.arange(18, dtype=np.uint8).reshape(2, 3, 3) + i for i in range(3)]
    cap = VideoCapture(frames, [500, 540, 620], fps=25)
    seen, initializations = install_fakes(monkeypatch, cap)
    monkeypatch.setattr(pose.time, "perf_counter", lambda: 50000.0)
    stream = pose.PoseStream("clip.mp4", device="cpu")
    assert not initializations
    items = [stream.read(), stream.read(), stream.read()]
    assert [item[0] for item in items] == pytest.approx([0, 0.04, 0.12])
    assert cap.index == 3
    for original, detected, item in zip(frames, seen, items, strict=True):
        np.testing.assert_array_equal(detected, original)
        assert item[3] is original
        assert item[1].dtype == item[2].dtype == np.float32
    assert stream.read() is None
    stream.close()
    assert cap.releases == 1
    assert not stream.used_fps_fallback


@pytest.mark.parametrize("timestamps", [[0, 0, 0], [0, float("nan"), -1]])
def test_video_falls_back_to_frame_period(monkeypatch, timestamps):
    cap = VideoCapture([np.zeros((2, 3, 3), np.uint8)] * 3, timestamps, fps=20)
    install_fakes(monkeypatch, cap)
    with pose.PoseStream("clip.mp4", device="cpu") as stream:
        assert [stream.read()[0] for _ in range(3)] == pytest.approx([0, 0.05, 0.1])
        assert stream.used_fps_fallback


def test_video_without_timestamps_or_fps_errors_and_releases(monkeypatch):
    cap = VideoCapture([np.zeros((2, 3, 3), np.uint8)] * 2, [0, 0], fps=0)
    install_fakes(monkeypatch, cap)
    stream = pose.PoseStream("clip.mp4", device="cpu")
    stream.read()
    with pytest.raises(ValueError, match="re-encode"):
        stream.read()
    assert cap.releases == 1


def test_bad_camera_and_failed_model_initialization_release(monkeypatch):
    cap = VideoCapture([], opened=False)
    install_fakes(monkeypatch, cap)
    with pytest.raises(OSError, match="Cannot open camera"):
        pose.PoseStream(0, device="cpu")
    assert cap.releases == 1

    cap = CameraCapture()

    def fail(**kwargs):
        raise RuntimeError("model initialization error")

    install_fakes(monkeypatch, cap, fail)
    stream = pose.PoseStream(0, device="cpu")
    with pytest.raises(RuntimeError, match="model initialization"):
        stream.read()
    assert cap.releases == 1
    assert not stream._thread.is_alive()


def test_camera_keeps_latest_frame_while_detector_is_busy(monkeypatch):
    cap = CameraCapture()
    entered, finish = threading.Event(), threading.Event()
    clock = iter([10, 10.1, 10.2, 10.3, 10.4])
    monkeypatch.setattr(pose.time, "perf_counter", lambda: next(clock))
    seen = []

    def factory(**kwargs):
        def model(frame):
            seen.append(int(frame[0, 0, 0]))
            if len(seen) == 1:
                entered.set()
                assert finish.wait(2)
            return [], []

        return model

    install_fakes(monkeypatch, cap, factory)
    stream = pose.PoseStream(0, device="cpu")
    cap.queue.put(np.full((2, 3, 3), 1, np.uint8))
    results, failures = [], []

    def read():
        try:
            results.append(stream.read())
        except BaseException as exc:
            failures.append(exc)

    reader = threading.Thread(target=read)
    reader.start()
    try:
        assert entered.wait(2)
        for value in (2, 3):
            cap.acquired.clear()
            cap.queue.put(np.full((2, 3, 3), value, np.uint8))
            assert cap.acquired.wait(2)
        with stream._condition:
            assert stream._condition.wait_for(
                lambda: stream._latest is not None and stream._latest[1][0, 0, 0] == 3,
                timeout=2,
            )
        finish.set()
        reader.join(2)
        assert not failures
        latest = stream.read()
        assert seen == [1, 3]
        assert results[0][0] == pytest.approx(0.1)
        assert latest[0] == pytest.approx(0.3)
        assert latest[1].shape == (0, 26, 2)
        assert latest[2].shape == (0, 26)
        assert (3, 1280) in cap.settings and (4, 720) in cap.settings and (5, 30) in cap.settings
    finally:
        finish.set()
        stream.close()
        reader.join(2)
    stream.close()
    assert cap.releases == 1
    assert not stream._thread.is_alive()


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), RuntimeError("inference error")])
def test_inference_error_or_keyboard_interrupt_stops_camera(monkeypatch, failure):
    cap = CameraCapture()

    def factory(**kwargs):
        def model(frame):
            raise failure

        return model

    install_fakes(monkeypatch, cap, factory)
    stream = pose.PoseStream(0, device="cpu")
    cap.queue.put(np.zeros((2, 3, 3), np.uint8))
    with pytest.raises(type(failure)):
        stream.read()
    assert cap.releases == 1
    assert not stream._thread.is_alive()


def test_camera_reader_exception_reaches_caller(monkeypatch):
    cap = CameraCapture()
    install_fakes(monkeypatch, cap)
    stream = pose.PoseStream(0, device="cpu")
    cap.queue.put(OSError("camera disconnected"))
    with pytest.raises(OSError, match="Camera capture failed"):
        stream.read()
    assert cap.releases == 1
    assert not stream._thread.is_alive()


def test_first_camera_frame_is_selected_after_model_initialization(monkeypatch):
    cap = CameraCapture()
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(pose.time, "perf_counter", lambda: clock.now)
    seen = []

    def factory(**kwargs):
        clock.now = 20.0  # model loading took longer than a typical attempt
        cap.queue.put(np.full((2, 3, 3), 2, np.uint8))
        with stream._condition:
            assert stream._condition.wait_for(
                lambda: stream._latest is not None and stream._latest[1][0, 0, 0] == 2,
                timeout=2,
            )

        def model(frame):
            seen.append(int(frame[0, 0, 0]))
            return [], []

        return model

    install_fakes(monkeypatch, cap, factory)
    stream = pose.PoseStream(0, device="cpu")
    try:
        cap.queue.put(np.full((2, 3, 3), 1, np.uint8))
        with stream._condition:
            assert stream._condition.wait_for(lambda: stream._latest is not None, timeout=2)
        item = stream.read()
        assert item[0] == 20.0 and seen == [2]
    finally:
        stream.close()
    assert not stream._thread.is_alive()


def test_blocked_camera_release_has_a_bounded_close(monkeypatch):
    entered, allow_release, completed = threading.Event(), threading.Event(), threading.Event()

    class BlockingRelease(CameraCapture):
        def release(self):
            entered.set()
            assert allow_release.wait(2)
            super().release()

    cap = BlockingRelease()
    install_fakes(monkeypatch, cap)
    monkeypatch.setattr(pose, "_SHUTDOWN_TIMEOUT_S", 0.1)
    stream = pose.PoseStream(0, device="cpu")
    failures = []

    def close():
        try:
            stream.close()
        except BaseException as exc:
            failures.append(exc)
        finally:
            completed.set()

    closer = threading.Thread(target=close)
    closer.start()
    try:
        assert entered.wait(1)
        assert completed.wait(1), "close must not wait indefinitely for the driver"
        assert len(failures) == 1 and isinstance(failures[0], OSError)
        assert "shutdown timeout" in str(failures[0])
    finally:
        allow_release.set()
        closer.join(2)
        stream.close()
    assert not stream._thread.is_alive()
    assert not stream._release_thread.is_alive()


def test_negative_camera_index_is_rejected_without_opening_cv2(monkeypatch):
    monkeypatch.setitem(sys.modules, "cv2", None)
    with pytest.raises(ValueError, match="Camera index"):
        pose.PoseStream(-1, device="cpu")


@pytest.mark.parametrize("failure", [ValueError("body error"), KeyboardInterrupt()])
def test_context_cleanup_failure_preserves_body_error(monkeypatch, failure):
    cap = VideoCapture([])
    install_fakes(monkeypatch, cap)
    stream = pose.PoseStream("clip.mp4", device="cpu")

    def failed_close():
        raise OSError("cleanup error")

    monkeypatch.setattr(stream, "close", failed_close)
    with pytest.raises(type(failure)) as exc:
        with stream:
            raise failure
    assert any("cleanup error" in note for note in exc.value.__notes__)


def test_context_cleanup_only_failure_propagates(monkeypatch):
    cap = VideoCapture([])
    install_fakes(monkeypatch, cap)
    stream = pose.PoseStream("clip.mp4", device="cpu")

    def failed_close():
        raise OSError("cleanup error")

    monkeypatch.setattr(stream, "close", failed_close)
    with pytest.raises(OSError, match="cleanup error"):
        with stream:
            pass


def test_fake_pose_stream_keeps_shared_interface():
    stream = pose.FakePoseStream(seconds=0.1, idle_before=0, idle_after=0)
    count = 0
    while (item := stream.read()) is not None:
        t, keypoints, scores, frame = item
        assert isinstance(t, float)
        assert keypoints.shape == (1, 26, 2)
        assert scores.shape == (1, 26)
        assert frame is None
        count += 1
    assert count == stream.tr.n_frames
    stream.close()
