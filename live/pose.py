"""Webcam (or video file) -> RTMPose Halpe-26 points per frame. Owner: Revallo.

Shared interface (final, change only through Daryn):

    stream = PoseStream(source=0, mode="balanced", device="cuda")
    item = stream.read()   # None when the video ends
    t, keypoints, scores, frame = item
        t          float, seconds since the stream started (time stamp of the frame)
        keypoints  float32 (P, 26, 2) pixels of the UNMIRRORED picture, P people (P may be 0)
        scores     float32 (P, 26)
        frame      the BGR picture (numpy uint8 H x W x 3), for display and video saving

FakePoseStream gives the same interface from common.format.make_fake_try, so the app and
tests run without a webcam, a GPU or rtmlib.
"""

from __future__ import annotations

import math
import os
import threading
import time

import numpy as np

from common.format import make_fake_try

# Halpe-26 index of every BODY_25 point (inverse use of live.convert.HALPE_TO_BODY25).
from live.convert import HALPE_TO_BODY25

# Operational timeouts, not measured camera performance requirements.
_SHUTDOWN_TIMEOUT_S = 2.0


class PoseStream:
    """Unmirrored frames, with camera acquisition independent of pose inference.

    Cameras keep one latest frame; recorded files decode every frame in media time.
    The model is initialized lazily, including when a benchmark accesses ``model``.
    Disable OS auto-framing manually before a session; OpenCV has no portable toggle.
    """

    def __init__(
        self, source=0, mode: str = "balanced", device: str = "cuda", backend: str = "onnxruntime"
    ):
        if isinstance(source, (int, np.integer)) and source < 0:
            raise ValueError("Camera index must be zero or greater")
        import cv2  # imported here so tests and CI don't need OpenCV GUI parts

        self._cv2 = cv2
        self._mode, self._device, self._backend = mode, device, backend
        self._model = None
        self._camera = isinstance(source, (int, np.integer))
        self._condition = threading.Condition()
        self._release_lock = threading.Lock()
        self._stop = threading.Event()
        self._latest = None
        self._capture_error = None
        self._capture_done = False
        self._closed = False
        self._released = False
        self._thread = None
        self._release_thread = None
        self._release_error = None
        self._video_index = 0
        self._media_origin = None
        self._fps_anchor = None
        self._last_video_t = None
        self.used_fps_fallback = False
        self.cap = cv2.VideoCapture(int(source) if self._camera else os.fspath(source))
        self.t0 = time.perf_counter()
        try:
            if not self.cap.isOpened():
                raise OSError(f"Cannot open {'camera' if self._camera else 'video'}: {source}")
            if self._camera:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
            reported_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
            self.source_fps = (
                reported_fps if math.isfinite(reported_fps) and reported_fps > 0 else None
            )
            if self._camera:
                self._thread = threading.Thread(
                    target=self._capture_camera, name="pose-camera-capture", daemon=True
                )
                self._thread.start()
        except BaseException as exc:
            self._close_after_error(exc)
            raise

    @property
    def model(self):
        """Expose the BodyWithFeet instance (and its detector/pose sessions) to benchmarks."""
        if self._model is None:
            if self._closed:
                raise RuntimeError("The pose stream is closed")
            try:
                if self._device == "cuda":
                    try:
                        import onnxruntime

                        onnxruntime.preload_dlls()
                    except (ImportError, AttributeError, OSError):
                        pass
                from rtmlib import BodyWithFeet

                self._model = BodyWithFeet(
                    mode=self._mode,
                    to_openpose=False,
                    backend=self._backend,
                    device=self._device,
                )
            except BaseException as exc:
                self._close_after_error(exc)
                raise
        return self._model

    def _close_after_error(self, error):
        try:
            self.close()
        except BaseException as cleanup_error:
            error.add_note(f"Stream cleanup also failed: {cleanup_error}")

    def _release_capture(self):
        with self._release_lock:
            if not self._released:
                self.cap.release()
                self._released = True

    def _capture_camera(self):
        previous_t = -math.inf
        try:
            while not self._stop.is_set():
                ok, frame = self.cap.read()
                acquired_t = time.perf_counter() - self.t0
                if self._stop.is_set():
                    break
                if not ok or frame is None:
                    raise OSError("Camera stopped providing frames; check its connection")
                if not math.isfinite(acquired_t) or acquired_t < previous_t:
                    raise OSError("Camera acquisition clock did not increase")
                if acquired_t == previous_t:
                    acquired_t = float(np.nextafter(previous_t, math.inf))
                previous_t = acquired_t
                with self._condition:
                    self._latest = acquired_t, frame
                    self._condition.notify_all()
        except BaseException as exc:
            if not self._stop.is_set():
                with self._condition:
                    self._capture_error = exc
        finally:
            with self._condition:
                self._capture_done = True
                self._condition.notify_all()
            try:
                self._release_capture()
            except BaseException as exc:
                self._release_error = exc
                with self._condition:
                    self._condition.notify_all()

    def _video_timestamp(self):
        media_s = float(self.cap.get(self._cv2.CAP_PROP_POS_MSEC)) / 1000.0
        valid_media = math.isfinite(media_s) and media_s >= 0
        if self._video_index == 0:
            if not valid_media and self.source_fps is None:
                raise ValueError(
                    "Video has no usable timestamps or fps; re-encode it with timestamps"
                )
            self._media_origin = media_s if valid_media else 0.0
            candidate = 0.0
            self.used_fps_fallback = not valid_media
            if not valid_media:
                self._fps_anchor = (0, 0.0)
        elif self._fps_anchor is not None:
            index, anchor_t = self._fps_anchor
            candidate = anchor_t + (self._video_index - index) / self.source_fps
        else:
            candidate = media_s - self._media_origin if valid_media else math.nan
            if not math.isfinite(candidate) or candidate <= self._last_video_t:
                if self.source_fps is None:
                    raise ValueError(
                        "Video timestamps repeat or go backwards and fps is unavailable; "
                        "re-encode the clip with timestamps"
                    )
                candidate = self._last_video_t + 1.0 / self.source_fps
                self.used_fps_fallback = True
                self._fps_anchor = (self._video_index - 1, self._last_video_t)
        self._last_video_t = candidate
        self._video_index += 1
        return candidate

    def read(self):
        if self._closed:
            return None
        try:
            model = self.model
            if self._camera:
                with self._condition:
                    available = self._condition.wait_for(
                        lambda: self._latest is not None or self._capture_done or self._closed,
                        timeout=5.0,
                    )
                    if not available:
                        raise OSError("Timed out waiting for a camera frame")
                    if self._closed:
                        return None
                    if self._latest is None:
                        if self._capture_error is not None:
                            raise OSError("Camera capture failed") from self._capture_error
                        return None
                    t, frame = self._latest
                    self._latest = None
            else:
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    self.close()
                    return None
                t = self._video_timestamp()
            keypoints, scores = model(frame)
            keypoints = np.asarray(keypoints, dtype=np.float32).reshape(-1, 26, 2)
            scores = np.asarray(scores, dtype=np.float32).reshape(-1, 26)
            if keypoints.shape[:-1] != scores.shape:
                raise ValueError("Pose detector returned mismatched keypoints and scores")
            return t, keypoints, scores, frame
        except BaseException as exc:
            self._close_after_error(exc)
            raise

    def close(self):
        self._closed = True
        self._stop.set()
        with self._condition:
            self._latest = None
            self._condition.notify_all()
        thread_started = self._thread is not None and self._thread.ident is not None
        if not thread_started:
            self._release_capture()
            return
        if threading.current_thread() is self._thread:
            return
        deadline = time.monotonic() + _SHUTDOWN_TIMEOUT_S
        # The capture worker owns normal release. If read/release blocks, a daemon helper
        # can ask the driver to release without making close() block indefinitely.
        self._thread.join(timeout=min(0.05, _SHUTDOWN_TIMEOUT_S))
        if self._thread.is_alive() and self._release_thread is None:

            def release():
                try:
                    self._release_capture()
                except BaseException as exc:
                    self._release_error = exc

            self._release_thread = threading.Thread(
                target=release, name="pose-camera-release", daemon=True
            )
            self._release_thread.start()
        threads = [self._thread, self._release_thread]
        for thread in threads:
            if thread is not None:
                thread.join(timeout=max(0, deadline - time.monotonic()))
        if any(thread is not None and thread.is_alive() for thread in threads):
            raise OSError(
                "Camera driver did not stop within the shutdown timeout; restart the camera"
            )
        if self._release_error is not None:
            raise OSError("Camera release failed") from self._release_error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_value is None:
            self.close()
        else:
            self._close_after_error(exc_value)


class FakePoseStream:
    """Same interface as PoseStream, built from a fake try (one person, no picture)."""

    def __init__(
        self,
        task: int = 1,
        leg: str = "L",
        seconds: float = 4.0,
        idle_before: float = 2.0,
        idle_after: float = 2.5,
        seed: int = 0,
    ):
        self.tr = make_fake_try(
            task=task,
            leg=leg,
            seconds=seconds,
            idle_before=idle_before,
            idle_after=idle_after,
            seed=seed,
        )
        self.i = 0
        idx = np.array(HALPE_TO_BODY25)
        n = self.tr.n_frames
        self.kp26 = np.zeros((n, 26, 2), dtype=np.float32)
        self.sc26 = np.zeros((n, 26), dtype=np.float32)
        self.kp26[:, idx, :] = self.tr.points[:, :, :2]
        self.sc26[:, idx] = self.tr.points[:, :, 2]
        self.kp26[:, 17, :] = self.tr.points[:, 0, :2] + np.array(
            [0, -20], dtype=np.float32
        )  # head
        self.sc26[:, 17] = 0.9

    def read(self):
        if self.i >= self.tr.n_frames:
            return None
        i = self.i
        self.i += 1
        return float(self.tr.t[i]), self.kp26[i : i + 1], self.sc26[i : i + 1], None

    def close(self):
        pass
