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

import time

import numpy as np

from common.format import make_fake_try

# Halpe-26 index of every BODY_25 point (inverse use of live.convert.HALPE_TO_BODY25).
from live.convert import HALPE_TO_BODY25


class PoseStream:
    """Real webcam / video stream. TODO Revallo (Nov 10-13): capture thread, 720p30, time stamps,
    auto-framing off, resample to 30 fps happens later on the whole try (not here)."""

    def __init__(
        self, source=0, mode: str = "balanced", device: str = "cuda", backend: str = "onnxruntime"
    ):
        import cv2  # imported here so tests and CI don't need OpenCV GUI parts

        if device == "cuda":
            try:  # loads the CUDA/cuDNN DLLs installed by onnxruntime-gpu[cuda,cudnn] (Windows)
                import onnxruntime

                onnxruntime.preload_dlls()
            except Exception:
                pass
        from rtmlib import BodyWithFeet  # lazy: only the live laptop needs rtmlib

        self.model = BodyWithFeet(mode=mode, to_openpose=False, backend=backend, device=device)
        self.cap = cv2.VideoCapture(source)
        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.t0 = time.perf_counter()

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None
        t = time.perf_counter() - self.t0
        keypoints, scores = self.model(frame)  # detector runs on the unmirrored picture
        keypoints = np.asarray(keypoints, dtype=np.float32).reshape(-1, 26, 2)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1, 26)
        return t, keypoints, scores, frame

    def close(self):
        self.cap.release()


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
