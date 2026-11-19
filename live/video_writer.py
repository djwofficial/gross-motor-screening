"""Per-attempt video recording for the live GUI. Owner: Revallo."""

from __future__ import annotations

import math
import re
from collections import deque
from pathlib import Path
from threading import Condition, Thread
from uuid import uuid4

import numpy as np


class TryVideoRecorder:
    """Offer unmirrored frames from the GUI and publish only on confirmation.

    Use one recorder per attempt, from a single GUI thread. The worker owns
    encoder initialization, writes and release. Overload drops incoming frames
    rather than blocking pose processing. An encoded clip is fixed-FPS and
    cannot recover dropped frames. The Try's timestamps remain authoritative.
    """

    def __init__(
        self,
        out_dir: str | Path,
        *,
        child: str,
        consent_level: int | None,
        fps: float,
        frame_size: tuple[int, int],
        queue_size: int = 4,
    ):
        if not re.fullmatch(r"[A-Za-z0-9]{1,8}", child):
            raise ValueError("Video filename needs an alphanumeric child code of 1–8 characters")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video FPS must be finite and positive")
        width, height = frame_size
        if width <= 0 or height <= 0 or width % 2 or height % 2:
            raise ValueError("Video width and height must be positive and even")
        if queue_size < 1:
            raise ValueError("Video queue size must be positive")
        token = uuid4().hex
        directory = Path(out_dir)
        self._staged = directory / f".{child}_{token}.partial.avi"
        self._destination = directory / f"{child}_{token}.avi"
        self._allowed = consent_level == 3
        self._fps = fps
        self._frame_size = frame_size
        self._shape = (height, width, 3)
        self._capacity = queue_size
        self._pending: deque[np.ndarray] = deque()
        self._changed = Condition()
        self._worker: Thread | None = None
        self._error: Exception | None = None
        self._closed = self._discard = self._finished = False
        self._result: Path | None = None
        self._offered = self._written = self._dropped = 0

    def offer(self, frame: np.ndarray, *, enabled: bool) -> bool:
        """Queue a snapshot if V is on, consent permits it and capacity remains."""
        if not enabled or not self._allowed:
            return False
        if frame.dtype != np.uint8 or frame.shape != self._shape:
            raise ValueError("Video frame must match the configured uint8 BGR frame size")
        with self._changed:
            if self._closed or self._error is not None:
                return False
            self._offered += 1
            if len(self._pending) >= self._capacity:
                self._dropped += 1
                return False
            self._pending.append(frame.copy())
            if self._worker is None:
                self._worker = Thread(target=self._run, name="try-video-writer", daemon=True)
                self._worker.start()
            self._changed.notify()
            return True

    @property
    def counts(self) -> tuple[int, int, int]:
        """Return offered, written and queue-overflow frame counts."""
        with self._changed:
            return self._offered, self._written, self._dropped

    def close(self, *, keep: bool, timeout: float = 2.0) -> Path | None:
        """Stop admission, then publish a confirmed clip or discard an attempt."""
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Video shutdown timeout must be finite and positive")
        with self._changed:
            if not self._closed:
                self._closed, self._discard = True, not keep
                if self._discard:
                    self._pending.clear()
                self._changed.notify()
            worker = self._worker
        if worker is None:
            return None
        worker.join(timeout)
        if worker.is_alive():
            with self._changed:
                self._discard = True
                self._pending.clear()
                finished = self._finished
                self._changed.notify()
            if finished:
                self._staged.unlink(missing_ok=True)
            raise TimeoutError("Video encoder has not stopped. The clip will be discarded.")
        if self._error is not None:
            raise OSError(f"Video encoding failed: {self._error}") from self._error
        if self._written and not self._discard and self._result is None:
            try:
                self._staged.replace(self._destination)
            except OSError:
                self._staged.unlink(missing_ok=True)
                raise
            self._result = self._destination
        return self._result

    def _run(self) -> None:
        try:
            import cv2

            self._staged.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(str(self._staged), fourcc, self._fps, self._frame_size)
            try:
                if not writer.isOpened():
                    raise OSError("OpenCV cannot open the MJPG video encoder")
                while True:
                    with self._changed:
                        self._changed.wait_for(lambda: self._pending or self._closed)
                        if not self._pending:
                            break
                        frame = self._pending.popleft()
                    writer.write(frame)
                    with self._changed:
                        self._written += 1
            finally:
                writer.release()
        except Exception as exc:
            with self._changed:
                self._error = exc
                self._pending.clear()
        finally:
            with self._changed:
                self._finished = True
                remove = self._discard or self._error is not None or not self._written
            if remove:
                try:
                    self._staged.unlink(missing_ok=True)
                except OSError as exc:
                    with self._changed:
                        if self._error is None:
                            self._error = exc
