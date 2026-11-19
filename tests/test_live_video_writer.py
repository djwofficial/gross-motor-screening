"""Video recording gate, asynchronous encoding and confirmed-file behavior."""

import sys
from pathlib import Path
from threading import Event, get_ident
from types import SimpleNamespace

import numpy as np
import pytest

from live.video_writer import TryVideoRecorder


def recorder(directory, consent_level=3, queue_size=4):
    return TryVideoRecorder(
        directory,
        child="TEAM01",
        consent_level=consent_level,
        fps=30,
        frame_size=(32, 24),
        queue_size=queue_size,
    )


@pytest.mark.parametrize(("consent", "enabled"), [(None, True), (1, True), (2, True), (3, False)])
def test_off_or_no_video_consent_creates_nothing(tmp_path, consent, enabled):
    output = tmp_path / "video"
    video = recorder(output, consent)
    assert not video.offer(np.zeros((24, 32, 3), np.uint8), enabled=enabled)
    assert video.close(keep=True) is None
    assert not output.exists()


def test_worker_encodes_without_waiting_in_offer_and_copies_frame(monkeypatch, tmp_path):
    entered, proceed = Event(), Event()
    seen, worker_ids = [], []

    class Writer:
        def __init__(self, path, *args):
            Path(path).touch()
            worker_ids.append(get_ident())

        def isOpened(self):
            return True

        def write(self, frame):
            worker_ids.append(get_ident())
            entered.set()
            if not proceed.wait(3):
                raise TimeoutError("test did not release the encoder")
            seen.append(int(frame[0, 0, 0]))

        def release(self):
            worker_ids.append(get_ident())

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(
            VideoWriter=Writer,
            VideoWriter_fourcc=lambda *args: 0,
        ),
    )
    video = recorder(tmp_path, queue_size=1)
    frame = np.full((24, 32, 3), 11, np.uint8)
    try:
        assert video.offer(frame, enabled=True)
        assert entered.wait(1)
        frame[:] = 22
        assert not video.offer(frame, enabled=False)
        assert video.offer(frame, enabled=True)
        frame[:] = 33
        assert not video.offer(frame, enabled=True)
    finally:
        proceed.set()
        path = video.close(keep=True)
    assert seen == [11, 22]
    assert len(set(worker_ids)) == 1 and get_ident() not in worker_ids
    assert video.counts == (3, 2, 1)
    assert path.name.startswith("TEAM01_") and list(tmp_path.iterdir()) == [path]
    assert not video.offer(frame, enabled=True)


@pytest.mark.parametrize("keep", [True, False])
def test_actual_encoder_saves_readable_clip_only_when_confirmed(tmp_path, keep):
    import cv2

    frame = np.zeros((24, 32, 3), np.uint8)
    frame[:, :16] = 200
    video = recorder(tmp_path)
    assert video.offer(frame, enabled=True)
    path = video.close(keep=keep)
    if not keep:
        assert path is None and not list(tmp_path.iterdir())
        return
    assert path is not None
    cap = cv2.VideoCapture(str(path))
    try:
        ok, picture = cap.read()
        assert ok and picture.shape == frame.shape
        assert picture[:, :16].mean() > picture[:, 16:].mean()
    finally:
        cap.release()


def test_codec_failure_discards_staged_file(monkeypatch, tmp_path):
    def fail_open(path, *args):
        Path(path).touch()
        return SimpleNamespace(isOpened=lambda: False, release=lambda: None)

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(
            VideoWriter=fail_open,
            VideoWriter_fourcc=lambda *args: 0,
        ),
    )
    video = recorder(tmp_path)
    assert video.offer(np.zeros((24, 32, 3), np.uint8), enabled=True)
    with pytest.raises(OSError, match="cannot open"):
        video.close(keep=True)
    assert not list(tmp_path.iterdir())


def test_stalled_encoder_cannot_publish_after_timeout(monkeypatch, tmp_path):
    entered, proceed = Event(), Event()

    def blocking_write(frame):
        entered.set()
        if not proceed.wait(3):
            raise TimeoutError("test did not release the encoder")

    def open_encoder(path, *args):
        Path(path).touch()
        return SimpleNamespace(
            isOpened=lambda: True,
            write=blocking_write,
            release=lambda: None,
        )

    monkeypatch.setitem(
        sys.modules,
        "cv2",
        SimpleNamespace(
            VideoWriter=open_encoder,
            VideoWriter_fourcc=lambda *args: 0,
        ),
    )
    video = recorder(tmp_path)
    assert video.offer(np.zeros((24, 32, 3), np.uint8), enabled=True)
    try:
        assert entered.wait(1)
        with pytest.raises(TimeoutError, match="discarded"):
            video.close(keep=True, timeout=0.05)
    finally:
        proceed.set()
    assert video.close(keep=True) is None
    assert not list(tmp_path.iterdir())
