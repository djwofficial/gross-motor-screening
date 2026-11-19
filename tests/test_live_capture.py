"""Capture and resampling behavior with artificial poses and generated video only."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from common.format import load, validate
from live import capture


def points(values):
    out = np.zeros((len(values), 25, 3), np.float32)
    for i, value in enumerate(values):
        if value:
            out[i, :, :2] = value
            out[i, :, 2] = 0.9
    return out


def install_stream(monkeypatch, timestamps=(0, 1 / 30, 2 / 30), empty=(), error=None):
    instances = []

    class Stream:
        source_fps = 30.0
        used_fps_fallback = False

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.i = 0
            self.closed = False
            instances.append(self)

        def read(self):
            if error is not None:
                raise error
            if self.i == len(timestamps):
                return None
            i = self.i
            self.i += 1
            n = 0 if i in empty else 1
            return (
                timestamps[i],
                np.full((n, 26, 2), i + 1, np.float32),
                np.full((n, 26), 1.5, np.float32),
                np.zeros((24, 32, 3), np.uint8),
            )

        def close(self):
            self.closed = True

    monkeypatch.setattr(capture, "PoseStream", Stream)
    return instances


def test_resampling_nearest_frames_zero_rows_dtypes_and_no_extrapolation():
    original = points([1, 0, 3])
    result, t = capture.resample_to_fps(original, [7.0, 7.1, 7.3], fps=20)
    np.testing.assert_allclose(t, np.arange(7) / 20)
    np.testing.assert_array_equal(result[:, 0, 0], [1, 1, 0, 0, 0, 3, 3])
    assert result.dtype == np.float32 and t.dtype == np.float64
    assert t[0] == 0 and np.all(np.diff(t) > 0)
    result[0] = 0
    assert original[0, 0, 0] == 1


def test_resampling_ties_choose_earlier_and_partial_endpoint_is_not_extended():
    result, t = capture.resample_to_fps(points([1, 3]), [0, 0.2], fps=10)
    np.testing.assert_array_equal(result[:, 0, 0], [1, 1, 3])
    _, t = capture.resample_to_fps(points([1, 2]), [0, 0.099], fps=30)
    assert len(t) == 3
    assert t[-1] <= 0.099
    result, t = capture.resample_to_fps(points([2]), [3.2])
    assert result.shape == (1, 25, 3) and t.tolist() == [0]


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf")])
def test_resampling_rejects_bad_fps(fps):
    with pytest.raises(ValueError, match="fps"):
        capture.resample_to_fps(points([1, 2]), [0, 1], fps)


@pytest.mark.parametrize(
    "bad_points,bad_t",
    [
        (np.zeros((0, 25, 3)), []),
        (np.zeros((2, 26, 3)), [0, 1]),
        (points([1, 2]), [0]),
        (points([1, 2]), [1, 1]),
        (points([1, 2]), [1, 0]),
        (points([1, 2]), [0, float("nan")]),
        (np.full((2, 25, 3), float("inf")), [0, 1]),
        (np.ones((2, 25, 3)) * 2, [0, 1]),
        (np.zeros((2, 25, 3), complex), [0, 1]),
    ],
)
def test_resampling_rejects_invalid_arrays(bad_points, bad_t):
    with pytest.raises(ValueError):
        capture.resample_to_fps(bad_points, bad_t)


def test_capture_uses_shared_picker_and_preserves_empty_frames(monkeypatch):
    instances = install_stream(monkeypatch, timestamps=[8, 8 + 1 / 30, 8 + 2 / 30], empty=[1])
    calls = []
    real_picker = capture.pick_child

    def picker(frames):
        calls.append([frame.shape for frame in frames])
        return real_picker(frames)

    monkeypatch.setattr(capture, "pick_child", picker)
    tr = validate(capture.capture_try("team.mp4", task=2, leg="R", device="cpu"))
    assert calls == [[(1, 25, 3), (0, 25, 3), (1, 25, 3)]]
    assert tr.points.shape == (3, 25, 3)
    assert not np.any(tr.points[1])
    assert tr.points[0, 0, 2] == 1.0
    assert tr.meta["width"] == 32 and tr.meta["height"] == 24
    assert tr.meta["source"] == "video" and tr.meta["source_fps"] == 30
    assert tr.meta["task"] == 2 and tr.meta["leg"] == "R"
    assert tr.t[0] == 0
    assert instances[0].closed


def test_camera_duration_uses_acquisition_time_and_is_bounded(monkeypatch):
    instances = install_stream(monkeypatch, timestamps=[15, 15.05, 15.1, 15.15])
    tr = capture.capture_try(0, task=1, leg="L", seconds=0.1, device="cpu")
    assert instances[0].i == 3 and instances[0].closed
    assert tr.meta["source"] == "webcam" and tr.meta["camera_index"] == 0
    assert tr.t[-1] < 0.1


@pytest.mark.parametrize("empty", [[], [0, 1, 2]])
def test_no_frames_or_all_empty_raises_and_closes(monkeypatch, empty):
    times = [] if not empty else [0, 1 / 30, 2 / 30]
    instances = install_stream(monkeypatch, timestamps=times, empty=empty)
    with pytest.raises(ValueError, match="no readable frames|No person"):
        capture.capture_try("team.mp4", task=1, leg="L", device="cpu")
    assert instances[0].closed


def test_resampling_cannot_save_an_attempt_whose_only_detection_was_dropped(monkeypatch):
    instances = install_stream(
        monkeypatch, timestamps=[0, 0.01, 0.02, 0.03, 0.04], empty=[0, 2, 3, 4]
    )
    with pytest.raises(ValueError, match="No person remains"):
        capture.capture_try("team.mp4", task=1, leg="L", device="cpu")
    assert instances[0].closed


def test_uppercase_output_suffix_is_rejected_before_capture(monkeypatch, tmp_path):
    instances = install_stream(monkeypatch)
    out = tmp_path / "result.NPZ"
    with pytest.raises(SystemExit) as exc:
        capture.main(["--out", str(out), "--task", "1", "--leg", "L"])
    assert exc.value.code == 2
    assert not instances
    assert not out.exists() and not out.with_suffix(".NPZ.npz").exists()


def test_cli_saves_loadable_try_and_all_empty_writes_nothing(monkeypatch, tmp_path):
    video = tmp_path / "team.mp4"
    video.touch()
    out = tmp_path / "result.npz"
    instances = install_stream(monkeypatch, empty=[1])
    args = [
        "--video",
        str(video),
        "--out",
        str(out),
        "--task",
        "3",
        "--leg",
        "L",
        "--device",
        "cpu",
    ]
    assert capture.main(args) == 0
    tr = load(out)
    assert tr.meta["task"] == 3 and tr.meta["leg"] == "L"
    assert tr.n_frames == 3 and not np.any(tr.points[1])
    assert instances[0].closed
    out.unlink()
    install_stream(monkeypatch, empty=[0, 1, 2])
    assert capture.main(args) == 1
    assert not out.exists()


def test_ctrl_c_closes_capture_and_cli_does_not_save(monkeypatch, tmp_path):
    instances = install_stream(monkeypatch, error=KeyboardInterrupt())
    out = tmp_path / "cancelled.npz"
    assert capture.main(["--out", str(out), "--task", "1", "--leg", "L"]) == 130
    assert instances[0].closed
    assert not out.exists()


@pytest.mark.parametrize(
    "failure,status", [(ValueError("pose error"), 1), (KeyboardInterrupt(), 130)]
)
def test_cleanup_failure_preserves_original_error_and_cli_status(
    monkeypatch, tmp_path, capsys, failure, status
):
    instances = install_stream(monkeypatch, error=failure)

    def failed_close(self):
        self.closed = True
        raise OSError("camera cleanup failed")

    monkeypatch.setattr(capture.PoseStream, "close", failed_close)
    with pytest.raises(type(failure)) as exc:
        capture.capture_try(0, task=1, leg="L", device="cpu")
    assert any("camera cleanup failed" in note for note in exc.value.__notes__)
    out = tmp_path / "not-saved.npz"
    assert capture.main(["--out", str(out), "--task", "1", "--leg", "L"]) == status
    error = capsys.readouterr().err
    assert "camera cleanup failed" in error
    if status == 1:
        assert "pose error" in error
    assert all(instance.closed for instance in instances) and not out.exists()


def test_cleanup_only_failure_still_prevents_a_saved_attempt(monkeypatch):
    instances = install_stream(monkeypatch)

    def failed_close(self):
        raise OSError("camera cleanup failed")

    monkeypatch.setattr(capture.PoseStream, "close", failed_close)
    with pytest.raises(OSError, match="camera cleanup failed"):
        capture.capture_try("team.mp4", task=1, leg="L", device="cpu")
    assert len(instances) == 1


def test_generated_recorded_clip_to_valid_try(monkeypatch, tmp_path):
    cv2 = pytest.importorskip("cv2")
    video = tmp_path / "generated.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 15, (32, 24))
    if not writer.isOpened():
        pytest.skip("This OpenCV installation cannot encode the generated MJPG fixture")
    try:
        for i in range(6):
            frame = np.zeros((24, 32, 3), np.uint8)
            frame[:, :16] = 40 + i
            writer.write(frame)
    finally:
        writer.release()
    seen = []

    def factory(**kwargs):
        def model(frame):
            seen.append(frame.copy())
            return np.ones((1, 26, 2), np.float32), np.ones((1, 26), np.float32)

        return model

    monkeypatch.setitem(sys.modules, "rtmlib", SimpleNamespace(BodyWithFeet=factory))
    out = tmp_path / "generated.npz"
    assert (
        capture.main(
            [
                "--video",
                str(video),
                "--out",
                str(out),
                "--task",
                "1",
                "--leg",
                "L",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    tr = load(out)
    assert len(seen) == 6
    assert tr.n_frames == 11
    assert tr.meta["source_fps"] == pytest.approx(15)
    assert tr.meta["width"] == 32 and tr.meta["height"] == 24
    assert all(frame[:, :16].mean() > frame[:, 16:].mean() for frame in seen)
    np.testing.assert_allclose(tr.t, np.arange(11) / 30)
