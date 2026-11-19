"""Benchmark behavior with deterministic clocks; these are not hardware performance results."""

from types import SimpleNamespace

import numpy as np
import pytest

from live.bench_fps import BenchmarkError, main, run_benchmark


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class Session:
    def __init__(self, providers=None):
        self.providers = list(providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.fallback_disabled = False

    def get_providers(self):
        return self.providers

    def disable_fallback(self):
        self.fallback_disabled = True


class Stream:
    def __init__(self, clock, *, step=0.25, people=None, error=None, empty=False, on_read=None):
        self.clock = clock
        self.step = step
        self.people = people or (lambda _i: 1)
        self.error = error
        self.empty = empty
        self.on_read = on_read
        self.reads = 0
        self.closed = False
        self.model = SimpleNamespace(
            det_model=SimpleNamespace(backend="onnxruntime", session=Session()),
            pose_model=SimpleNamespace(backend="onnxruntime", session=Session()),
        )

    def read(self):
        if self.error:
            raise self.error
        if self.empty:
            return None
        self.clock.now += self.step
        self.reads += 1
        if self.on_read:
            self.on_read(self)
        count = self.people(self.reads)
        return (
            self.clock.now,
            np.zeros((count, 26, 2), np.float32),
            np.zeros((count, 26), np.float32),
            None,
        )

    def close(self):
        self.closed = True


def test_warmup_excluded_from_measured_counts_and_actual_elapsed():
    clock = Clock()
    stream = Stream(clock)
    kwargs_seen = {}

    def factory(**kwargs):
        kwargs_seen.update(kwargs)
        clock.now += 100  # model initialization is excluded too
        return stream

    result = run_benchmark(
        source=2, seconds=0.8, warmup_seconds=0.5, stream_factory=factory, clock=clock
    )
    assert kwargs_seen == {"source": 2, "mode": "balanced", "device": "cuda"}
    assert result.warmup.frames == 2
    assert result.warmup.seconds == 0.5
    assert result.measured.frames == 4
    assert result.measured.person_frames == 4
    assert result.measured.seconds == 1.0  # includes the final completed read
    assert result.measured.mean_fps == 4.0
    assert stream.reads == 6 and stream.closed
    assert all(
        model.session.fallback_disabled
        for model in (stream.model.det_model, stream.model.pose_model)
    )


@pytest.mark.parametrize("model_name", ["det_model", "pose_model"])
def test_cpu_only_session_rejected_before_camera_read(model_name):
    clock = Clock()
    stream = Stream(clock)
    getattr(stream.model, model_name).session.providers = ["CPUExecutionProvider"]
    with pytest.raises(BenchmarkError, match="CPU fallback"):
        run_benchmark(stream_factory=lambda **_: stream, clock=clock)
    assert stream.reads == 0 and stream.closed


def test_provider_fallback_after_inference_is_detected():
    clock = Clock()

    def fallback(stream):
        stream.model.pose_model.session.providers = ["CPUExecutionProvider"]

    stream = Stream(clock, on_read=fallback)
    with pytest.raises(BenchmarkError, match="pose session"):
        run_benchmark(
            seconds=0.25, warmup_seconds=0, stream_factory=lambda **_: stream, clock=clock
        )
    assert stream.reads == 1 and stream.closed


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"empty": True}, "Camera returned no frame"),
        ({"people": lambda _i: 0}, "No person detected"),
    ],
)
def test_missing_frames_or_people_do_not_produce_fps_pass(kwargs, message):
    clock = Clock()
    stream = Stream(clock, **kwargs)
    with pytest.raises(BenchmarkError, match=message):
        run_benchmark(seconds=0.5, warmup_seconds=0, stream_factory=lambda **_: stream, clock=clock)
    assert stream.closed


@pytest.mark.parametrize(
    ("error", "status"), [(RuntimeError("read failed"), 2), (KeyboardInterrupt(), 130)]
)
def test_cli_reports_read_failure_or_interrupt_and_releases_camera(error, status, capsys):
    clock = Clock()
    stream = Stream(clock, error=error)
    assert (
        main(
            ["--seconds", "1", "--warmup-seconds", "0"],
            stream_factory=lambda **_: stream,
            clock=clock,
            gpu_info=lambda: "test fixture",
        )
        == status
    )
    output = capsys.readouterr()
    assert "PASS" not in output.out
    assert output.err
    assert stream.closed


def test_cli_success_reports_both_sessions_and_separate_windows(capsys):
    clock = Clock()
    stream = Stream(clock, step=0.03125)
    assert (
        main(
            ["--seconds", "0.125", "--warmup-seconds", "0.0625"],
            stream_factory=lambda **_: stream,
            clock=clock,
            gpu_info=lambda: "test fixture, not real GPU evidence",
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "GPU inventory: test fixture" in output
    assert "detector session providers: CUDAExecutionProvider" in output
    assert "pose session providers: CUDAExecutionProvider" in output
    assert "warmup: 2 frames" in output and "measured: 4 frames" in output
    assert "mean fps: 32.00" in output and "person frames: 4/4" in output
    assert "no kernel profiling" in output and "PASS" in output


def test_empty_scene_reads_cannot_inflate_pass():
    clock = Clock()
    stream = Stream(clock, step=0.03125, people=lambda i: int(i % 4 == 0))
    result = run_benchmark(
        seconds=1, warmup_seconds=0, stream_factory=lambda **_: stream, clock=clock
    )
    assert result.measured.mean_fps == 32
    assert result.measured.person_fps == 8
    assert not result.passes


def test_cpu_run_is_diagnostic_even_with_high_person_fps():
    clock = Clock()
    stream = Stream(clock, step=0.03125)
    for model in (stream.model.det_model, stream.model.pose_model):
        model.session.providers = ["CPUExecutionProvider"]
    result = run_benchmark(
        device="cpu",
        seconds=0.125,
        warmup_seconds=0,
        stream_factory=lambda **_: stream,
        clock=clock,
    )
    assert result.measured.person_fps == 32
    assert not result.passes


@pytest.mark.parametrize(
    "args",
    [
        ["--seconds", "0"],
        ["--seconds", "-1"],
        ["--seconds", "nan"],
        ["--seconds", "inf"],
        ["--warmup-seconds", "-1"],
        ["--source", "-1"],
    ],
)
def test_invalid_cli_arguments_do_not_open_camera(args):
    def factory(**_):
        pytest.fail("invalid arguments must not open the webcam")

    with pytest.raises(SystemExit) as exc:
        main(args, stream_factory=factory)
    assert exc.value.code == 2
