"""Quick check: does RTMPose run fast enough on the live laptop? Owner: Revallo.

Run it once on Freya's laptop (the live laptop), with a person in the picture:
    python -m live.bench_fps --device cuda --mode balanced --seconds 30

Model loading and warmup are excluded. Duration is checked between completed reads, so the
last read can extend the requested window. Nothing is displayed or saved. Provider inspection
checks both actual ONNXRuntime sessions; it does not measure individual GPU kernel placement.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time
from dataclasses import dataclass

# Issue #27 / docs/people/revallo.md: at least 15 fps, ideally 30, on Freya's laptop.
MIN_FPS = 15.0
CUDA_PROVIDER = "CUDAExecutionProvider"


class BenchmarkError(RuntimeError):
    """The run cannot supply a valid live performance check."""


@dataclass(frozen=True)
class Window:
    frames: int
    person_frames: int
    seconds: float

    @property
    def mean_fps(self) -> float:
        return self.frames / self.seconds if self.seconds else 0.0

    @property
    def person_fps(self) -> float:
        return self.person_frames / self.seconds if self.seconds else 0.0


@dataclass(frozen=True)
class Result:
    device: str
    mode: str
    warmup: Window
    measured: Window
    providers: dict[str, tuple[str, ...]]

    @property
    def passes(self) -> bool:
        # Empty-scene reads still cost time, but cannot inflate the person-present FPS gate.
        return self.device == "cuda" and self.measured.person_fps >= MIN_FPS


def _duration(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds < 0:
        raise argparse.ArgumentTypeError("duration must be finite and non-negative")
    return seconds


def _open_stream(**kwargs):
    # Keep rtmlib, OpenCV and ONNXRuntime optional when importing the benchmark or running tests.
    import onnxruntime

    if kwargs["device"] == "cuda" and CUDA_PROVIDER not in onnxruntime.get_available_providers():
        raise BenchmarkError(
            "ONNXRuntime has no CUDAExecutionProvider. Use the Windows NVIDIA environment "
            "from README (onnxruntime-gpu, without the CPU onnxruntime package)."
        )
    from live.pose import PoseStream

    return PoseStream(**kwargs)


def _model_sessions(stream) -> dict:
    # rtmlib BodyWithFeet -> det_model/pose_model -> BaseTool.session. Sources in the task card.
    try:
        models = {"detector": stream.model.det_model, "pose": stream.model.pose_model}
        if any(model.backend != "onnxruntime" for model in models.values()):
            raise BenchmarkError("This benchmark requires the onnxruntime backend for both models.")
        return {name: model.session for name, model in models.items()}
    except AttributeError as exc:
        raise BenchmarkError(
            "Cannot inspect both rtmlib model sessions; check the pinned rtmlib installation."
        ) from exc


def _providers(sessions: dict, device: str) -> dict[str, tuple[str, ...]]:
    providers = {name: tuple(session.get_providers()) for name, session in sessions.items()}
    if device == "cuda":
        for name, names in providers.items():
            if not names or names[0] != CUDA_PROVIDER:
                raise BenchmarkError(
                    f"{name} session is not using CUDA as its first provider: {names}. "
                    "Check NVIDIA driver/CUDA/cuDNN and the GPU ONNXRuntime installation; "
                    "CPU fallback is not a passing result."
                )
    return providers


def _measure(stream, seconds: float, clock) -> Window:
    if seconds == 0:
        return Window(0, 0, 0.0)
    start = clock()
    elapsed, frames, person_frames = 0.0, 0, 0
    while elapsed < seconds:
        item = stream.read()
        if item is None:
            raise BenchmarkError(
                "Camera returned no frame before the benchmark finished. "
                "Check camera access, close other camera apps, and try --source INDEX."
            )
        _t, keypoints, _scores, _frame = item
        frames += 1
        person_frames += int(len(keypoints) > 0)
        elapsed = clock() - start
    return Window(frames, person_frames, elapsed)


def run_benchmark(
    *,
    source: int = 0,
    device: str = "cuda",
    mode: str = "balanced",
    seconds: float = 30.0,
    warmup_seconds: float = 3.0,
    stream_factory=None,
    clock=None,
) -> Result:
    """Measure capture plus inference; factory/clock injection keeps tests hardware independent."""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    if not math.isfinite(warmup_seconds) or warmup_seconds < 0:
        raise ValueError("warmup_seconds must be finite and non-negative")
    stream_factory = stream_factory or _open_stream
    clock = clock or time.perf_counter
    stream = stream_factory(source=source, mode=mode, device=device)
    try:
        cap = getattr(stream, "cap", None)
        if cap is not None and not cap.isOpened():
            raise BenchmarkError(f"Cannot open webcam {source}; check access or --source INDEX.")
        sessions = _model_sessions(stream)
        _providers(sessions, device)
        for session in sessions.values():
            session.disable_fallback()  # ORT public API: don't reset providers after a run failure.
        warmup = _measure(stream, warmup_seconds, clock)
        measured = _measure(stream, seconds, clock)
        providers = _providers(sessions, device)  # inspect the sessions after actual inference too
        if measured.person_frames == 0:
            raise BenchmarkError(
                "No person detected during measurement. Stand fully in the picture and rerun; "
                "empty-scene FPS is not a valid pose benchmark."
            )
        return Result(device, mode, warmup, measured, providers)
    finally:
        stream.close()


def gpu_inventory() -> str:
    """Inventory only: CUDA_VISIBLE_DEVICES may make CUDA ordinal 0 a different GPU."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return "; ".join(result.stdout.strip().splitlines()) or "not reported"
    except (OSError, subprocess.SubprocessError):
        return "unavailable (nvidia-smi could not report GPU names)"


def main(argv=None, *, stream_factory=None, clock=None, gpu_info=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=int, default=0, help="webcam index (default: 0)")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    ap.add_argument(
        "--mode", choices=["balanced", "lightweight", "performance"], default="balanced"
    )
    ap.add_argument(
        "--seconds", type=_duration, default=30.0, help="measurement seconds (default: 30)"
    )
    ap.add_argument(
        "--warmup-seconds", type=_duration, default=3.0, help="excluded warmup (default: 3)"
    )
    args = ap.parse_args(argv)
    if args.seconds == 0:
        ap.error("--seconds must be positive")
    if args.source < 0:
        ap.error("--source must be a non-negative webcam index")
    print("Keep a person in the unmirrored camera picture; turn off automatic framing.", flush=True)
    try:
        gpu = (gpu_info or gpu_inventory)()
        result = run_benchmark(**vars(args), stream_factory=stream_factory, clock=clock)
    except KeyboardInterrupt:
        print("Benchmark interrupted; camera released.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Benchmark error: {exc}", file=sys.stderr)
        return 2
    print(f"GPU inventory: {gpu}")
    print(f"device: {result.device} | backend: onnxruntime | mode: {result.mode}")
    for name, names in result.providers.items():
        print(f"{name} session providers: {', '.join(names)}")
    print("Provider evidence: registered session providers after inference; no kernel profiling.")
    print(f"warmup: {result.warmup.frames} frames in {result.warmup.seconds:.3f} s (excluded)")
    measured = result.measured
    print(f"measured: {measured.frames} frames in {measured.seconds:.3f} s")
    print(f"mean fps: {measured.mean_fps:.2f}")
    print(
        f"person frames: {measured.person_frames}/{measured.frames} | person fps: {measured.person_fps:.2f}"
    )
    if result.passes:
        print(
            "PASS: both sessions register CUDA first; person-frame throughput >= 15 fps (target 30)."
        )
    else:
        print(
            "FAIL: needs CUDA and >= 15 person frames/s on Freya's laptop. Share this output with Daryn."
        )
    return 0 if result.passes else 1


if __name__ == "__main__":
    sys.exit(main())
