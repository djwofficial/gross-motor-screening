"""Headless webcam or recorded-video capture into the shared Try format. Owner: Revallo.

Example: python -m live.capture --video team.mp4 --out runs/team.npz --task 1 --leg L --device cpu
Camera acquisition lives in PoseStream's latest-frame thread; this module selects the child
and resamples completed attempts. Video saving belongs to the separate Nov 15 task.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

from common.format import Try, save, validate
from common.pick import pick_child
from live.convert import halpe_to_body25
from live.pose import PoseStream


def resample_to_fps(points, t, fps: float = 30.0):
    """Return (float32 points, float64 seconds) on a grid starting at zero.

    Choose the nearest observed frame, choosing the earlier frame at an exact midpoint.
    Whole frames (including zero detections) are copied; missing points are never interpolated.
    The final grid time does not extend beyond the observed attempt, to timestamp precision.
    """
    if np.iscomplexobj(points) or np.iscomplexobj(t):
        raise ValueError("points and timestamps must be real numbers")
    points = np.asarray(points, dtype=np.float32)
    t = np.asarray(t, dtype=np.float64)
    validate(Try(points=points, t=t))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and greater than zero")
    relative = t - t[0]
    steps = relative[-1] * fps
    if not math.isfinite(steps):
        raise ValueError("Attempt duration and fps must produce a finite sample count")
    tolerance = 4 * np.finfo(np.float64).eps * max(1.0, np.max(np.abs(t)))
    count = int(np.floor(steps + tolerance * fps)) + 1
    grid = np.arange(count, dtype=np.float64) / fps
    right = np.minimum(np.searchsorted(relative, grid), len(relative) - 1)
    left = np.maximum(right - 1, 0)
    midpoint = relative[left] + (relative[right] - relative[left]) / 2.0
    indices = np.where(grid > midpoint + tolerance, right, left)
    return points[indices].copy(), grid


def capture_try(
    source=0,
    *,
    task: int,
    leg: str,
    mode: str = "balanced",
    device: str = "cuda",
    seconds: float = 10.0,
) -> Try:
    """Capture one attempt. Video files use their complete media timeline.

    ``seconds`` bounds webcam acquisition time, independently of inference speed. Empty
    detections remain in the timeline; an entirely unseen person cannot form an attempt.
    """
    if task not in (1, 2, 3) or leg not in ("L", "R"):
        raise ValueError("task must be 1/2/3 and leg must be the support leg L/R")
    camera = isinstance(source, (int, np.integer))
    if camera and source < 0:
        raise ValueError("Camera index must be zero or greater")
    if camera and (not math.isfinite(seconds) or seconds <= 0):
        raise ValueError("Webcam seconds must be finite and greater than zero")
    detections, times = [], []
    width = height = None
    stream = PoseStream(source=source, mode=mode, device=device)
    try:
        while True:
            item = stream.read()
            if item is None:
                break
            t, keypoints, scores, frame = item
            if camera and times and t >= times[0] + seconds:
                break
            if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("Capture must provide a BGR frame with shape (height,width,3)")
            h, w = frame.shape[:2]
            if width is None:
                width, height = int(w), int(h)
            elif (w, h) != (width, height):
                raise ValueError("Frame size changed during capture; start a new attempt")
            detections.append(halpe_to_body25(keypoints, scores))
            times.append(t)
        if not times:
            raise ValueError("The source contained no readable frames")
        points = pick_child(detections)
        if not np.any(points[..., 2] > 0):
            raise ValueError("No person was detected; no attempt was saved")
        points[points[..., 2] == 0] = 0
        points, t = resample_to_fps(points, times)
        if not np.any(points[..., 2] > 0):
            raise ValueError("No person remains at the output sample times; no attempt was saved")
        meta = {
            "source": "webcam" if camera else "video",
            "width": width,
            "height": height,
            "fps": 30.0,
            "source_fps": stream.source_fps,
            "task": task,
            "leg": leg,
            "mode": mode,
            "device": device,
            "timestamp_source": (
                "acquisition_clock"
                if camera
                else "media_with_fps_fallback"
                if stream.used_fps_fallback
                else "media"
            ),
        }
        if camera:
            meta["camera_index"] = int(source)
        else:
            meta["video_file"] = os.fspath(source)
        return validate(Try(points=points, t=t, meta=meta))
    finally:
        original_error = sys.exception()
        try:
            stream.close()
        except BaseException as cleanup_error:
            if original_error is None:
                raise
            original_error.add_note(f"Stream cleanup also failed: {cleanup_error}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Capture unmirrored pose points into a Try .npz")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--video", type=Path, help="Recorded clip; process every frame")
    source.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--task", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument("--leg", choices=("L", "R"), required=True, help="Support leg")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--mode", choices=("balanced", "lightweight", "performance"), default="balanced"
    )
    parser.add_argument("--seconds", type=float, help="Webcam duration (default: 10 seconds)")
    args = parser.parse_args(argv)
    if args.out.suffix != ".npz":
        parser.error("--out must end in .npz")
    if args.video is not None and args.seconds is not None:
        parser.error("--seconds applies to webcam capture; video mode processes the complete clip")
    if args.video is not None and not args.video.is_file():
        parser.error("--video must name an existing recorded clip")
    if args.video is None:
        print(
            "Before capture, turn off OS automatic framing (Center Stage/Studio Effects).",
            file=sys.stderr,
        )
    try:
        tr = capture_try(
            args.video if args.video is not None else args.camera,
            task=args.task,
            leg=args.leg,
            mode=args.mode,
            device=args.device,
            seconds=10.0 if args.seconds is None else args.seconds,
        )
        save(tr, args.out)
    except KeyboardInterrupt as exc:
        print("Capture cancelled; no attempt was saved.", file=sys.stderr)
        for note in getattr(exc, "__notes__", []):
            print(note, file=sys.stderr)
        return 130
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"Capture failed: {exc}", file=sys.stderr)
        for note in getattr(exc, "__notes__", []):
            print(note, file=sys.stderr)
        return 1
    print(f"Saved {tr.n_frames} frames at {tr.meta['fps']:g} fps to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
