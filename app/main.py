"""Joins the parts: pose stream -> track child -> start/end -> task guess -> clues -> score -> log.

Owner: Daryn. Run from the repo root:

    python -m app.main --fake                      # automatic start/end on a fake try (task 2)
    python -m app.main --fake --task 1 --mode manual
    python -m app.main                             # live webcam GUI (testing mode: no score shown)
    python -m app.main --show-score                # live webcam GUI with 0/1/2 score overlay

In --fake mode there is no operator, so the "confirmation" uses the task and leg the fake try
was made with; if the guess differs, the row is marked corrected (as if the operator pressed
1/2/3 or L/R). No window opens, so this also runs in CI.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

import numpy as np

from app.session import Session
from common.format import Try
from features.clues import explain, timeline_and_compute
from live.convert import halpe_to_body25
from live.pose import FakePoseStream
from live.segment import Segmenter
from live.track import ChildTracker
from model.predict import predict
from model.task_classifier import predict_task

PRE_ROLL_S = 0.5  # keep this much before the detected start


def _score_try_with_events(tr: Try, task: int) -> tuple[dict, int, float, str, list[dict]]:
    events, clues = timeline_and_compute(task, tr.points, tr.t)
    score, confidence = predict(task, clues)
    return clues, score, confidence, explain(task, clues, events), events


def score_try(tr: Try, task: int) -> tuple[dict, int, float, str]:
    """Clues, score, confidence and reason text for one try (also used by rescore.py)."""
    clues, score, confidence, reasons, _events = _score_try_with_events(tr, task)
    return clues, score, confidence, reasons


def finish_try(
    session: Session,
    points: np.ndarray,
    t: np.ndarray,
    raw26: np.ndarray,
    meta: dict,
    true_task: int,
    true_leg: str,
) -> Path:
    tr = Try(points=points.astype(np.float32), t=(t - t[0]).astype(np.float64), meta=meta)
    guessed_task, guessed_leg, task_conf = predict_task(tr.points, tr.t)
    # Operator step (GUI: Enter confirms, 1/2/3 and L/R correct). In --fake mode we know the truth.
    task, leg = true_task, true_leg
    corrected = (guessed_task != task) or (task != 3 and guessed_leg != leg)
    clues, score, confidence, reasons, events = _score_try_with_events(tr, task)
    return session.save_try(
        tr,
        raw26=raw26,
        task=task,
        leg=leg,
        guessed_task=guessed_task,
        guessed_leg=guessed_leg,
        task_confidence=task_conf,
        corrected=corrected,
        score=score,
        confidence=confidence,
        clues=clues,
        reasons=reasons,
        events=events,
    )


def run_fake(args) -> list[Path]:
    stream = FakePoseStream(task=args.task, leg=args.leg, seed=args.seed)
    session = Session(args.out, child=args.child, age_months=args.age, video=False)
    tracker = ChildTracker(frame_size=(1280, 720))
    seg = Segmenter()
    buf_t, buf_p, buf_raw = [], [], []
    recording, start_i = args.mode == "manual", 0
    saved: list[Path] = []
    meta = {"source": "fake", "mode": args.mode, "fps": 30.0, "width": 1280, "height": 720}

    while (item := stream.read()) is not None:
        t, kp26, sc26, _frame = item
        people25 = halpe_to_body25(kp26, sc26)  # (P, 25, 3)
        idx = tracker.update(people25, t)
        found = idx is not None
        child25 = people25[idx] if found else np.zeros((25, 3), np.float32)
        raw = (
            np.concatenate([kp26[idx], sc26[idx, :, None]], axis=-1)
            if found
            else np.zeros((26, 3), np.float32)
        )
        buf_t.append(t)
        buf_p.append(child25)
        buf_raw.append(raw)
        if args.mode == "manual":
            continue
        event = seg.update(child25 if found else None, t)
        if event == "start":
            recording = True
            start_i = max(0, int(np.searchsorted(buf_t, t - PRE_ROLL_S)))
        elif event == "end" and recording:
            saved.append(
                finish_try(
                    session,
                    np.stack(buf_p[start_i:]),
                    np.array(buf_t[start_i:]),
                    np.stack(buf_raw[start_i:]),
                    meta,
                    args.task,
                    args.leg,
                )
            )
            recording = False
            buf_t, buf_p, buf_raw = [], [], []

    if args.mode == "manual" and buf_t:  # Space at the first and last frame
        saved.append(
            finish_try(
                session,
                np.stack(buf_p),
                np.array(buf_t),
                np.stack(buf_raw),
                meta,
                args.task,
                args.leg,
            )
        )
    stream.close()
    return saved


def pick_camera(candidates: tuple[int, ...] = (0, 1, 2), max_frames: int = 10) -> int:
    """Return the first camera index producing non-black frames on macOS; 0 elsewhere.

    On macOS, iPhone Continuity Camera can claim index 0 and return all-zero frames while
    the built-in MacBook camera shifts to index 1.
    """
    if sys.platform != "darwin":
        return 0
    import time

    import cv2

    for idx in candidates:
        cap = cv2.VideoCapture(idx)
        try:
            if not cap.isOpened():
                continue
            for _ in range(max_frames):
                ok, frame = cap.read()
                if ok and frame is not None and int(frame.max()) > 0:
                    return idx
                time.sleep(0.03)
        finally:
            cap.release()
    return 0


def _frame_size(stream) -> tuple[int, int]:
    """Read (width, height) from stream.cap, falling back to 1280x720."""
    cap = getattr(stream, "cap", None)
    if cap is None:
        return (1280, 720)
    import cv2

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    return (w, h)


def run_live(args) -> list[Path]:
    """Open the live webcam or video pipeline and return saved try JSON paths."""
    from app.demo import run_demo
    from app.ui import run_gui
    from live.pose import PoseStream

    if args.video is not None:
        source = args.video
        source_name = os.fspath(args.video)
    else:
        cam = args.camera if args.camera is not None else pick_camera()
        source = cam
        source_name = f"camera:{cam}"
    device = args.device or ("cpu" if sys.platform == "darwin" else "cuda")
    pose_mode = args.pose_mode or ("lightweight" if sys.platform == "darwin" else "balanced")
    view = args.view or ("camera" if args.show_score else "stick")
    stream = PoseStream(source=source, mode=pose_mode, device=device)
    try:
        frame_size = _frame_size(stream)
        runner = run_demo if args.show_score else run_gui
        return runner(
            stream,
            args.out,
            mode=args.mode,
            view=view,
            frame_size=frame_size,
            source=source_name,
        )
    finally:
        stream.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Gross motor screening app (screening aid, not a diagnosis)"
    )
    ap.add_argument("--fake", action="store_true", help="use a fake try instead of the webcam")
    ap.add_argument(
        "--camera",
        type=int,
        default=None,
        help="webcam index for live mode (auto-picked on Mac if omitted)",
    )
    ap.add_argument("--video", type=Path, default=None, help="video file instead of a live webcam")
    ap.add_argument("--mode", choices=["auto", "manual"], default="auto")
    ap.add_argument("--view", choices=["stick", "camera", "neutral"], default=None)
    ap.add_argument("--device", default=None, help="pose device: cpu or cuda (default by OS)")
    ap.add_argument(
        "--pose-mode",
        choices=["lightweight", "balanced", "performance"],
        default=None,
        help="RTMPose speed/accuracy preset (default: lightweight on Mac, balanced elsewhere)",
    )
    ap.add_argument(
        "--show-score",
        action="store_true",
        help="show the live 0/1/2 score overlay (demo mode; not for testing days)",
    )
    ap.add_argument("--task", type=int, choices=[1, 2, 3], default=2)
    ap.add_argument("--leg", choices=["L", "R"], default="L")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--child", default="K00", help="child code (--fake mode)")
    ap.add_argument("--age", type=int, default=48, help="age in months (--fake mode)")
    ap.add_argument("--out", type=Path, default=Path("runs") / date.today().isoformat())
    args = ap.parse_args(argv)

    saved = run_fake(args) if args.fake else run_live(args)
    for p in saved:
        print(f"saved {p}")
    print(f"{len(saved)} try saved; log: {args.out / 'log.csv'}")
    return (0 if saved else 1) if args.fake else 0


if __name__ == "__main__":
    sys.exit(main())
