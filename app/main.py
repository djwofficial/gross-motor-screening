"""Joins the parts: pose stream -> pick child -> start/end -> task guess -> clues -> score -> log.

Owner: Daryn. Run from the repo root:

    python -m app.main --fake                      # automatic start/end on a fake try (task 2)
    python -m app.main --fake --task 1 --mode manual
    python -m app.main                             # live webcam: joined on Nov 22

In --fake mode there is no operator, so the "confirmation" uses the task and leg the fake try
was made with; if the guess differs, the row is marked corrected (as if the operator pressed
1/2/3 or L/R). No window opens, so this also runs in CI.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np

from app.session import Session
from common.format import Try
from common.pick import pick_child
from features.clues import compute, explain, timeline
from live.convert import halpe_to_body25
from live.pose import FakePoseStream
from live.segment import Segmenter
from model.predict import predict
from model.task_classifier import predict_task

PRE_ROLL_S = 0.5  # keep this much before the detected start


def score_try(tr: Try, task: int) -> tuple[dict, int, float, str]:
    """Clues, score, confidence and reason text for one try (also used by rescore.py)."""
    clues = compute(task, tr.points, tr.t)
    score, confidence = predict(task, clues)
    return clues, score, confidence, explain(task, clues, timeline(task, tr.points, tr.t))


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
    clues, score, confidence, reasons = score_try(tr, task)
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
        events=timeline(task, tr.points, tr.t),
    )


def run_fake(args) -> list[Path]:
    stream = FakePoseStream(task=args.task, leg=args.leg, seed=args.seed)
    session = Session(args.out, child=args.child, age_months=args.age, video=False)
    seg = Segmenter()
    buf_t, buf_p, buf_raw = [], [], []
    recording, start_i = args.mode == "manual", 0
    saved: list[Path] = []
    meta = {"source": "fake", "mode": args.mode, "fps": 30.0, "width": 1280, "height": 720}

    while (item := stream.read()) is not None:
        t, kp26, sc26, _frame = item
        people25 = halpe_to_body25(kp26, sc26)  # (P, 25, 3)
        child25 = pick_child([people25])[0]  # live: Revallo replaces this with a tracked pick
        found = people25.shape[0] > 0
        raw = (
            np.concatenate([kp26, sc26[..., None]], axis=-1)[0]
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Gross motor screening app (screening aid, not a diagnosis)"
    )
    ap.add_argument("--fake", action="store_true", help="use a fake try instead of the webcam")
    ap.add_argument("--mode", choices=["auto", "manual"], default="auto")
    ap.add_argument("--task", type=int, choices=[1, 2, 3], default=2)
    ap.add_argument("--leg", choices=["L", "R"], default="L")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--child", default="K00")
    ap.add_argument("--age", type=int, default=48, help="age in months")
    ap.add_argument("--out", type=Path, default=Path("runs") / date.today().isoformat())
    args = ap.parse_args(argv)

    if not args.fake:
        print(
            "Live mode is joined on Nov 22 (Revallo's PoseStream + Freya's GUI). Use --fake for now."
        )
        return 2
    saved = run_fake(args)
    for p in saved:
        print(f"saved {p}")
    print(f"{len(saved)} try saved; log: {args.out / 'log.csv'}")
    return 0 if saved else 1


if __name__ == "__main__":
    sys.exit(main())
