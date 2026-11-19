"""Clue table for the random forest (Louise, #22): one row per group B file.

Columns: file, child, task, attempt, camera, label, n_frames, max_people, picked, then the clue
names in features.clues.CLUE_NAMES[task] (the format model/train_rf.py reads).

picked = "single"  every frame had at most one person
         "temp"    2+ people somewhere; the child was chosen by the TEMPORARY picker below until
                   Marty's common.pick.pick_child (#16) lands. Re-run the table after that.
Clips under 30 frames are left out (Marty's loader rule, #14); the script prints how many.

Run:  python scripts/make_clue_table.py --tasks 1 --out runs/clue_table.csv
Then: python -m model.train_rf --table runs/clue_table.csv --task 1
"""

from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import body25 as b  # noqa: E402
from common.geometry import valid  # noqa: E402
from data.load_b import DEFAULT_ZIP, FPS, list_files, parse_name, read_file  # noqa: E402
from features.clues import CLUE_NAMES, compute  # noqa: E402

MIN_FRAMES = 30
WIDTH, HEIGHT = 1920, 1080  # dataset picture size (team doc, Info > 3)


def _anchor(person: np.ndarray) -> np.ndarray | None:
    """Neck, or mid-hip when the neck is missing; None when both are missing."""
    for i in (b.NECK, b.MID_HIP):
        if valid(person[i]):
            return person[i, :2]
    return None


def pick_temp(frames: list[np.ndarray]) -> np.ndarray:
    """TEMPORARY stand-in for Marty's pick_child: lock on the person nearest the picture centre
    in the first frame with anyone in it, then follow the nearest neck frame by frame.
    (0, 0, 0) where nobody is found."""
    out = np.zeros((len(frames), b.N_POINTS, 3), dtype=np.float32)
    last = None
    for i, people in enumerate(frames):
        cands = [(p, a) for p in people if (a := _anchor(p)) is not None]
        if not cands:
            continue
        ref = last if last is not None else np.array([WIDTH / 2, HEIGHT / 2])
        p, a = min(cands, key=lambda c: float(np.linalg.norm(c[1] - ref)))
        out[i], last = p, a
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, nargs="+", default=[1])
    ap.add_argument("--out", default="runs/clue_table.csv")
    a = ap.parse_args()

    clue_cols: list[str] = []
    for task in a.tasks:
        clue_cols += [c for c in CLUE_NAMES[task] if c not in clue_cols]
    cols = ["file", "child", "task", "attempt", "camera", "label", "n_frames", "max_people"]
    cols += ["picked", *clue_cols]

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows, short, no_torso = 0, 0, 0
    with zipfile.ZipFile(DEFAULT_ZIP) as z, out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for name in list_files():
            m = parse_name(name)
            if m["task"] not in a.tasks:
                continue
            frames, label, _ = read_file(z, name)
            if len(frames) < MIN_FRAMES:
                short += 1
                continue
            max_people = max(len(fr) for fr in frames)
            if max_people <= 1:
                pts = np.stack(
                    [fr[0] if len(fr) else np.zeros((25, 3), np.float32) for fr in frames]
                )
                picked = "single"
            else:
                pts, picked = pick_temp(frames), "temp"
            t = np.arange(len(pts)) / FPS
            try:
                clues = compute(m["task"], pts, t)
            except ValueError:  # no frame with both neck and mid-hip
                no_torso += 1
                continue
            row = {
                "file": name.split("/")[-1],
                **m,
                "label": label,
                "n_frames": len(pts),
                "max_people": max_people,
                "picked": picked,
            }
            row.update({k: round(v, 4) for k, v in clues.items()})
            w.writerow(row)
            rows += 1
    print(
        f"wrote {out}: {rows} rows | left out: {short} under {MIN_FRAMES} frames, "
        f"{no_torso} without a torso"
    )


if __name__ == "__main__":
    main()
