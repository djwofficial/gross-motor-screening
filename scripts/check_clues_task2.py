"""Task 2 (hop) clues on group B (Louise, #21): median of each clue per expert label.

One-person files only (pick_child is Marty's, not merged yet), clips of 30+ frames.
Run: python scripts/check_clues_task2.py [--bounce 0.1]
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import features.clues as fc  # noqa: E402
from data.load_b import DEFAULT_ZIP, FPS, list_files, parse_name, read_file  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bounce", type=float, default=fc.HOP_BOUNCE)
    a = ap.parse_args()
    fc.HOP_BOUNCE = a.bounce

    rows = []
    with zipfile.ZipFile(DEFAULT_ZIP) as z:
        for n in list_files():
            if parse_name(n)["task"] != 2:
                continue
            frames, label, _ = read_file(z, n)
            if len(frames) < 30 or any(f.shape[0] != 1 for f in frames):
                continue
            p = np.stack([f[0] for f in frames])
            t = np.arange(len(p)) / FPS
            c = fc.compute(2, p, t)
            rows.append((label, [c[k] for k in fc.CLUE_NAMES[2]]))

    print(f"bounce {fc.HOP_BOUNCE} | one-person files: {len(rows)}")
    print("label    n  " + "  ".join(f"{k:>10}" for k in fc.CLUE_NAMES[2]))
    for lab, name in enumerate(["Bad", "Good", "Perfect"]):
        v = np.array([r[1] for r in rows if r[0] == lab])
        med = np.median(v, axis=0)
        print(f"{name:<7} {len(v):>4}  " + "  ".join(f"{m:>10.2f}" for m in med))


if __name__ == "__main__":
    main()
