"""Movement clues taken from the scoring rules (paper Table 3). Owner: Revallo.

Shared interface (final, change only through Daryn):

    compute(task, points, t) -> dict[str, float]
        task:   1 (stand on one leg), 2 (hop on one leg), 3 (standing long jump)
        points: float32 (T, 25, 3) of the child, BODY_25, pixels (dataset or live)
        t:      float64 (T,) seconds
        returns {name: value} with exactly the names in CLUE_NAMES[task], all finite floats.
        Units: seconds, counts and torso lengths only (use common.geometry.normalize).

    explain(task, clues) -> str
        short reason text for log.csv and the operator, e.g. "4.2 s on one foot, 0 touch-downs".

CLUE_NAMES is also the feature order of Marty's random forest. Adding or renaming a clue is
fine, but tell Marty and update CLUE_NAMES in the same PR.

STUB: only duration_s is real; the other values are 0.0 until Revallo's versions land
(task 1 by Nov 14, tasks 2-3 by Nov 17, explain by Nov 19). Ideas per clue: Job Card 2 in the
team Google Doc.
"""

from __future__ import annotations

import numpy as np

from common.geometry import normalize

CLUE_NAMES = {
    1: ["duration_s", "one_foot_s", "touchdowns", "sway"],
    2: ["duration_s", "hops", "longest_pause_s", "touchdowns"],
    3: ["duration_s", "feet_gap_takeoff", "feet_gap_landing", "air_s", "jump_size"],
}


def compute(task: int, points: np.ndarray, t: np.ndarray) -> dict[str, float]:
    if task not in CLUE_NAMES:
        raise ValueError("task must be 1, 2 or 3")
    _ = normalize(points)  # torso lengths, y up; Revallo computes the clues from this
    out = {name: 0.0 for name in CLUE_NAMES[task]}
    out["duration_s"] = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    return out


def explain(task: int, clues: dict[str, float]) -> str:
    c = clues
    if task == 1:
        return f"{c['one_foot_s']:.1f} s on one foot, {c['touchdowns']:.0f} touch-downs, sway {c['sway']:.2f}"
    if task == 2:
        return f"{c['hops']:.0f} hops, longest pause {c['longest_pause_s']:.1f} s, {c['touchdowns']:.0f} touch-downs"
    if task == 3:
        return f"feet gap {c['feet_gap_takeoff']:.2f} at take-off, {c['air_s']:.2f} s in the air, jump size {c['jump_size']:.2f}"
    raise ValueError("task must be 1, 2 or 3")
