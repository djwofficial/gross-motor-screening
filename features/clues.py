"""Movement clues and the moment timeline, from the scoring rules (paper Table 3).

Owners: Louise (tasks 1 and 2), Revallo (task 3).

Shared interface (final, change only through Daryn):

    timeline(task, points, t) -> list[dict]
        What happened when in one try, in time order:
        [{"start_s": 1.1, "end_s": 2.0, "state": "steady"}, ...]
        state is one of STATES[task], or UNKNOWN. Times are seconds, taken from t.

    compute(task, points, t) -> dict[str, float]
        task:   1 (stand on one leg), 2 (hop on one leg), 3 (standing long jump)
        points: float32 (T, 25, 3) of the child, BODY_25, pixels (dataset or live)
        t:      float64 (T,) seconds
        returns {name: value} with exactly the names in CLUE_NAMES[task], all finite floats.
        Units: seconds, counts and torso lengths only (use common.geometry.normalize).

    explain(task, clues, events=None) -> str
        short reason text for log.csv and the operator, e.g. "4.2 s on one foot, 0 touch-downs".
        events is the timeline; use it to name moments, e.g. "foot down at 2.0 s".

The dataset has one expert score per try and none per moment, so the timeline is measured from
the points, not learned. Build the timeline first and compute the clues from it, so every number
has one source. It records what happened and never says that a child failed.

CLUE_NAMES is also the feature order of the random forest (Daryn, model/train_rf.py), except
duration_s: live, the length of a try is set by our automatic start and end, so the model must
not learn from it. Adding or renaming a clue is fine, but tell Daryn and Marty and update
CLUE_NAMES in the same PR.

STUB: only duration_s is real; the other values are 0.0 and the timeline is one UNKNOWN block
until the real versions land (task 1 by Nov 14, tasks 2-3 by Nov 17, explain by Nov 19). Ideas
per clue: Job Card 2 in the team Google Doc.
"""

from __future__ import annotations

import numpy as np

from common.geometry import normalize

CLUE_NAMES = {
    1: ["duration_s", "one_foot_s", "touchdowns", "sway", "wobble_s"],
    2: ["duration_s", "hops", "longest_pause_s", "touchdowns"],
    3: ["duration_s", "feet_gap_takeoff", "feet_gap_landing", "air_s", "jump_size"],
}

UNKNOWN = "unknown"
STATES = {
    1: ("both_feet", "steady", "wobble"),
    2: ("hop", "pause", "both_feet"),
    3: ("before", "air", "after"),
}


def timeline(task: int, points: np.ndarray, t: np.ndarray) -> list[dict]:
    if task not in CLUE_NAMES:
        raise ValueError("task must be 1, 2 or 3")
    _ = normalize(points)  # torso lengths, y up; the owners build the moments from this
    return [{"start_s": float(t[0]), "end_s": float(t[-1]), "state": UNKNOWN}]


def compute(task: int, points: np.ndarray, t: np.ndarray) -> dict[str, float]:
    if task not in CLUE_NAMES:
        raise ValueError("task must be 1, 2 or 3")
    _ = timeline(task, points, t)  # the owners compute the clues from this
    out = {name: 0.0 for name in CLUE_NAMES[task]}
    out["duration_s"] = float(t[-1] - t[0]) if len(t) > 1 else 0.0
    return out


def explain(task: int, clues: dict[str, float], events: list[dict] | None = None) -> str:
    # TODO Louise, Nov 19: use events to name moments ("foot down at 2.0 s").
    c = clues
    if task == 1:
        return f"{c['one_foot_s']:.1f} s on one foot, {c['touchdowns']:.0f} touch-downs, sway {c['sway']:.2f}"
    if task == 2:
        return f"{c['hops']:.0f} hops, longest pause {c['longest_pause_s']:.1f} s, {c['touchdowns']:.0f} touch-downs"
    if task == 3:
        return f"feet gap {c['feet_gap_takeoff']:.2f} at take-off, {c['air_s']:.2f} s in the air, jump size {c['jump_size']:.2f}"
    raise ValueError("task must be 1, 2 or 3")
