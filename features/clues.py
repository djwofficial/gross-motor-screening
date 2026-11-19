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

Tasks 1 and 2 still have placeholder timeline and clue values.
"""

from __future__ import annotations

import numpy as np

from common import body25 as b
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

# Initial engineering guesses in seconds and torso lengths. See the task card.
_JUMP_WINDOW_S = 0.4
_MIN_AIR_S = 0.08
_MAX_FRAME_GAP_S = 0.10
_ANKLE_LIFT = 0.035
_HIP_RISE = 0.12


def _task3_events(p: np.ndarray, t: np.ndarray) -> list[dict]:
    """Find short two-foot lifts relative to nearby poses on either side."""
    ankles = p[:, [b.R_ANKLE, b.L_ANKLE], 1]
    hip = p[:, b.MID_HIP, 1]
    signals = np.column_stack((ankles, hip))
    valid = np.isfinite(signals).all(axis=1)
    airborne = np.zeros(len(t), dtype=bool)
    states = np.full(len(t), UNKNOWN, dtype=object)
    cuts = np.flatnonzero((np.diff(t) > _MAX_FRAME_GAP_S) | (valid[:-1] != valid[1:])) + 1
    for section in np.split(np.arange(len(t)), cuts):
        if not valid[section[0]]:
            continue
        lo, hi = int(section[0]), int(section[-1]) + 1
        for i in range(lo + 1, hi - 1):
            left = max(lo, np.searchsorted(t, t[i] - _JUMP_WINDOW_S))
            right = min(hi, np.searchsorted(t, t[i] + _JUMP_WINDOW_S, side="right"))
            if left == i or right == i + 1:
                continue
            floor = np.maximum(signals[left:i].min(axis=0), signals[i + 1 : right].min(axis=0))
            lift = signals[i] - floor
            airborne[i] = bool((lift[:2] >= _ANKLE_LIFT).all() and lift[2] >= _HIP_RISE)

        changes = np.flatnonzero(np.diff(airborne[lo:hi], prepend=False, append=False))
        for start, end in changes.reshape(-1, 2):
            if t[lo + end] - t[lo + start] < _MIN_AIR_S:
                airborne[lo + start : lo + end] = False

    has_jump = airborne.any()
    if has_jump:
        first = int(np.flatnonzero(airborne)[0])
        states[valid] = "after"
        states[valid & (np.arange(len(t)) < first)] = "before"
        states[airborne] = "air"

    events = []
    start = 0
    for end in range(1, len(t) + 1):
        if end == len(t):
            end_s = t[-1]
        elif t[end] - t[end - 1] > _MAX_FRAME_GAP_S:
            end_s = t[end - 1]
        elif states[end] != states[start]:
            end_s = t[end]
        else:
            continue
        events.append({"start_s": float(t[start]), "end_s": float(end_s), "state": states[start]})
        if end < len(t) and end_s == t[end - 1]:
            events.append({"start_s": float(end_s), "end_s": float(t[end]), "state": UNKNOWN})
        start = end
    return events


def _task3_pose(points: np.ndarray) -> np.ndarray:
    try:
        return normalize(points)
    except ValueError:
        return np.full(points.shape, np.nan)


def _task3_clues(p: np.ndarray, t: np.ndarray, events: list[dict]) -> dict[str, float]:
    out = dict.fromkeys(CLUE_NAMES[3], 0.0)
    out["duration_s"] = float(t[-1] - t[0])
    for i in range(1, len(events) - 1):
        before, air, after = events[i - 1 : i + 2]
        if air["state"] != "air" or before["state"] not in ("before", "after"):
            continue
        if after["state"] != "after":
            continue
        takeoff = np.searchsorted(t, air["start_s"]) - 1
        landing = np.searchsorted(t, air["end_s"])
        boundary = p[[takeoff, landing]]
        gaps = np.linalg.norm(boundary[:, b.R_ANKLE, :2] - boundary[:, b.L_ANKLE, :2], axis=1)
        widths = np.linalg.norm(boundary[:, b.R_HIP, :2] - boundary[:, b.L_HIP, :2], axis=1)
        for key, gap, width in zip(CLUE_NAMES[3][1:3], gaps, widths, strict=True):
            if np.isfinite(gap) and np.isfinite(width) and width > 0:
                out[key] = float(gap / width)
        out["air_s"] = air["end_s"] - air["start_s"]
        hips = boundary[:, b.MID_HIP, 0]
        if np.isfinite(hips).all():
            out["jump_size"] = float(abs(hips[1] - hips[0]))
        break
    return out


def timeline(task: int, points: np.ndarray, t: np.ndarray) -> list[dict]:
    if task not in CLUE_NAMES:
        raise ValueError("task must be 1, 2 or 3")
    if task == 3:
        return _task3_events(_task3_pose(points), t)
    _ = normalize(points)  # torso lengths, y up; the owners build the moments from this
    return [{"start_s": float(t[0]), "end_s": float(t[-1]), "state": UNKNOWN}]


def compute(task: int, points: np.ndarray, t: np.ndarray) -> dict[str, float]:
    if task not in CLUE_NAMES:
        raise ValueError("task must be 1, 2 or 3")
    if task == 3:
        p = _task3_pose(points)
        return _task3_clues(p, t, _task3_events(p, t))
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
