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

Task 1 is real (Louise, #20) and task 3 is real (Revallo, #32). STUB for task 2: only
duration_s is real; the other values are 0.0 and the timeline is one UNKNOWN block until it
lands. Ideas per clue: Job Card 2 in the team Google Doc.
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


# --- Task 1 (stand on one leg), Louise -------------------------------------------------------
# On one foot: the two ankles differ in height by more than ONE_FOOT_GAP torso lengths.
# 0.15 is the start value from Job Card 2 and the 2026-11-09 rough test; tune it by eye.
ONE_FOOT_GAP = 0.15
# A one-foot stretch counts only if the gap reaches LIFT_PEAK somewhere inside it; otherwise it was
# a step toward the camera (a foot closer to camera 1 sits lower in the picture), not a lift.
# By eye 2026-11-19 (Louise): Bad clip B040_GMS_1_4_1 peaks at 0.22 and the child never stood on
# one foot; real lifts in B249_GMS_1_1_1 and B210_GMS_1_5_1 reach 0.6-0.9.
LIFT_PEAK = 0.3
# Runs shorter than this are flickers and are merged into a neighbour (2026-11-09 rough test).
MIN_RUN_S = 0.2
# Wobble: the mid-hip moves sideways faster than WOBBLE_SPEED torso lengths per second, after
# smoothing over SMOOTH_S. Chosen with scripts/check_clues_task1.py on 906 one-person group B
# task 1 files (2026-11-19): at 0.25 the median wobble time is 0.40 s (Bad), 0.57 s (Good),
# 0.00 s (Perfect); at 0.35 and 0.5 Bad and Perfect both drop to 0. Not yet checked by eye.
WOBBLE_SPEED = 0.25
SMOOTH_S = 0.2

_BOTH, _ONE, _WOBBLE, _UNK = "both_feet", "steady", "wobble", UNKNOWN
_ON_ONE_FOOT = (_ONE, _WOBBLE)


def _duration(t: np.ndarray) -> float:
    return float(t[-1] - t[0]) if len(t) > 1 else 0.0


def _whole(t: np.ndarray, state: str = UNKNOWN) -> list[dict]:
    return [{"start_s": float(t[0]), "end_s": float(t[-1]), "state": state}]


def _foot_height(q: np.ndarray, ankle: int, heel: int) -> np.ndarray:
    """Ankle height per frame (torso lengths, y up); the heel stands in when the ankle is missing."""
    y = q[:, ankle, 1].copy()
    miss = np.isnan(y)
    y[miss] = q[miss, heel, 1]
    return y


def _fill_nan(x: np.ndarray) -> np.ndarray:
    ok = ~np.isnan(x)
    if not ok.any():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    return np.interp(idx, idx[ok], x[ok])


def _smooth(x: np.ndarray, t: np.ndarray, window_s: float) -> np.ndarray:
    """Moving average over a time window (works for any frame rate)."""
    out = np.empty_like(x)
    half = window_s / 2
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right")
    c = np.concatenate([[0.0], np.cumsum(x)])
    out[:] = (c[hi] - c[lo]) / np.maximum(hi - lo, 1)
    return out


def _runs(states: list[str]) -> list[list]:
    """Frame states -> [[state, first_frame, last_frame], ...]."""
    runs: list[list] = []
    for i, s in enumerate(states):
        if runs and runs[-1][0] == s:
            runs[-1][2] = i
        else:
            runs.append([s, i, i])
    return runs


def _run_s(run: list, t: np.ndarray) -> float:
    """Seconds a run lasts: from its first frame to the first frame of the next run."""
    end = min(run[2] + 1, len(t) - 1)
    return float(t[end] - t[run[1]])


def _merge_short(runs: list[list], t: np.ndarray, min_s: float) -> list[list]:
    """Merge runs shorter than min_s into their longer neighbour, shortest first."""
    runs = [r[:] for r in runs]
    while len(runs) > 1:
        lengths = [_run_s(r, t) for r in runs]
        i = int(np.argmin(lengths))
        if lengths[i] >= min_s:
            break
        if i == 0:
            j = 1
        elif i == len(runs) - 1:
            j = i - 1
        else:
            j = i - 1 if lengths[i - 1] >= lengths[i + 1] else i + 1
        lo, hi = min(i, j), max(i, j)
        merged = [runs[j][0], runs[lo][1], runs[hi][2]]
        runs[lo : hi + 1] = [merged]
        # neighbours that now share a state become one run
        k = lo
        if k > 0 and runs[k - 1][0] == runs[k][0]:
            runs[k - 1 : k + 1] = [[runs[k][0], runs[k - 1][1], runs[k][2]]]
            k -= 1
        if k < len(runs) - 1 and runs[k + 1][0] == runs[k][0]:
            runs[k : k + 2] = [[runs[k][0], runs[k][1], runs[k + 1][2]]]
    return runs


def _frame_states_task1(q: np.ndarray, t: np.ndarray) -> tuple[list[str], np.ndarray]:
    yl = _foot_height(q, b.L_ANKLE, b.L_HEEL)
    yr = _foot_height(q, b.R_ANKLE, b.R_HEEL)
    gap = np.abs(yl - yr)
    hip_x = _smooth(_fill_nan(q[:, b.MID_HIP, 0]), t, SMOOTH_S)
    speed = np.abs(np.gradient(hip_x, t)) if len(t) > 1 else np.zeros(len(t))
    speed = _smooth(speed, t, SMOOTH_S)
    states = []
    for g, v in zip(gap, speed, strict=True):
        if np.isnan(g):
            states.append(_UNK)
        elif g <= ONE_FOOT_GAP:
            states.append(_BOTH)
        else:
            states.append(_WOBBLE if v > WOBBLE_SPEED else _ONE)
    return states, gap


def _drop_low_lifts(runs: list[list], gap: np.ndarray) -> list[list]:
    """One-foot stretches (neighbouring steady/wobble runs) whose gap never reaches LIFT_PEAK
    become both_feet; neighbouring runs that then share a state are joined."""
    runs = [r[:] for r in runs]
    k = 0
    while k < len(runs):
        if runs[k][0] not in _ON_ONE_FOOT:
            k += 1
            continue
        j = k
        while j + 1 < len(runs) and runs[j + 1][0] in _ON_ONE_FOOT:
            j += 1
        peak = np.nanmax(gap[runs[k][1] : runs[j][2] + 1])
        if not peak >= LIFT_PEAK:
            for r in runs[k : j + 1]:
                r[0] = _BOTH
        k = j + 1
    joined: list[list] = []
    for r in runs:
        if joined and joined[-1][0] == r[0]:
            joined[-1][2] = r[2]
        else:
            joined.append(r)
    return joined


def _timeline_task1(points: np.ndarray, t: np.ndarray) -> list[dict]:
    try:
        q = normalize(points)
    except ValueError:  # no neck/mid-hip anywhere: torso length unknown
        return _whole(t)
    states, gap = _frame_states_task1(q, t)
    runs = _drop_low_lifts(_merge_short(_runs(states), t, MIN_RUN_S), gap)
    events = []
    for k, (state, i0, _i1) in enumerate(runs):
        end = t[runs[k + 1][1]] if k + 1 < len(runs) else t[-1]
        events.append({"start_s": float(t[i0]), "end_s": float(end), "state": state})
    return events


def _clues_task1(points: np.ndarray, t: np.ndarray, events: list[dict]) -> dict[str, float]:
    """Every clue comes from the timeline, except sway, which needs the hip inside the
    longest one-foot stretch that the timeline marks."""
    out = {name: 0.0 for name in CLUE_NAMES[1]}
    out["duration_s"] = _duration(t)

    # one-foot stretches = neighbouring steady/wobble events joined
    stretches: list[list[float]] = []
    prev_on = False
    for e in events:
        on = e["state"] in _ON_ONE_FOOT
        if on and prev_on:
            stretches[-1][1] = e["end_s"]
        elif on:
            stretches.append([e["start_s"], e["end_s"]])
        prev_on = on
    if not stretches:
        return out

    longest = max(stretches, key=lambda s: s[1] - s[0])
    out["one_foot_s"] = float(longest[1] - longest[0])
    # the lifted foot came down and went up again: both-feet events between two stretches
    out["touchdowns"] = float(len(stretches) - 1)
    out["wobble_s"] = float(sum(e["end_s"] - e["start_s"] for e in events if e["state"] == _WOBBLE))

    q = normalize(points)
    inside = (t >= longest[0]) & (t <= longest[1])
    hip_x = q[inside, b.MID_HIP, 0]
    hip_x = hip_x[~np.isnan(hip_x)]
    if hip_x.size >= 2:
        lo, hi = np.percentile(hip_x, [5, 95])
        out["sway"] = float(hi - lo)
    return out


# --- Shared interface ------------------------------------------------------------------------


def timeline(task: int, points: np.ndarray, t: np.ndarray) -> list[dict]:
    if task not in CLUE_NAMES:
        raise ValueError("task must be 1, 2 or 3")
    if task == 1:
        return _timeline_task1(points, t)
    if task == 3:
        return _task3_events(_task3_pose(points), t)
    _ = normalize(points)  # torso lengths, y up; the owners build the moments from this
    return _whole(t)


def compute(task: int, points: np.ndarray, t: np.ndarray) -> dict[str, float]:
    if task not in CLUE_NAMES:
        raise ValueError("task must be 1, 2 or 3")
    if task == 3:
        p = _task3_pose(points)
        return _task3_clues(p, t, _task3_events(p, t))
    events = timeline(task, points, t)
    if task == 1:
        return _clues_task1(points, t, events)
    out = {name: 0.0 for name in CLUE_NAMES[task]}
    out["duration_s"] = _duration(t)
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
