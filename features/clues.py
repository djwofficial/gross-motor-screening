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

Tasks 1 and 2 are real (Louise, #20 and #21). STUB for task 3: only duration_s is real; the other
values are 0.0 and the timeline is one UNKNOWN block until Revallo's version lands. Ideas
per clue: Job Card 2 in the team Google Doc.
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


# --- Task 2 (hop on one leg), Louise ---------------------------------------------------------
# A hop = one fast up-and-down bounce of the mid-hip while one foot is clearly above the other.
# Fast part of the hip = hip height minus its HOP_WINDOW_S moving average; a peak above HOP_BOUNCE
# (torso lengths), at least HOP_SEP_S after the previous one, is a hop. Foot height alone misses
# small hops (feet only 0.05-0.12 torso lengths off the floor in B054_GMS_2_1_2).
# By eye 2026-11-19 (Louise, camera 2): B054_GMS_2_1_2 10 hops (we find 10), B209_GMS_2_2_2 2 (2),
# B210_GMS_2_2_2 0 (1). Group B one-person task 2 files: median hops 1 / 3 / 6 (Bad/Good/Perfect)
# (scripts/check_clues_task2.py). 0.15 would give B210 0 but B209 1 and B054 4.
HOP_BOUNCE = 0.1
HOP_WINDOW_S = 0.5
HOP_SEP_S = 0.2
# Both feet down: both ankles within ONE_FOOT_GAP of the floor (lowest ankle within FLOOR_WIN_S;
# the floor line drifts as the child hops toward the camera). Task 1's "ankles level" fails here:
# in the air the hopping foot rises level with the lifted one.
FLOOR_WIN_S = 1.0
# Two hops further apart than this have a pause between them (normal rhythm in B054: about 0.35 s;
# start value, tune by eye).
PAUSE_GAP_S = 0.7

_HOP, _PAUSE = "hop", "pause"


def _rolling_min(x: np.ndarray, t: np.ndarray, window_s: float) -> np.ndarray:
    half = window_s / 2
    lo = np.searchsorted(t, t - half, side="left")
    hi = np.searchsorted(t, t + half, side="right")
    return np.array([x[a:z].min() for a, z in zip(lo, hi, strict=True)])


def _hop_frames(q: np.ndarray, t: np.ndarray):
    """-> (on_one_foot, missing, ankle gap, fast hip height), each (T,)."""
    yl = _foot_height(q, b.L_ANKLE, b.L_HEEL)
    yr = _foot_height(q, b.R_ANKLE, b.R_HEEL)
    missing = np.isnan(yl) | np.isnan(yr)
    yl, yr = _fill_nan(yl), _fill_nan(yr)
    floor = _rolling_min(np.minimum(yl, yr), t, FLOOR_WIN_S)
    gap = np.abs(yl - yr)
    both_down = (yl - floor < ONE_FOOT_GAP) & (yr - floor < ONE_FOOT_GAP)
    on_one = ~both_down & (gap > ONE_FOOT_GAP)
    hip = _smooth(_fill_nan(q[:, b.MID_HIP, 1]), t, 0.1)
    fast = hip - _smooth(hip, t, HOP_WINDOW_S)
    return on_one, missing, gap, fast


def _hop_peaks(fast: np.ndarray, t: np.ndarray, i0: int, i1: int) -> list[int]:
    peaks: list[int] = []
    for i in range(max(i0, 1), min(i1, len(t) - 2) + 1):
        if not (fast[i] >= fast[i - 1] and fast[i] > fast[i + 1] and fast[i] > HOP_BOUNCE):
            continue
        if peaks and t[i] - t[peaks[-1]] < HOP_SEP_S:
            if fast[i] > fast[peaks[-1]]:
                peaks[-1] = i
            continue
        peaks.append(i)
    return peaks


def _timeline_task2(points: np.ndarray, t: np.ndarray) -> list[dict]:
    """Each hop is its own event, so the hop count is the number of hop events. A hop runs from
    the low point before its bounce to the next hop's start (or to the low point after it when a
    pause or both feet follow)."""
    try:
        q = normalize(points)
    except ValueError:
        return _whole(t)
    on_one, missing, gap, fast = _hop_frames(q, t)
    base = [_UNK if m else (_ONE if o else _BOTH) for m, o in zip(missing, on_one, strict=True)]
    runs = _drop_low_lifts(_merge_short(_runs(base), t, MIN_RUN_S), gap)

    spans: list[tuple[int, str]] = []  # (first frame, state); each span runs to the next one
    for s, i0, i1 in runs:
        if s != _ONE:
            spans.append((i0, s))
            continue
        peaks = _hop_peaks(fast, t, i0, i1)
        if not peaks:
            spans.append((i0, _PAUSE))
            continue
        # low point before each peak (bounded by the previous peak and the stretch start)
        starts = []
        for k, pk in enumerate(peaks):
            lo = peaks[k - 1] + 1 if k else i0
            starts.append(lo + int(np.argmin(fast[lo : pk + 1])))
        if starts[0] > i0:
            spans.append((i0, _PAUSE))
        for k, pk in enumerate(peaks):
            spans.append((starts[k], _HOP))
            nxt_peak = peaks[k + 1] if k + 1 < len(peaks) else None
            if nxt_peak is None or t[nxt_peak] - t[pk] > PAUSE_GAP_S:
                hi = starts[k + 1] if nxt_peak is not None else i1 + 1
                after = pk + int(np.argmin(fast[pk:hi])) if hi > pk else pk
                if after + 1 < hi:
                    spans.append((after + 1, _PAUSE))

    events = []
    for k, (i0, s) in enumerate(spans):
        end = t[spans[k + 1][0]] if k + 1 < len(spans) else t[-1]
        events.append({"start_s": float(t[i0]), "end_s": float(end), "state": s})
    return events


def _clues_task2(t: np.ndarray, events: list[dict]) -> dict[str, float]:
    out = {name: 0.0 for name in CLUE_NAMES[2]}
    out["duration_s"] = _duration(t)
    hops = [e for e in events if e["state"] == _HOP]
    out["hops"] = float(len(hops))
    # longest time between two hops: from a hop's end to the next hop's start (pauses, both feet)
    gaps = [b_["start_s"] - a["end_s"] for a, b_ in zip(hops, hops[1:], strict=False)]
    out["longest_pause_s"] = float(max(gaps)) if gaps else 0.0
    # the lifted foot came down and went up again: both_feet between two one-foot stretches
    on = [e["state"] in (_HOP, _PAUSE) for e in events]
    stretches = sum(1 for k, x in enumerate(on) if x and (k == 0 or not on[k - 1]))
    out["touchdowns"] = float(max(stretches - 1, 0))
    return out


# --- Shared interface ------------------------------------------------------------------------


def timeline(task: int, points: np.ndarray, t: np.ndarray) -> list[dict]:
    if task not in CLUE_NAMES:
        raise ValueError("task must be 1, 2 or 3")
    if task == 1:
        return _timeline_task1(points, t)
    if task == 2:
        return _timeline_task2(points, t)
    _ = normalize(points)  # torso lengths, y up; the owners build the moments from this
    return _whole(t)


def compute(task: int, points: np.ndarray, t: np.ndarray) -> dict[str, float]:
    if task not in CLUE_NAMES:
        raise ValueError("task must be 1, 2 or 3")
    events = timeline(task, points, t)
    if task == 1:
        return _clues_task1(points, t, events)
    if task == 2:
        return _clues_task2(t, events)
    out = {name: 0.0 for name in CLUE_NAMES[task]}
    out["duration_s"] = _duration(t)
    return out


MAX_MOMENTS = 4  # moments named in one reason text; the full timeline is in the try's JSON


def _plural(n: float, word: str) -> str:
    n = int(round(n))
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _moments_task1(events: list[dict]) -> list[str]:
    """'foot up at 0.8 s', 'wobble at 1.4 s', 'foot down at 3.1 s', in time order."""
    out = []
    prev = None
    for e in events:
        s = e["state"]
        if s in _ON_ONE_FOOT and prev not in _ON_ONE_FOOT:
            extra = " with a wobble" if s == _WOBBLE else ""
            out.append(f"foot up at {e['start_s']:.1f} s{extra}")
        elif s == _WOBBLE and prev != _WOBBLE:
            out.append(f"wobble at {e['start_s']:.1f} s")
        if s == _BOTH and prev in _ON_ONE_FOOT:
            out.append(f"foot down at {e['start_s']:.1f} s")
        prev = s
    return out


def _explain_task1(c: dict[str, float], events: list[dict] | None) -> str:
    """Says what was measured, never that the child failed (team rule)."""
    if c["one_foot_s"] <= 0:
        parts = ["no clear one-foot stand seen"]
    else:
        parts = [f"{c['one_foot_s']:.1f} s on one foot", _plural(c["touchdowns"], "touch-down")]
        if c["wobble_s"] > 0:
            parts.append(f"{c['wobble_s']:.1f} s wobbling")
    if events:
        moments = _moments_task1(events)
        if len(moments) > MAX_MOMENTS:
            moments = moments[:MAX_MOMENTS] + ["..."]
        parts += moments
    return ", ".join(parts)


def _moments_task2(events: list[dict]) -> list[str]:
    """'hops from 0.9 to 4.6 s', 'pause at 2.0 s', 'foot down at 3.1 s', in time order."""
    out = []
    prev = None
    run_start = run_end = None  # current run of back-to-back hops
    for e in events:
        s = e["state"]
        if s == _HOP:
            if prev != _HOP:
                run_start = e["start_s"]
            run_end = e["end_s"]
        elif prev == _HOP:
            out.append(f"hops from {run_start:.1f} to {run_end:.1f} s")
        if s == _PAUSE and prev == _HOP:
            out.append(f"pause at {e['start_s']:.1f} s")
        if s == _BOTH and prev in (_HOP, _PAUSE):
            out.append(f"foot down at {e['start_s']:.1f} s")
        prev = s
    if prev == _HOP:
        out.append(f"hops from {run_start:.1f} to {run_end:.1f} s")
    return out


def _explain_task2(c: dict[str, float], events: list[dict] | None) -> str:
    """Says what was measured, never that the child failed (team rule)."""
    if c["hops"] <= 0:
        parts = ["no hop seen"]
    else:
        parts = [_plural(c["hops"], "hop"), _plural(c["touchdowns"], "touch-down")]
        if c["longest_pause_s"] > 0:
            parts.append(f"longest pause {c['longest_pause_s']:.1f} s")
    if events:
        moments = _moments_task2(events)
        if len(moments) > MAX_MOMENTS:
            moments = moments[:MAX_MOMENTS] + ["..."]
        parts += moments
    return ", ".join(parts)


def explain(task: int, clues: dict[str, float], events: list[dict] | None = None) -> str:
    c = clues
    if task == 1:
        return _explain_task1(c, events)
    if task == 2:
        return _explain_task2(c, events)
    if task == 3:
        return f"feet gap {c['feet_gap_takeoff']:.2f} at take-off, {c['air_s']:.2f} s in the air, jump size {c['jump_size']:.2f}"
    raise ValueError("task must be 1, 2 or 3")
