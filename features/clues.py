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

All three tasks are real: task 1 (Louise, #20), task 2 (Louise, #21), task 3 (Revallo, #32).
Ideas per clue: Job Card 2 in the team Google Doc.
"""

from __future__ import annotations

import warnings

import numpy as np

from common import body25 as b
from common.geometry import normalize

CLUE_NAMES = {
    1: [
        "duration_s",
        "one_foot_s",
        "touchdowns",
        "sway",
        "wobble_s",
        "hip_speed",
        "neck_speed",
        "lift_std",
        "wrist_speed",
    ],
    2: ["duration_s", "hops", "longest_pause_s", "touchdowns"],
    3: [
        "duration_s",
        "feet_gap_takeoff",
        "feet_gap_landing",
        "air_s",
        "jump_size",
        # 13 extra clues "A" (Marty, 2026-11-23, docs/task_cards/marty_2026-11-23_t3_experiments.md):
        # 5-seed macro-F1 0.522 -> 0.558, kappa 0.464 -> 0.529. Keep the 5 names above in place.
        *(
            "n_air",
            "best_jump_size",
            "dx",
            "dy",
            "scale_gain",
            "crouch",
            "hip_peak",
            "arm_up",
            "arm_range",
            "land_squat",
            "land_move",
            "land_step",
            "hands_low",
        ),
    ],
}

UNKNOWN = "unknown"
STATES = {
    1: ("both_feet", "steady", "wobble"),
    2: ("hop", "pause", "both_feet"),
    3: ("before", "air", "after"),
}

# Task 3 (standing long jump) thresholds in seconds and torso lengths.
# Hybrid-C: chord baseline for airborne membership, validated by peak clearance above both
# the chord and strict max(left_min, right_min) to reject flat->crouch phantom jumps.
_JUMP_WINDOW_S = 0.45
_MIN_AIR_S = 0.06
_MAX_FRAME_GAP_S = 0.25
_MAX_INTERP_GAP_S = 0.15
_ANKLE_CHORD_LIFT = 0.04
_HIP_CHORD_RISE = 0.08
_PEAK_ANKLE_CHORD = 0.06
_PEAK_ANKLE_MAX = 0.03
_PEAK_HIP_MAX = 0.12
_JUMP_END_WIN_S = 0.35
_TIME_EPS = 1e-6  # avoids IEEE-754 boundary flips at exact fps multiples (e.g. 3/20 == 0.15 s)


def _mid_hip_xy(p: np.ndarray) -> np.ndarray:
    """Mid-hip (x, y) per frame, falling back to (L_HIP + R_HIP) / 2 when MID_HIP is NaN."""
    hip = p[:, b.MID_HIP, :2].copy()
    lr = p[:, [b.L_HIP, b.R_HIP], :2]
    miss = (~np.isfinite(hip).all(axis=1)) & np.isfinite(lr).all(axis=(1, 2))
    hip[miss] = 0.5 * (lr[miss, 0] + lr[miss, 1])
    return hip


def _interp_short_gaps(values: np.ndarray, t: np.ndarray, max_gap_s: float) -> np.ndarray:
    """Linearly interpolate interior NaN runs whose bounding valid frames are <= max_gap_s apart."""
    out = values.copy()
    ok = np.flatnonzero(np.isfinite(out))
    for a, b_idx in zip(ok, ok[1:], strict=False):
        if b_idx > a + 1 and t[b_idx] - t[a] <= max_gap_s + _TIME_EPS:
            out[a + 1 : b_idx] = np.interp(t[a + 1 : b_idx], [t[a], t[b_idx]], [out[a], out[b_idx]])
    return out


def _window_median(
    series: np.ndarray, mask: np.ndarray, fallback_idx: int, *, positive: bool = False
) -> float | None:
    """Median of finite (and optionally > 0) samples in mask, falling back to a single frame."""
    vals = series[mask]
    ok = np.isfinite(vals) & (vals > 0) if positive else np.isfinite(vals)
    if np.any(ok):
        return float(np.median(vals[ok]))
    fb = float(series[fallback_idx])
    if np.isfinite(fb) and (not positive or fb > 0):
        return fb
    return None


def _task3_events(p: np.ndarray, t: np.ndarray) -> list[dict]:
    """Find short two-foot lifts relative to nearby grounded poses on either side."""
    ankles = p[:, [b.R_ANKLE, b.L_ANKLE], 1].copy()
    hip_y = _mid_hip_xy(p)[:, 1]
    signals = np.column_stack((ankles, hip_y))
    time_cuts = np.flatnonzero(np.diff(t) > _MAX_FRAME_GAP_S + _TIME_EPS) + 1
    for sec in np.split(np.arange(len(t)), time_cuts):
        for col in range(3):
            signals[sec, col] = _interp_short_gaps(signals[sec, col], t[sec], _MAX_INTERP_GAP_S)

    valid = np.isfinite(signals).all(axis=1)
    airborne = np.zeros(len(t), dtype=bool)
    states = np.full(len(t), UNKNOWN, dtype=object)
    lift_chord = np.full((len(t), 3), -np.inf)
    lift_max = np.full((len(t), 3), -np.inf)
    cuts = (
        np.flatnonzero((np.diff(t) > _MAX_FRAME_GAP_S + _TIME_EPS) | (valid[:-1] != valid[1:])) + 1
    )
    cols = np.arange(3)
    for section in np.split(np.arange(len(t)), cuts):
        if not valid[section[0]]:
            continue
        lo, hi = int(section[0]), int(section[-1]) + 1
        for i in range(lo + 1, hi - 1):
            left = max(lo, int(np.searchsorted(t, t[i] - _JUMP_WINDOW_S - _TIME_EPS)))
            right = min(
                hi, int(np.searchsorted(t, t[i] + _JUMP_WINDOW_S + _TIME_EPS, side="right"))
            )
            if left == i or right == i + 1:
                continue
            l_idx = left + np.argmin(signals[left:i], axis=0)
            r_idx = (i + 1) + np.argmin(signals[i + 1 : right], axis=0)
            l_min = signals[l_idx, cols]
            r_min = signals[r_idx, cols]
            weight = (t[i] - t[l_idx]) / (t[r_idx] - t[l_idx])
            chord = l_min + (r_min - l_min) * weight
            lc = signals[i] - chord
            lm = signals[i] - np.maximum(l_min, r_min)
            lift_chord[i] = lc
            lift_max[i] = lm
            airborne[i] = bool((lc[:2] >= _ANKLE_CHORD_LIFT).all() and lc[2] >= _HIP_CHORD_RISE)

        changes = np.flatnonzero(np.diff(airborne[lo:hi], prepend=False, append=False))
        for start, end in changes.reshape(-1, 2):
            s, e = lo + int(start), lo + int(end)
            if t[e] - t[s] < _MIN_AIR_S - _TIME_EPS:
                airborne[s:e] = False
                continue
            pk_ac = float(lift_chord[s:e, :2].min(axis=1).max())
            pk_am = float(lift_max[s:e, :2].min(axis=1).max())
            pk_hm = float(lift_max[s:e, 2].max())
            if not (
                pk_ac >= _PEAK_ANKLE_CHORD and pk_am >= _PEAK_ANKLE_MAX and pk_hm >= _PEAK_HIP_MAX
            ):
                airborne[s:e] = False

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
        elif t[end] - t[end - 1] > _MAX_FRAME_GAP_S + _TIME_EPS:
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
        takeoff = int(np.searchsorted(t, air["start_s"])) - 1
        landing = int(np.searchsorted(t, air["end_s"]))
        boundary = p[[takeoff, landing]]
        gaps = np.linalg.norm(boundary[:, b.R_ANKLE, :2] - boundary[:, b.L_ANKLE, :2], axis=1)
        widths = np.linalg.norm(boundary[:, b.R_HIP, :2] - boundary[:, b.L_HIP, :2], axis=1)
        for key, gap, width in zip(CLUE_NAMES[3][1:3], gaps, widths, strict=True):
            if np.isfinite(gap) and np.isfinite(width) and width > 0:
                out[key] = float(gap / width)
        out["air_s"] = float(air["end_s"] - air["start_s"])

        pre = (t >= max(before["start_s"], air["start_s"] - _JUMP_END_WIN_S) - _TIME_EPS) & (
            t < air["start_s"]
        )
        is_last = i + 1 == len(events) - 1
        end_idx = int(np.searchsorted(t, after["end_s"]))
        gap_after = (
            not is_last
            and events[i + 2]["state"] == UNKNOWN
            and end_idx + 1 < len(t)
            and t[end_idx] == after["end_s"]
            and t[end_idx + 1] == events[i + 2]["end_s"]
            and t[end_idx + 1] - t[end_idx] > _MAX_FRAME_GAP_S + _TIME_EPS
        )
        in_after = (t >= air["end_s"]) & (
            t <= after["end_s"] if (is_last or gap_after) else t < after["end_s"]
        )
        post = in_after & (t <= air["end_s"] + _JUMP_END_WIN_S + _TIME_EPS)

        hip_xy = _mid_hip_xy(p)
        hx_pre = _window_median(hip_xy[:, 0], pre, takeoff)
        hx_post = _window_median(hip_xy[:, 0], post, landing)
        dx = abs(hx_post - hx_pre) if hx_pre is not None and hx_post is not None else 0.0

        ankles_y = p[:, [b.R_ANKLE, b.L_ANKLE], 1]
        both_ankles = np.isfinite(ankles_y).all(axis=1)
        foot_y = np.where(both_ankles, 0.5 * (ankles_y[:, 0] + ankles_y[:, 1]), np.nan)
        fy_pre = _window_median(foot_y, pre, takeoff)
        fy_post = _window_median(foot_y, post, landing)
        dy = fy_pre - fy_post if fy_pre is not None and fy_post is not None else 0.0

        torso = np.linalg.norm(p[:, b.NECK, :2] - hip_xy, axis=1)
        torso_pre = _window_median(torso, pre, takeoff, positive=True)
        torso_post = _window_median(torso, post, landing, positive=True)
        scale_gain = (
            torso_post / torso_pre - 1.0
            if torso_pre is not None and torso_post is not None
            else 0.0
        )

        out["jump_size"] = float(np.hypot(dx, 1.5 * max(0.0, dy) + 2.5 * max(0.0, scale_gain)))
        break
    return out


# Task 3 extra clues "A": ported from scripts/experiments/exp_t3_best.py (jump_feats, bug 1 fixed:
# the pre-jump windows stop before the first air frame). Windows in seconds around the first full
# jump (air start a0, air end a1); every value in torso lengths, from BODY_25 points only.
_T3_EXTRA_NAMES = CLUE_NAMES[3][5:]
_T3_FILL = (
    -9.0
)  # a clue that can't be measured (no jump, joint missing); same fill as the experiment


def _nan_stat(x: np.ndarray, fn) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(fn(x)) if x.size else float("nan")


def _nanmed(x: np.ndarray) -> float:
    return _nan_stat(x, np.median)


def _nanmin(x: np.ndarray) -> float:
    return _nan_stat(x, np.min)


def _nanmax(x: np.ndarray) -> float:
    return _nan_stat(x, np.max)


def _task3_extra_raw(p: np.ndarray, t: np.ndarray, events: list[dict]) -> dict[str, float]:
    nan = float("nan")
    f = dict.fromkeys(_T3_EXTRA_NAMES, nan)
    airs = [i for i in range(1, len(events) - 1) if events[i]["state"] == "air"]
    f["n_air"] = float(sum(e["state"] == "air" for e in events))
    best = nan
    for i in airs:
        s = events[i - 1 : i + 2]
        if s[0]["state"] in ("before", "after") and s[2]["state"] in ("before", "after"):
            c = _task3_clues(p, t, [dict(s[0], state="before"), s[1], dict(s[2], state="after")])
            if c["air_s"] > 0:
                best = c["jump_size"] if np.isnan(best) else max(best, c["jump_size"])
    f["best_jump_size"] = best
    first = next(
        (
            i
            for i in airs
            if events[i - 1]["state"] in ("before", "after") and events[i + 1]["state"] == "after"
        ),
        None,
    )
    if first is None:
        return f
    hip = _mid_hip_xy(p)
    neck = p[:, b.NECK, :2]
    torso = np.linalg.norm(neck - hip, axis=1)
    ank_y = np.nanmean(p[:, [b.L_ANKLE, b.R_ANKLE], 1], axis=1)
    knee_y = np.nanmean(p[:, [b.L_KNEE, b.R_KNEE], 1], axis=1)
    wr_y = p[:, [b.L_WRIST, b.R_WRIST], 1]
    h_rel = (hip[:, 1] - ank_y) / torso
    a0, a1 = events[first]["start_s"], events[first]["end_s"]

    # _TIME_EPS: window edges such as a1 + 1.2 s fall exactly on frames (30 fps); keep those frames
    # in, whatever the clock offset (the experiment used no eps; see the task card for the effect).
    def win(s0: float, s1: float) -> np.ndarray:
        return (t >= s0 - _TIME_EPS) & (t <= s1 + _TIME_EPS)

    def before_air(s0: float) -> np.ndarray:  # pre-jump windows stop before the first air frame
        return (t >= s0 - _TIME_EPS) & (t < a0)  # a0 is a frame time itself

    stand_h = _nanmed(h_rel[win(t[0], max(t[0] + 0.3, a0 - 0.8))])
    pre, post = before_air(a0 - 0.35), win(a1, a1 + 0.35)
    f["dx"] = abs(_nanmed(hip[post, 0]) - _nanmed(hip[pre, 0]))
    f["dy"] = _nanmed(ank_y[pre]) - _nanmed(ank_y[post])
    f["scale_gain"] = _nanmed(torso[post]) / _nanmed(torso[pre]) - 1.0
    if np.isfinite(stand_h) and stand_h > 0:
        f["crouch"] = (stand_h - _nanmin(h_rel[before_air(a0 - 0.8)])) / stand_h
        f["land_squat"] = (stand_h - _nanmin(h_rel[win(a1, a1 + 1.0)])) / stand_h
    f["hip_peak"] = _nanmax(hip[win(a0, a1), 1]) - _nanmed(hip[pre, 1])
    w = win(a0 - 0.3, a1 + 0.1)
    f["arm_up"] = _nanmax((np.nanmax(wr_y, axis=1) - neck[:, 1])[w] / torso[w])
    seg = ((np.nanmean(wr_y, axis=1) - hip[:, 1]) / torso)[win(a0 - 1.0, a0 + 0.2)]
    f["arm_range"] = _nanmax(seg) - _nanmin(seg)
    late = win(a1 + 0.3, a1 + 1.2)
    hx = hip[late, 0] / torso[late]
    f["land_move"] = _nanmax(hx) - _nanmin(hx)
    ground = {j: _nanmed(p[post, j, 1]) for j in (b.L_ANKLE, b.R_ANKLE)}
    f["land_step"] = _nanmax(
        np.array([_nanmax((p[late, j, 1] - g) / torso[late]) for j, g in ground.items()])
    )
    w = win(a1, a1 + 1.2)
    f["hands_low"] = _nanmin((np.nanmin(wr_y, axis=1) - knee_y)[w] / torso[w])
    return f


def _task3_extra_clues(p: np.ndarray, t: np.ndarray, events: list[dict]) -> dict[str, float]:
    """The 13 extra task 3 clues, every value a finite float (_T3_FILL where unmeasured)."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = _task3_extra_raw(p, t, events)
    return {k: float(v) if np.isfinite(v) else _T3_FILL for k, v in raw.items()}


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


def _smoothed_speed_1d(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Double-smoothed 1D speed |dx/dt| in torso lengths per second."""
    pos = _smooth(_fill_nan(x), t, SMOOTH_S)
    speed = np.abs(np.gradient(pos, t)) if len(t) > 1 else np.zeros(len(t))
    return _smooth(speed, t, SMOOTH_S)


def _smoothed_speed_2d(xy: np.ndarray, t: np.ndarray) -> np.ndarray:
    """2D speed hypot(vx, vy) of a smoothed trajectory, in torso lengths per second."""
    if len(t) <= 1:
        return np.zeros(len(t))
    sx = _smooth(_fill_nan(xy[:, 0]), t, SMOOTH_S)
    sy = _smooth(_fill_nan(xy[:, 1]), t, SMOOTH_S)
    return np.hypot(np.gradient(sx, t), np.gradient(sy, t))


def _frame_states_task1(q: np.ndarray, t: np.ndarray) -> tuple[list[str], np.ndarray]:
    yl = _foot_height(q, b.L_ANKLE, b.L_HEEL)
    yr = _foot_height(q, b.R_ANKLE, b.R_HEEL)
    gap = np.abs(yl - yr)
    speed = _smoothed_speed_1d(q[:, b.MID_HIP, 0], t)
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
    """Every clue comes from the timeline, except sway and the core stagger clues, which are
    measured inside the longest one-foot stretch that the timeline marks."""
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

    # Trimmed core inside the longest one-foot stretch to exclude lift-up / foot-down transients.
    t0, t1 = longest
    trim = min(0.2, 0.15 * (t1 - t0))
    core = (t >= t0 + trim) & (t <= t1 - trim)
    if int(core.sum()) < 4:
        core = inside
    if int(core.sum()) >= 2:
        if np.isfinite(q[core, b.MID_HIP, 0]).sum() >= 2:
            out["hip_speed"] = float(np.median(_smoothed_speed_1d(q[:, b.MID_HIP, 0], t)[core]))
        if np.isfinite(q[core, b.NECK, 0]).sum() >= 2:
            out["neck_speed"] = float(np.median(_smoothed_speed_1d(q[:, b.NECK, 0], t)[core]))
        yl = _foot_height(q, b.L_ANKLE, b.L_HEEL)
        yr = _foot_height(q, b.R_ANKLE, b.R_HEEL)
        gap_core = np.abs(yl[core] - yr[core])
        gap_core = gap_core[np.isfinite(gap_core)]
        if gap_core.size >= 2:
            out["lift_std"] = float(np.std(gap_core))
        wrist_p90: list[float] = []
        for w_idx in (b.L_WRIST, b.R_WRIST):
            ok_w = np.isfinite(q[core, w_idx, 0]) & np.isfinite(q[core, w_idx, 1])
            if ok_w.sum() >= 2:
                wrist_p90.append(
                    float(np.percentile(_smoothed_speed_2d(q[:, w_idx, :2], t)[core], 90))
                )
        if wrist_p90:
            out["wrist_speed"] = float(np.mean(wrist_p90))
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
# The lifted foot must be at least HOP_LIFT torso lengths above the hopping foot at the bounce;
# otherwise the bounce is a body bob, not a hop (same height as task 1's LIFT_PEAK). By eye
# 2026-11-19 (Louise, 7 camera-2 clips, by_eye_sheet.csv): without it the program found one
# extra hop in 4 of 5 Bad clips; with 0.3, 6 of 7 clips match (B213_GMS_2_5_2: 0 vs 1 by eye).
# Task 2 random forest macro-F1 / kappa: 0.60 / 0.59 without, 0.66 / 0.66 with.
HOP_LIFT = 0.3
# Both feet down: both ankles within ONE_FOOT_GAP of the floor (lowest ankle within FLOOR_WIN_S;
# the floor line drifts as the child hops toward the camera). Task 1's "ankles level" fails here:
# in the air the hopping foot rises level with the lifted one.
FLOOR_WIN_S = 1.0
# Two hops further apart than this have a pause between them (normal child hop cadence is
# ~0.37-0.40 s; troughs before/after each peak are bounded within HALF_HOP_S so a pause is
# not absorbed into the neighbouring hop).
HALF_HOP_S = 0.25
PAUSE_GAP_S = 0.45

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


def _hop_peaks(fast: np.ndarray, gap: np.ndarray, t: np.ndarray, i0: int, i1: int) -> list[int]:
    peaks: list[int] = []
    for i in range(max(i0, 1), min(i1, len(t) - 2) + 1):
        if not (fast[i] >= fast[i - 1] and fast[i] > fast[i + 1] and fast[i] > HOP_BOUNCE):
            continue
        if peaks and t[i] - t[peaks[-1]] < HOP_SEP_S:
            if fast[i] > fast[peaks[-1]]:
                peaks[-1] = i
            continue
        peaks.append(i)
    # a bounce while the lifted foot is barely up is a body bob, not a hop
    return [i for i in peaks if gap[i] >= HOP_LIFT]


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
        peaks = _hop_peaks(fast, gap, t, i0, i1)
        if not peaks:
            spans.append((i0, _PAUSE))
            continue
        # low point before each peak (bounded by HALF_HOP_S, previous peak, and stretch start)
        starts = []
        for k, pk in enumerate(peaks):
            lo_bound = int(np.searchsorted(t, t[pk] - HALF_HOP_S, side="left"))
            lo = max(peaks[k - 1] + 1 if k else i0, lo_bound)
            starts.append(lo + int(np.argmin(fast[lo : pk + 1])))
        if starts[0] > i0:
            spans.append((i0, _PAUSE))
        for k, pk in enumerate(peaks):
            spans.append((starts[k], _HOP))
            nxt_peak = peaks[k + 1] if k + 1 < len(peaks) else None
            if nxt_peak is None or t[nxt_peak] - t[pk] > PAUSE_GAP_S:
                hi = starts[k + 1] if nxt_peak is not None else i1 + 1
                hi_hop = min(hi, int(np.searchsorted(t, t[pk] + HALF_HOP_S, side="right")))
                after = pk + int(np.argmin(fast[pk:hi_hop])) if hi_hop > pk else pk
                if after + 1 < hi and after + 1 < len(t) - 1:
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
    # touch-downs between hops: both_feet events between the first hop's end and last hop's start
    if len(hops) >= 2:
        t_start, t_end = hops[0]["end_s"], hops[-1]["start_s"]
        out["touchdowns"] = float(
            sum(
                1
                for e in events
                if e["state"] == _BOTH and e["start_s"] >= t_start and e["end_s"] <= t_end
            )
        )
    return out


# --- Shared interface ------------------------------------------------------------------------


def timeline(task: int, points: np.ndarray, t: np.ndarray) -> list[dict]:
    if task not in CLUE_NAMES:
        raise ValueError("task must be 1, 2 or 3")
    if task == 1:
        return _timeline_task1(points, t)
    if task == 2:
        return _timeline_task2(points, t)
    if task == 3:
        return _task3_events(_task3_pose(points), t)
    _ = normalize(points)  # torso lengths, y up; the owners build the moments from this
    return _whole(t)


def timeline_and_compute(
    task: int, points: np.ndarray, t: np.ndarray
) -> tuple[list[dict], dict[str, float]]:
    """Compute the moment timeline once and derive all clues for the task from it."""
    if task not in CLUE_NAMES:
        raise ValueError("task must be 1, 2 or 3")
    if task == 3:
        p = _task3_pose(points)
        events = _task3_events(p, t)
        out = _task3_clues(p, t, events)
        out.update(_task3_extra_clues(p, t, events))
        return events, out
    events = timeline(task, points, t)
    if task == 1:
        return events, _clues_task1(points, t, events)
    if task == 2:
        return events, _clues_task2(t, events)
    out = {name: 0.0 for name in CLUE_NAMES[task]}
    out["duration_s"] = _duration(t)
    return events, out


def compute(task: int, points: np.ndarray, t: np.ndarray) -> dict[str, float]:
    return timeline_and_compute(task, points, t)[1]


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
