"""EXPERIMENT (Marty, 2026-11-23): two task 3 ideas, separately and together. Changes no repo file.

B = knee fallback in the jump finder. When an ankle is missing, its height is filled from the same
    side's knee: knee height - (that leg's median knee-ankle height / torso, from this file's
    standing frames) x this frame's torso length. Jumps whose window touches a filled frame are
    marked "found with the knee".
A = the 13 extra clues of scripts/experiments/t3_more_clues.py (n_air ... hands_low), with bug 1 fixed: the
    pre-jump windows (pre, crouch) stop before the first air frame (t < a0, was t <= a0).

_task3_events and _task3_clues are copied from features/clues.py (2026-11-23) so B can feed them a
filled pose; the script checks the copies give the same timeline as features.clues on every file.
Load and pick the child exactly like scripts/make_clue_table.py.

Run:  .venv\\Scripts\\python.exe scripts/experiments/exp_t3_best.py      (a few minutes, writes no files)
"""

from __future__ import annotations

import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import body25 as b  # noqa: E402
from common.pick import pick_child  # noqa: E402
from data.load_b import DEFAULT_ZIP, FPS, list_files, parse_name, read_file  # noqa: E402
from features import clues as fc  # noqa: E402
from features.clues import CLUE_NAMES, UNKNOWN, _mid_hip_xy, _task3_pose  # noqa: E402
from model.train_rf import _kappa, _macro_f1, features_for, make_forest  # noqa: E402

MIN_FRAMES = 30  # same as scripts/make_clue_table.py
NAN = float("nan")

# --- B thresholds: all GUESSES, not tuned -----------------------------------------------------
STAND_FRAC = 0.9  # GUESS: "standing" = hip-ankle height >= 0.9 x its 90th percentile in the file
STAND_PCT = 90  # GUESS (see above)
MIN_STAND_FRAMES = 5  # GUESS: fewer standing frames on a side -> no fallback for that side
MARK_WIN_S = fc._JUMP_WINDOW_S  # GUESS: a jump is "found with the knee" when a filled frame lies
#                                 within 0.45 s (the jump finder's own window) of its air part

# --- copied from features/clues.py (2026-11-23), unchanged except names ------------------------
_JUMP_WINDOW_S = fc._JUMP_WINDOW_S
_MIN_AIR_S = fc._MIN_AIR_S
_MAX_FRAME_GAP_S = fc._MAX_FRAME_GAP_S
_MAX_INTERP_GAP_S = fc._MAX_INTERP_GAP_S
_ANKLE_CHORD_LIFT = fc._ANKLE_CHORD_LIFT
_HIP_CHORD_RISE = fc._HIP_CHORD_RISE
_PEAK_ANKLE_CHORD = fc._PEAK_ANKLE_CHORD
_PEAK_ANKLE_MAX = fc._PEAK_ANKLE_MAX
_PEAK_HIP_MAX = fc._PEAK_HIP_MAX
_JUMP_END_WIN_S = fc._JUMP_END_WIN_S
_TIME_EPS = fc._TIME_EPS
_interp_short_gaps = fc._interp_short_gaps
_window_median = fc._window_median


def task3_events(p: np.ndarray, t: np.ndarray) -> list[dict]:
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


def task3_clues(p: np.ndarray, t: np.ndarray, events: list[dict]) -> dict[str, float]:
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


# --- B: knee fallback --------------------------------------------------------------------------
def knee_fill(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Copy of p with missing ankle heights filled from the knee; (filled, filled mask (T,), sides
    without enough standing frames)."""
    q = p.copy()
    hip = _mid_hip_xy(p)
    torso = np.linalg.norm(p[:, b.NECK, :2] - hip, axis=1)
    filled = np.zeros(len(p), dtype=bool)
    no_stand = 0
    for knee, ankle in ((b.L_KNEE, b.L_ANKLE), (b.R_KNEE, b.R_ANKLE)):
        ky, ay = p[:, knee, 1], p[:, ankle, 1]
        both = np.isfinite(ky) & np.isfinite(ay) & np.isfinite(torso) & (torso > 0)
        if not both.any():
            no_stand += 1
            continue
        h = np.full(len(p), np.nan)
        h[both] = (hip[both, 1] - ay[both]) / torso[both]
        stand = both & (h >= STAND_FRAC * np.nanpercentile(h[both], STAND_PCT))
        if stand.sum() < MIN_STAND_FRAMES:
            no_stand += 1
            continue
        ratio = float(np.median((ky[stand] - ay[stand]) / torso[stand]))
        miss = ~np.isfinite(ay) & np.isfinite(ky) & np.isfinite(torso) & (torso > 0)
        q[miss, ankle, 1] = ky[miss] - ratio * torso[miss]
        filled |= miss
    return q, filled, no_stand


def air_events(events: list[dict]) -> list[dict]:
    return [e for e in events if e["state"] == "air"]


def n_knee_jumps(events: list[dict], t: np.ndarray, filled: np.ndarray) -> int:
    tf = t[filled]
    return sum(
        bool(((tf >= e["start_s"] - MARK_WIN_S) & (tf <= e["end_s"] + MARK_WIN_S)).any())
        for e in air_events(events)
    )


# --- A: the 13 extra clues (scripts/experiments/t3_more_clues.py, bug 1 fixed) ----------------------------
A_NAMES = [
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
]


def _f(x, fn):
    x = x[np.isfinite(x)]
    return float(fn(x)) if x.size else NAN


def med(x):
    return _f(x, np.median)


def mn(x):
    return _f(x, np.min)


def mx(x):
    return _f(x, np.max)


def jump_feats(p: np.ndarray, t: np.ndarray, events: list[dict]) -> dict[str, float]:
    f = dict.fromkeys(A_NAMES, NAN)
    airs = [i for i in range(1, len(events) - 1) if events[i]["state"] == "air"]
    f["n_air"] = sum(e["state"] == "air" for e in events)
    best = NAN
    for i in airs:
        s = events[i - 1 : i + 2]
        if s[0]["state"] in ("before", "after") and s[2]["state"] in ("before", "after"):
            c = task3_clues(p, t, [dict(s[0], state="before"), s[1], dict(s[2], state="after")])
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

    def win(s, e):
        return (t >= s) & (t <= e)

    def before_air(s):  # bug 1 fix: pre-jump windows stop before the first air frame
        return (t >= s) & (t < a0)

    stand_h = med(h_rel[win(t[0], max(t[0] + 0.3, a0 - 0.8))])
    pre, post = before_air(a0 - 0.35), win(a1, a1 + 0.35)
    f["dx"] = abs(med(hip[post, 0]) - med(hip[pre, 0]))
    f["dy"] = med(ank_y[pre]) - med(ank_y[post])
    f["scale_gain"] = med(torso[post]) / med(torso[pre]) - 1.0
    if np.isfinite(stand_h) and stand_h > 0:
        f["crouch"] = (stand_h - mn(h_rel[before_air(a0 - 0.8)])) / stand_h
        f["land_squat"] = (stand_h - mn(h_rel[win(a1, a1 + 1.0)])) / stand_h
    f["hip_peak"] = mx(hip[win(a0, a1), 1]) - med(hip[pre, 1])
    w = win(a0 - 0.3, a1 + 0.1)
    f["arm_up"] = mx((np.nanmax(wr_y, axis=1) - neck[:, 1])[w] / torso[w])
    seg = ((np.nanmean(wr_y, axis=1) - hip[:, 1]) / torso)[win(a0 - 1.0, a0 + 0.2)]
    f["arm_range"] = mx(seg) - mn(seg)
    late = win(a1 + 0.3, a1 + 1.2)
    hx = hip[late, 0] / torso[late]
    f["land_move"] = mx(hx) - mn(hx)
    ground = {j: med(p[post, j, 1]) for j in (b.L_ANKLE, b.R_ANKLE)}
    f["land_step"] = mx(
        np.array([mx((p[late, j, 1] - g) / torso[late]) for j, g in ground.items()])
    )
    w = win(a1, a1 + 1.2)
    f["hands_low"] = mn((np.nanmin(wr_y, axis=1) - knee_y)[w] / torso[w])
    return f


# --- data --------------------------------------------------------------------------------------
def load() -> pd.DataFrame:
    rows, short, mismatch, no_stand_sides = [], 0, 0, 0
    with zipfile.ZipFile(DEFAULT_ZIP) as z:
        for name in list_files():
            m = parse_name(name)
            if m["task"] != 3:
                continue
            frames, label, _ = read_file(z, name)
            if len(frames) < MIN_FRAMES:
                short += 1
                continue
            if max(len(fr) for fr in frames) <= 1:
                pts = np.stack(
                    [fr[0] if len(fr) else np.zeros((25, 3), np.float32) for fr in frames]
                )
            else:
                pts = pick_child(frames)
            t = np.arange(len(pts)) / FPS
            p = _task3_pose(pts)
            ev = task3_events(p, t)
            if ev != fc.timeline(3, pts, t):
                mismatch += 1
            q, filled, ns = knee_fill(p)
            no_stand_sides += ns
            ev_b = task3_events(q, t)
            row = {"file": name.split("/")[-1], **m, "label": label}
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                row.update(task3_clues(p, t, ev))
                row.update(jump_feats(p, t, ev))
                cb = task3_clues(q, t, ev_b)
                ab = jump_feats(q, t, ev_b)
            row.update({f"{k}_B": v for k, v in cb.items()})
            row.update({f"{k}_B": v for k, v in ab.items()})
            row["jumps"] = len(air_events(ev))
            row["jumps_B"] = len(air_events(ev_b))
            row["knee_jumps_B"] = n_knee_jumps(ev_b, t, filled)
            row["any_filled"] = bool(filled.any())
            rows.append(row)
    d = pd.DataFrame(rows)
    print(
        f"task 3 files: {len(d)} | left out {short} under {MIN_FRAMES} frames | copied "
        f"timeline differs from features.clues on {mismatch} files | leg sides without "
        f"{MIN_STAND_FRAMES} standing frames (no fallback): {no_stand_sides}"
    )
    return d


def b_check(d: pd.DataFrame) -> None:
    for what, col in (("no air event", "jumps"), ("air_s == 0 (no full jump)", "air_s")):
        none = d[d[col] == 0]
        colb = f"{col}_B" if col == "air_s" else "jumps_B"
        gp = none[none["label"] > 0]
        bad = none[none["label"] == 0]
        print(
            f"B check, no-jump = {what}: Good/Perfect {len(gp)} -> {int((gp[colb] > 0).sum())} now"
            f" get a jump | Bad {len(bad)} -> {int((bad[colb] == 0).sum())} stay no-jump"
        )
    print(
        f"  jumps found with the knee (all files): {int(d['knee_jumps_B'].sum())} in "
        f"{int((d['knee_jumps_B'] > 0).sum())} files; files with any filled ankle: "
        f"{int(d['any_filled'].sum())}"
    )
    eye = pd.read_csv("docs/eye_check/by_eye_sheet.csv")
    eye = eye[eye["task"] == 3]
    parts = []
    for _, r in eye.iterrows():
        row = d[d["file"] == f"{r['clip']}.json"]
        if row.empty:
            parts.append(f"{r['clip']} missing")
            continue
        v = row.iloc[0]
        parts.append(f"{r['clip']} eye {r['jumps']} now {v['jumps']} B {v['jumps_B']}")
    print("  by-eye clips: " + " | ".join(parts))


def cv(d: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = d[cols].fillna(-9).to_numpy(dtype=np.float64)  # -9 fill: same as t3_more_clues.py
    y = d["label"].to_numpy(dtype=int)
    g = d["child"].astype(str).to_numpy()
    folds = list(GroupKFold(n_splits=5).split(x, y, g))
    F, K, R = [], [], []
    for seed in range(5):
        for tr, te in folds:
            pr = make_forest(seed).fit(x[tr], y[tr]).predict(x[te])
            F.append(_macro_f1(y[te], pr))
            K.append(_kappa(y[te], pr))
            R.append(recall_score(y[te], pr, labels=[1], average="macro", zero_division=0))
    return np.array(F), np.array(K), np.array(R)


def main() -> None:
    d = load()
    b_check(d)
    base = features_for(3)
    sets = {
        "current 4 clues": base,
        "4 clues with B": [f"{c}_B" for c in base],
        "4 + A": base + A_NAMES,
        "4 + A with B": [f"{c}_B" for c in base + A_NAMES],
    }
    print(
        f"\nrandom forest (make_forest), GroupKFold(5) by child, seeds 0-4 = 25 runs, {len(d)} rows"
    )
    print(f"  {'version':<17} macro-F1  kappa  Good recall  better than current (F1 / kappa)")
    ref = None
    for title, cols in sets.items():
        F, K, R = cv(d, cols)
        ref = ref or (F, K)
        wins = f"{int((F > ref[0]).sum())}/25 / {int((K > ref[1]).sum())}/25"
        print(f"  {title:<17} {F.mean():.3f}     {K.mean():.3f}  {R.mean():.2f}         {wins}")


if __name__ == "__main__":
    main()
