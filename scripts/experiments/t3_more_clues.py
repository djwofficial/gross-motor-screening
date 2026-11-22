"""EXPERIMENT (2026-11-23): 13 more task 3 clue ideas on group B. Changes no repo file.

From the repo root (with .venv on):
    python scripts/experiments/t3_more_clues.py
Needs data/raw/B_skel.zip. Takes a few minutes (5 feature sets x 5 seeds x 5 folds).
Ideas (all in torso lengths and seconds, from the first jump the timeline finds):
  n_air            how many separate jumps the timeline finds in the clip
  best_jump_size   the largest jump_size over all jumps, not just the first
  dx, dy, scale    the 3 parts of jump_size (sideways, up the screen, body getting bigger)
  crouch           how deep the child bends before take-off
  hip_peak         how high the hips go in the air
  arm_up, arm_range  arm swing: wrists above the neck around the jump / wrist travel before it
  land_squat, land_move, land_step, hands_low   landing: squat depth, stepping, foot lifts, hands low
  size_change, size_max, travel_x, hip_rise, ankles_ok   whole-clip clues that need no jump found
"""

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, recall_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import body25 as b  # noqa: E402
from common.pick import pick_child  # noqa: E402
from data.load_b import DEFAULT_ZIP, FPS, list_files, parse_name, read_file  # noqa: E402
from features.clues import _mid_hip_xy, _task3_clues, _task3_events, _task3_pose  # noqa: E402
from model.train_rf import make_forest  # noqa: E402

NAN = float("nan")


def _f(x, fn):
    x = x[np.isfinite(x)]
    return float(fn(x)) if x.size else NAN


def med(x):
    return _f(x, np.median)


def mn(x):
    return _f(x, np.min)


def mx(x):
    return _f(x, np.max)


def jump_feats(p, t, events):
    f = dict.fromkeys(
        [
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
        ],
        NAN,
    )
    airs = [i for i in range(1, len(events) - 1) if events[i]["state"] == "air"]
    f["n_air"] = sum(e["state"] == "air" for e in events)
    best = NAN
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

    def win(s, e):
        return (t >= s) & (t <= e)

    stand_h = med(h_rel[win(t[0], max(t[0] + 0.3, a0 - 0.8))])
    pre, post = win(a0 - 0.35, a0), win(a1, a1 + 0.35)
    f["dx"] = abs(med(hip[post, 0]) - med(hip[pre, 0]))
    f["dy"] = med(ank_y[pre]) - med(ank_y[post])
    f["scale_gain"] = med(torso[post]) / med(torso[pre]) - 1.0
    if np.isfinite(stand_h) and stand_h > 0:
        f["crouch"] = (stand_h - mn(h_rel[win(a0 - 0.8, a0)])) / stand_h
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


def clip_feats(p):
    hip = _mid_hip_xy(p)
    torso = np.linalg.norm(p[:, b.NECK, :2] - hip, axis=1)
    k = max(5, len(p) // 6)
    ts = pd.Series(torso).rolling(5, center=True, min_periods=2).median().to_numpy()
    hy = pd.Series(hip[:, 1]).rolling(3, center=True, min_periods=1).median().to_numpy()
    hx = pd.Series(hip[:, 0]).rolling(3, center=True, min_periods=1).median().to_numpy()
    t0, t1, tm = med(ts[:k]), med(ts[-k:]), med(ts)
    ok = np.isfinite(t0) and t0 > 0
    return {
        "size_change": t1 / t0 if ok else NAN,
        "size_max": mx(ts) / t0 if ok else NAN,
        "travel_x": abs(med(hx[-k:]) - med(hx[:k])) / tm,
        "hip_rise": (mx(hy) - med(hy[:k])) / tm,
        "ankles_ok": float(np.isfinite(p[:, [b.L_ANKLE, b.R_ANKLE], 1]).all(axis=1).mean()),
    }


rows = []
with zipfile.ZipFile(DEFAULT_ZIP) as z:
    for name in list_files():
        m = parse_name(name)
        if m["task"] != 3:
            continue
        frames, label, _ = read_file(z, name)
        if len(frames) < 30:
            continue
        if max(len(fr) for fr in frames) <= 1:
            pts = np.stack([fr[0] if len(fr) else np.zeros((25, 3), np.float32) for fr in frames])
        else:
            pts = pick_child(frames)
        t = np.arange(len(pts)) / FPS
        try:
            p = _task3_pose(pts)
            ev = _task3_events(p, t)
            base = _task3_clues(p, t, ev)
        except ValueError:
            continue
        with np.errstate(all="ignore"):
            rows.append(
                {
                    "file": name.split("/")[-1],
                    **m,
                    "label": label,
                    **base,
                    **jump_feats(p, t, ev),
                    **clip_feats(p),
                }
            )
d = pd.DataFrame(rows)
print(f"task 3 files: {len(d)}")
NEW = [
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
    "size_change",
    "size_max",
    "travel_x",
    "hip_rise",
    "ankles_ok",
]
print("\nMedian per label (0 Bad / 1 Good / 2 Perfect):")
print(d.groupby("label")[NEW].median().round(2).T.to_string())

y, g = d["label"].to_numpy(), d["child"].to_numpy()
folds = list(GroupKFold(5).split(d, y, g))
BASE = ["feet_gap_takeoff", "feet_gap_landing", "air_s", "jump_size"]
LAND = ["land_squat", "land_move", "land_step", "hands_low"]
ARMS = ["n_air", "arm_up", "arm_range"]
SETS = {
    "current 4 clues": BASE,
    "+ landing 4": BASE + LAND,
    "+ jumps count + arms": BASE + ARMS,
    "+ landing + jumps count + arms": BASE + LAND + ARMS,
    "+ the first 13 ideas": BASE + NEW[:13],
}
ref = None
print("\nRandom forest (model.train_rf.make_forest), GroupKFold(5) by child, seeds 0-4 (25 runs):")
for label, cols in SETS.items():
    X = d[cols].fillna(-9).to_numpy()
    F, K, R = [], [], []
    for s in range(5):
        for tr, te in folds:
            pr = make_forest(s).fit(X[tr], y[tr]).predict(X[te])
            F.append(f1_score(y[te], pr, average="macro"))
            K.append(cohen_kappa_score(y[te], pr, weights="quadratic"))
            R.append(recall_score(y[te], pr, labels=[1], average="macro"))
    F, K = np.array(F), np.array(K)
    if ref is None:
        ref = (F, K)
        wins = ""
    else:
        wins = f"  better than current in F1 {int((F > ref[0]).sum())}/25, kappa {int((K > ref[1]).sum())}/25"
    print(
        f"  {label:32s} macro-F1 {F.mean():.3f}  kappa {K.mean():.3f}  Good recall {np.mean(R):.2f}{wins}"
    )
