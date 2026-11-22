"""EXPERIMENT (Marty, 2026-11-23): does "feet together" in task 3 mean TIME rather than SPACE?

For every task 3 group B file: find the first "air" event with features.clues.timeline(3, ...),
then per ankle the moment it leaves the ground and touches down again.

    takeoff_stagger_s = |lift-off time left ankle - lift-off time right ankle|
    landing_stagger_s = |touch-down time left ankle - touch-down time right ankle|

"Off the ground": the ankle is more than LIFT torso lengths above its own ground level
(median height over the 0.35 s before take-off / after landing). Ankle height: short gaps
(<= 0.15 s) interpolated, then a 3-frame median. Crossing times are interpolated between the
two frames on either side (sub-frame, one frame = 0.033 s at 30 fps). FILL (-1) when there is
no jump, an ankle is missing, or no crossing is found.

Joined with the task 3 rows of runs/clue_table.csv. Prints medians per label, random forest
macro-F1 (4 clues vs 4 + 2 new) and the numbers for the by-eye clips. Writes no files.

Run:  .venv\\Scripts\\python.exe scripts/experiments/exp_t3_stagger.py
"""

from __future__ import annotations

import sys
import warnings
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import body25 as b  # noqa: E402
from common.geometry import normalize  # noqa: E402
from common.pick import pick_child  # noqa: E402
from data.load_b import DEFAULT_ZIP, FPS, list_files, parse_name, read_file  # noqa: E402
from features.clues import timeline  # noqa: E402
from model.train_rf import _macro_f1, features_for, make_forest  # noqa: E402

MIN_FRAMES = 30  # same as scripts/make_clue_table.py
LIFT = 0.04  # torso lengths, = features.clues._ANKLE_CHORD_LIFT
WIN_S = 0.35  # = features.clues._JUMP_END_WIN_S
MAX_GAP_S = 0.15  # = features.clues._MAX_INTERP_GAP_S
FILL = -1.0
NEW = ["takeoff_stagger_s", "landing_stagger_s"]
ONE_FRAME_S = 1.0 / FPS


def _ankle_height(y: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Interpolate interior gaps <= MAX_GAP_S, then a centred 3-frame median."""
    out = y.copy()
    ok = np.flatnonzero(np.isfinite(out))
    for a, c in zip(ok, ok[1:], strict=False):
        if c > a + 1 and t[c] - t[a] <= MAX_GAP_S + 1e-6:
            out[a + 1 : c] = np.interp(t[a + 1 : c], [t[a], t[c]], [out[a], out[c]])
    padded = np.pad(out, 1, mode="edge")
    stack = np.stack([padded[:-2], padded[1:-1], padded[2:]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(stack, axis=0)


def _cross(t: np.ndarray, y: np.ndarray, i0: int, i1: int, thr: float) -> float:
    """Time where y crosses thr between frames i0 and i1 (linear)."""
    return float(t[i0] + (thr - y[i0]) / (y[i1] - y[i0]) * (t[i1] - t[i0]))


def _ankle_times(y: np.ndarray, t: np.ndarray, s: int, e: int) -> tuple[object, object]:
    """(lift-off time or reason, touch-down time or reason) for one ankle; air frames [s, e)."""
    pre = (t >= t[s] - WIN_S - 1e-6) & (np.arange(len(t)) < s)
    post = (np.arange(len(t)) >= e) & (t <= t[min(e, len(t) - 1)] + WIN_S + 1e-6)
    air = y[s:e]
    if not np.isfinite(air).any():
        return "missing_ankle", "missing_ankle"
    pk = s + int(np.nanargmax(air))
    lo = int(np.flatnonzero(pre)[0]) if pre.any() else s
    hi = int(np.flatnonzero(post)[-1]) if post.any() else e - 1

    lift: object = "missing_ankle"
    g = y[pre]
    if np.isfinite(g).any():
        thr = float(np.nanmedian(g)) + LIFT
        lift = "no_crossing"
        if y[pk] > thr:
            i = pk
            while i > lo and np.isfinite(y[i - 1]) and y[i - 1] > thr:
                i -= 1
            if i > lo and np.isfinite(y[i - 1]):
                lift = _cross(t, y, i - 1, i, thr)
            elif i > lo:
                lift = "missing_ankle"

    down: object = "missing_ankle"
    g = y[post]
    if np.isfinite(g).any():
        thr = float(np.nanmedian(g)) + LIFT
        down = "no_crossing"
        if y[pk] > thr:
            i = pk
            while i < hi and np.isfinite(y[i + 1]) and y[i + 1] > thr:
                i += 1
            if i < hi and np.isfinite(y[i + 1]):
                down = _cross(t, y, i, i + 1, thr)
            elif i < hi:
                down = "missing_ankle"
    return lift, down


def staggers(points: np.ndarray, t: np.ndarray) -> tuple[dict[str, float], dict[str, str]]:
    """The two new numbers (FILL where unknown) and the fill reason per number."""
    events = timeline(3, points, t)
    air = next((ev for ev in events if ev["state"] == "air"), None)
    if air is None:
        return dict.fromkeys(NEW, FILL), dict.fromkeys(NEW, "no_jump")
    s = int(np.searchsorted(t, air["start_s"] - 1e-6))
    e = int(np.searchsorted(t, air["end_s"] - 1e-6))
    e = max(e, s + 1)
    n = normalize(points)
    times = [_ankle_times(_ankle_height(n[:, j, 1], t), t, s, e) for j in (b.L_ANKLE, b.R_ANKLE)]
    out, why = {}, {}
    for k, name in enumerate(NEW):
        a, c = times[0][k], times[1][k]
        if isinstance(a, float) and isinstance(c, float):
            out[name] = abs(a - c)
        else:
            out[name] = FILL
            why[name] = a if isinstance(a, str) else c
    return out, why


def load_staggers() -> pd.DataFrame:
    rows, reasons = [], {name: Counter() for name in NEW}
    short = no_torso = 0
    with zipfile.ZipFile(DEFAULT_ZIP) as z:
        for name in list_files():
            m = parse_name(name)
            if m["task"] != 3:
                continue
            frames, _label, _ = read_file(z, name)
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
            try:
                vals, why = staggers(pts, t)
            except ValueError:  # no frame with both neck and mid-hip
                no_torso += 1
                continue
            for k, r in why.items():
                reasons[k][r] += 1
            rows.append({"file": name.split("/")[-1], **vals})
    print(
        f"task 3 files: {len(rows)} | left out: {short} under {MIN_FRAMES} frames, {no_torso} no torso"
    )
    for k in NEW:
        print(f"  fill {FILL} in {k}: {sum(reasons[k].values())} {dict(reasons[k])}")
    return pd.DataFrame(rows)


def cv_f1(table: pd.DataFrame, cols: list[str], seed: int) -> list[float]:
    x = table[cols].to_numpy(dtype=np.float64)
    y = table["label"].to_numpy(dtype=int)
    groups = table["child"].astype(str).to_numpy()
    out = []
    for train, test in GroupKFold(n_splits=5).split(x, y, groups):
        model = make_forest(seed).fit(x[train], y[train])
        out.append(_macro_f1(y[test], model.predict(x[test])))
    return out


def main() -> None:
    new = load_staggers()
    clue = pd.read_csv("runs/clue_table.csv", dtype={"child": str})
    clue = clue[clue["task"] == 3]
    table = clue.merge(new, on="file", how="inner")
    print(f"joined: {len(table)} rows | clue table rows without a match: {len(clue) - len(table)}")

    real = {k: table[k] != FILL for k in NEW}
    for k in NEW:
        below = int((table.loc[real[k], k] < ONE_FRAME_S).sum())
        print(f"  {k} below one frame (< {ONE_FRAME_S:.3f} s): {below} of {int(real[k].sum())}")

    print("\nmedian (s) per label, fills left out  [n]")
    for title, sub in (("all cameras", table), ("camera 1", table[table["camera"] == 1])):
        print(f"  {title}:")
        for label in (0, 1, 2):
            g = sub[sub["label"] == label]
            parts = []
            for k in NEW:
                v = g.loc[g[k] != FILL, k]
                parts.append(f"{k} {v.median():.3f} [{len(v)}]")
            print(f"    label {label}: " + "   ".join(parts))

    base = features_for(3)
    print(f"\nrandom forest macro-F1, GroupKFold(5) by child, make_forest, {len(table)} rows")
    for title, cols in (("4 clues", base), ("4 clues + 2 new", base + NEW)):
        per_seed = [cv_f1(table, cols, seed) for seed in range(5)]
        means = [float(np.mean(f)) for f in per_seed]
        folds = " ".join(f"{v:.3f}" for v in per_seed[0])
        print(
            f"  {title:<16} mean over seeds 0-4: {np.mean(means):.4f} ± {np.std(means):.4f}"
            f" | seed 0 folds: {folds}"
        )

    eye = pd.read_csv("docs/eye_check/by_eye_sheet.csv")
    eye = eye[eye["task"] == 3]
    print("\nby-eye task 3 clips (Louise: feet together takeoff/landing)")
    for _, r in eye.iterrows():
        row = table[table["file"] == f"{r['clip']}.json"]
        if row.empty:
            print(f"  {r['clip']}: not in the joined table")
            continue
        v = row.iloc[0]
        print(
            f"  {r['clip']} {r['label']:<8} eye {r['feet_together_takeoff']}/"
            f"{r['feet_together_landing']} | takeoff_stagger_s {v[NEW[0]]:.3f}"
            f"  landing_stagger_s {v[NEW[1]]:.3f}  (feet_gap {v['feet_gap_takeoff']:.2f}/"
            f"{v['feet_gap_landing']:.2f})"
        )


if __name__ == "__main__":
    main()
