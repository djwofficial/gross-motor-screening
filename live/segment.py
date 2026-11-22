"""Automatic start and end of a try from the child's points. Owner: Theresia (tune by Nov 19).

Shared interface (final, change only through Daryn):

    seg = Segmenter()
    event = seg.update(points, t)    # once per frame
        points: float32 (25, 3) of the picked child in this frame, or None if nobody was found
        t:      seconds (frame time stamp)
        event:  "idle" | "ready" | "start" | "recording" | "end"
    seg.reset()

Rules (agreed 2026-11-08):
    ready  = standing still with both feet down for about 1 s (and, when a start area is given,
             with the feet in it: Segmenter(start_area=live.start_area.load(frame_size=...)))
    start  = clear movement, or the pose changes away from the ready pose (e.g. one foot lifted)
    end    = still again for about 1.5 s AND back in a both-feet-down pose (or lost for 1 s)

Tuned on the 27 adult team clips (docs/task_cards/team_clips_27_try.zip, 2026-11-22): standing
on phone video measures 0.2-0.5 torso lengths per second, so still_speed went from 0.35 (guess)
to 0.5. Per-clip table and misses: python -m live.segment --team-clips (task card
docs/task_cards/theresia_2026-11-19_segment.md). Units: torso lengths and seconds, never pixels (the one
exception is the start area, which is a place in the picture, in unmirrored pixels).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from common import body25 as b
from live.start_area import feet_inside

EVENTS = ("idle", "ready", "start", "recording", "end")
_LEG = [b.MID_HIP, b.R_HIP, b.R_KNEE, b.R_ANKLE, b.L_HIP, b.L_KNEE, b.L_ANKLE]


class Segmenter:
    def __init__(
        self,
        ready_still_s: float = 1.0,
        end_still_s: float = 1.5,
        still_speed: float = 0.5,  # torso lengths per second (team clips, 2026-11-22)
        move_speed: float = 0.8,
        pose_tol: float = 0.25,  # torso lengths, largest point distance from the ready pose
        feet_tol: float = 0.15,  # torso lengths, ankle height difference for "both feet down"
        lost_s: float = 1.0,
        window_s: float = 0.4,  # speed = change between the older and newer half of this window
        min_score: float = 0.3,
        start_area=None,  # (x0, y0, x1, y1) unmirrored pixels, from live.start_area.load
    ):
        self.p = dict(locals())
        self.p.pop("self")
        self.reset()

    def reset(self):
        self.state = "idle"
        self.hist: deque = deque()
        self.still_since: float | None = None
        self.rest: np.ndarray | None = None
        self.last_seen: float | None = None
        self.torso: float | None = None  # smoothed torso length in pixels

    def _features(self, points):
        if points is None or np.any(points[[b.NECK] + _LEG, 2] < self.p["min_score"]):
            return None
        torso = float(np.linalg.norm(points[b.NECK, :2] - points[b.MID_HIP, :2]))
        if torso < 1.0:
            return None
        # Smooth the ruler: dividing a far-from-origin position by a noisy per-frame torso
        # would turn 1 px of noise into fake movement.
        self.torso = torso if self.torso is None else 0.9 * self.torso + 0.1 * torso
        pos = points[b.MID_HIP, :2] / self.torso
        shape = (points[_LEG, :2] - points[b.MID_HIP, :2]) / self.torso
        return pos, shape

    def _feet_down(self, shape) -> bool:
        ly, ry = shape[_LEG.index(b.L_ANKLE), 1], shape[_LEG.index(b.R_ANKLE), 1]
        return abs(ly - ry) < self.p["feet_tol"]

    def _in_area(self, points) -> bool:
        """No start area set, or the found foot points are in it."""
        area = self.p["start_area"]
        return area is None or feet_inside(points, area, self.p["min_score"]) is not False

    def _speed(self, t: float) -> float:
        """Torso lengths per second between the mean of the older and newer half of the window
        (averaging keeps 1-2 px of detector noise from looking like movement)."""
        if len(self.hist) < 4 or t - self.hist[0][0] < 0.5 * self.p["window_s"]:
            return 0.0
        mid = self.hist[0][0] + (t - self.hist[0][0]) / 2
        old = [h for h in self.hist if h[0] < mid]
        new = [h for h in self.hist if h[0] >= mid]
        if not old or not new:
            return 0.0
        dt = np.mean([h[0] for h in new]) - np.mean([h[0] for h in old])
        dpos = np.linalg.norm(
            np.mean([h[1] for h in new], axis=0) - np.mean([h[1] for h in old], axis=0)
        )
        dshape = np.max(
            np.linalg.norm(
                np.mean([h[2] for h in new], axis=0) - np.mean([h[2] for h in old], axis=0), axis=1
            )
        )
        return float((dpos + dshape) / dt) if dt > 0 else 0.0

    def update(self, points, t: float) -> str:
        f = self._features(points)
        if f is None:
            if (
                self.state == "recording"
                and self.last_seen is not None
                and t - self.last_seen > self.p["lost_s"]
            ):
                self.state = "idle"
                self.still_since = None
                return "end"
            return self.state
        self.last_seen = t
        pos, shape = f
        self.hist.append((t, pos, shape))
        while self.hist and t - self.hist[0][0] > self.p["window_s"]:
            self.hist.popleft()
        speed = self._speed(t)
        if speed > self.p["still_speed"]:
            self.still_since = None
        elif self.still_since is None:
            self.still_since = t
        still_for = 0.0 if self.still_since is None else t - self.still_since
        off_rest = (
            None if self.rest is None else float(np.max(np.linalg.norm(shape - self.rest, axis=1)))
        )

        if self.state == "idle":
            if (
                still_for >= self.p["ready_still_s"]
                and self._feet_down(shape)
                and self._in_area(points)
            ):
                self.state, self.rest = "ready", shape.copy()
            return self.state
        if self.state == "ready":
            # a slow lift stays under move_speed and the ready pose would follow it, so a foot
            # coming off the floor is a start of its own (team clips folder1_Move_1_clip3,
            # folder3_Move_1_clip1)
            if (
                speed > self.p["move_speed"]
                or (off_rest is not None and off_rest > self.p["pose_tol"])
                or not self._feet_down(shape)
            ):
                self.state = "recording"
                return "start"
            if still_for >= self.p["ready_still_s"] and self._feet_down(shape):
                self.rest = shape.copy()  # follow small changes while waiting
            return "ready"
        # recording
        if (
            still_for >= self.p["end_still_s"]
            and self._feet_down(shape)
            and off_rest is not None
            and off_rest < self.p["pose_tol"]
        ):
            self.state = "idle"
            self.still_since = None
            return "end"
        return "recording"


# ---------------------------------------------------------------- checking on recorded clips

TAIL_S = 2.0  # seconds added after a clip to test the end (the team clips stop right after moving)
MATCH_S = 0.5  # a start this close to the first movement in features.clues.timeline counts as found
_NOT_MOVING = ("unknown", "both_feet", "before", "after")


def run(points, t, **kw) -> list[str]:
    """Events of a fresh Segmenter(**kw) for frames points (T, 25, 3) or a list with None."""
    seg = Segmenter(**kw)
    return [seg.update(p, float(ti)) for p, ti in zip(points, t, strict=True)]


def with_tail(points, t, how: str, tail_s: float = TAIL_S, fps: float = 30.0):
    """SIMULATED end of a clip: 'hold' = the last pose stays still, 'leave' = nobody in picture."""
    n = int(round(tail_s * fps))
    tt = float(t[-1]) + np.arange(1, n + 1) / fps
    extra = [points[-1]] * n if how == "hold" else [None] * n
    return list(points) + extra, np.concatenate([np.asarray(t, float), tt])


def first_movement(task: int, points, t) -> float | None:
    """Second of the first lift, hop or jump in features.clues.timeline (Louise), or None."""
    from features.clues import timeline

    moving = [e for e in timeline(task, points, t) if e["state"] not in _NOT_MOVING]
    return float(moving[0]["start_s"]) if moving else None


def check_clip(points, t, task: int, **kw) -> dict:
    """Start and end on one recorded try, with a reason for each miss."""
    t = np.asarray(t, float)
    move = first_movement(task, points, t)
    ev = run(points, t, **kw)
    row = {"move_s": move, "starts": ev.count("start"), "start_s": None, "start": False}
    if "start" in ev:
        row["start_s"] = float(t[ev.index("start")])
    if "ready" not in ev and "start" not in ev:
        row["start_why"] = "never ready: not still with both feet down for 1 s before moving"
    elif "start" not in ev:
        row["start_why"] = "ready, but no start"
    elif ev.count("start") > 1:
        row["start_why"] = f"{ev.count('start')} starts"
    elif move is None or abs(row["start_s"] - move) > MATCH_S:
        mv = "none" if move is None else f"{move:.2f} s"
        row["start_why"] = f"start at {row['start_s']:.2f} s, first movement at {mv}"
    else:
        row["start"], row["start_why"] = True, ""
    i0 = ev.index("start") if "start" in ev else None
    for how in ("clip", "hold", "leave"):
        if how == "clip":
            e2 = ev
        else:
            e2 = run(*with_tail(points, t, how), **kw)
        found = i0 is not None and "end" in e2[i0:]
        row[f"end_{how}"] = found
    if i0 is None:
        row["end_why"] = "no start"
    elif not row["end_hold"]:
        seg = Segmenter(**kw)
        f = seg._features(points[-1])
        if f is None:
            row["end_why"] = "last frame: legs not found"
        elif not seg._feet_down(f[1]):
            row["end_why"] = "clip stops with one foot up (try not over)"
        else:
            row["end_why"] = "last pose differs from the ready pose"
    else:
        row["end_why"] = ""
    return row


def check_team_clips(zip_path, **kw) -> list[dict]:
    """check_clip on every try in the team clips zip (task from manifest.csv)."""
    import csv
    import io
    import tempfile
    import zipfile
    from pathlib import Path

    from common.format import load

    rows = []
    with zipfile.ZipFile(zip_path) as z, tempfile.TemporaryDirectory() as tmp:
        manifest = list(csv.DictReader(io.TextIOWrapper(z.open("manifest.csv"), "utf-8-sig")))
        for m in manifest:
            z.extract(m["try_file"], tmp)
            tr = load(Path(tmp) / m["try_file"])
            row = check_clip(tr.points, tr.t, int(m["task"]), **kw)
            rows.append({"clip": Path(m["try_file"]).stem, "task": int(m["task"]), **row})
    return rows


def _main(argv=None) -> int:
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description="Automatic start/end on the team clips")
    default = Path(__file__).resolve().parents[1] / "docs/task_cards/team_clips_27_try.zip"
    ap.add_argument("--team-clips", type=Path, nargs="?", const=default, default=default)
    args = ap.parse_args(argv)
    rows = check_team_clips(args.team_clips)
    print("clip                    task  move_s start_s  start  end: clip hold leave  miss")
    for r in rows:
        why = "; ".join(w for w in (r["start_why"], r["end_why"]) if w)
        mv = "-" if r["move_s"] is None else f"{r['move_s']:.2f}"
        st = "-" if r["start_s"] is None else f"{r['start_s']:.2f}"
        yn = {True: "yes", False: "no"}
        print(
            f"{r['clip']:<24}{r['task']:>4}  {mv:>6} {st:>7}  {yn[r['start']]:>5}"
            f"  {yn[r['end_clip']]:>9} {yn[r['end_hold']]:>4} {yn[r['end_leave']]:>5}  {why}"
        )
    n = len(rows)
    print(
        f"start found {sum(r['start'] for r in rows)}/{n}; end found in the clip itself "
        f"{sum(r['end_clip'] for r in rows)}/{n}, with a SIMULATED 2 s tail: still "
        f"{sum(r['end_hold'] for r in rows)}/{n}, leaving {sum(r['end_leave'] for r in rows)}/{n}"
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
