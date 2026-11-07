"""Automatic start and end of a try from the child's points. Owner: Theresia (tune by Nov 19).

Shared interface (final, change only through Daryn):

    seg = Segmenter()
    event = seg.update(points, t)    # once per frame
        points: float32 (25, 3) of the picked child in this frame, or None if nobody was found
        t:      seconds (frame time stamp)
        event:  "idle" | "ready" | "start" | "recording" | "end"
    seg.reset()

Rules (agreed 2026-11-08):
    ready  = standing still with both feet down for about 1 s
    start  = clear movement, or the pose changes away from the ready pose (e.g. one foot lifted)
    end    = still again for about 1.5 s AND back in a both-feet-down pose (or lost for 1 s)

This first version is a PLACEHOLDER that works on fake tries. Theresia tunes the thresholds on
the 27 team clips and lists the misses. Units: torso lengths and seconds, never pixels.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from common import body25 as b

EVENTS = ("idle", "ready", "start", "recording", "end")
_LEG = [b.MID_HIP, b.R_HIP, b.R_KNEE, b.R_ANKLE, b.L_HIP, b.L_KNEE, b.L_ANKLE]


class Segmenter:
    def __init__(
        self,
        ready_still_s: float = 1.0,
        end_still_s: float = 1.5,
        still_speed: float = 0.35,  # torso lengths per second
        move_speed: float = 0.8,
        pose_tol: float = 0.25,  # torso lengths, largest point distance from the ready pose
        feet_tol: float = 0.15,  # torso lengths, ankle height difference for "both feet down"
        lost_s: float = 1.0,
        window_s: float = 0.4,  # speed = change between the older and newer half of this window
        min_score: float = 0.3,
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
            if still_for >= self.p["ready_still_s"] and self._feet_down(shape):
                self.state, self.rest = "ready", shape.copy()
            return self.state
        if self.state == "ready":
            if speed > self.p["move_speed"] or (
                off_rest is not None and off_rest > self.p["pose_tol"]
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
