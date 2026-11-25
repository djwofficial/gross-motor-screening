"""Split one long recording into task attempts ("bouts"). Owner: Daryn.

One recording in the gesture flow (app.demo) can hold several tasks in a row: stand on one leg,
hop, jump. A bout starts when the child moves away from a still, both-feet-down pose and ends when
they have been still WITH BOTH FEET DOWN for END_STILL_S (BoutSegmenter: live.segment.Segmenter by
Theresia, with the stillness counted only while both feet are down, so a still one-leg stand does
not end the moment the foot comes down, and a short touch-down does not end it either). Right after
an end the child already stands still, so the next movement starts the next bout at once. A stand
that goes straight into hopping, without putting the foot down, is cut at the first hop
(split_stand_then_hop, from features.clues.timeline(2)).

    sp = BoutSplitter()
    done = sp.update(t, points25 or None, raw26)   # finished bouts in this frame (usually [])
    sp.current()                                   # frames of the bout so far, or None
    sp.flush(until_t)                              # close the open bout when the recording stops

A frame is (t, points25 float32 (25, 3), raw26 float32 (26, 3)); points are unmirrored pixels.
Units inside: seconds and torso lengths (Segmenter, timeline). Thresholds are GUESSES (2026-11-25).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from live.segment import Segmenter

READY_STILL_S = 0.6  # guess: still this long before a bout can start (Segmenter default 1.0)
END_STILL_S = 1.0  # guess: still with both feet down this long ends a bout (default 1.5)
MAX_BOUT_S = 20.0  # a bout never runs longer (e.g. a child walking around)
STAND_BEFORE_HOP_S = 2.0  # guess: a one-foot stand at least this long before hops is its own task
MIN_HOPS = 2


@dataclass
class Bout:
    frames: list  # [(t, points25, raw26), ...]

    @property
    def t(self) -> np.ndarray:
        return np.array([f[0] for f in self.frames], dtype=np.float64)

    @property
    def points(self) -> np.ndarray:
        return np.stack([f[1] for f in self.frames]).astype(np.float32)

    @property
    def raw26(self) -> np.ndarray:
        return np.stack([f[2] for f in self.frames]).astype(np.float32)

    @property
    def duration_s(self) -> float:
        return float(self.frames[-1][0] - self.frames[0][0])


def split_stand_then_hop(points: np.ndarray, t: np.ndarray) -> list[tuple[int, int]]:
    """Frame ranges [(i0, i1), ...]: one range, or two when a long stand runs into hops."""
    whole = [(0, len(t))]
    from features.clues import timeline

    try:
        events = timeline(2, points, t - t[0])
    except ValueError:
        return whole
    hops = [e for e in events if e["state"] == "hop"]
    if len(hops) < MIN_HOPS:
        return whole
    first = hops[0]["start_s"]
    before = [e for e in events if e["end_s"] <= first + 1e-6]
    if not before or before[-1]["state"] != "pause":
        return whole
    if before[-1]["end_s"] - before[-1]["start_s"] < STAND_BEFORE_HOP_S:
        return whole
    cut = int(np.searchsorted(t - t[0], first))
    if cut < 2 or len(t) - cut < 2:
        return whole
    return [(0, cut), (cut, len(t))]


class BoutSegmenter(Segmenter):
    """Segmenter events (idle/ready/start/recording/end) for bouts inside a recording.

    Uses the Segmenter's own torso-length speed and feet checks; only the stillness rule differs:
    "calm" = still AND both feet down, and both ready and end count calm time.
    """

    def reset(self):
        super().reset()
        self.calm_since: float | None = None

    def update(self, points, t: float) -> str:
        f = self._features(points)
        if f is None:
            if (
                self.state == "recording"
                and self.last_seen is not None
                and t - self.last_seen > self.p["lost_s"]
            ):
                self.state, self.calm_since = "idle", None
                return "end"
            return self.state
        self.last_seen = t
        pos, shape = f
        self.hist.append((t, pos, shape))
        while self.hist and t - self.hist[0][0] > self.p["window_s"]:
            self.hist.popleft()
        speed = self._speed(t)
        feet = self._feet_down(shape)
        if speed <= self.p["still_speed"] and feet:
            self.calm_since = t if self.calm_since is None else self.calm_since
        else:
            self.calm_since = None
        calm_for = 0.0 if self.calm_since is None else t - self.calm_since
        if self.state == "recording":
            if calm_for >= self.p["end_still_s"]:
                # calm with both feet down = the ready pose of the next bout: no second wait
                self.state, self.rest = "ready", shape.copy()
                return "end"
            return "recording"
        if self.state == "ready":
            off_rest = float(np.max(np.linalg.norm(shape - self.rest, axis=1)))
            if speed > self.p["move_speed"] or off_rest > self.p["pose_tol"] or not feet:
                self.state = "recording"
                return "start"
        if calm_for >= self.p["ready_still_s"]:
            self.state, self.rest = "ready", shape.copy()
        return self.state


class BoutSplitter:
    def __init__(
        self,
        pre_roll_s: float = 0.5,
        ready_still_s: float = READY_STILL_S,
        end_still_s: float = END_STILL_S,
        max_bout_s: float = MAX_BOUT_S,
    ):
        self.pre_roll_s, self.max_bout_s = pre_roll_s, max_bout_s
        self._seg_kw = {"ready_still_s": ready_still_s, "end_still_s": end_still_s}
        self.reset()

    def reset(self):
        self.seg = BoutSegmenter(start_area=None, **self._seg_kw)
        self.pre: deque = deque()
        self.cur: list | None = None

    def restart(self):
        """Drop the bout in progress (a recording starts now); keep the stillness seen so far."""
        self.cur = None
        if self.seg.state == "recording":
            self.seg.reset()

    def current(self) -> list | None:
        return self.cur

    def update(self, t: float, points, raw26=None) -> list[Bout]:
        found = points is not None
        p = points.astype(np.float32) if found else np.zeros((25, 3), np.float32)
        raw = raw26 if raw26 is not None else np.zeros((26, 3), np.float32)
        frame = (float(t), p, raw)
        self.pre.append(frame)
        while self.pre and t - self.pre[0][0] > self.pre_roll_s:
            self.pre.popleft()
        ev = self.seg.update(p if found else None, float(t))
        if self.cur is None:
            if ev == "start":
                self.cur = list(self.pre)
            return []
        self.cur.append(frame)
        if ev == "end":
            return self._close()
        if t - self.cur[0][0] > self.max_bout_s:
            self.seg.reset()
            return self._close()
        return []

    def flush(self, until_t: float | None = None) -> list[Bout]:
        """Close the bout in progress, keeping only frames before until_t (e.g. the X gesture)."""
        if self.cur is None:
            return []
        if until_t is not None:
            self.cur = [f for f in self.cur if f[0] < until_t]
        self.seg.reset()
        return self._close()

    def _close(self) -> list[Bout]:
        frames, self.cur = self.cur, None
        if not frames or len(frames) < 2:
            return []
        t = np.array([f[0] for f in frames], dtype=np.float64)
        pts = np.stack([f[1] for f in frames]).astype(np.float32)
        return [Bout(frames[a:z]) for a, z in split_stand_then_hop(pts, t)]
