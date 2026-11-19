"""Follow the same child live, frame by frame (#33). Owner: Freya (handed over by Revallo).

common.pick.pick_child works on a whole clip: it locks on the child at the start and then follows
the nearest neck. Live, the app gets one frame at a time, and calling pick_child on a single
frame forgets the lock, so with a parent or another child in the picture the choice can jump to
the other person in any frame. ChildTracker keeps the lock between frames with Marty's rules:

    tracker = ChildTracker(frame_size=(1280, 720))
    i = tracker.update(people25, t)     # index of the child in people25, or None (not found)
    tracker.reset()                     # new child, readiness accepted, or a new clip

- No lock yet: one person -> that person; several -> the best start-line candidate
  (common.pick's start score: well-detected legs and feet, near the centre, not at the edge).
- Locked: the person whose neck (or mid-hip) is nearest the last one, if it moved less than
  the step limit (Marty's 220 px + 60 px per missed frame, for a 1920 px wide picture; scaled to
  the live picture width). Anyone else is ignored, even if they look like a better start-line
  candidate. If nobody is near enough, the child counts as not found in that frame.
- Live, the step grows for at most MAX_MISSES = 2 missed frames (common.pick: 5). With 5, a
  person standing about 250 px away (1280 px picture) took over after 3 frames in which the
  child was hidden (tests/test_live_track.py). 2 is a guess: tune it at the dry run.
- Lost for longer than lost_s (the child left the picture): the lock is dropped and taken again.

Units: the step limit is in pixels of the picture, like common.pick, scaled by width / 1920.
"""

from __future__ import annotations

import numpy as np

from common.pick import MAX_STEP_BASE_PX, MAX_STEP_PER_MISS_PX, _anchor, _initial_score

DATASET_WIDTH = 1920.0
MAX_MISSES = 2  # common.pick uses 5; live, 5 let a nearby person take over (see above)
LOST_S = 2.0  # not found for this long: drop the lock (a guess; tune at the dry run)


class ChildTracker:
    def __init__(self, frame_size=(1280, 720), lost_s: float = LOST_S):
        self.width, self.height = float(frame_size[0]), float(frame_size[1])
        self.scale = self.width / DATASET_WIDTH
        self.lost_s = lost_s
        self.reset()

    def reset(self) -> None:
        self.anchor: np.ndarray | None = None  # neck (or mid-hip) of the child, last time seen
        self.missed = 0
        self.last_seen: float | None = None

    @property
    def locked(self) -> bool:
        return self.anchor is not None

    def max_step(self) -> float:
        return (MAX_STEP_BASE_PX + MAX_STEP_PER_MISS_PX * min(self.missed, MAX_MISSES)) * self.scale

    def update(self, people: np.ndarray, t: float) -> int | None:
        """people: float32 (P, 25, 3) BODY_25 of everyone found in this frame (P may be 0)."""
        if (
            self.anchor is not None
            and self.last_seen is not None
            and t - self.last_seen > self.lost_s
        ):
            self.reset()  # the child left the picture: take the lock again
        if len(people) == 0:
            self.missed += 1
            return None
        anchors = [_anchor(p) for p in people]
        if self.anchor is None:
            if len(people) == 1:
                i = 0
            else:
                i = int(np.argmax([_initial_score(p, self.width, self.height) for p in people]))
            if anchors[i] is None:
                return None
            return self._take(i, anchors[i], t)
        dists = [
            float(np.linalg.norm(a - self.anchor)) if a is not None else np.inf for a in anchors
        ]
        i = int(np.argmin(dists))
        if dists[i] > self.max_step():
            self.missed += 1
            return None
        return self._take(i, anchors[i], t)

    def _take(self, i: int, anchor: np.ndarray, t: float) -> int:
        self.anchor, self.missed, self.last_seen = anchor, 0, t
        return i
