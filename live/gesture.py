"""Hand gestures that start and stop a recording in the score window (app.demo). Owner: Daryn.

    det = GestureDetector(hand_checker=HandChecker(device="cpu"))   # or GestureDetector()
    event = det.update(points25 or None, t, want="palm" | "cross" | None, frame=bgr)
        event: None, "palm" or "cross"

palm   one hand up (wrist above the elbow, at chest height or higher) with the palm open to the
       camera, held PALM_HOLD_S. Either hand. Also works on the readiness screen. Open fingers come from the rtmlib Hand model (RTMPose-m, 21 hand points), run in
       its own thread on a crop around the raised wrist (HandChecker). Without the model (offline,
       fake stream, load error) a raised hand is enough.
cross  both forearms crossed (an X), held CROSS_HOLD_S. det.cross_since is when the X began, so the
       caller can cut the gesture off the recording.
Neither fires while one foot is clearly up (standing on one leg, hopping); feet out of the picture
do not block a gesture. Without the hips in the picture the shoulder width is the ruler.

Points are BODY_25 (25, 3) of the UNMIRRORED picture; distances are in torso lengths (neck to
mid-hip), times in seconds. Every threshold below is a GUESS (Daryn, 2026-11-25), to be tuned on the
live demo.

    python -m live.gesture --download        # fetch the hand model into the rtmlib cache once
"""

from __future__ import annotations

import threading

import numpy as np

from common import body25 as b

MIN_SCORE = 0.3  # guess: body points below this are ignored
FEET_TOL = 0.15  # torso lengths, ankle height difference for "both feet down" (as live.segment)
# guess, lowered 2026-11-25 after the first live try (0.5 = above the head never fired): wrist height
# above its shoulder in torso lengths, negative = below; a palm shown at face or chest height counts
PALM_MIN_HEIGHT = -0.15
SHOULDER_TO_TORSO = 1.8  # guess: torso ~ 1.8 x shoulder width (ruler when the hips are not seen)
PALM_HOLD_S = 0.7  # guess
CROSS_HOLD_S = 0.8  # guess
GAP_S = 0.2  # a gesture may drop out this long (missed frames) and still count as held
MIN_SHOULDER_SPAN = 0.3  # torso lengths; below this the child is side-on and left/right is unsure
HAND_FRESH_S = 0.6  # a hand-model answer older than this is not used
HAND_EVERY_S = 0.1  # send at most this often to the hand model
HAND_MIN_SCORE = 0.3  # guess: a finger counts only if its PIP and tip scores reach this
FINGER_RATIO = 1.15  # guess: straight finger = tip->wrist longer than this x PIP->wrist

_ARM = {
    "L": (b.L_SHOULDER, b.L_ELBOW, b.L_WRIST),
    "R": (b.R_SHOULDER, b.R_ELBOW, b.R_WRIST),
}
# rtmlib hand order (COCO-WholeBody hand): 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle,
# 13-16 ring, 17-20 little finger; per finger MCP, PIP, DIP, tip.
_FINGERS = ((6, 8), (10, 12), (14, 16), (18, 20))  # (PIP, tip) of the four fingers


def _ok(p: np.ndarray, *idx: int) -> bool:
    return bool(np.all(p[list(idx), 2] >= MIN_SCORE))


def _torso(p: np.ndarray) -> float | None:
    """Torso length in pixels; from the shoulder width when the hips are not in the picture."""
    if _ok(p, b.NECK, b.MID_HIP):
        d = float(np.linalg.norm(p[b.NECK, :2] - p[b.MID_HIP, :2]))
        if d >= 1.0:
            return d
    if _ok(p, b.L_SHOULDER, b.R_SHOULDER):
        span = float(np.linalg.norm(p[b.L_SHOULDER, :2] - p[b.R_SHOULDER, :2]))
        if span >= 1.0:
            return SHOULDER_TO_TORSO * span
    return None


def feet_down(p: np.ndarray, torso: float) -> bool:
    """False only when both ankles are found and one is clearly higher (one leg, a hop)."""
    if not _ok(p, b.L_ANKLE, b.R_ANKLE):
        return True  # feet not in the picture: they do not block a gesture
    return abs(float(p[b.L_ANKLE, 1] - p[b.R_ANKLE, 1])) / torso < FEET_TOL


def raised_sides(p: np.ndarray | None) -> list[str]:
    """Sides ("L"/"R") whose hand is up (above its elbow, chest height or higher), highest first."""
    if p is None:
        return []
    torso = _torso(p)
    if torso is None or not feet_down(p, torso):
        return []
    out = []
    for side, (sh, el, wr) in _ARM.items():
        if not _ok(p, sh, el, wr):
            continue
        above = (p[sh, 1] - p[wr, 1]) / torso  # pixel y points down
        if above >= PALM_MIN_HEIGHT and p[wr, 1] < p[el, 1]:
            out.append(side)
    return sorted(out, key=lambda s: float(p[_ARM[s][2], 1]))


def _orient(a, c, d) -> float:
    return float((c[0] - a[0]) * (d[1] - a[1]) - (c[1] - a[1]) * (d[0] - a[0]))


def _intersection(a, b_, c, d):
    """Crossing point of segments a-b_ and c-d, or None."""
    o1, o2 = _orient(a, b_, c), _orient(a, b_, d)
    o3, o4 = _orient(c, d, a), _orient(c, d, b_)
    if o1 * o2 >= 0 or o3 * o4 >= 0:
        return None
    r, s = b_ - a, d - c
    den = r[0] * s[1] - r[1] * s[0]
    if den == 0:
        return None
    u = ((c[0] - a[0]) * s[1] - (c[1] - a[1]) * s[0]) / den
    return a + u * r


def cross_point(p: np.ndarray | None):
    """Pixel point (x, y) where the forearms cross, or None when the arms do not form an X."""
    if p is None:
        return None
    torso = _torso(p)
    if torso is None or not feet_down(p, torso):
        return None
    if not _ok(p, b.L_SHOULDER, b.R_SHOULDER, b.L_ELBOW, b.R_ELBOW, b.L_WRIST, b.R_WRIST):
        return None
    le, lw = p[b.L_ELBOW, :2], p[b.L_WRIST, :2]
    re, rw = p[b.R_ELBOW, :2], p[b.R_WRIST, :2]
    span = float(p[b.L_SHOULDER, 0] - p[b.R_SHOULDER, 0])
    pt = _intersection(le, lw, re, rw)
    if pt is None:
        # The forearms may cross between two points the model put a little short; the wrists
        # then sit on the "wrong" side of each other (left wrist on the child's right).
        if abs(span) / torso < MIN_SHOULDER_SPAN or np.sign(lw[0] - rw[0]) != -np.sign(span):
            return None
        pt = (lw + rw) / 2
    if _ok(p, b.MID_HIP) and pt[1] > p[b.MID_HIP, 1]:  # below the hips: arms hanging, not an X
        return None
    return float(pt[0]), float(pt[1])


def hand_box(p: np.ndarray, side: str, frame_size=None):
    """(x0, y0, x1, y1) pixels around the hand beyond the wrist of side, or None."""
    _sh, el, wr = _ARM[side]
    torso = _torso(p)
    if torso is None or not _ok(p, el, wr):
        return None
    fore = p[wr, :2] - p[el, :2]
    c = p[wr, :2] + 0.4 * fore  # guess: a hand is about 0.75 forearm long; its middle is here
    half = max(0.5 * float(np.linalg.norm(fore)), 0.3 * torso, 16.0)
    x0, y0, x1, y1 = c[0] - half, c[1] - half, c[0] + half, c[1] + half
    if frame_size is not None:
        w, h = frame_size
        x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return float(x0), float(y0), float(x1), float(y1)


def open_fingers(kp: np.ndarray, sc: np.ndarray) -> int:
    """How many of the 4 fingers (index..little) are straight and seen (rtmlib hand points)."""
    kp, sc = np.asarray(kp, float), np.asarray(sc, float)
    if kp.shape[0] < 21:
        return 0
    w = kp[0]
    return int(
        sum(
            sc[pip] >= HAND_MIN_SCORE
            and sc[tip] >= HAND_MIN_SCORE
            and np.linalg.norm(kp[tip] - w) > FINGER_RATIO * np.linalg.norm(kp[pip] - w)
            for pip, tip in _FINGERS
        )
    )


def hand_is_open(kp: np.ndarray, sc: np.ndarray) -> bool:
    """21 hand points (rtmlib order): at least 3 of the 4 fingers straight and seen."""
    return open_fingers(kp, sc) >= 3


class HandChecker:
    """The rtmlib hand model (RTMPose-m hand, 256x256) in its own thread.

    submit() hands over a crop around one wrist; only the newest request is kept, so a slow model
    never makes the window wait. latest() is the last answer: {"t", "side", "open", "score"}.
    available: None while the model loads, True when ready, False if it could not load (.error).
    model: anything called as model(image, bboxes=[box]) -> (keypoints (1, 21, 2), scores (1, 21));
    tests pass a stub, the app leaves it None (rtmlib loads in the thread).
    """

    def __init__(self, device: str = "cpu", backend: str = "onnxruntime", model=None):
        self.device, self.backend = device, backend
        self.model = model
        self.available: bool | None = True if model is not None else None
        self.error: str | None = None
        self._cond = threading.Condition()
        self._req = None
        self._res: dict | None = None
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="hand-check", daemon=True)
        self._thread.start()

    def load(self):
        """Build the rtmlib model (downloads it into the rtmlib cache the first time)."""
        if self.model is None:
            from rtmlib import Hand, RTMPose

            cfg = Hand.MODE["lightweight"]
            self.model = RTMPose(
                cfg["pose"],
                model_input_size=cfg["pose_input_size"],
                backend=self.backend,
                device=self.device,
            )
        return self.model

    def submit(self, t: float, frame, box, side: str) -> None:
        if self.available is False or frame is None or box is None:
            return
        # Copy a padded crop, so the camera thread can reuse its frame buffer.
        h, w = frame.shape[:2]
        x0, y0, x1, y1 = box
        mx, my = 0.3 * (x1 - x0), 0.3 * (y1 - y0)
        cx0, cy0 = max(0, int(x0 - mx)), max(0, int(y0 - my))
        cx1, cy1 = min(w, int(x1 + mx) + 1), min(h, int(y1 + my) + 1)
        if cx1 - cx0 < 8 or cy1 - cy0 < 8:
            return
        crop = frame[cy0:cy1, cx0:cx1].copy()
        local = [x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0]
        with self._cond:
            self._req = (t, crop, local, side)
            self._cond.notify()

    def latest(self) -> dict | None:
        with self._cond:
            return self._res

    def _run(self):
        if self.available is None:
            try:
                self.load()
                self.available = True
            except Exception as e:  # noqa: BLE001 (offline, missing onnxruntime, bad download...)
                self.available, self.error = False, f"{type(e).__name__}: {e}"
                return
        while True:
            with self._cond:
                while self._req is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                (t, crop, box, side), self._req = self._req, None
            try:
                kp, sc = self.model(crop, bboxes=[box])
                kp, sc = np.asarray(kp)[0], np.asarray(sc)[0]
            except Exception:  # noqa: BLE001 (one bad crop: no answer for this request)
                continue
            n = open_fingers(kp, sc)
            res = {"t": t, "side": side, "open": n >= 3, "fingers": n, "score": float(np.mean(sc))}
            with self._cond:
                self._res = res

    def close(self, timeout: float = 1.0) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify()
        self._thread.join(timeout)


class _Hold:
    """How long a yes/no has been yes, forgiving gaps up to gap_s."""

    def __init__(self):
        self.since: float | None = None
        self.last: float | None = None

    def step(self, ok: bool, t: float, gap_s: float) -> float:
        if ok:
            if self.since is None or (self.last is not None and t - self.last > gap_s):
                self.since = t
            self.last = t
            return t - self.since
        if self.last is not None and t - self.last > gap_s:
            self.since = self.last = None
        return 0.0

    def clear(self):
        self.since = self.last = None


class GestureDetector:
    """Palm (start) and X (stop) from the tracked child's points, one frame at a time.

    For the screen: kind ("palm"/"cross"/None), progress 0..1 of the hold, anchor (x, y) unmirrored
    pixels of the wrist or the crossing, waiting_for_open (hand up, fingers not open yet) and
    status (None, "loading" or "no hand model").
    """

    def __init__(
        self,
        hand_checker: HandChecker | None = None,
        palm_hold_s: float = PALM_HOLD_S,
        cross_hold_s: float = CROSS_HOLD_S,
        gap_s: float = GAP_S,
    ):
        self.hand = hand_checker
        self.palm_hold_s, self.cross_hold_s, self.gap_s = palm_hold_s, cross_hold_s, gap_s
        self._last_submit = -1e9
        self.reset()

    def reset(self):
        self._palm, self._cross = _Hold(), _Hold()
        self._armed = True  # the hand must come down before the next palm
        self._last_raised: float | None = None
        self._cross_armed = True  # the arms must open before the next X
        self._last_cross: float | None = None
        self.kind: str | None = None
        self.progress = 0.0
        self.anchor: tuple[float, float] | None = None
        self.waiting_for_open = False
        self.cross_since: float | None = None
        self._side: str | None = None
        self._t = 0.0

    @property
    def hint(self) -> str:
        """Short text under the ring: what the gesture still needs (also for tuning live)."""
        if self.kind == "cross":
            return "Hold to stop"
        if self.kind != "palm":
            return ""
        if self.status == "loading":
            return "Loading hand model…"
        if not self.waiting_for_open:
            return "Hold to start"
        r = self.hand.latest() if self.hand is not None else None
        if not r or r["side"] != self._side or self._t - r["t"] > HAND_FRESH_S:
            return "Open your hand · looking…"
        return f"Open your hand · {r.get('fingers', 0)}/4 fingers"

    @property
    def status(self) -> str | None:
        if self.hand is None or self.hand.available is True:
            return None
        return "loading" if self.hand.available is None else "no hand model"

    def _open_ok(self, side: str, t: float) -> bool | None:
        """True/False from the hand model; None while it loads (wait); True without a model."""
        if self.hand is None or self.hand.available is False:
            return True
        if self.hand.available is None:
            return None
        r = self.hand.latest()
        return bool(r and r["side"] == side and t - r["t"] <= HAND_FRESH_S and r["open"])

    def update(self, points, t: float, want: str | None, frame=None) -> str | None:
        self.kind, self.progress, self.anchor, self.waiting_for_open = None, 0.0, None, False
        if want == "palm":
            return self._palm_step(points, t, frame)
        self._palm.clear()
        if want == "cross":
            return self._cross_step(points, t)
        self._cross.clear()
        return None

    def _palm_step(self, points, t, frame) -> str | None:
        self._cross.clear()
        sides = raised_sides(points)
        if not sides:
            if self._last_raised is None or t - self._last_raised > self.gap_s:
                self._armed = True
            self._palm.step(False, t, self.gap_s)
            return None
        self._last_raised = t
        side = sides[0]
        self._side, self._t = side, t
        wr = _ARM[side][2]
        self.kind, self.anchor = "palm", (float(points[wr, 0]), float(points[wr, 1]))
        if self.hand is not None and self.hand.available and t - self._last_submit >= HAND_EVERY_S:
            size = None if frame is None else (frame.shape[1], frame.shape[0])
            self.hand.submit(t, frame, hand_box(points, side, size), side)
            self._last_submit = t
        is_open = self._open_ok(side, t)
        self.waiting_for_open = not is_open
        held = self._palm.step(bool(is_open) and self._armed, t, self.gap_s)
        self.progress = min(held / self.palm_hold_s, 1.0) if self._armed else 0.0
        if self._armed and held >= self.palm_hold_s:
            self._armed = False
            self._palm.clear()
            return "palm"
        return None

    def _cross_step(self, points, t) -> str | None:
        pt = cross_point(points)
        if pt is None:
            if self._last_cross is None or t - self._last_cross > self.gap_s:
                self._cross_armed = True  # the arms opened: the next X counts again
            self._cross.step(False, t, self.gap_s)
            return None
        self._last_cross = t
        self.kind, self.anchor = "cross", pt
        if not self._cross_armed:
            return None
        held = self._cross.step(True, t, self.gap_s)
        self.cross_since = self._cross.since
        self.progress = min(held / self.cross_hold_s, 1.0)
        if held >= self.cross_hold_s:
            self._cross.clear()
            self._cross_armed = False
            return "cross"
        return None


def _main(argv=None) -> int:
    import argparse
    import sys

    default_device = "cpu" if sys.platform == "darwin" else "cuda"
    ap = argparse.ArgumentParser(description="Hand model for the open-palm start gesture")
    ap.add_argument("--download", action="store_true", help="fetch the model into the cache")
    ap.add_argument("--device", default=default_device)
    args = ap.parse_args(argv)
    if not args.download:
        ap.print_help()
        return 0
    import time

    checker = HandChecker(device=args.device)
    deadline = time.monotonic() + 300
    while checker.available is None and time.monotonic() < deadline:
        time.sleep(0.2)
    if not checker.available:
        print(f"Hand model NOT ready: {checker.error}")
        return 1
    print("Hand model ready (rtmlib cache, ~/.cache/rtmlib/hub/checkpoints).")
    checker.close()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
