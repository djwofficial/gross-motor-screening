"""The shared format every part of the code passes around: one Try = one attempt by one child.

Owner: Daryn. Change only through Daryn (see AGENTS.md), because every part depends on it.

    Try.points  float32 array (T, 25, 3): x, y, score per frame, BODY_25 order (common/body25.py).
                x, y are pixels of the source picture (dataset: 1920x1080, webcam: usually 1280x720).
                A missing point is (0, 0, 0), the same as in the dataset.
    Try.t       float64 array (T,): seconds since the first frame, strictly increasing.
    Try.meta    dict, JSON-serialisable: source, width, height, fps, task, leg, child, ...

Clues must use seconds (from t) and torso lengths (common/geometry.py), never raw pixels, so
the same code works on dataset points and on live webcam points.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from common import body25 as b


@dataclass
class Try:
    points: np.ndarray
    t: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return int(self.points.shape[0])

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0]) if self.n_frames > 1 else 0.0


def validate(tr: Try) -> Try:
    """Raise ValueError with a plain message if tr breaks the format; return tr if it is fine."""
    p, t = tr.points, tr.t
    if not isinstance(p, np.ndarray) or p.dtype != np.float32:
        raise ValueError("points must be a float32 numpy array")
    if p.ndim != 3 or p.shape[1:] != (b.N_POINTS, 3) or p.shape[0] < 1:
        raise ValueError(f"points must have shape (T, 25, 3) with T >= 1, got {p.shape}")
    if not isinstance(t, np.ndarray) or t.dtype != np.float64 or t.shape != (p.shape[0],):
        raise ValueError(f"t must be a float64 array of shape ({p.shape[0]},)")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(t)):
        raise ValueError("points and t must not contain NaN or inf")
    if p.shape[0] > 1 and not np.all(np.diff(t) > 0):
        raise ValueError("t must be strictly increasing (seconds)")
    if np.any(p[..., 2] < 0) or np.any(p[..., 2] > 1):
        raise ValueError("scores (points[..., 2]) must be between 0 and 1")
    if not isinstance(tr.meta, dict):
        raise ValueError("meta must be a dict")
    try:
        json.dumps(tr.meta)
    except TypeError as e:
        raise ValueError(f"meta must be JSON-serialisable: {e}") from e
    return tr


def save(tr: Try, path: str | Path) -> Path:
    """Save as a compressed .npz (points, t, meta as JSON text)."""
    validate(tr)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, points=tr.points, t=tr.t, meta=np.array(json.dumps(tr.meta)))
    return path


def load(path: str | Path) -> Try:
    with np.load(Path(path), allow_pickle=False) as z:
        tr = Try(
            points=z["points"].astype(np.float32),
            t=z["t"].astype(np.float64),
            meta=json.loads(str(z["meta"])),
        )
    return validate(tr)


# Standing pose in pixels, relative to the mid-hip, for a child facing the camera.
# The child's right side is on the left of the picture. Torso (neck to mid-hip) = 110 px.
_STAND = {
    b.NOSE: (0, -150),
    b.NECK: (0, -110),
    b.R_SHOULDER: (-30, -105),
    b.R_ELBOW: (-38, -60),
    b.R_WRIST: (-40, -20),
    b.L_SHOULDER: (30, -105),
    b.L_ELBOW: (38, -60),
    b.L_WRIST: (40, -20),
    b.MID_HIP: (0, 0),
    b.R_HIP: (-18, 0),
    b.R_KNEE: (-20, 55),
    b.R_ANKLE: (-20, 110),
    b.L_HIP: (18, 0),
    b.L_KNEE: (20, 55),
    b.L_ANKLE: (20, 110),
    b.R_EYE: (-6, -158),
    b.L_EYE: (6, -158),
    b.R_EAR: (-12, -152),
    b.L_EAR: (12, -152),
    b.L_BIG_TOE: (28, 118),
    b.L_SMALL_TOE: (34, 116),
    b.L_HEEL: (18, 116),
    b.R_BIG_TOE: (-28, 118),
    b.R_SMALL_TOE: (-34, 116),
    b.R_HEEL: (-18, 116),
}


def make_fake_try(
    task: int = 1,
    leg: str = "L",
    seconds: float = 4.0,
    fps: float = 30.0,
    idle_before: float = 0.0,
    idle_after: float = 0.0,
    seed: int = 0,
    width: int = 1280,
    height: int = 720,
) -> Try:
    """A believable fake try for tests and for working before real data is ready.

    task 1: stands on `leg` (the support leg) and lifts the other foot.
    task 2: hops 3 times on `leg`, the other foot stays up.
    task 3: jumps once with both feet, moving sideways in the picture (a 45-degree camera).
    idle_before / idle_after: seconds of standing still around the movement (for segment.py).
    """
    if task not in (1, 2, 3):
        raise ValueError("task must be 1, 2 or 3")
    if leg not in ("L", "R"):
        raise ValueError("leg must be 'L' or 'R'")
    rng = np.random.default_rng(seed)
    total = idle_before + seconds + idle_after
    n = max(2, int(round(total * fps)))
    t = np.arange(n) / fps + rng.uniform(-0.002, 0.002, n)
    t[0] = 0.0
    t = np.maximum.accumulate(t + np.arange(n) * 1e-6)

    base = np.array([_STAND[i] for i in range(b.N_POINTS)], dtype=np.float64)
    hip0 = np.array([width / 2, height * 0.62])
    lifted = b.RIGHT_FOOT + [b.R_KNEE] if leg == "L" else b.LEFT_FOOT + [b.L_KNEE]
    pts = np.zeros((n, b.N_POINTS, 3), dtype=np.float64)

    for i, ti in enumerate(t):
        xy = base.copy()
        s = ti - idle_before  # time inside the movement
        moving = 0.0 <= s <= seconds
        if moving and task == 1:
            if 0.4 <= s <= seconds - 0.4:
                xy[lifted, 1] -= 40.0
                xy[:, 0] += 3.0 * np.sin(2 * np.pi * 0.7 * s)  # small sway
        elif moving and task == 2:
            xy[lifted, 1] -= 40.0
            hop = 3.0 * s / seconds  # 3 hops
            xy[:, 1] -= 25.0 * abs(np.sin(np.pi * hop))
            xy[:, 0] += 15.0 * hop
        elif task == 3 and s >= 0.0:
            jump_start, air = 1.0, 0.45
            if moving and jump_start <= s <= jump_start + air:
                u = (s - jump_start) / air
                xy[:, 1] -= 160.0 * u * (1 - u)
                xy[:, 0] += 120.0 * u
            elif s > jump_start + air:
                xy[:, 0] += 120.0
        pts[i, :, :2] = xy + hip0 + rng.normal(0.0, 1.0, (b.N_POINTS, 2))
        pts[i, :, 2] = rng.uniform(0.8, 0.95, b.N_POINTS)

    meta = {
        "source": "fake",
        "task": task,
        "leg": leg,
        "fps": fps,
        "width": width,
        "height": height,
        "seed": seed,
        "idle_before": idle_before,
        "idle_after": idle_after,
    }
    return validate(Try(points=pts.astype(np.float32), t=t.astype(np.float64), meta=meta))
