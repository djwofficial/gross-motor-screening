"""Readiness check before the first child (and between children). Owner: Theresia (Nov 17).

Shared interface (final, change only through Daryn):

    problems = check(points, fps, frame_size=(1280, 720), start_area=None)
        points:     float32 (T, 25, 3) of the picked person over the last ~1-2 s (or (25, 3))
        fps:        measured frames per second
        frame_size: (width, height) of the picture
        start_area: optional (x0, y0, x1, y1) in pixels where both feet must be
        returns a list of plain-language problems; [] means ready

This first version does simple checks; Theresia tests it on deliberately bad setups from the
team clips (too close, too dark, feet outside the tape) and improves the messages.
"""

from __future__ import annotations

import numpy as np

from common import body25 as b

MIN_FPS = 15.0
_MUST_SEE = {
    "head": [b.NOSE],
    "shoulders": [b.R_SHOULDER, b.L_SHOULDER],
    "hips": [b.R_HIP, b.L_HIP],
    "knees": [b.R_KNEE, b.L_KNEE],
    "ankles": [b.R_ANKLE, b.L_ANKLE],
    "feet": [b.R_HEEL, b.L_HEEL, b.R_BIG_TOE, b.L_BIG_TOE],
}


def check(
    points, fps: float, frame_size=(1280, 720), start_area=None, min_score: float = 0.3
) -> list[str]:
    problems: list[str] = []
    if fps < MIN_FPS:
        problems.append(
            f"Frame rate is {fps:.0f} fps; at least {MIN_FPS:.0f} is needed (more light, or close other programs)."
        )
    if points is None or len(points) == 0:
        return problems + ["Nobody is in the picture."]
    p = np.asarray(points, dtype=np.float32)
    p = p[None] if p.ndim == 2 else p
    seen = p[..., 2] > min_score
    if not np.any(seen):
        return problems + ["Nobody is in the picture."]

    missing = [name for name, idx in _MUST_SEE.items() if np.mean(seen[:, idx]) < 0.8]
    w, h = frame_size
    margin = 0.02
    xy = p[..., :2]
    near_edge = seen & (
        (xy[..., 0] < margin * w)
        | (xy[..., 0] > (1 - margin) * w)
        | (xy[..., 1] < margin * h)
        | (xy[..., 1] > (1 - margin) * h)
    )
    if missing or np.mean(near_edge) > 0.05:
        what = ", ".join(missing) if missing else "part of the body is at the edge"
        problems.append(
            f"The whole body is not in the picture ({what}). Move the laptop back or the child to the start line."
        )

    mean_score = float(np.mean(p[..., 2][seen]))
    if mean_score < 0.5:
        problems.append(
            f"Points are unsure (average confidence {mean_score:.2f}): the room may be too dark or the picture blurry."
        )

    if start_area is not None:
        x0, y0, x1, y1 = start_area
        feet = p[:, [b.L_ANKLE, b.R_ANKLE], :]
        ok = (
            (feet[..., 0] >= x0)
            & (feet[..., 0] <= x1)
            & (feet[..., 1] >= y0)
            & (feet[..., 1] <= y1)
        )
        if np.mean(ok) < 0.8:
            problems.append("The feet are not in the start area (the taped line).")
    return problems
