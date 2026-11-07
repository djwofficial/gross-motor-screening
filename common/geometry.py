"""Torso length and normalisation: the only "ruler" in this project. Owner: Daryn.

Every clue is measured in seconds and torso lengths (neck to mid-hip), never raw pixels,
so dataset points (1920x1080, OpenPose) and live points (1280x720, RTMPose) can be compared.
The data loader, the clues and the live side all call these functions; don't write a second copy.
"""

from __future__ import annotations

import numpy as np

from common import body25 as b

MIN_SCORE = 0.1


def valid(points: np.ndarray, min_score: float = MIN_SCORE) -> np.ndarray:
    """Boolean mask (..., 25): the point was found (score above min_score)."""
    return points[..., 2] > min_score


def torso_length(points: np.ndarray, min_score: float = MIN_SCORE) -> float:
    """Median neck-to-mid-hip distance in pixels over the frames where both points were found.

    points: (T, 25, 3) or (25, 3). Raises ValueError if no frame has both points.
    """
    p = points[None] if points.ndim == 2 else points
    ok = valid(p[:, b.NECK], min_score) & valid(p[:, b.MID_HIP], min_score)
    if not np.any(ok):
        raise ValueError("no frame has both neck and mid-hip; torso length unknown")
    d = np.linalg.norm(p[ok, b.NECK, :2] - p[ok, b.MID_HIP, :2], axis=-1)
    d = d[d > 0]
    if d.size == 0:
        raise ValueError("neck and mid-hip are at the same place in every frame")
    return float(np.median(d))


def normalize(points: np.ndarray, min_score: float = MIN_SCORE) -> np.ndarray:
    """(T, 25, 3) pixels -> float64 (T, 25, 3) in torso lengths.

    - Origin: the neck in the first frame where the neck was found (so movement is kept:
      a jump still moves, sway still shows).
    - Scale: divided by torso_length(points).
    - y points UP (y = -(pixel_y - origin_y) / torso), so "foot higher" means larger y.
    - Points that were not found (score <= min_score) get NaN for x and y; the score is kept.
    """
    p = points.astype(np.float64)
    torso = torso_length(p, min_score)
    neck_ok = np.nonzero(valid(p[:, b.NECK], min_score))[0]
    origin = p[neck_ok[0], b.NECK, :2]
    out = np.empty_like(p)
    out[..., 0] = (p[..., 0] - origin[0]) / torso
    out[..., 1] = -(p[..., 1] - origin[1]) / torso
    out[..., 2] = p[..., 2]
    miss = ~valid(p, min_score)
    out[miss, 0] = np.nan
    out[miss, 1] = np.nan
    return out
