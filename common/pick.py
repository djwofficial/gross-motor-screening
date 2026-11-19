"""Pick the child when 2+ people are in the picture. Owner: Marty (Nov 15). Revallo reuses it live.

Shared interface (final, change only through Daryn):

    pick_child(frames) -> np.ndarray float32 (T, 25, 3)
        frames: list of length T; frames[i] is a float32 array (P_i, 25, 3) with every person
                found in frame i (BODY_25, pixels). P_i may be 0.
        returns the child's points per frame; (0, 0, 0) for frames where the child wasn't found.

Plan (Marty): lock on the person at the start (dataset: the child near the centre / start line),
then follow the nearest neck frame by frame. 28% of group B files have 2+ people in some frame.
"""

from __future__ import annotations

import numpy as np

from common import body25 as b

MIN_SCORE = 0.1
MAX_STEP_BASE_PX = 220.0
MAX_STEP_PER_MISS_PX = 60.0


def _anchor(person: np.ndarray) -> np.ndarray | None:
    """Return a 2D (x, y) reference point for tracking: NECK, fallback MID_HIP, then valid mean."""
    if person[b.NECK, 2] > MIN_SCORE:
        return person[b.NECK, :2].astype(np.float64)
    if person[b.MID_HIP, 2] > MIN_SCORE:
        return person[b.MID_HIP, :2].astype(np.float64)
    ok = person[:, 2] > MIN_SCORE
    if np.any(ok):
        return person[ok, :2].astype(np.float64).mean(axis=0)
    return None


def _initial_score(person: np.ndarray, width: float = 1920.0, height: float = 1080.0) -> float:
    """Score a candidate at the start frame: prefer well-detected foreground person near center."""
    anc = _anchor(person)
    if anc is None:
        return -1e9
    mean_conf = float(person[:, 2].mean())
    lower_idx = b.LEFT_FOOT + b.RIGHT_FOOT + [b.L_KNEE, b.R_KNEE, b.MID_HIP, b.NECK]
    body_conf = float(person[lower_idx, 2].mean())
    x, y = float(anc[0]), float(anc[1])
    edge_penalty = 0.0
    if x < 0.06 * width or x > 0.94 * width:
        edge_penalty += 0.5
    if y < 0.08 * height:
        edge_penalty += 0.4
    center_dist = abs(x - 0.5 * width) / (0.5 * width)
    return body_conf + 0.5 * mean_conf - 0.25 * center_dist - edge_penalty


def pick_child(frames: list[np.ndarray]) -> np.ndarray:
    out = np.zeros((len(frames), b.N_POINTS, 3), dtype=np.float32)
    if not frames:
        return out

    has_multi = any(len(p) >= 2 for p in frames)
    last_anc: np.ndarray | None = None
    missed = 0

    for i, people in enumerate(frames):
        n_people = len(people)
        if n_people == 0:
            missed += 1
            continue

        if not has_multi:
            out[i] = people[0]
            continue

        if last_anc is None:
            if n_people == 1:
                chosen = people[0]
            else:
                max_x = max(float(people[..., 0].max()), 1280.0)
                w = 1920.0 if max_x > 1280.0 else 1280.0
                h = 1080.0 if w == 1920.0 else 720.0
                scores = [_initial_score(p, w, h) for p in people]
                chosen = people[int(np.argmax(scores))]
            anc = _anchor(chosen)
            if anc is not None:
                last_anc = anc
            out[i] = chosen
            missed = 0
        else:
            dists = []
            for p in people:
                anc = _anchor(p)
                dists.append(float(np.linalg.norm(anc - last_anc)) if anc is not None else 1e9)
            best = int(np.argmin(dists))
            max_step = MAX_STEP_BASE_PX + MAX_STEP_PER_MISS_PX * min(missed, 5)
            if dists[best] > max_step:
                missed += 1
                continue
            chosen = people[best]
            anc = _anchor(chosen)
            if anc is not None:
                last_anc = anc
            out[i] = chosen
            missed = 0

    return out
