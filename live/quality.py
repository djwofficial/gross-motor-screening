"""Readiness check before the first child (and between children). Owner: Theresia (Nov 17).

Shared interface (final, change only through Daryn):

    problems = check(points, fps, frame_size=(1280, 720), start_area=None)
        points:     float32 (T, 25, 3) of the picked person over the last ~1-2 s (or (25, 3))
        fps:        measured frames per second (0 or less = not measured yet, no message)
        frame_size: (width, height) of the picture
        start_area: optional (x0, y0, x1, y1) in pixels where both feet must be
        task:       optional (keyword, added 2026-11-24, old calls unchanged): 3 adds the
                    long-jump landing-room check
        returns a list of plain-language problems; [] means ready

One message per problem, each saying what the teacher can do:
- nobody found                -> "Nobody is in the picture."
- frame rate too low          -> "... fps ..."
- body cut off or parts lost  -> "The whole body is not in the picture (...)"  (too close)
- body small in the picture   -> "The child looks small ..."                   (too far)
- points unsure               -> "Points are unsure ..."                       (too dark / blurry)
- feet outside start_area     -> "The feet are not in the start area ..."
- task=3 (optional, added 2026-11-24): too little floor below the feet for the long jump
                              -> "Not enough floor in the picture below the feet ..."

Task 3 landing room (Marty, docs/task_cards/marty_2026-11-23_t3_experiments.md: in most missed
Good/Perfect jumps the feet leave the bottom of the picture). Measured on group B task 3 (1,091
files, scratch script 2026-11-24): the lowest foot point moves down by a median 0.5-0.9 and a
95th percentile 1.1-1.8 start torso lengths (cameras 3 / 2 / 1). With at least 1.5 torso lengths
of picture below the feet at the start line, the files whose feet reach the bottom edge drop from
497 to 33. So ROOM_T3 = 1.5 torso lengths.

Positions here are in picture pixels on purpose: the edges and the start area are picture
coordinates. Body size is compared as a fraction of the picture height, so it works for the
1280x720 webcam, the 1920x1080 dataset and the 1080x1920 team clips alike.

Checked on the 27 team clips (docs/task_cards/team_clips_27_try.zip) and on 3 bad setups made
from them (too close, too dark, feet outside the tape): tests/test_live_quality.py.
"""

from __future__ import annotations

import numpy as np

from common import body25 as b
from live.start_area import feet_inside

MIN_FPS = 15.0  # guess: the clue timings need about 15 or more frames per second
MIN_SCORE = 0.3  # a point with a lower score counts as not found
MIN_SEEN = 0.8  # a body part must be found in at least 80% of the frames
MIN_MEAN_SCORE = 0.5  # guess: team clips in normal light average 0.87-0.92
MIN_BODY_FRAC = 0.25  # guess: body (head to feet) shorter than this part of the picture = too far
MAX_BODY_FRAC = 0.65
EDGE = 0.02  # a point closer than 2% of the picture size to an edge counts as cut off
MAX_EDGE_FRAC = 0.05  # more than 5% of the found points at an edge = body cut off
ROOM_T3 = 1.5  # torso lengths of picture below the feet for the long jump (group B, see above)
_FEET = [b.R_ANKLE, b.L_ANKLE, b.R_HEEL, b.L_HEEL, b.R_BIG_TOE, b.L_BIG_TOE]
_MUST_SEE = {
    "head": [b.NOSE],
    "shoulders": [b.R_SHOULDER, b.L_SHOULDER],
    "hips": [b.R_HIP, b.L_HIP],
    "knees": [b.R_KNEE, b.L_KNEE],
    "ankles": [b.R_ANKLE, b.L_ANKLE],
    "feet": [b.R_HEEL, b.L_HEEL, b.R_BIG_TOE, b.L_BIG_TOE],
}
# body parts used for the confidence (light) check: the big joints, not the small face/foot points
_CORE = [b.NOSE, b.NECK, b.R_SHOULDER, b.L_SHOULDER, b.R_HIP, b.L_HIP]
_CORE += [b.R_KNEE, b.L_KNEE, b.R_ANKLE, b.L_ANKLE]

NOBODY = "Nobody is in the picture."


def _cut_sides(xy: np.ndarray, seen: np.ndarray, w: float, h: float) -> list[str]:
    """Which picture edges the body touches: 'top', 'bottom', 'left', 'right'."""
    n = max(int(seen.sum()), 1)
    sides = {
        "top": xy[..., 1] < EDGE * h,
        "bottom": xy[..., 1] > (1 - EDGE) * h,
        "left": xy[..., 0] < EDGE * w,
        "right": xy[..., 0] > (1 - EDGE) * w,
    }
    return [s for s, m in sides.items() if (m & seen).sum() / n > MAX_EDGE_FRAC]


def landing_room(points, frame_size=(1280, 720), min_score: float = MIN_SCORE) -> float | None:
    """Picture below the lowest foot point, in torso lengths (median over the frames), or None.

    Pixels only to measure the place in the picture; the result is in torso lengths, so it works
    on the webcam, the dataset and the team clips alike.
    """
    p = np.asarray(points, dtype=np.float32)
    p = p[None] if p.ndim == 2 else p
    ok = p[..., 2] > min_score
    fy = np.where(ok[:, _FEET], p[:, _FEET, 1], -np.inf)
    low = fy.max(axis=1)
    torso = np.linalg.norm(p[:, b.NECK, :2] - p[:, b.MID_HIP, :2], axis=1)
    use = np.isfinite(low) & ok[:, b.NECK] & ok[:, b.MID_HIP] & (torso > 1.0)
    if not np.any(use):
        return None
    h = float(frame_size[1])
    return float(np.median((h - low[use]) / torso[use]))


def feet_visible(points, frame_size=(1280, 720), min_score: float = MIN_SCORE) -> bool:
    """True when both ankles are detected and safely inside the picture edges."""
    if points is None or len(points) == 0:
        return False
    p = np.asarray(points, dtype=np.float32)
    frame = p[-1] if p.ndim == 3 else p
    if frame.shape[0] <= max(b.L_ANKLE, b.R_ANKLE):
        return False
    w, h = float(frame_size[0]), float(frame_size[1])
    for idx in (b.L_ANKLE, b.R_ANKLE):
        x, y, s = float(frame[idx, 0]), float(frame[idx, 1]), float(frame[idx, 2])
        if s <= min_score:
            return False
        if x < EDGE * w or x > (1 - EDGE) * w or y < EDGE * h or y > (1 - EDGE) * h:
            return False
    return True


def check(
    points,
    fps: float,
    frame_size=(1280, 720),
    start_area=None,
    min_score: float = MIN_SCORE,
    task: int | None = None,
    max_body_frac: float | None = None,
) -> list[str]:
    problems: list[str] = []
    if 0 < fps < MIN_FPS:
        problems.append(
            f"Frame rate is {fps:.0f} fps; at least {MIN_FPS:.0f} is needed "
            "(turn on more light, plug in the laptop, or close other programs)."
        )
    if points is None or len(points) == 0:
        return problems + [NOBODY]
    p = np.asarray(points, dtype=np.float32)
    p = p[None] if p.ndim == 2 else p
    seen = p[..., 2] > min_score
    if not np.any(seen):
        return problems + [NOBODY]
    w, h = float(frame_size[0]), float(frame_size[1])
    xy = p[..., :2]

    # light: average confidence of the big joints that were found at all (score > 0)
    core = p[:, _CORE, 2]
    mean_score = float(np.mean(core[core > 0])) if np.any(core > 0) else 0.0
    dark = mean_score < MIN_MEAN_SCORE

    # whole body: cut off at an edge, or body parts not found
    sides = _cut_sides(xy, seen, w, h)
    missing = [name for name, idx in _MUST_SEE.items() if np.mean(seen[:, idx]) < MIN_SEEN]
    if dark and not sides:
        missing = []  # parts lost because the picture is too dark: the light message says it
    if sides or missing:
        details = []
        if "top" in sides:
            details.append("the head is cut off at the top")
        if "bottom" in sides:
            details.append("the feet are cut off at the bottom")
        if "left" in sides or "right" in sides:
            details.append(
                f"the body touches the {' and '.join(s for s in sides if s in ('left', 'right'))} edge"
            )
        if missing:
            details.append("not found: " + ", ".join(missing))
        problems.append(
            f"The whole body is not in the picture ({'; '.join(details)}). Move the laptop "
            "further back, or have the child stand on the start line."
        )

    # too far: body height (top of head to lowest foot point) as a part of the picture height
    elif not dark:
        ys = np.where(seen, xy[..., 1], np.nan)
        height = np.nanmax(ys, axis=1) - np.nanmin(ys, axis=1)
        frac = float(np.nanmedian(height)) / h
        if frac < MIN_BODY_FRAC:
            problems.append(
                f"The child looks small in the picture (body is {frac:.0%} of the picture "
                f"height, at least {MIN_BODY_FRAC:.0%} is needed). Move the laptop closer."
            )
        elif max_body_frac is not None and frac > max_body_frac:
            problems.append(
                f"Too close for a forward jump (body is {frac:.0%} of the picture height; "
                "aim for about 50%). Step back a little."
            )

    if dark:
        problems.append(
            f"Points are unsure (average confidence {mean_score:.2f}): the room may be too dark "
            "or the picture blurry. Turn on more light or open the curtains."
        )

    # start area (live/start_area.py): only foot points that were found count
    if start_area is not None and feet_inside(p, start_area, min_score) is False:
        problems.append(
            "The feet are not in the start area (the taped area). "
            "Ask the child to stand on the tape."
        )

    # long jump: the landing spot must be in the picture too (only when the feet are seen)
    if task == 3:
        room = landing_room(p, frame_size, min_score)
        if room is not None and room < ROOM_T3:
            problems.append(
                f"Not enough floor in the picture below the feet for the long jump ({room:.1f} "
                f"torso lengths, at least {ROOM_T3:.1f} needed): tilt the laptop screen down a "
                "little or move the laptop back, so the landing spot is in the picture."
            )
    return problems
