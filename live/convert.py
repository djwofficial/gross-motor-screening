"""Halpe-26 (RTMPose / rtmlib BodyWithFeet) -> BODY_25 (dataset order). Owner: Louise.

Drop Halpe point 17 (head) and re-order. No point is made up.

HALPE26_NAMES is copied from rtmlib 0.0.16, rtmlib/visualization/skeleton/halpe26.py (checked
2026-11-08 by printing it from the installed package). tests/contracts/test_convert.py checks the
mapping name by name against common/body25.py.

Louise, Nov 11: also check against mmpose configs/_base_/datasets/halpe26.py and plot one labelled
webcam frame to confirm left and right on a real picture (unmirrored).
"""

from __future__ import annotations

import numpy as np

HALPE26_NAMES = [
    "nose",  # 0
    "left_eye",  # 1
    "right_eye",  # 2
    "left_ear",  # 3
    "right_ear",  # 4
    "left_shoulder",  # 5
    "right_shoulder",  # 6
    "left_elbow",  # 7
    "right_elbow",  # 8
    "left_wrist",  # 9
    "right_wrist",  # 10
    "left_hip",  # 11
    "right_hip",  # 12
    "left_knee",  # 13
    "right_knee",  # 14
    "left_ankle",  # 15
    "right_ankle",  # 16
    "head",  # 17  (dropped)
    "neck",  # 18
    "hip",  # 19  (= BODY_25 mid_hip)
    "left_big_toe",  # 20
    "right_big_toe",  # 21
    "left_small_toe",  # 22
    "right_small_toe",  # 23
    "left_heel",  # 24
    "right_heel",  # 25
]

# BODY_25 index i takes Halpe-26 index HALPE_TO_BODY25[i].
HALPE_TO_BODY25 = [
    0,
    18,
    6,
    8,
    10,
    5,
    7,
    9,
    19,
    12,
    14,
    16,
    11,
    13,
    15,
    2,
    1,
    4,
    3,
    20,
    22,
    24,
    21,
    23,
    25,
]


def halpe_to_body25(keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """keypoints (..., 26, 2) in pixels, scores (..., 26) -> float32 (..., 25, 3) as x, y, score.

    Works for one person (26, 2), several people (P, 26, 2) or a clip (T, 26, 2).
    Scores are clipped to 0..1 (RTMPose SimCC scores are not guaranteed to stay in that range).
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if keypoints.shape[-2:] != (26, 2) or scores.shape != keypoints.shape[:-1]:
        raise ValueError(
            f"expected keypoints (...,26,2) and scores (...,26), got {keypoints.shape} and {scores.shape}"
        )
    idx = np.array(HALPE_TO_BODY25)
    out = np.concatenate(
        [keypoints[..., idx, :], np.clip(scores[..., idx], 0.0, 1.0)[..., None]], axis=-1
    )
    return out.astype(np.float32)
