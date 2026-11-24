"""Task 3 landing room in live/quality.py (owner Theresia): the landing spot must be in the picture.

Marty's task 3 experiments (2026-11-23): most missed Good/Perfect jumps lose the feet at the
bottom of the picture. ROOM_T3 = 1.5 torso lengths comes from group B (see live/quality.py).
"""

import numpy as np

from common import body25 as b
from common.format import make_fake_try
from live.quality import ROOM_T3, check, landing_room

W, H = 1280, 720


def _standing(room_torso: float, torso_px: float = 100.0):
    """1 s of a still person whose lowest foot point is room_torso torso lengths above the bottom."""
    p = make_fake_try(task=3).points[:30].copy()
    ok = p[..., 2] > 0
    torso = np.linalg.norm(p[:, b.NECK, :2] - p[:, b.MID_HIP, :2], axis=1)[:, None]
    p[..., :2] = (p[..., :2] - p[:, [b.NECK], :2]) / torso[..., None] * torso_px
    feet = [b.R_ANKLE, b.L_ANKLE, b.R_HEEL, b.L_HEEL, b.R_BIG_TOE, b.L_BIG_TOE]
    low = p[:, feet, 1].max(axis=1, keepdims=True)
    p[..., 1] += (H - room_torso * torso_px) - low  # put the feet at the wanted height
    p[..., 0] += W / 2
    p[~ok] = 0
    return p


def test_landing_room_in_torso_lengths():
    assert abs(landing_room(_standing(2.0), (W, H)) - 2.0) < 0.05
    assert abs(landing_room(_standing(0.5), (W, H)) - 0.5) < 0.05
    assert landing_room(np.zeros((5, 25, 3)), (W, H)) is None


def test_long_jump_needs_floor_below_the_feet():
    near_edge = _standing(0.8)
    msgs = check(near_edge, 30, (W, H), task=3)
    assert any(m.startswith("Not enough floor") for m in msgs)
    assert not any(
        m.startswith("Not enough floor") for m in check(_standing(2.0), 30, (W, H), task=3)
    )


def test_other_tasks_and_old_calls_unchanged():
    near_edge = _standing(0.8)
    for task in (None, 1, 2):
        assert not any(
            m.startswith("Not enough floor") for m in check(near_edge, 30, (W, H), task=task)
        )
    assert check(near_edge, 30, (W, H)) == check(near_edge, 30, (W, H), task=None)


def test_threshold_is_the_group_b_value():
    assert ROOM_T3 == 1.5
