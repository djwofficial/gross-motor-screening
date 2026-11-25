"""Behaviour tests for the 13 extra task 3 clues "A" (Marty, 2026-11-25; from
scripts/experiments/exp_t3_best.py)."""

import numpy as np
import pytest

from common.format import make_fake_try
from features.clues import CLUE_NAMES, compute

OLD_T3 = ["duration_s", "feet_gap_takeoff", "feet_gap_landing", "air_s", "jump_size"]
EXTRA = [
    "n_air",
    "best_jump_size",
    "dx",
    "dy",
    "scale_gain",
    "crouch",
    "hip_peak",
    "arm_up",
    "arm_range",
    "land_squat",
    "land_move",
    "land_step",
    "hands_low",
    "jump_size_local",
    "hip_peak_local",
]


def test_names_keep_the_old_five_first():
    assert CLUE_NAMES[3] == OLD_T3 + EXTRA


@pytest.mark.parametrize("leg", ["L", "R"])
def test_fake_try_gives_finite_values_for_all_20_names(leg):
    tr = make_fake_try(task=3, leg=leg)
    c = compute(3, tr.points, tr.t)
    assert list(c) == CLUE_NAMES[3]
    assert len(c) == 20
    assert all(isinstance(v, float) and np.isfinite(v) for v in c.values())


def test_no_pose_fills_extra_clues_with_minus_9():
    t = np.arange(40) / 30.0
    points = np.zeros((40, 25, 3), dtype=np.float32)
    c = compute(3, points, t)
    assert all(np.isfinite(v) for v in c.values())
    assert c["n_air"] == 0.0
    assert all(c[k] == -9.0 for k in EXTRA if k != "n_air")
