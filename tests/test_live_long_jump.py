"""Behavior of the long jump clues on generated poses."""

import numpy as np
import pytest

from common import body25 as b
from common.format import make_fake_try
from features.clues import UNKNOWN, compute, timeline


def test_jump_has_grounded_phases_and_bounded_air_time():
    tr = make_fake_try(task=3)
    events = timeline(3, tr.points, tr.t)
    assert [e["state"] for e in events] == ["before", "air", "after"]
    air = events[1]
    clues = compute(3, tr.points, tr.t)
    # The synthetic fixture starts its 0.45 s jump at 1.0 s, with noisy
    # measurements. The threshold-based interval sits inside that motion.
    assert 1.0 <= air["start_s"] < 1.15
    assert 1.3 < air["end_s"] <= 1.5
    assert clues["air_s"] == pytest.approx(air["end_s"] - air["start_s"])
    assert 0.8 < clues["jump_size"] < 1.2


def test_stationary_and_missing_pose_have_no_invented_air_time():
    tr = make_fake_try(task=3)
    stationary = np.repeat(tr.points[:1], tr.n_frames, axis=0)
    for points in (stationary, np.zeros_like(stationary)):
        assert timeline(3, points, tr.t)[0]["state"] == UNKNOWN
        assert compute(3, points, tr.t)["air_s"] == 0.0


def test_missing_ankle_creates_an_unknown_interval():
    tr = make_fake_try(task=3)
    missing = (tr.t >= 0.3) & (tr.t < 0.55)
    tr.points[missing, b.L_ANKLE] = 0
    events = timeline(3, tr.points, tr.t)
    assert any(e["state"] == UNKNOWN and e["end_s"] > e["start_s"] for e in events)
    assert sum(e["state"] == "air" for e in events) == 1


def test_repeated_jumps_keep_separate_air_intervals_and_first_jump_clues():
    tr = make_fake_try(task=3)
    points = np.concatenate([tr.points, tr.points])
    t = np.concatenate([tr.t, tr.t + tr.t[-1] + 1 / 30])
    events = timeline(3, points, t)
    assert sum(e["state"] == "air" for e in events) == 2
    first = compute(3, tr.points, tr.t)
    repeated = compute(3, points, t)
    for key in ("air_s", "jump_size", "feet_gap_takeoff", "feet_gap_landing"):
        assert repeated[key] == pytest.approx(first[key])
