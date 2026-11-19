"""Behaviour tests for the task 1 timeline and clues (Louise, #20). Fake tries only."""

import numpy as np
import pytest

from common import body25 as b
from common.format import make_fake_try
from features.clues import compute, timeline


def _both_feet(seconds=4.0):
    """A fake try with the lifted foot put back level with the support foot."""
    tr = make_fake_try(task=1, leg="L", seconds=seconds)
    p = tr.points.copy()
    for i_r, i_l in zip(b.RIGHT_FOOT + [b.R_KNEE], b.LEFT_FOOT + [b.L_KNEE], strict=True):
        p[:, i_r, 1] = p[:, i_l, 1]
    return p, tr.t


def _lift(p, t, start, end, px=40.0):
    """Raise the right foot by px pixels between start and end seconds."""
    p = p.copy()
    on = (t >= start) & (t < end)
    for i in b.RIGHT_FOOT + [b.R_KNEE]:
        p[on, i, 1] -= px
    return p


def test_one_lift_of_two_seconds():
    p, t = _both_feet(4.0)
    c = compute(1, _lift(p, t, 1.0, 3.0), t)
    assert c["one_foot_s"] == pytest.approx(2.0, abs=0.07)
    assert c["touchdowns"] == 0.0
    assert c["wobble_s"] == 0.0  # the fake hip does not move sideways


def test_two_lifts_give_one_touchdown():
    p, t = _both_feet(5.0)
    p = _lift(_lift(p, t, 0.5, 1.5), t, 2.5, 4.5)
    c = compute(1, p, t)
    assert c["touchdowns"] == 1.0
    assert c["one_foot_s"] == pytest.approx(2.0, abs=0.07)


def test_both_feet_down_all_the_time():
    p, t = _both_feet()
    c = compute(1, p, t)
    assert c["one_foot_s"] == 0.0 and c["touchdowns"] == 0.0 and c["sway"] == 0.0
    assert [e["state"] for e in timeline(1, p, t)] == ["both_feet"]


def test_flicker_shorter_than_min_run_is_ignored():
    p, t = _both_feet(4.0)
    p = _lift(p, t, 1.0, 3.0)
    dip = (t >= 2.0) & (t < 2.1)  # foot touches down for 0.1 s
    for i in b.RIGHT_FOOT + [b.R_KNEE]:
        p[dip, i, 1] = p[dip, b.L_ANKLE, 1]
    c = compute(1, p, t)
    assert c["touchdowns"] == 0.0
    assert c["one_foot_s"] == pytest.approx(2.0, abs=0.07)


def test_sideways_hip_movement_is_wobble_and_sway():
    p, t = _both_feet(4.0)
    p = _lift(p, t, 0.5, 3.5)
    torso = 110.0  # pixels, torso length of the fake child
    shift = 0.3 * torso * np.sin(2 * np.pi * 1.5 * t)  # 1.5 Hz, 0.3 torso lengths
    p[:, :, 0] += shift[:, None]
    c = compute(1, p, t)
    assert c["wobble_s"] > 1.0
    assert c["sway"] > 0.4
    states = {e["state"] for e in timeline(1, p, t)}
    assert "wobble" in states


def test_timeline_covers_the_whole_try_without_gaps():
    tr = make_fake_try(task=1)
    ev = timeline(1, tr.points, tr.t)
    assert ev[0]["start_s"] == tr.t[0] and ev[-1]["end_s"] == tr.t[-1]
    for a, c in zip(ev, ev[1:], strict=False):
        assert a["end_s"] == c["start_s"]


def test_missing_ankles_are_unknown():
    p, t = _both_feet()
    p[:, [b.L_ANKLE, b.L_HEEL], 2] = 0.0
    assert {e["state"] for e in timeline(1, p, t)} == {"unknown"}
    assert compute(1, p, t)["one_foot_s"] == 0.0


def test_small_gap_like_a_step_is_not_a_lift():
    """0.2 torso lengths (a step toward the camera) never reaches LIFT_PEAK."""
    p, t = _both_feet(4.0)
    c = compute(1, _lift(p, t, 1.0, 3.0, px=0.2 * 110.0), t)
    assert c["one_foot_s"] == 0.0


def test_explain_names_moments():
    from features.clues import explain

    p, t = _both_feet(4.0)
    p = _lift(p, t, 1.0, 3.0)
    text = explain(1, compute(1, p, t), timeline(1, p, t))
    assert "2.0 s on one foot" in text and "0 touch-downs" in text
    assert "foot up at 1.0 s" in text and "foot down at 3.0 s" in text


def test_explain_without_a_lift_is_neutral():
    from features.clues import explain

    p, t = _both_feet()
    text = explain(1, compute(1, p, t), timeline(1, p, t))
    assert text == "no clear one-foot stand seen"
    assert "fail" not in text
