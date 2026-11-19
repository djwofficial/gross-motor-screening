"""Behaviour tests for the task 2 (hop) timeline and clues (Louise, #21). Fake tries only."""

import numpy as np

from common import body25 as b
from common.format import make_fake_try
from features.clues import compute, explain, timeline

LIFTED = b.RIGHT_FOOT + [b.R_KNEE]  # fake tries hop on the left leg


def _hops(n_hops, seconds, hop_s=0.4, start=0.5, pause_after=None, pause_s=0.0):
    """Hop on the left leg: the whole body bounces 30 px (about 0.27 torso lengths) per hop."""
    tr = make_fake_try(task=1, leg="L", seconds=seconds)  # task 1 = standing still on the left leg
    p, t = tr.points.copy(), tr.t
    p[:, LIFTED, 1] = p[:, b.L_ANKLE, 1][:, None] - 40.0  # right foot up the whole try
    t0 = start
    for k in range(n_hops):
        on = (t >= t0) & (t < t0 + hop_s)
        u = (t[on] - t0) / hop_s
        p[on, :, 1] -= (30.0 * np.sin(np.pi * u))[:, None]
        t0 += hop_s + (pause_s if pause_after is not None and k == pause_after else 0.0)
    return p, t


def test_counts_three_hops():
    p, t = _hops(3, 2.5)
    c = compute(2, p, t)
    assert c["hops"] == 3.0
    assert c["touchdowns"] == 0.0


def test_standing_on_one_foot_is_not_a_hop():
    p, t = _hops(0, 2.5)
    assert compute(2, p, t)["hops"] == 0.0
    assert explain(2, compute(2, p, t), timeline(2, p, t)).startswith("no hop seen")


def test_two_foot_jump_is_not_a_hop():
    tr = make_fake_try(task=3)  # jumps once with both feet
    assert compute(2, tr.points, tr.t)["hops"] == 0.0


def test_pause_between_hops():
    p, t = _hops(4, 4.0, pause_after=1, pause_s=1.0)
    c = compute(2, p, t)
    assert c["hops"] == 4.0
    assert c["longest_pause_s"] > 0.5
    assert "pause" in {e["state"] for e in timeline(2, p, t)}


def test_each_hop_is_one_event_and_explain_names_them():
    p, t = _hops(3, 2.5)
    ev = timeline(2, p, t)
    assert sum(e["state"] == "hop" for e in ev) == 3
    text = explain(2, compute(2, p, t), ev)
    assert text.startswith("3 hops, 0 touch-downs") and "hops from" in text


def test_clues_do_not_depend_on_pixels():
    p, t = _hops(3, 2.5)
    big = p.copy()
    big[..., :2] *= 1.5
    assert compute(2, p, t) == compute(2, big, t)
