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


def test_mid_sequence_06s_pause_is_detected():
    """A 0.6 s pause between hops must show up as a pause event and longest_pause_s > 0.2."""
    p, t = _hops(4, 4.0, pause_after=1, pause_s=0.6)
    c = compute(2, p, t)
    assert c["hops"] == 4.0
    assert c["longest_pause_s"] > 0.2
    assert "pause" in {e["state"] for e in timeline(2, p, t)}


def test_stray_pre_and_post_hop_foot_lifts_do_not_count_as_touchdowns():
    """Foot lifts / both-feet spans before the first hop or after the last hop are not
    touch-downs between hops."""
    p, t = _hops(3, 5.0, hop_s=0.4, start=1.8)
    # Put right foot down before the hops (0.8-1.5 s) and after the hops (3.3-4.0 s),
    # leaving stray non-hop one-foot lifts at 0.0-0.8 s and 4.0-5.0 s.
    down = ((t >= 0.8) & (t < 1.5)) | ((t >= 3.3) & (t < 4.0))
    for i_r, i_l in zip(b.RIGHT_FOOT + [b.R_KNEE], b.LEFT_FOOT + [b.L_KNEE], strict=True):
        p[down, i_r, 1] = p[down, i_l, 1]
    c = compute(2, p, t)
    assert c["hops"] == 3.0
    assert c["touchdowns"] == 0.0


def test_touchdown_between_hops_is_counted():
    """Putting the lifted foot down between hops counts as a touchdown."""
    p, t = _hops(4, 4.5, hop_s=0.4, start=0.5, pause_after=1, pause_s=0.9)
    # Put the lifted foot down during the 0.9 s gap between hop 2 and hop 3 (1.3-2.2 s).
    down = (t >= 1.45) & (t < 2.05)
    for i_r, i_l in zip(b.RIGHT_FOOT + [b.R_KNEE], b.LEFT_FOOT + [b.L_KNEE], strict=True):
        p[down, i_r, 1] = p[down, i_l, 1]
    c = compute(2, p, t)
    assert c["hops"] == 4.0
    assert c["touchdowns"] == 1.0


def test_single_hop_always_has_zero_touchdowns():
    """With <= 1 hop there are no hops on both sides of a touchdown, so touchdowns == 0."""
    p, t = _hops(1, 4.0, hop_s=0.4, start=1.8)
    down = ((t >= 0.7) & (t < 1.4)) | ((t >= 2.5) & (t < 3.2))
    for i_r, i_l in zip(b.RIGHT_FOOT + [b.R_KNEE], b.LEFT_FOOT + [b.L_KNEE], strict=True):
        p[down, i_r, 1] = p[down, i_l, 1]
    c = compute(2, p, t)
    assert c["hops"] == 1.0
    assert c["touchdowns"] == 0.0


def test_hop_ending_near_clip_end_emits_no_zero_duration_trailing_pause():
    """A hop whose post-peak trough lands near the final frame must not emit a 0-second pause."""
    p, t = _hops(3, 2.0, hop_s=0.4, start=0.5)
    # Trim the clip right after the 3rd hop's landing trough so after + 1 would hit len(t) - 1.
    keep = t <= 1.72
    p_cut, t_cut = p[keep], t[keep]
    ev = timeline(2, p_cut, t_cut)
    c = compute(2, p_cut, t_cut)
    assert c["hops"] == 3.0
    assert c["longest_pause_s"] == 0.0
    assert all(e["end_s"] > e["start_s"] for e in ev)
    assert "pause at" not in explain(2, c, ev)
