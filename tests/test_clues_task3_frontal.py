"""Behaviour tests for Task 3 Hybrid-C jump detection and unified frontal/side jump_size."""

from __future__ import annotations

import numpy as np
import pytest

from common import body25 as b
from common.format import make_fake_try
from features.clues import UNKNOWN, compute, timeline


def _frontal_jump(
    *,
    seconds: float = 3.5,
    fps: float = 30.0,
    jump_start: float = 1.0,
    air_s: float = 0.4,
    hop_px: float = 140.0,
    drop_px: float = 55.0,
    end_scale: float = 1.15,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic frontal standing long jump straight toward the camera (dx == 0).

    As the child jumps toward the camera, their grounded feet end up lower in the image
    (larger pixel y -> smaller normalized y) and their apparent body size grows by end_scale.
    """
    tr = make_fake_try(task=3, seconds=seconds, fps=fps, seed=0)
    t = tr.t
    # Clean stationary standing pose from frame 0, then animate cleanly
    stand = tr.points[0].copy()
    neck0 = stand[b.NECK, :2].copy()
    pts = np.repeat(stand[None], len(t), axis=0)
    for i, ti in enumerate(t):
        if ti < jump_start:
            u = 0.0
            lift = 0.0
        elif ti <= jump_start + air_s:
            u = (ti - jump_start) / air_s
            lift = hop_px * 4.0 * u * (1.0 - u)
        else:
            u = 1.0
            lift = 0.0
        scale = 1.0 + (end_scale - 1.0) * u
        xy = neck0 + (stand[:, :2] - neck0) * scale
        # Pure frontal forward progression: no lateral x shift, feet move down in image
        xy[:, 1] += drop_px * u - lift
        pts[i, :, :2] = xy
    return pts.astype(np.float32), t


def test_one_frame_nan_dropout_during_air_still_detects_jump():
    tr = make_fake_try(task=3)
    pts = tr.points.copy()
    mid_air = int(np.argmin(np.abs(tr.t - 1.2)))
    # Drop L_ANKLE for 2 frames (~0.067 s <= 0.15 s) right in the middle of the jump
    pts[mid_air : mid_air + 2, b.L_ANKLE] = 0.0
    events = timeline(3, pts, tr.t)
    assert [e["state"] for e in events] == ["before", "air", "after"]
    clues = compute(3, pts, tr.t)
    assert clues["air_s"] > 0.2
    assert 0.8 < clues["jump_size"] < 1.2


def test_mid_hip_missing_during_air_falls_back_to_lr_hips():
    tr = make_fake_try(task=3)
    pts = tr.points.copy()
    air_mask = (tr.t >= 0.9) & (tr.t <= 1.6)
    pts[air_mask, b.MID_HIP] = 0.0
    events = timeline(3, pts, tr.t)
    assert sum(e["state"] == "air" for e in events) == 1
    clues = compute(3, pts, tr.t)
    assert clues["air_s"] > 0.2
    assert 0.8 < clues["jump_size"] < 1.2


def test_flat_to_crouch_does_not_trigger_phantom_jump():
    """A flat stance followed by a downward crouch or monotonic drop must not invent an air event."""
    tr = make_fake_try(task=3, seconds=3.5, fps=30.0)
    t = tr.t
    base = np.repeat(tr.points[:1], len(t), axis=0)

    # Case 1: upper body crouches down while feet stay on the floor
    crouch = base.copy()
    torso = 110.0
    ramp = np.clip((t - 1.2) / 0.4, 0.0, 1.0)
    for idx in (b.NOSE, b.NECK, b.MID_HIP, b.L_HIP, b.R_HIP, b.L_SHOULDER, b.R_SHOULDER):
        crouch[:, idx, 1] += 0.45 * torso * ramp
    assert all(e["state"] != "air" for e in timeline(3, crouch, t))
    assert compute(3, crouch, t)["air_s"] == 0.0
    assert compute(3, crouch, t)["jump_size"] == 0.0

    # Case 2: entire body drops monotonically (flat -> lower level, no upward peak above start)
    drop = base.copy()
    drop[:, :, 1] += (0.35 * torso * ramp)[:, None]
    assert all(e["state"] != "air" for e in timeline(3, drop, t))
    assert compute(3, drop, t)["air_s"] == 0.0
    assert compute(3, drop, t)["jump_size"] == 0.0


def test_frontal_forward_jump_gives_nonzero_scale_invariant_jump_size():
    short_pts, t = _frontal_jump(drop_px=25.0, end_scale=1.06)
    long_pts, _ = _frontal_jump(drop_px=75.0, end_scale=1.22)

    ev_long = timeline(3, long_pts, t)
    assert [e["state"] for e in ev_long] == ["before", "air", "after"]

    c_short = compute(3, short_pts, t)
    c_long = compute(3, long_pts, t)
    assert c_short["air_s"] > 0.15
    assert c_long["air_s"] > 0.15
    assert c_short["jump_size"] > 0.25
    assert c_long["jump_size"] > c_short["jump_size"] + 0.5

    # Pixel-scale invariance on frontal jump with perspective scale_gain
    scaled = long_pts.copy()
    scaled[..., :2] = scaled[..., :2] * 1.8 + np.array([80.0, 40.0], dtype=np.float32)
    c_scaled = compute(3, scaled, t)
    for k in c_long:
        assert c_long[k] == pytest.approx(c_scaled[k], rel=1e-4, abs=1e-6), k


def test_low_fps_20fps_clip_does_not_fracture_into_unknown():
    tr = make_fake_try(task=3, fps=20.0, seed=1)
    # Introduce a realistic 0.15 s dropped-frame hiccup (< _MAX_FRAME_GAP_S = 0.25 s)
    keep = np.ones(len(tr.t), dtype=bool)
    keep[10:12] = False  # gap of 0.15 s before the jump
    pts = tr.points[keep]
    t = tr.t[keep]
    events = timeline(3, pts, t)
    assert all(e["state"] != UNKNOWN for e in events)
    assert [e["state"] for e in events] == ["before", "air", "after"]
    assert compute(3, pts, t)["air_s"] > 0.2


def test_exact_20fps_015s_nan_gap_is_clock_offset_invariant():
    """At exact 20 fps, a 2-frame NaN dropout spans 3/20 = 0.15 s (== _MAX_INTERP_GAP_S).
    IEEE-754 rounding must not flip jump detection across clock offsets."""
    tr = make_fake_try(task=3, fps=20.0, seed=0)
    t_exact = np.arange(len(tr.t), dtype=np.float64) / 20.0
    pts = tr.points.copy()
    mid = int(np.argmin(np.abs(t_exact - 1.2)))
    pts[mid : mid + 2, b.L_ANKLE] = 0.0  # gap = 3 / 20 = 0.15 s

    base = compute(3, pts, t_exact)
    assert base["air_s"] > 0.2
    assert 0.8 < base["jump_size"] < 1.2
    for offset in (10.0, 100.0, 100000.0):
        shifted = compute(3, pts, t_exact + offset)
        for key in base:
            assert shifted[key] == pytest.approx(base[key], rel=1e-6, abs=1e-9), (offset, key)


def test_frame_gap_after_landing_keeps_last_after_frame_in_post_window():
    """When a camera frame gap (> 0.25 s) follows a short 'after' span, after['end_s'] is
    inclusive (t[end-1]), so that last grounded frame must stay inside the post window."""
    tr = make_fake_try(task=3, fps=30.0, seed=0)
    ev0 = timeline(3, tr.points, tr.t)
    air0 = next(e for e in ev0 if e["state"] == "air")
    land = int(np.searchsorted(tr.t, air0["end_s"]))

    # Two grounded landing frames (land, land + 1), then a 0.5 s camera freeze.
    t_gap = tr.t.copy()
    t_gap[land + 2 :] += 0.5
    pts_a = tr.points.copy()
    pts_b = tr.points.copy()
    # Shift MID_HIP x on frame land + 1 only: if land + 1 is included in post, jump_size changes.
    pts_b[land + 1, b.MID_HIP, 0] += 80.0

    ev_a = timeline(3, pts_a, t_gap)
    after_ev = [e for e in ev_a if e["state"] == "after"][0]
    assert after_ev["end_s"] == pytest.approx(t_gap[land + 1])

    ca = compute(3, pts_a, t_gap)
    cb = compute(3, pts_b, t_gap)
    assert cb["jump_size"] > ca["jump_size"] + 0.2
