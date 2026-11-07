"""Contract: common/geometry.py (owner Daryn). The torso-length ruler used by every clue."""

import numpy as np
import pytest

from common import body25 as b
from common.format import make_fake_try
from common.geometry import normalize, torso_length


def test_torso_length_of_fake_child():
    assert torso_length(make_fake_try().points) == pytest.approx(110, abs=3)


def test_normalize_origin_scale_and_direction():
    tr = make_fake_try()
    n = normalize(tr.points)
    assert n.shape == tr.points.shape and n.dtype == np.float64
    np.testing.assert_allclose(n[0, b.NECK, :2], [0, 0], atol=1e-9)
    assert n[0, b.MID_HIP, 1] == pytest.approx(-1.0, abs=0.05)  # one torso below the neck, y up
    assert n[0, b.L_ANKLE, 1] < n[0, b.MID_HIP, 1] < n[0, b.NOSE, 1]


def test_normalize_is_pixel_free():
    p = make_fake_try(task=3).points
    big = p.copy()
    big[..., :2] *= 1.5  # same child seen bigger (e.g. 1920x1080 instead of 1280x720)
    np.testing.assert_allclose(normalize(big)[..., :2], normalize(p)[..., :2], atol=1e-5)


def test_missing_points_become_nan():
    p = make_fake_try().points.copy()
    p[3, b.L_ANKLE] = 0  # dataset style missing point
    n = normalize(p)
    assert np.isnan(n[3, b.L_ANKLE, 0]) and np.isnan(n[3, b.L_ANKLE, 1])
