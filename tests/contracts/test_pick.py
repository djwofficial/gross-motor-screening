"""Contract: common/pick.py (owner Marty). Add behaviour tests (2 people, crossing) in tests/."""

import numpy as np

from common.format import make_fake_try
from common.pick import pick_child


def test_shape_and_missing_frames():
    child = make_fake_try().points
    frames = [child[i : i + 1] for i in range(10)] + [np.zeros((0, 25, 3), np.float32)]
    out = pick_child(frames)
    assert out.shape == (11, 25, 3) and out.dtype == np.float32
    np.testing.assert_array_equal(out[:10], child[:10])
    assert np.all(out[10] == 0)  # nobody found -> (0, 0, 0)
