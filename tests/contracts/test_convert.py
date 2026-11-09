"""Contract: live/convert.py (owner Revallo). Halpe-26 -> BODY_25, checked name by name."""

import numpy as np

from common import body25 as b
from live.convert import HALPE26_NAMES, HALPE_TO_BODY25, halpe_to_body25


def _same_point(body25_name: str, halpe_name: str) -> bool:
    if body25_name == "mid_hip":
        return halpe_name == "hip"
    side = {"l": "left", "r": "right"}
    if body25_name[:2] in ("l_", "r_"):
        return halpe_name == f"{side[body25_name[0]]}_{body25_name[2:]}"
    return halpe_name == body25_name


def test_mapping_matches_names():
    assert len(HALPE_TO_BODY25) == 25 and len(set(HALPE_TO_BODY25)) == 25
    assert 17 not in HALPE_TO_BODY25  # Halpe "head" is dropped
    for i, h in enumerate(HALPE_TO_BODY25):
        assert _same_point(b.NAMES[i], HALPE26_NAMES[h]), (i, b.NAMES[i], HALPE26_NAMES[h])


def test_shapes_values_and_scores():
    kp = np.stack([np.arange(26), np.arange(26) + 100], axis=-1).astype(np.float32)  # x = index
    sc = np.linspace(-0.5, 1.5, 26).astype(np.float32)
    out = halpe_to_body25(kp, sc)
    assert out.shape == (25, 3) and out.dtype == np.float32
    np.testing.assert_array_equal(out[:, 0], np.array(HALPE_TO_BODY25, dtype=np.float32))
    assert out[:, 2].min() >= 0 and out[:, 2].max() <= 1  # clipped
    many = halpe_to_body25(np.stack([kp, kp, kp]), np.stack([sc, sc, sc]))
    assert many.shape == (3, 25, 3)
