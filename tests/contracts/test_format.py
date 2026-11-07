"""Contract: common/format.py (owner Daryn). Every part relies on this format."""

import numpy as np
import pytest

from common.format import Try, load, make_fake_try, save, validate


@pytest.mark.parametrize("task", [1, 2, 3])
@pytest.mark.parametrize("leg", ["L", "R"])
def test_fake_try_is_valid(task, leg):
    tr = make_fake_try(task=task, leg=leg)
    assert tr.points.dtype == np.float32 and tr.points.shape[1:] == (25, 3)
    assert tr.t.dtype == np.float64 and tr.t.shape == (tr.n_frames,)
    assert tr.t[0] == 0.0 and np.all(np.diff(tr.t) > 0)
    assert tr.meta["task"] == task and tr.meta["leg"] == leg


def test_save_load_round_trip(tmp_path):
    tr = make_fake_try(task=2)
    back = load(save(tr, tmp_path / "x.npz"))
    np.testing.assert_array_equal(back.points, tr.points)
    np.testing.assert_array_equal(back.t, tr.t)
    assert back.meta == tr.meta


def test_validate_rejects_bad_tries():
    tr = make_fake_try()
    with pytest.raises(ValueError):
        validate(Try(tr.points[:, :24], tr.t, {}))  # 24 points
    with pytest.raises(ValueError):
        validate(Try(tr.points.astype(np.float64), tr.t, {}))  # wrong dtype
    with pytest.raises(ValueError):
        validate(Try(tr.points, tr.t[::-1].copy(), {}))  # time going backwards
    bad = tr.points.copy()
    bad[0, 0, 2] = 1.5
    with pytest.raises(ValueError):
        validate(Try(bad, tr.t, {}))  # score above 1
