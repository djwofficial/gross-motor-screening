"""Contract: features/clues.py (owner Revallo). Put value tests (by-eye clips) in tests/."""

import numpy as np
import pytest

from common.format import make_fake_try
from features.clues import CLUE_NAMES, compute, explain


@pytest.mark.parametrize("task", [1, 2, 3])
def test_names_types_and_explain(task):
    tr = make_fake_try(task=task)
    c = compute(task, tr.points, tr.t)
    assert list(c) == CLUE_NAMES[task]
    assert all(isinstance(v, float) and np.isfinite(v) for v in c.values())
    assert c["duration_s"] == pytest.approx(tr.duration)
    text = explain(task, c)
    assert isinstance(text, str) and text


@pytest.mark.parametrize("task", [1, 2, 3])
def test_clues_do_not_depend_on_pixels(task):
    """Seconds and torso lengths only: a bigger picture must give the same clues."""
    tr = make_fake_try(task=task)
    big = tr.points.copy()
    big[..., :2] *= 1.5
    a, b = compute(task, tr.points, tr.t), compute(task, big, tr.t)
    for k in a:
        assert a[k] == pytest.approx(b[k], rel=1e-4, abs=1e-6), k
