"""Contract: live/segment.py and live/quality.py (owner Theresia)."""

import pytest

from common.format import make_fake_try
from live.quality import check
from live.segment import EVENTS, Segmenter


@pytest.mark.parametrize("task", [1, 2, 3])
@pytest.mark.parametrize("leg", ["L", "R"])
def test_one_start_and_one_end_per_fake_try(task, leg):
    tr = make_fake_try(task=task, leg=leg, idle_before=2.0, idle_after=2.5)
    seg = Segmenter()
    events = [seg.update(p, t) for p, t in zip(tr.points, tr.t, strict=True)]
    assert set(events) <= set(EVENTS)
    assert events.count("start") == 1 and events.count("end") == 1
    assert events.index("start") < events.index("end")


def test_segmenter_handles_nobody():
    seg = Segmenter()
    assert seg.update(None, 0.0) in EVENTS


def test_quality_messages():
    pts = make_fake_try().points[:30]
    assert check(pts, fps=30) == []
    assert any("fps" in m for m in check(pts, fps=10))
    no_feet = pts.copy()
    no_feet[:, 11:15, 2] = 0  # knees and ankles not found
    no_feet[:, 19:25, 2] = 0
    assert any("whole body" in m for m in check(no_feet, fps=30))
    assert check(None, fps=30) == ["Nobody is in the picture."]
