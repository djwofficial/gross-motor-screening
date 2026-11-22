"""Behaviour tests for live/segment.py on the 27 adult team clips (#50).

The clips stop 0.1-1 s after the movement, so the end is tested on a SIMULATED 2 s tail: the
last pose held still, or nobody in the picture (leaving). Numbers from
`python -m live.segment --team-clips` on 2026-11-22; see
docs/task_cards/theresia_2026-11-19_segment.md for the per-clip table and the misses.
"""

from pathlib import Path

import numpy as np
import pytest

from common.format import make_fake_try
from live import segment

ZIP = Path(__file__).resolve().parents[1] / "docs" / "task_cards" / "team_clips_27_try.zip"


@pytest.fixture(scope="module")
def rows():
    zp = ZIP
    if not zp.exists():
        pytest.skip("team clips zip not found")
    return segment.check_team_clips(zp)


def test_starts_found_on_most_team_clips(rows):
    assert len(rows) == 27
    assert sum(r["start"] for r in rows) >= 24
    assert all(r["starts"] <= 1 for r in rows)  # never two starts in one try


def test_ends_found_with_simulated_tail(rows):
    assert sum(r["end_leave"] for r in rows) >= 25
    assert sum(r["end_hold"] for r in rows) >= 19
    # holding still never ends a try while one foot is still up
    for r in rows:
        if r["end_why"].startswith("clip stops with one foot up"):
            assert not r["end_hold"]


def test_with_tail():
    tr = make_fake_try()
    pts, t = segment.with_tail(tr.points, tr.t, "leave", tail_s=1.0)
    assert len(pts) == len(t) == tr.n_frames + 30 and pts[-1] is None
    assert np.all(np.diff(t) > 0)
    pts, _ = segment.with_tail(tr.points, tr.t, "hold", tail_s=1.0)
    assert np.array_equal(pts[-1], tr.points[-1])


def test_slow_foot_lift_starts_the_try():
    """A foot lifted too slowly for move_speed still starts the try."""
    tr = make_fake_try(task=1, leg="L", idle_before=2.0)
    still = tr.points[0].copy()
    frames, times = [], []
    for i in range(90):  # 3 s: 2 s still, then the right ankle rises slowly (0.3 torso/s)
        p = still.copy()
        if i >= 60:
            torso = float(np.linalg.norm(still[1, :2] - still[8, :2]))
            p[[11, 22, 23, 24], 1] -= 0.3 * torso * (i - 60) / 30  # y down = up in the picture
        frames.append(p)
        times.append(i / 30)
    ev = segment.run(frames, times)
    assert "ready" in ev and ev.count("start") == 1
    assert times[ev.index("start")] < 2.8
