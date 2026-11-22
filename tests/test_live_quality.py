"""Behaviour tests for live/quality.py (owner Theresia, issue #48).

Three deliberately bad setups are made from the adult team clips (docs/task_cards/
team_clips_27_try.zip): too close (picture zoomed in, points outside the picture lost), too dark
(confidence lowered) and feet outside the tape (start area placed beside the person). Each must get
its own message, and only that message, on every clip whose normal setup is clean.
"""

import zipfile
from pathlib import Path

import numpy as np
import pytest

from common.format import load, make_fake_try
from live.quality import NOBODY, check
from live.start_area import FEET

ZIP = Path(__file__).resolve().parents[1] / "docs" / "task_cards" / "team_clips_27_try.zip"
WINDOW = 45  # 1.5 s at 30 fps, as the app's readiness window


def _zoom(p, w, h, k):
    """Zoom the picture by k around its centre; points that leave the picture are lost."""
    q = p.copy()
    c = np.array([w / 2, h / 2], np.float32)
    q[..., :2] = (q[..., :2] - c) * k + c
    out = (q[..., 0] < 0) | (q[..., 0] > w) | (q[..., 1] < 0) | (q[..., 1] > h)
    q[out] = 0
    return q


def _foot_box(p, pad):
    f = p[:, FEET][p[:, FEET, 2] > 0.3][:, :2]
    (x0, y0), (x1, y1) = f.min(0), f.max(0)
    return x0 - pad, y0 - pad, x1 + pad, y1 + pad


@pytest.fixture(scope="module")
def team_clips(tmp_path_factory):
    if not ZIP.exists():
        pytest.skip("team clips zip not found")
    out = tmp_path_factory.mktemp("team")
    with zipfile.ZipFile(ZIP) as z:
        names = [n for n in z.namelist() if n.endswith(".npz")]
        z.extractall(out, members=names)
    return [load(out / n) for n in sorted(names)]


def test_team_clips_mostly_ready(team_clips):
    flagged = []
    for tr in team_clips:
        size = (tr.meta["width"], tr.meta["height"])
        if check(tr.points[:WINDOW], 30, size):
            flagged.append(tr)
    assert len(team_clips) == 27
    # 4 clips are framed so tightly that the feet touch the bottom edge: a correct message
    assert len(flagged) <= 4
    for tr in flagged:
        size = (tr.meta["width"], tr.meta["height"])
        (msg,) = check(tr.points[:WINDOW], 30, size)
        assert "feet are cut off at the bottom" in msg


def test_three_bad_setups_get_the_right_message(team_clips):
    tested = 0
    for tr in team_clips:
        w, h = tr.meta["width"], tr.meta["height"]
        p = tr.points[:WINDOW]
        area = _foot_box(p, 60)
        if check(p, 30, (w, h), area):
            continue  # normal setup not clean: nothing to compare against
        tested += 1

        (close,) = check(_zoom(p, w, h, 1.6), 30, (w, h))
        assert close.startswith("The whole body is not in the picture")
        assert "Move the laptop further back" in close

        dark = p.copy()
        dark[..., 2] *= 0.45
        (msg,) = check(dark, 30, (w, h))
        assert msg.startswith("Points are unsure")

        x0, y0, x1, y1 = area
        beside = (x1 + 200, y0, x1 + 500, y1)
        (msg,) = check(p, 30, (w, h), beside)
        assert msg.startswith("The feet are not in the start area")

        (msg,) = check(_zoom(p, w, h, 0.2), 30, (w, h))
        assert msg.startswith("The child looks small")
    assert tested >= 23


def test_fps_not_measured_yet_is_quiet():
    pts = make_fake_try().points[:30]
    assert check(pts, fps=0) == []
    assert check(None, fps=0) == [NOBODY]


def test_feet_not_found_are_not_outside_the_start_area():
    tr = make_fake_try()
    pts = tr.points[:30].copy()
    pts[:, FEET, 2] = 0
    msgs = check(pts, 30, (1280, 720), start_area=(0, 0, 10, 10))
    assert not any("start area" in m for m in msgs)
    assert any("whole body" in m for m in msgs)


def test_start_area_corners_in_any_order():
    pts = make_fake_try().points[:30]
    box = _foot_box(pts, 30)
    assert check(pts, 30, start_area=box) == []
    assert check(pts, 30, start_area=(box[2], box[3], box[0], box[1])) == []


def test_one_frame_works():
    pts = make_fake_try().points[0]
    assert check(pts, 30) == []
