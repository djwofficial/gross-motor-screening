"""Behaviour tests for live/start_area.py and its use in the readiness check and Segmenter (#49)."""

import numpy as np
import pytest

from common.format import make_fake_try
from live import start_area as sa
from live.quality import check
from live.segment import Segmenter


def _foot_box(points, pad=30.0):
    f = points[:, sa.FEET][points[:, sa.FEET, 2] > 0.3][:, :2]
    (x0, y0), (x1, y1) = f.min(0), f.max(0)
    return x0 - pad, y0 - pad, x1 + pad, y1 + pad


def _events(tr, **kw):
    seg = Segmenter(**kw)
    return [seg.update(p, t) for p, t in zip(tr.points, tr.t, strict=True)]


def test_tidy_unmirror_rescale():
    assert sa.tidy((10, 20, 5, 2)) == (5.0, 2.0, 10.0, 20.0)
    # a box at the left of the mirrored picture is at the right of the real one
    assert sa.unmirror((0, 10, 100, 50), 1280) == (1180.0, 10.0, 1280.0, 50.0)
    assert sa.unmirror(sa.unmirror((3, 4, 50, 60), 640), 640) == (3.0, 4.0, 50.0, 60.0)
    assert sa.rescale((960, 540, 1920, 1080), (1920, 1080), (1280, 720)) == (640, 360, 1280, 720)


def test_save_and_load(tmp_path):
    path = tmp_path / "area.json"
    assert sa.load(path) is None  # nothing saved yet
    sa.save((500, 600, 300, 400), (1280, 720), path)
    assert sa.load(path) == (300.0, 400.0, 500.0, 600.0)
    assert sa.load(path, (1280, 720)) == (300.0, 400.0, 500.0, 600.0)
    assert sa.load(path, (640, 360)) == (150.0, 200.0, 250.0, 300.0)


@pytest.mark.parametrize("bad", [(0, 0, 0, 10), (100, 100, 1300, 200), (-5, 0, 10, 10)])
def test_save_refuses_empty_or_outside(tmp_path, bad):
    with pytest.raises(ValueError):
        sa.save(bad, (1280, 720), tmp_path / "a.json")


def test_feet_inside():
    pts = make_fake_try().points[:30]
    assert sa.feet_inside(pts, _foot_box(pts)) is True
    assert sa.feet_inside(pts[0], _foot_box(pts)) is True
    assert sa.feet_inside(pts, (0, 0, 50, 50)) is False
    no_feet = pts.copy()
    no_feet[:, sa.FEET, 2] = 0
    assert sa.feet_inside(no_feet, (0, 0, 50, 50)) is None


def test_readiness_check_uses_the_saved_area(tmp_path):
    pts = make_fake_try().points[:30]
    path = tmp_path / "area.json"
    sa.save(_foot_box(pts), (1280, 720), path)
    assert check(pts, 30, (1280, 720), start_area=sa.load(path, (1280, 720))) == []
    sa.save((0, 0, 100, 100), (1280, 720), path)
    (msg,) = check(pts, 30, (1280, 720), start_area=sa.load(path, (1280, 720)))
    assert msg.startswith("The feet are not in the start area")


@pytest.mark.parametrize("task", [1, 2, 3])
@pytest.mark.parametrize("leg", ["L", "R"])
def test_segmenter_with_area_around_the_feet_works_as_before(task, leg):
    tr = make_fake_try(task=task, leg=leg, idle_before=2.0, idle_after=2.5)
    area = _foot_box(tr.points[:30])
    ev = _events(tr, start_area=area)
    assert ev.count("start") == 1 and ev.count("end") == 1
    # same events as without an area up to the end; after a long jump (task 3) the child stands
    # outside the area, so it does not become "ready" there again
    end = ev.index("end") + 1
    assert ev[:end] == _events(tr)[:end]


def test_segmenter_not_ready_outside_the_area():
    tr = make_fake_try(task=1, idle_before=2.0, idle_after=2.5)
    ev = _events(tr, start_area=(0, 0, 100, 100))
    assert "ready" not in ev and "start" not in ev


def test_mark_returns_unmirrored_box(monkeypatch):
    import cv2

    frame = np.zeros((720, 1280, 3), np.uint8)
    # the operator drags x 100-300, y 500-700 on the MIRRORED picture
    monkeypatch.setattr(cv2, "selectROI", lambda *a, **k: (100, 500, 200, 200))
    monkeypatch.setattr(cv2, "destroyWindow", lambda *a: None)
    assert sa.mark(frame) == (980.0, 500.0, 1180.0, 700.0)
    monkeypatch.setattr(cv2, "selectROI", lambda *a, **k: (0, 0, 0, 0))  # Esc
    assert sa.mark(frame) is None


def test_draw_on_mirrored_picture():
    img = np.zeros((720, 1280, 3), np.uint8)
    sa.draw(img, (980, 500, 1180, 700), mirrored=True)
    cols = np.nonzero(img[600].any(axis=1))[0]
    assert 95 <= cols.min() <= 105 and 295 <= cols.max() <= 305  # drawn at x 100-300
    assert sa.draw(img, None) is img
