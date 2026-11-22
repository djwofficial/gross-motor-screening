"""Behaviour tests for viz/blur.py (#52) on synthetic frames (no real faces, no children)."""

import cv2
import numpy as np
import pytest

from common import body25 as b
from common.format import make_fake_try
from live.convert import HALPE_TO_BODY25
from viz import blur

W, H = 320, 240


def _person(dx=0.0, scale=40.0, head=True):
    """A standing person (25, 3) around x = 160 + dx, head near y = 60."""
    p = make_fake_try(task=1).points[0].copy()
    ok = p[:, 2] > 0
    xy = p[:, :2]
    neck = xy[b.NECK].copy()
    torso = float(np.linalg.norm(xy[b.NECK] - xy[b.MID_HIP]))
    xy[ok] = (xy[ok] - neck) / torso * scale + np.array([W / 2 + dx, 90.0])
    p[:, :2] = xy
    if not head:
        p[blur.HEAD] = 0
    return p


def _noise_frame(seed=0):
    return np.random.default_rng(seed).integers(0, 255, (H, W, 3), dtype=np.uint8)


def test_head_ellipse_covers_all_head_points():
    p = _person()
    cx, cy, ax, ay = blur.head_ellipse(p)
    for i in blur.HEAD:
        if p[i, 2] > 0:
            x, y = p[i, :2]
            assert ((x - cx) / ax) ** 2 + ((y - cy) / ay) ** 2 < 0.5


def test_no_head_points_puts_the_ellipse_above_the_neck():
    p = _person(head=False)
    cx, cy, ax, ay = blur.head_ellipse(p)
    assert cy < p[b.NECK, 1]  # picture y down: above the neck
    assert abs(cx - p[b.NECK, 0]) < ax


def test_frame_status():
    assert blur.frame_plan(np.zeros((0, 25, 3)))[1] == "nobody"
    assert blur.frame_plan(_person()[None])[1] == "ok"
    feet_only = np.zeros((25, 3), np.float32)
    feet_only[b.L_ANKLE] = (100, 200, 0.9)
    assert blur.frame_plan(np.stack([_person(), feet_only]))[1] == "unsure"


def test_neighbour_frames_cover_a_missed_frame():
    frames = [_person()[None], np.zeros((0, 25, 3)), _person()[None]]
    steps = blur.plan(frames)
    assert len(steps[1][0]) == 2  # ellipses from the frames before and after
    assert steps[1][1] == "nobody"  # still blurred whole (fail-safe)


def test_blur_changes_only_inside_the_ellipse():
    f = _noise_frame()
    e = blur.head_ellipse(_person())
    out = blur.blur_ellipses(f, [e])
    cx, cy, ax, ay = e
    assert np.abs(out[int(cy), int(cx)].astype(int) - f[int(cy), int(cx)]).sum() > 0
    assert np.array_equal(out[H - 5, 5], f[H - 5, 5])
    # inside the ellipse the noise is gone (mosaic + blur is smooth)
    inner = out[int(cy - ay / 2) : int(cy + ay / 2), int(cx - ax / 2) : int(cx + ax / 2)]
    assert inner.std() < 0.5 * f.std()


def test_unsure_frame_is_blurred_whole():
    f = _noise_frame()
    out = blur.blur_frame(f, [], "unsure")
    assert out.std() < 0.5 * f.std()


class _Stream:
    """Like PoseStream: every frame of the video with the given people."""

    def __init__(self, people_per_frame):
        self.people = people_per_frame
        self.i = 0
        self.source_fps = 30.0

    def read(self):
        if self.i >= len(self.people):
            return None
        pp = self.people[self.i]
        kp = np.zeros((len(pp), 26, 2), np.float32)
        sc = np.zeros((len(pp), 26), np.float32)
        for j, p in enumerate(pp):
            for i25, i26 in enumerate(HALPE_TO_BODY25):
                if i26 is not None and i26 >= 0:
                    kp[j, i26], sc[j, i26] = p[i25, :2], p[i25, 2]
        self.i += 1
        return self.i / 30.0, kp, sc, None

    def close(self):
        pass


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "T01_abc.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30.0, (W, H))
    if not w.isOpened():
        pytest.skip("MJPG encoder not available")
    for k in range(10):
        w.write(_noise_frame(k))
    w.release()
    return path


def test_blur_video_writes_clip_and_check_csv(video):
    people = [[_person(), _person(dx=80)]] * 8 + [[]] * 2
    res = blur.blur_video(video, stream=_Stream(people))
    assert res["frames"] == 10 and res["ok"] == 8 and res["whole_nobody"] == 2
    rows = blur.read_check(res["check"])
    assert [r["status"] for r in rows][-2:] == ["whole_nobody"] * 2
    assert rows[0]["people"] == "2" and rows[0]["heads"] == "2"
    cap = cv2.VideoCapture(str(res["out"]))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    assert n == 10


def test_json_must_name_this_video(video, tmp_path):
    j = tmp_path / "try.json"
    j.write_text('{"meta": {"video_file": "other.avi"}}', encoding="utf-8")
    with pytest.raises(ValueError, match="names video"):
        blur.blur_video(video, j, stream=_Stream([[_person()]] * 10))


def test_eye_check_passes_only_when_every_frame_is_seen_and_no_face():
    rows = [{"frame": str(i), "eye_check": ""} for i in range(5)]
    assert not blur.check_summary(rows)["passed"]
    blur.apply_marks(rows, seen={0, 1, 2, 3, 4}, faces=set())
    assert blur.check_summary(rows)["passed"]
    blur.apply_marks(rows, seen={0, 1, 2, 3, 4}, faces={3})
    s = blur.check_summary(rows)
    assert not s["passed"] and s["faces"] == [3]
