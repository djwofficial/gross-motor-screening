"""live/track.py + app/ui.py (#33): the live window keeps following the same child.

Done when (#33): with 2 people in the picture the track doesn't jump to the other person.
"""

import json
import zipfile

import numpy as np
import pytest

from common.format import make_fake_try
from common.pick import pick_child
from live.convert import HALPE_TO_BODY25
from live.track import ChildTracker

W, H = 1280, 720


def scene(n=120, child_x=0.7 * W, other_from=-0.1 * W, other_to=0.5 * W, other_enters=30):
    """Per frame: the child standing at child_x, and from frame other_enters a second person who
    walks in from the left and stops in the centre (a better start-line candidate than the child)."""
    child = make_fake_try(task=1, seconds=n / 30, width=W, height=H).points[:n].copy()
    child[..., 0] += child_x - W / 2
    other = make_fake_try(task=1, seconds=n / 30, width=W, height=H, seed=5).points[:n].copy()
    frames = []
    for k in range(n):
        people = [child[k]]
        if k >= other_enters:
            u = min(1.0, (k - other_enters) / 40)
            o = other[k].copy()
            o[:, 0] += other_from + u * (other_to - other_from) - W / 2
            people.append(o)
        frames.append(np.stack(people).astype(np.float32))
    return frames


def neck_x(p):
    return float(p[1, 0])


def test_old_per_frame_pick_jumps_to_the_other_person():
    picked = [pick_child([f])[0] for f in scene()]
    assert abs(neck_x(picked[-1]) - 0.7 * W) > 100  # it ended on the other person


def test_tracker_stays_on_the_child():
    tr = ChildTracker((W, H))
    for k, f in enumerate(scene()):
        i = tr.update(f, k / 30)
        assert i is not None and abs(neck_x(f[i]) - 0.7 * W) < 30, k


def test_tracker_ignores_someone_passing_close_by():
    frames = scene(other_from=0.1 * W, other_to=0.7 * W - 160)  # ends 160 px beside the child
    tr = ChildTracker((W, H))
    for k, f in enumerate(frames):
        i = tr.update(f, k / 30)
        assert i is not None and abs(neck_x(f[i]) - 0.7 * W) < 30, k


def test_child_missing_for_a_moment_is_not_replaced():
    frames = scene()
    tr = ChildTracker((W, H), lost_s=2.0)
    for k, f in enumerate(frames):
        if 60 <= k < 75:  # child not detected for 0.5 s, the other person still is
            f = f[1:]
        i = tr.update(f, k / 30)
        if 60 <= k < 75:
            assert i is None, k
        else:
            assert abs(neck_x(f[i]) - 0.7 * W) < 30, k


def test_lock_is_taken_again_after_the_child_left():
    tr = ChildTracker((W, H), lost_s=1.0)
    frames = scene()
    for k in range(40):
        tr.update(frames[k], k / 30)
    other_only = frames[100][1:]
    assert tr.update(other_only, 2.0) is None  # 0.7 s later: still waiting for the child
    assert tr.update(other_only, 3.0) == 0  # more than 1 s gone: lock on whoever is there


def _items(frames):
    idx = np.array(HALPE_TO_BODY25)
    for k, f in enumerate(frames):
        kp = np.zeros((len(f), 26, 2), np.float32)
        sc = np.zeros((len(f), 26), np.float32)
        kp[:, idx] = f[..., :2]
        sc[:, idx] = f[..., 2]
        yield k / 30, kp, sc, None


def test_window_saves_the_child_not_the_other_person(tmp_path):
    from app.ui import Runner

    r = Runner(tmp_path, mode="manual", frame_size=(W, H), source="webcam")
    for key in ["K", "0", "1", "ENTER", "4", "5", "ENTER", "1", "ENTER"]:
        r.press(key)
    items = list(_items(scene(n=150)))
    for it in items[:20]:
        r.feed(it)
    r.press("ENTER", 100.0)  # readiness accepted: lock on the child at the start line
    r.press("SPACE", 110.0)
    for it in items[20:]:
        r.feed(it)
    r.press("SPACE", 120.0)
    r.press("ENTER", 130.0)
    rec = json.loads(r.saved[0].read_text(encoding="utf-8"))
    p25 = np.array(rec["points25"])
    p26 = np.array(rec["points26"])
    assert np.all(np.abs(p25[:, 1, 0] - 0.7 * W) < 30)  # neck always on the child
    assert np.all(np.abs(p26[:, 18, 0] - 0.7 * W) < 30)  # raw Halpe neck (18) of the same person


def test_dataset_multi_person_files_follow_the_offline_pick():
    """Group B files with 2+ people: live frame-by-frame tracking agrees with Marty's whole-clip
    pick_child (they share the same rules), on at least 95% of the frames."""
    from data.load_b import DEFAULT_ZIP, list_files, parse_name, read_file

    if not DEFAULT_ZIP.exists():
        pytest.skip("data/raw/B_skel.zip not present")
    agree = total = files = 0
    with zipfile.ZipFile(DEFAULT_ZIP) as z:
        for name in list_files():
            if parse_name(name)["task"] not in (1, 2, 3):
                continue
            frames, _, _ = read_file(z, name)
            if max(len(f) for f in frames) < 2:
                continue
            offline = pick_child(frames)
            tr = ChildTracker((1920, 1080))
            for k, f in enumerate(frames):
                i = tr.update(f, k / 30)
                live = f[i] if i is not None else np.zeros((25, 3), np.float32)
                if offline[k, 1, 2] > 0 or live[1, 2] > 0:
                    total += 1
                    agree += bool(np.allclose(live, offline[k]))
            files += 1
            if files >= 80:
                break
    assert files == 80 and agree / total >= 0.95, (files, agree, total)
