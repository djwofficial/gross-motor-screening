"""viz/replay.py (owner Freya): Try -> stick-figure mp4, with the timeline bar."""

import json

import cv2
import numpy as np

from common.format import make_fake_try
from viz.replay import BAR_H, STATE_COLOURS, TOP_H, demo_events, load_any, main, render


def _read_all(path):
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, img = cap.read()
        if not ok:
            break
        frames.append(img)
    cap.release()
    return frames


def test_fake_try_plays(tmp_path):
    tr = make_fake_try(task=2, leg="R")
    out = render(tr, tmp_path / "fake.mp4", size=(640, 360))
    frames = _read_all(out)
    assert len(frames) == tr.n_frames
    assert frames[0].shape == (TOP_H + 360, 640, 3)
    assert frames[0].max() > 100  # something was drawn


def _dataset_like_json(tr, extra_person=True):
    """The dataset's JSON layout, built from a fake try (2 people in some frames)."""
    data = []
    for i in range(tr.n_frames):
        sk = [{"pose": tr.points[i, :, :2].ravel().tolist(), "score": tr.points[i, :, 2].tolist()}]
        if extra_person and i % 3 == 0:
            other = tr.points[i].copy()
            other[:, 0] += 400
            sk.append({"pose": other[:, :2].ravel().tolist(), "score": other[:, 2].tolist()})
        data.append({"frame_index": i, "skeleton": sk})
    return {"data": data, "label": "Good", "label_index": 1}


def test_dataset_file_plays(tmp_path):
    tr = make_fake_try(task=1, width=1920, height=1080)
    path = tmp_path / "B010_GMS_1_1_1.json"
    path.write_text(json.dumps(_dataset_like_json(tr)))
    loaded, events = load_any(str(path))
    assert events is None
    assert loaded.meta["child"] == "B010" and loaded.meta["task"] == 1
    assert loaded.n_frames == tr.n_frames
    np.testing.assert_allclose(loaded.t[1], 1 / 30)
    assert main([str(path), "-o", str(tmp_path / "ds.mp4")]) == 0
    assert len(_read_all(tmp_path / "ds.mp4")) == tr.n_frames


def test_timeline_bar_shows_the_right_colour_at_the_right_second(tmp_path):
    tr = make_fake_try(task=1, seconds=4.0)
    events = [
        {"start_s": 0.0, "end_s": 1.0, "state": "both_feet"},
        {"start_s": 1.0, "end_s": 3.0, "state": "steady"},
        {"start_s": 3.0, "end_s": float(tr.t[-1]), "state": "wobble"},
    ]
    w, h = 640, 360
    frames = _read_all(render(tr, tmp_path / "bar.mp4", size=(w, h), events=events))
    assert frames[0].shape[0] == TOP_H + h + BAR_H
    y = TOP_H + h + 10  # inside the coloured strip
    t1 = float(tr.t[-1])
    img = frames[0]
    for second, state in [(0.5, "both_feet"), (2.0, "steady"), (3.6, "wobble")]:
        x = int(second / t1 * (w - 1))
        got = img[y, x].astype(int)
        want = np.array(STATE_COLOURS[state])
        assert np.abs(got - want).max() < 40, (second, state, got, want)


def test_session_json_with_events(tmp_path):
    from app.session import Session

    tr = make_fake_try(task=3)
    ev = demo_events(tr)
    p = Session(tmp_path, child="K01", age_months=40).save_try(
        tr,
        task=3,
        leg="L",
        guessed_task=3,
        guessed_leg="L",
        task_confidence=0.5,
        corrected=False,
        score=2,
        confidence=0.9,
        clues={},
        reasons="test",
        events=ev,
    )
    loaded, events = load_any(str(p))
    assert events == ev
    assert main([str(p)]) == 0
    assert p.with_suffix(".mp4").exists()
