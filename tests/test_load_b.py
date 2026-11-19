"""Behaviour tests for data/load_b.py and common/pick.py (owner Marty)."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from common import body25 as b
from common.format import make_fake_try, validate
from common.geometry import normalize
from common.pick import pick_child
from data.load_b import load_try, parse_name, read_file


def _make_skeleton_entry(points_per_frame: list[list[np.ndarray]], label_index: int = 2) -> str:
    label_words = {0: "Bad", 1: "Good", 2: "Perfect"}
    data = []
    for i, people in enumerate(points_per_frame):
        skeleton = []
        for person in people:
            pose = person[:, :2].reshape(-1).astype(float).tolist()
            score = person[:, 2].reshape(-1).astype(float).tolist()
            skeleton.append({"pose": pose, "score": score})
        data.append({"frame_index": i + 1, "skeleton": skeleton})
    return json.dumps(
        {
            "data": data,
            "label": label_words[label_index],
            "label_index": label_index,
        }
    )


def test_load_try_rejects_clips_under_30_frames(tmp_path: Path):
    tr = make_fake_try(task=1, seconds=0.5, fps=30.0)  # 15 frames < 30
    zip_path = tmp_path / "short.zip"
    name = "B/B010_GMS_1_1_1.json"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr(name, _make_skeleton_entry([[tr.points[i]] for i in range(tr.n_frames)]))
    with zipfile.ZipFile(zip_path, "r") as z:
        with pytest.raises(ValueError, match="30-frame minimum"):
            load_try(z, name)


def test_load_try_keeps_clips_over_300_frames_uncut_and_normalizes(tmp_path: Path):
    tr = make_fake_try(task=2, seconds=11.5, fps=30.0)  # 345 frames > 300
    assert tr.n_frames > 300
    zip_path = tmp_path / "long.zip"
    name = "B/B012_GMS_2_3_1.json"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr(name, _make_skeleton_entry([[tr.points[i]] for i in range(tr.n_frames)], 1))
    with zipfile.ZipFile(zip_path, "r") as z:
        loaded = load_try(z, name)
    validate(loaded)
    assert loaded.n_frames == tr.n_frames
    assert loaded.meta["child"] == "B012"
    assert loaded.meta["task"] == 2
    assert loaded.meta["attempt"] == 3
    assert loaded.meta["camera"] == 1
    assert loaded.meta["label_index"] == 1
    assert loaded.meta["label"] == "Good"
    norm = normalize(loaded.points)
    assert norm.shape == (tr.n_frames, b.N_POINTS, 3)


def test_pick_child_locks_on_center_child_and_follows_nearest_neck():
    child = make_fake_try(task=1, seconds=1.5, fps=30.0, width=1920, height=1080).points
    # Bystander standing near right edge (x ~ 1880), lower confidence
    bystander = child.copy()
    bystander[:, :, 0] += 900.0
    bystander[:, :, 2] = 0.35

    frames = []
    for i in range(len(child)):
        # Swap person order every other frame to simulate OpenPose reordering detections
        if i % 2 == 0:
            frames.append(np.stack([bystander[i], child[i]]))
        else:
            frames.append(np.stack([child[i], bystander[i]]))

    picked = pick_child(frames)
    assert picked.shape == child.shape and picked.dtype == np.float32
    np.testing.assert_allclose(picked[:, b.NECK, :2], child[:, b.NECK, :2], atol=1e-4)


def test_pick_child_ignores_bystander_when_child_disappears_for_one_frame():
    child = make_fake_try(task=1, seconds=1.2, fps=30.0, width=1920, height=1080).points
    bystander = child.copy()
    bystander[:, :, 0] += 800.0
    bystander[:, :, 2] = 0.40

    frames = [np.stack([child[i], bystander[i]]) for i in range(len(child))]
    # At frame 5, child is occluded; only the distant bystander is detected
    frames[5] = bystander[5:6]

    picked = pick_child(frames)
    assert np.all(picked[5] == 0)  # bystander rejected by step gate
    np.testing.assert_allclose(picked[6, b.NECK, :2], child[6, b.NECK, :2], atol=1e-4)


def test_read_file_and_parse_name_roundtrip(tmp_path: Path):
    tr = make_fake_try(task=3, seconds=1.2, fps=30.0)
    zip_path = tmp_path / "round.zip"
    name = "B/B099_GMS_3_2_2.json"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr(name, _make_skeleton_entry([[tr.points[i]] for i in range(tr.n_frames)], 0))
    with zipfile.ZipFile(zip_path, "r") as z:
        frames, label_idx, label = read_file(z, name)
    assert len(frames) == tr.n_frames
    assert label_idx == 0 and label == "Bad"
    assert parse_name(name) == {"child": "B099", "task": 3, "attempt": 2, "camera": 2}
