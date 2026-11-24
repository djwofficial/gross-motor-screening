"""Behaviour tests for viz/kee_clips.py (#51) on a FAKE zip in the group B file format."""

import csv
import json
import zipfile

import cv2
import pytest

from common.format import make_fake_try
from viz import kee_clips as kc

LABELS = ["Bad", "Good", "Perfect"]


@pytest.fixture
def fake_zip(tmp_path):
    """30 fake children x tasks 1-3 (+ task 4, ignored), scores spread, 2 cameras."""
    p = tmp_path / "B_skel.zip"
    with zipfile.ZipFile(p, "w") as z:
        for c in range(30):
            for task in (1, 2, 3, 4):
                tr = make_fake_try(task=min(task, 3), seed=c)
                score = (c + task) % 3
                data = [
                    {
                        "frame_index": i,
                        "skeleton": [
                            {
                                "pose": tr.points[i, :, :2].ravel().tolist(),
                                "score": tr.points[i, :, 2].tolist(),
                            }
                        ],
                    }
                    for i in range(tr.n_frames)
                ]
                for cam in (1, 2):
                    z.writestr(
                        f"B/B{c:03d}_GMS_{task}_1_{cam}.json",
                        json.dumps({"data": data, "label": LABELS[score], "label_index": score}),
                    )
    return p


def test_choose_spreads_tasks_and_scores(fake_zip):
    names = kc.choose(fake_zip)
    assert len(names) == 20
    children = [n.split("/")[-1][:4] for n in names]
    assert len(set(children)) == 20  # one clip per child
    tasks = [int(n.split("_GMS_")[1][0]) for n in names]
    assert set(tasks) == {1, 2, 3}
    with zipfile.ZipFile(fake_zip) as z:
        cells = {
            (t, json.loads(z.read(n))["label_index"]) for n, t in zip(names, tasks, strict=True)
        }
    assert len(cells) == 9  # every task x score
    assert sorted(tasks.count(t) for t in (1, 2, 3)) == [6, 7, 7]  # extras in different tasks


def test_clips_hide_the_score(fake_zip, tmp_path):
    res = kc.make_clips(fake_zip, tmp_path / "out", n=6)
    mp4 = sorted(res["clips"].glob("*.mp4"))
    assert [p.stem for p in mp4] == [f"clip_{k:02d}" for k in range(1, 7)]
    cap = cv2.VideoCapture(str(mp4[0]))
    assert cap.read()[0]
    cap.release()
    with open(res["key"], newline="", encoding="utf-8") as f:
        key = list(csv.DictReader(f))
    assert [r["clip"] for r in key] == [p.stem for p in mp4]
    assert all(r["score"] in ("0", "1", "2") and r["file"].startswith("B") for r in key)
    # the key is not inside the folder given to KEE
    assert res["key"].parent != res["clips"]
    with open(res["sheet"], newline="", encoding="utf-8") as f:
        assert all(r["score"] == "" for r in csv.DictReader(f))


def test_hidden_meta_has_no_label_or_child():
    tr = make_fake_try(task=2)
    tr.meta.update(child="B010", label="Good", label_index=1, file="B010_GMS_2_1_1.json")
    m = kc.hidden(tr).meta
    assert set(m) <= {"task", "width", "height"}
