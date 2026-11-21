"""Tests for app/main.py (--fake ChildTracker path, live dispatch, and 3-task scoring)."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest

import app.main as app_main
from app.session import load_record
from live.pose import FakePoseStream
from rescore import rescore


def test_fake_mode_does_not_import_rtmlib_or_cv2(tmp_path):
    code = (
        "import sys, app.main; "
        f"rc = app.main.main(['--fake', '--task', '1', '--out', {str(tmp_path)!r}]); "
        "assert rc == 0; "
        "assert 'rtmlib' not in sys.modules; "
        "assert 'cv2' not in sys.modules"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_run_fake_uses_child_tracker_on_multiperson_frames(tmp_path, monkeypatch):
    base = FakePoseStream(task=2, leg="L", seed=1)

    class TwoPersonFakeStream:
        def __init__(self, **_kwargs):
            self._inner = base

        def read(self):
            item = self._inner.read()
            if item is None:
                return None
            t, kp26, sc26, frame = item
            # Small distant decoy person at index 0; real centered child at index 1.
            decoy_kp = np.full((1, 26, 2), 80.0, dtype=np.float32)
            decoy_kp[0, 18, :] = [80.0, 70.0]  # Neck
            decoy_kp[0, 19, :] = [80.0, 90.0]  # Hip
            decoy_sc = np.full((1, 26), 0.9, dtype=np.float32)
            return (
                t,
                np.concatenate([decoy_kp, kp26], axis=0),
                np.concatenate([decoy_sc, sc26], axis=0),
                frame,
            )

        def close(self):
            self._inner.close()

    monkeypatch.setattr(app_main, "FakePoseStream", TwoPersonFakeStream)
    args = argparse.Namespace(
        task=2, leg="L", seed=1, mode="auto", child="K99", age=48, out=tmp_path
    )
    saved = app_main.run_fake(args)
    assert len(saved) == 1
    record = load_record(saved[0])
    tr = record["try"]
    # Selected person is the centered child, not the corner decoy around x=80.
    valid_x = tr.points[:, 1, 0][tr.points[:, 1, 2] > 0]
    assert float(np.median(valid_x)) > 300.0
    if "raw26" in record and record["raw26"] is not None:
        raw26 = np.asarray(record["raw26"], dtype=np.float32)
        assert float(np.median(raw26[:, 18, 0][raw26[:, 18, 2] > 0])) > 300.0
    assert rescore(tmp_path) == (1, 1)


def test_live_dispatch_testing_and_show_score(tmp_path, monkeypatch):
    import cv2

    import app.demo as app_demo
    import app.ui as app_ui
    import live.pose as live_pose

    calls = []
    created = []

    class FakeCap:
        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return 640
            if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return 480
            return 0

    class StubPoseStream:
        def __init__(self, source=0, mode="balanced", device="cuda"):
            self.init_args = (source, mode, device)
            self.cap = FakeCap()
            self.closed = False
            created.append(self)

        def close(self):
            self.closed = True

    def fake_run_gui(stream, out_dir, **kwargs):
        calls.append(("gui", stream, out_dir, kwargs))
        return [tmp_path / "try_001.json"]

    def fake_run_demo(stream, out_dir, **kwargs):
        calls.append(("demo", stream, out_dir, kwargs))
        return []

    monkeypatch.setattr(live_pose, "PoseStream", StubPoseStream)
    monkeypatch.setattr(app_ui, "run_gui", fake_run_gui)
    monkeypatch.setattr(app_demo, "run_demo", fake_run_demo)
    monkeypatch.setattr(app_main, "pick_camera", lambda: 1)

    # Default testing mode: auto-picked webcam (1), view="stick", platform defaults.
    rc1 = app_main.main(["--out", str(tmp_path)])
    assert rc1 == 0
    kind, stream1, out_dir1, kw1 = calls[-1]
    assert kind == "gui"
    assert stream1.closed is True
    assert stream1.init_args[0] == 1
    assert out_dir1 == tmp_path
    assert kw1 == {
        "mode": "auto",
        "view": "stick",
        "frame_size": (640, 480),
        "source": "camera:1",
    }

    # Explicit --camera 0 bypasses pick_camera.
    rc_cam0 = app_main.main(["--camera", "0", "--out", str(tmp_path)])
    assert rc_cam0 == 0
    assert calls[-1][1].init_args[0] == 0
    assert calls[-1][3]["source"] == "camera:0"

    # Explicit video + show-score routes to run_demo and defaults view="camera".
    clip = tmp_path / "demo.mp4"
    rc2 = app_main.main(
        [
            "--video",
            str(clip),
            "--mode",
            "manual",
            "--device",
            "cpu",
            "--pose-mode",
            "balanced",
            "--show-score",
            "--out",
            str(tmp_path),
        ]
    )
    assert rc2 == 0
    kind, stream2, out_dir2, kw2 = calls[-1]
    assert kind == "demo"
    assert stream2.closed is True
    assert stream2.init_args == (clip, "balanced", "cpu")
    assert out_dir2 == tmp_path
    assert kw2 == {
        "mode": "manual",
        "view": "camera",
        "frame_size": (640, 480),
        "source": str(clip),
    }

    # Stream is still closed even if the UI raises.
    def boom(*_args, **_kwargs):
        raise RuntimeError("window failed")

    monkeypatch.setattr(app_ui, "run_gui", boom)
    with pytest.raises(RuntimeError, match="window failed"):
        app_main.main(["--out", str(tmp_path)])
    assert len(created) == 4
    assert created[-1].closed is True


def test_pick_camera_skips_black_mac_camera(monkeypatch):
    import cv2

    opened = []

    class FakeCap:
        def __init__(self, idx):
            self.idx = idx
            self.released = False
            opened.append(self)

        def isOpened(self):
            return True

        def read(self):
            if self.idx == 0:
                return True, np.zeros((4, 4, 3), dtype=np.uint8)
            return True, np.full((4, 4, 3), 120, dtype=np.uint8)

        def release(self):
            self.released = True

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(cv2, "VideoCapture", FakeCap)
    assert app_main.pick_camera(candidates=(0, 1), max_frames=2) == 1
    assert [c.idx for c in opened] == [0, 1]
    assert all(c.released for c in opened)

    # Non-macOS returns 0 immediately without probing VideoCapture.
    opened.clear()
    monkeypatch.setattr(sys, "platform", "win32")
    assert app_main.pick_camera(candidates=(0, 1), max_frames=2) == 0
    assert opened == []


@pytest.mark.parametrize("task", [1, 2, 3])
def test_score_try_and_finish_try_compute_timeline_once(task, tmp_path, monkeypatch):
    import features.clues as fc
    from app.session import Session
    from common.format import make_fake_try

    fn_name = {1: "_timeline_task1", 2: "_timeline_task2", 3: "_task3_events"}[task]
    orig = getattr(fc, fn_name)
    calls = 0

    def spy(*args, **kwargs):
        nonlocal calls
        calls += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(fc, fn_name, spy)

    tr = make_fake_try(task=task, leg="L", seed=0)
    app_main.score_try(tr, task)
    assert calls == 1

    # Mock predict_task so finish_try only counts the scoring/saving timeline pass.
    monkeypatch.setattr(app_main, "predict_task", lambda _p, _t: (task, "L", 0.9))
    calls = 0
    session = Session(tmp_path, child="K01", age_months=48, video=False)
    raw26 = np.zeros((tr.n_frames, 26, 3), dtype=np.float32)
    app_main.finish_try(session, tr.points, tr.t, raw26, tr.meta, task, "L")
    assert calls == 1


def test_demo_runner_uses_predict_task(tmp_path, monkeypatch):
    import app.demo as app_demo
    import model.task_classifier as tc

    called = []
    orig = tc.predict_task

    def spy(points, t):
        called.append(len(t))
        return orig(points, t)

    monkeypatch.setattr(tc, "predict_task", spy)
    runner = app_demo.DemoRunner(tmp_path, mode="manual", frame_size=(1280, 720), source="fake")
    stream = FakePoseStream(task=1, leg="L", seed=0)
    while (item := stream.read()) is not None:
        runner.feed(item)
    stream.close()
    assert called, "DemoRunner.feed should call model.task_classifier.predict_task"
    assert runner.live is not None and "score" in runner.live


def test_all_three_tasks_score_quickly_and_rescore_matches(tmp_path):
    t0 = time.perf_counter()
    for task in (1, 2, 3):
        rc = app_main.main(["--fake", "--task", str(task), "--out", str(tmp_path)])
        assert rc == 0
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0
    df = pd.read_csv(tmp_path / "log.csv")
    assert set(df["task"].tolist()) == {1, 2, 3}
    assert rescore(tmp_path) == (3, 3)
