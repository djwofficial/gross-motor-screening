# Task card: Revallo, 2026-11-13: webcam and recorded-video capture

**Issue:** #29. **Branch:** `revallo/capture-video`.

**Goal:** Capture unmirrored webcam frames independently of inference, and turn a recorded
team clip into a valid shared `Try` file without using inference wall time as video time.

**Input → output:** A webcam index or recorded video; Halpe-26 detections become BODY_25
`float32 (T,25,3)` points and `float64 (T,)` seconds, sampled at 30 fps and saved as `.npz`.

**Signatures (don't change):** `PoseStream(source, mode, device, backend).read()` returns
`(t, keypoints (P,26,2), scores (P,26), unmirrored_frame)` or `None`;
`resample_to_fps(points, t, fps=30.0)`; `FakePoseStream` remains compatible.

**Files you may change:** `live/pose.py`, `live/capture.py`,
`tests/test_live_capture.py`, `tests/test_live_pose.py`, and this card.

**Steps:**
1. Request 1280×720 at 30 fps for the webcam; keep only the newest timestamped frame in a
   capture thread. Release resources on initialization, inference, and camera errors.
2. Decode every recorded-video frame sequentially. Use its media timestamp, with a
   frame-index/source-fps fallback, and reject clips that cannot be timestamped.
3. Reuse `halpe_to_body25` and Marty's `pick_child`; keep empty frames as zeros and reject
   a completely empty attempt. Resample to uniform seconds with nearest-frame selection
   (ties choose the earlier frame), then validate and save a shared `Try`.
4. Add a headless CLI with `--video`, `--out`, `--task`, `--leg`, and `--device`; bound
   webcam capture with `--seconds`. Remind operators to disable OS auto-framing manually.

**Done when:** A recorded clip produces a file accepted by `common.format.load`/`validate`;
tests establish media timing, unmirrored inference, latest-frame capture, safe shutdown,
missing detections, resampling, and CLI behavior.

**Watch out:** This task does not save camera video, change tracking, or measure GPU fps.
No testing-day child data, changes to dependencies, shared modules, or contract tests.
720p30 is a camera request; actual device settings may differ. Auto-framing is an OS setting,
not a portable OpenCV toggle.

**Sources:** OpenCV's [video properties](https://docs.opencv.org/5.0/main_modules/videoio_flags_base.html)
define `CAP_PROP_POS_MSEC` in milliseconds and `CAP_PROP_FPS` as source fps. Its read timeout
is only supported by FFmpeg/GStreamer. RTMLib's
[BodyWithFeet source](https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/solution/body_with_feet.py)
defines the detector interface. 720p30 and 30 fps output come from issue #29/project requirements.

**Evidence (2026-11-13):** Python 3.12.13, pinned NumPy 2.5.4 and OpenCV contrib 5.0.0.93.

- `python -m pytest tests/test_live_capture.py tests/test_live_pose.py`:
  **43 passed**. Coverage includes media timestamps independent of inference time,
  latest-frame capture, slow model initialization, zero detections, bounded driver-release
  failure, preservation of original errors/Ctrl+C during cleanup, and output filenames.
- `python -m pytest`: **92 passed in 27.87s**, including the group B dataset test.
- `ruff format --no-cache .`, `ruff check --no-cache .`, `git diff --check`, and
  `python -m live.capture --help` passed.
- The recorded-clip test writes and reads a generated MJPG video through actual OpenCV,
  processes every frame, then loads and validates the saved Try. Its detector is injected
  artificial data; it does not establish real RTMPose quality or live-camera behavior.
- No camera or GPU was opened during validation. The operator confirmed that no adult team
  clip is available yet. Keep this work draft until a team clip has been processed with the
  real detector; actual camera resolution/FPS, OS auto-framing and driver shutdown also need
  checks on the live laptop.

**CLI example (from this worktree):**

```bat
.venv\Scripts\python.exe -m live.capture --video path\to\team.mp4 --out runs\team_task1.npz --task 1 --leg L --device cuda
```

Use the README's pinned live environment before running the real detector. Local setup for
these tests used only the pinned packages needed by the checks; full GPU runtime installation
and model downloads are separate from the generated-video proof.

**Shutdown limitation:** Normal release belongs to the camera worker. A bounded fallback
release attempt prevents the caller from hanging on a stuck driver; if the backend refuses
to stop, the stream reports an error and the camera application must be restarted. The
configured timeout is an operational choice, not a measured camera-performance result.
