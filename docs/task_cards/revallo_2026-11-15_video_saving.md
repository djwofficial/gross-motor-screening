# Task card: Revallo, 2026-11-15: save consented video off the pose thread

**Goal.** Write a camera clip in a separate thread for
[#30](https://github.com/djwofficial/gross-motor-screening/issues/30).
The filename carries the child code. Video requires consent level 3
and the current V setting to be on.

**Interface.** Freya's Runner can create one `TryVideoRecorder` per attempt
with the agreed session directory, code, consent, frame size and FPS.
It can submit original camera frames through `offer(frame, enabled=...)`
and finish with `close(keep=True)` only on confirmation. Use
`close(keep=False)` on cancellation or unwinding. The GUI can report a
writer error without losing an otherwise valid points record.

**Files.** Revallo's new `live/video_writer.py` and tests in
`tests/test_live_video_writer.py`. Freya or Daryn owns the GUI connection.

**Done when.** The local writer tests pass and the person-present live
laptop comparison shows the measured FPS with V on and off. Compare the
same camera, mode, driver, scene and recording duration. Record the number
of offered, written and dropped video frames as well as pose FPS.

**GUI hookup.** Create a recorder when the attempt starts, using the agreed
session storage and the current participant's consent. During recording,
send the original BGR frame from the pose stream, before mirroring for
display. Pass the current V flag on each call. Stop offering at the end of
the attempt. On confirmation, finalize the clip and record whether it was
actually saved. On cancel or exit, discard the staged clip. If the writer
fails, let the operator save the points and show the video error separately.

**Validation on 2026-11-19.** Nine writer tests passed, including actual
MJPG write and readback, V and consent checks, cancellation, slow and
failed encoders, and queue overflow. The full local suite passed all 87
tests with the Group B archive. Ruff format and lint passed.

**Operational defaults.** The four-frame queue and two-second shutdown
timeout are implementation choices. The queue drops incoming frames
when full to bound memory and keep encoding off the producer thread.
An AVI clip uses the configured playback FPS. Its timing can differ
from the attempt if frames drop or arrive unevenly. Use Try timestamps
for movement analysis.

**Live FPS with V off and on, measured 2026-11-19.** Laptop: Revallo's
(NVIDIA GeForce RTX 3050 Laptop GPU, driver 610.62), not Freya's live laptop.
Camera: built-in ACER HD User Facing at 1280x720. rtmlib BodyWithFeet balanced
on CUDA, manual mode, camera view, the real app window from #66 (ae3a711).
Both runs used `python -m live.measure_video_fps` (branch
revallo/measure-video-fps): same scene, light and movement (one adult team
member: one-leg stands, hops, jumps), setup T1 / 45 / consent 3, about 25 s
per try.

| Run | Frames / duration_s (log.csv) | Pose fps | Recorder offered / written / dropped | AVI |
|---|---|---|---|---|
| V off | 346 / 25.49 s | 13.6 | (V off: nothing written) | none |
| V on | 325 / 25.41 s | 12.8 | 318 / 318 / 0 | opens, 318 frames read back, playback 13.0 fps |

The writer dropped no frames and the clip opens. Done when (V on at 15 fps or
more) is **not met**: 12.8 fps with V on. V off is also below 15 (13.6), so
most of the gap comes from pose estimation plus the window on this laptop;
video saving costs about 0.8 fps here. The 7 frames between 325 and 318 are
the pre-roll, which holds points only. Freya and Daryn are told; fallbacks
in the plan: a lower saved-video resolution or `--pose-mode lightweight`, and
the same check on Freya's live laptop. The run folder was deleted after the
numbers were copied.
The writer uses the existing
[OpenCV VideoWriter interface](https://docs.opencv.org/4.x/dd/d9e/classcv_1_1VideoWriter.html).
