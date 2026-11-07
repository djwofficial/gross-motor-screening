# Louise: live pose module (critical path)

**Your part:** webcam → RTMPose → 25 BODY_25 points, fast enough and stable, with the right
child followed. Everything live depends on this.

**Your files:** `live/pose.py`, `live/convert.py`, `live/capture.py`, `live/bench_fps.py`,
new tests `tests/test_live_*.py`.

**Signatures (AGENTS.md):** `PoseStream(source, mode, device).read() -> (t, keypoints (P,26,2),
scores (P,26), frame)` or `None`; `halpe_to_body25(keypoints, scores) -> (...,25,3)`;
`HALPE_TO_BODY25`. Task card for Nov 10: `docs/task_cards/louise_2026-11-10_bench_fps.md`.

**Your laptop is a Mac (no CUDA):** develop on CPU; official fps is measured on the NVIDIA
laptops (Marty, Revallo, Freya, Theresia run your `bench_fps.py`).

**You need from others:**
| What | From | Stand-in now |
|---|---|---|
| `pick_child` | Marty (Nov 15) | `common.pick.pick_child` stub (first person) |
| 27 team clips | Theresia (Nov 13) | your own webcam recordings |

**Who waits for you:** Daryn (integration Nov 22), Revallo (team clips through `--video`),
Freya (live points in the GUI).

## Tasks (write `Done YYYY-MM-DD` in Status in the same PR)
| Due | Task | Level | Done when | Status |
|---|---|---|---|---|
| Nov 8 | Milestone 1 | Easy | First line `files: 4661 \| children: 106 \| attempts: 1555` | |
| Nov 10 | rtmlib BodyWithFeet (balanced) on your Mac webcam (CPU); write `live/bench_fps.py`; 4 NVIDIA owners run it with CUDA; pick the live laptop | Medium | fps table for the 4 NVIDIA laptops; live laptop chosen | |
| Nov 11 | `live/convert.py`: check `HALPE_TO_BODY25` against mmpose halpe26.py; plot 1 labelled frame | Easy | Plot correct (right wrist = point 4) | |
| Nov 15 | `capture.py` + `pose.py`: 720p30 capture thread, auto-framing off, unmirrored, time stamp per frame, resample to 30 fps, output a Try; plus `--video` file mode | Hard | Recorded clip → valid Try file | |
| Nov 19 | Pick the child live with Marty's `pick_child()`: lock on the person at the start line, hold the track when someone walks in | Hard | 2 people in the picture: the track doesn't jump | |
| Nov 22 | Performance on the live laptop (with its owner): fps with video saving on, dim-room test; help Daryn join the parts | Medium | 15 fps or more with everything on (ideally 30) | |

**Watch out:** don't import rtmlib at module level (CI has none). Run detection on the
unmirrored frame. rtmlib scores can exceed 1 (`halpe_to_body25` clips).
