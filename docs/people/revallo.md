# Revallo: live pose module (critical path) and long-jump clues

**Your part:** the eyes of the app: webcam → RTMPose → 25 BODY_25 points, fast enough and stable, with
the right child followed. Everything live depends on this. After capture works, you also take the
timeline and clues for the standing long jump.

**Your files:** `live/pose.py`, `live/convert.py`, `live/capture.py`, `live/bench_fps.py`,
`features/clues.py` (task 3, in its own functions), new tests `tests/test_live_*.py`.

**Signatures (AGENTS.md):** `PoseStream(source, mode, device).read() -> (t, keypoints (P,26,2),
scores (P,26), frame)` or `None`; `halpe_to_body25(keypoints, scores) -> (...,25,3)`;
`HALPE_TO_BODY25`; for task 3 the `timeline` and `compute` signatures of `features/clues.py`.

**Your laptop has NVIDIA:** develop live on it. The official check is on Freya's laptop.

**You need from others:**
| What | From | Stand-in now |
|---|---|---|
| `pick_child` | Marty (Nov 15) | `common.pick.pick_child` stub (first person) |
| 27 team clips | Theresia (Nov 13) | your own webcam recordings |
| dataset in shared format | Marty (Nov 11) | `make_fake_try` |

**Who waits for you:** Daryn (integration Nov 22), Louise and Theresia (team clips as points Nov 15),
Louise (task 3 columns Nov 17), Freya (live points and video saving in the GUI).

## Tasks (status is on the [project board](https://github.com/users/djwofficial/projects/3); `Closes #<issue>` in your PR closes the task)
| Due | Task | Level | Done when | Issue |
|---|---|---|---|---|
| Nov 8 | Milestone 1 | Easy | Same first line | #26 |
| Nov 10 | 10-minute check on Freya's laptop (the live laptop): rtmlib BodyWithFeet (balanced) with CUDA and a person in the picture (live/bench_fps.py) | Easy | CUDA is used and fps is 15 or more | #27 |
| Nov 11 | live/convert.py: check HALPE_TO_BODY25 against mmpose halpe26.py; plot 1 labelled frame to check left and right | Easy | Plot is correct (right wrist = point 4) | #28 |
| Nov 13 | live/capture.py and pose.py: 720p30 capture thread, auto-framing off, unmirrored picture, time stamp per frame, resample to 30 fps, output a Try; plus a --video file mode for recorded clips (moved up from Nov 15) | Hard | Recorded clip → valid Try file | #29 |
| Nov 15 | Video saving in its own thread, only when V is on; file name carries the child code | Medium | fps doesn't drop compared with V off | #30 |
| Nov 15 | Run the 27 team clips through --video mode and hand the points to Louise and Theresia | Easy | 27 valid Try files | #31 |
| Nov 17 | Task 3 timeline and clues, in their own functions in features/clues.py: moments (before, air, after), feet together, time in the air, jump size. Task 3 was the weakest in the first test, so it comes first after capture | Medium | 3 clips match the by-eye counts | #32 |
| Nov 19 | Pick the child live: use Marty's pick_child(), lock on the person at the start line, hold the track when someone else walks into the picture | Hard | Test with 2 people in the picture: the track doesn't jump to the other person | #33 |
| Nov 22 | Performance on the live laptop (with Freya): fps with video saving on, a dim-room test if time allows; help Daryn join the parts | Medium | 15 fps or more with everything on (ideally 30) | #34 |

**Watch out:** don't import rtmlib at module level (CI has none). Run detection on the unmirrored
frame. rtmlib scores can exceed 1 (`halpe_to_body25` clips them). If the Nov 10 check fails, tell
Daryn the same day: the fallback is MediaPipe.

Task 3 was the weakest in the first rough test (2026-11-09, one-person files): macro-F1 0.40, against
0.27 for always guessing the most common score. The paper never defines "a long distance", and one
camera sees forward distance poorly (that second reason is a guess).
