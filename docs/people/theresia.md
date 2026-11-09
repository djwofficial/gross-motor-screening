# Theresia: test clips, readiness, start and end, privacy, testing-day tables

**Your part:** the test clips and by-eye answers everyone checks against, the readiness check, the
start area, automatic start and end of a try, face blurring, and the tables that give the testing-day
result.

**Your files:** `viz/blur.py`, `live/segment.py`, `live/quality.py`, and new files
`live/start_area.py`, `eval/agreement.py`, `eval/dry_run_count.py`, new tests
`tests/test_segment_*.py`, `tests/test_viz_*.py`, `tests/test_eval_*.py`.

**Signatures (AGENTS.md):** `Segmenter().update(points (25,3) or None, t) ->
"idle"/"ready"/"start"/"recording"/"end"`; `check(points, fps, frame_size, start_area) ->
list[str]`.
The current `Segmenter` is a working placeholder (finds 18/18 fake tries); tune it on real clips.

**You need from others:**
| What | From | Stand-in now |
|---|---|---|
| replay | Freya (`viz/replay.py`, Nov 10) | plot frames with matplotlib |
| dataset in shared format | Marty (Nov 11) | `data.load_b.read_file`, `make_fake_try` |
| team clips as points | Revallo (`--video`, Nov 15) | fake tries |

**Who waits for you:** Louise (by-eye sheet Nov 11), Revallo and Louise (team clips Nov 13), Freya
(readiness and start/end for the GUI), KEE (20 practice clips Nov 19), Daryn (dry-run count Nov 25).

## Tasks (status is on the [project board](https://github.com/users/djwofficial/projects/3); `Closes #<issue>` in your PR closes the task)
| Due | Task | Level | Done when | Issue |
|---|---|---|---|---|
| Nov 8 | Milestone 1 | Easy | Same first line | #45 |
| Nov 11 | By-eye sheet for 9 dataset clips (3 per task) using Freya's replay: seconds on one foot, the second a foot comes down, number of hops, feet together or not | Easy | A 9-row table exists | #46 |
| Nov 13 | Record team clips (720p30, auto-framing off): 3 people × 3 tasks × 3 tries, some with another person walking past, plus a by-eye count sheet. This is the test data for Revallo and Louise | Medium | 27 videos and 1 by-eye count table | #47 |
| Nov 17 | live/quality.py: readiness check from the points (whole body in the picture, fps, light from point confidence, feet in the start area) | Easy | 3 deliberately bad setups from team clips each get the right message | #48 |
| Nov 17 | live/start_area.py: the operator marks the start area on the camera picture; the readiness check and automatic start use it | Easy | The area is saved and the readiness check uses it | #49 |
| Nov 19 | live/segment.py: automatic start and end from the points (still about 1 s at the start line = ready, movement = start, still about 1.5 s or leaving = end); tune on the 27 team clips | Medium | Start and end found for most of the 27 team clips; misses listed | #50 |
| Nov 19 | About 20 group B stick-figure clips for KEE calibration, spread over the 3 tasks and 3 scores, expert score hidden | Easy | 20 mp4 files and a separate answer key | #51 |
| Nov 22 | viz/blur.py: blur faces using RTMPose head points (nose, eyes, ears), then check frame by frame | Hard | 3 team videos: no frame shows a face | #52 |
| Nov 25 | eval/dry_run_count.py: count how many of the 27 dry-run tries automatic mode caught correctly. The result decides the mode for Nov 29 | Easy | One count per task | #53 |
| Nov 25 | eval/agreement.py: app vs KEE and KEE vs KEE tables, and KEE's 20 practice clips vs the answer key. Built and tested on fake data only | Medium | Tables come out of a fake score file | #54 |
| Nov 25 | Replays from the dry-run JSON files (to spot bugs by eye) | Easy | 27 replays made | #55 |

**Watch out:** `*.mp4` is gitignored on purpose. Positions in torso lengths, times in seconds.
`eval/agreement.py` is built and tested on fake data only: testing-day scores never go into an AI.
