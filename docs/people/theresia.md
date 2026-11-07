# Theresia: replay, team clips, readiness, start and end, privacy

**Your part:** the tools that let everyone see the points (replay), the team test clips, the
readiness check, automatic start/end of a try, and face blurring.

**Your files:** `viz/replay.py`, `viz/blur.py`, `live/segment.py`, `live/quality.py`,
new tests `tests/test_viz_*.py`, `tests/test_segment_*.py`.

**Signatures (AGENTS.md):** `Segmenter().update(points (25,3) or None, t) ->
"idle"/"ready"/"start"/"recording"/"end"`; `check(points, fps, frame_size, start_area) ->
list[str]`; `render(tr, out_path)`.
The current `Segmenter` is a working placeholder (finds 18/18 fake tries); tune it on real clips.

**You need from others:**
| What | From | Stand-in now |
|---|---|---|
| dataset in shared format | Marty (Nov 11) | `data.load_b.read_file`, `make_fake_try` |
| team clips as points | Louise (`--video`, Nov 15) | fake tries |

**Who waits for you:** Revallo, Marty, Louise (replay Nov 10), Louise + Revallo (team clips
Nov 13), Freya (readiness + segment for the GUI).

## Tasks (write `Done YYYY-MM-DD` in Status in the same PR)
| Due | Task | Level | Done when | Status |
|---|---|---|---|---|
| Nov 8 | Milestone 1 | Easy | First line `files: 4661 \| children: 106 \| attempts: 1555` | |
| Nov 10 | `viz/replay.py`: Try file → stick-figure mp4 with point numbers | Easy | Plays 1 dataset file and 1 fake file | |
| Nov 13 | Record team clips (720p30, auto-framing off): 3 people × 3 tasks × 3 tries, plus a by-eye count sheet | Medium | 27 videos and 1 by-eye table (team clips are adults: not child data, but keep them out of git) | |
| Nov 17 | `live/quality.py`: readiness check (whole body in picture, fps, light from point confidence, feet in start area) | Easy | 3 deliberately bad setups each get the right message | |
| Nov 19 | `live/segment.py`: automatic start/end (still ~1 s = ready, movement = start, still ~1.5 s or leaving = end); tune on the 27 team clips | Medium | Start and end found for most of the 27 clips; misses listed | |
| Nov 19 | About 20 group B stick-figure clips for KEE calibration, expert score hidden | Easy | 20 mp4 files and a separate answer key | |
| Nov 22 | `viz/blur.py`: blur faces using head points (nose, eyes, ears), check frame by frame | Hard | 3 team videos: no frame shows a face | |
| Nov 25 | Replays from the dry-run JSON files | Easy | 27 replays made | |

**Watch out:** `*.mp4` is gitignored on purpose. Positions in torso lengths, times in seconds.
