# Louise: moment timeline, movement clues, evaluation

**Your part:** turn a try into "what happened when" (the timeline: foot down at 2.0 s, wobble at
4.1 s) and into the measurements the model scores on, for stand on one leg and hop on one leg. You
also put together the clue table, the reason text and the evaluation report.

**Your files:** `features/clues.py` (tasks 1 and 2; Revallo adds task 3 in its own functions),
`eval/report.py`, new tests `tests/test_clues_*.py`.

**Your laptop is a Mac:** this work needs no GPU.

**Signatures (AGENTS.md):** `timeline(task, points, t) -> list[{"start_s", "end_s", "state"}]`;
`compute(task, points, t) -> dict` with exactly `CLUE_NAMES[task]`;
`explain(task, clues, events=None) -> str`.
Units: seconds and torso lengths (`common.geometry.normalize`, y points up), never pixels.

**You need from others:**
| What | From | Stand-in now |
|---|---|---|
| dataset in shared format | Marty (`data/load_b.py`, Nov 11) | `data.load_b.read_file` + `common.format.make_fake_try` |
| replay | Freya (`viz/replay.py`, Nov 10; timeline bar Nov 13) | plot frames with matplotlib |
| by-eye sheet | Theresia (Nov 11) | count 3 clips yourself |
| team clips as points | Revallo (Nov 15) | dataset clips |
| task 3 columns of the clue table | Revallo (Nov 17) | leave them for last |
| confusion matrices | Marty (Nov 22) | — |

**Who waits for you:** Daryn (clue table Nov 18 for the model: the tightest handoff in the plan),
Marty (task classifier), Freya and Daryn (`explain` for the log).

## Tasks (status is on the [project board](https://github.com/users/djwofficial/projects/3); `Closes #<issue>` in your PR closes the task)
| Due | Task | Level | Done when | Issue |
|---|---|---|---|---|
| Nov 8 | Milestone 1 | Easy | Same first line | #19 |
| Nov 14 | Task 1 timeline and clues in features/clues.py: moments (both feet, steady, wobble), longest time on one foot (s), sway (torso lengths), touch-downs, wobble time. Build the timeline first and compute the clues from it | Medium | 3 clips match the by-eye counts, including the second a foot comes down | #20 |
| Nov 17 | Task 2 timeline and clues: moments (hop, pause, both feet), hop count, longest pause | Hard | 3 clips match the by-eye counts | #21 |
| Nov 18 | Clue table for all 3,359 files, one row per file (task 3 columns from Revallo), sent to Daryn and Marty | Medium | No empty values, or every filled-in value is written down | #22 |
| Nov 19 | features/clues.py: explain(task, clues, events) → short reason text for the log, for example 4.2 s on one foot, foot down at 2.0 s | Easy | Reason text for 3 example clips matches the by-eye sheet | #23 |
| Nov 22 | Check the clues and timeline on our own team clips (Theresia's 27 recordings, through Revallo's --video mode); tune thresholds | Medium | Most clips match the by-eye counts; clips that differ are listed | #24 |
| Nov 25 | eval/report.py: confusion matrix per task, per-fold numbers, mix-ups compared with the scoring rules | Medium | One report covering all 3 tasks | #25 |

**Watch out:** build the timeline first and compute the clues from it, so every number has one
source. The dataset has one expert score per try and none per moment, so the timeline is measured,
not learned, and it never says a child failed. The same code must work on dataset points (1920x1080,
OpenPose) and live points (1280x720, RTMPose). `leg` is the support leg.

First rough test on task 1 (one-off script, 2026-11-09, one-person files, not checked by eye): the
height gap between the ankles over 0.15 torso lengths, ignoring flickers under 0.2 s, gave a median
longest time on one foot of 1.2 s (Bad), 2.3 s (Good), 3.7 s (Perfect). That 0.15 is a guess: tune it.
