# Revallo: movement clues and evaluation

**Your part:** the clues (features) the model scores on, the short reason text in the log,
and the evaluation report.

**Your files:** `features/clues.py`, `eval/report.py`, new tests `tests/test_clues_*.py`.

**Signatures (AGENTS.md):** `compute(task, points, t) -> dict` with exactly
`CLUE_NAMES[task]`; `explain(task, clues) -> str`.
Units: seconds and torso lengths (`common.geometry.normalize`, y points up), never pixels.

**You need from others:**
| What | From | Stand-in now |
|---|---|---|
| dataset in shared format | Marty (`data/load_b.py`, Nov 11) | `data.load_b.read_file` + `common.format.make_fake_try` |
| replay for the by-eye sheet | Theresia (`viz/replay.py`, Nov 10) | plot frames with matplotlib |
| team clips through the live pipeline | Theresia (Nov 13), Louise (`--video`, Nov 15) | dataset clips |
| confusion matrices | Marty (Nov 22) | — |

**Who waits for you:** Marty (clue table Nov 18 for the forest), Freya/Daryn (`explain` for the log).

## Tasks (write `Done YYYY-MM-DD` in Status in the same PR)
| Due | Task | Level | Done when | Status |
|---|---|---|---|---|
| Nov 8 | Milestone 1 | Easy | First line `files: 4661 \| children: 106 \| attempts: 1555` | |
| Nov 11 | By-eye sheet for 9 dataset clips (3 per task) with Theresia's replay: seconds on one foot, hops, feet together or not | Easy | A 9-row table exists | |
| Nov 14 | Task 1 clues: longest time on one foot (s), sway (torso lengths), touch-downs | Medium | 3 clips match the by-eye counts | |
| Nov 17 | Task 2 clues (hop count, longest pause) and task 3 clues (feet together, time in the air, jump size) | Hard | 6 clips match the by-eye counts | |
| Nov 18 | Clue table for all 3,359 files, one row per file, sent to Marty | Medium | No empty values, or every filled-in value written down | |
| Nov 19 | `explain(task, clues)` → short reason text, e.g. "4.2 s on one foot, 0 touch-downs" | Easy | Reasons for 3 example clips match the by-eye sheet | |
| Nov 22 | Check clues on the 27 team clips (through Louise's `--video` mode); tune thresholds | Medium | Most clips match; differing clips listed | |
| Nov 25 | `eval/report.py`: confusion matrix per task, per-fold numbers, mix-ups vs the scoring rules | Medium | One report covering all 3 tasks | |

**Watch out:** the same code must work on dataset points (1920x1080, OpenPose) and live points
(1280x720, RTMPose): that is why everything is in torso lengths. `leg` is the support leg.
