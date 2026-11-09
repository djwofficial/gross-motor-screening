# Freya: replay, the app window and testing mode

**Your part:** the stick-figure replay everyone uses to check things by eye, and the one GUI window
the operator uses on testing day: keys, states, saving every try, testing mode, automatic mode with
confirm/correct. You also lead the dry run. Your laptop is the live laptop.

**Your files:** `app/ui.py`, `app/session.py`, `viz/replay.py`, new tests `tests/test_app_*.py`,
`tests/test_viz_*.py`. Not yours: `app/main.py` (Daryn joins the parts there; ask Daryn for a hook).

**Signatures (AGENTS.md):** `Session(out_dir, child, age_months, video).save_try(tr, *, task,
leg, guessed_task, guessed_leg, task_confidence, corrected, score, confidence, clues, reasons,
raw26=None, events=None) -> Path`; `load_record(path)`; `LOG_FIELDS`;
`render(tr, out_path, fps=30, size=None, labels=True, events=None)`.

**You need from others (use the stand-in until it lands):**
| What | From | Stand-in now |
|---|---|---|
| live points | Revallo (`live/pose.py`) | `live.pose.FakePoseStream` |
| saving video | Revallo (Nov 15) | keep the V key working, no file yet |
| start/end events | Theresia (`live/segment.py`) | `Segmenter` placeholder (works on fake data) |
| readiness messages | Theresia (`live/quality.py`) | `check()` placeholder |
| task guess, score, reasons | Marty, Daryn, Louise | stubs in `model/`, `features/` |

**Who waits for you:** Louise, Marty, Theresia (replay, Nov 10: it is due first), Daryn (integration
Nov 22), everyone (dry run Nov 25).

## Tasks (status is on the [project board](https://github.com/users/djwofficial/projects/3); `Closes #<issue>` in your PR closes the task)
| Due | Task | Level | Done when | Issue |
|---|---|---|---|---|
| Nov 8 | Milestone 1 | Easy | Same first line | #35 |
| Nov 10 | viz/replay.py: Try file → stick-figure mp4 with point numbers (used by Louise, Marty and Theresia for checking by eye) | Easy | Plays 1 dataset file and 1 fake file | #36 |
| Nov 11 | GUI skeleton (app/ui.py, session.py) with fake points: keys 1/2/3, L/R, Space, Enter, Backspace, N, V; state machine for ready / recording / confirm | Medium | All keys work and the state is correct | #37 |
| Nov 13 | One JSON per try (K03_task2_try4_L.json: time stamps, 26 raw points, 25 converted points, events) and log.csv (child, age, task, try, leg, guessed task, corrected, score, confidence, reasons) | Easy | 5 fake tries → 5 JSON files and 5 log rows | #38 |
| Nov 13 | Replay: coloured bar under the stick figure showing the timeline, one colour per state, for checking moments by eye | Easy | A try with hand-made events shows the right colours at the right seconds | #39 |
| Nov 19 | Testing mode: no score and no timeline on screen; enter child code and age in months; show Theresia's readiness check before the first child; all log fields filled | Easy | The screen never shows a score or timeline; readiness problems are shown in plain words | #40 |
| Nov 21 | Automatic mode: show ready / recording / guessed task and leg; Enter confirms, 1/2/3 and L/R correct, Backspace cancels the try; manual Start/Stop stays as fallback | Medium | Both modes write the same JSON and log rows on fake and recorded clips | #41 |
| Nov 22 | Crash-safe: every try is saved immediately, no double key presses; help Daryn join the parts | Medium | Kill the app mid-session and earlier tries are safe | #42 |
| Nov 25 | Lead the dry run: 3 people × 3 tasks × 3 tries, in automatic mode with the operator correcting | Medium | 27 JSON files and 27 log rows | #43 |
| Optional | Demo mode: score, stick figure and timeline on screen | Easy | Only after CP6 | #44 |

**Watch out:** keep the GUI light so pose keeps 15+ fps. Draw the mirrored picture for display
only. Testing mode never shows a score or the timeline.
