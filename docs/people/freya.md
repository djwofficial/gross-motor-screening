# Freya: the app window and testing mode

**Your part:** the one GUI window the operator uses on testing day: keys, states, saving every
try, video, testing mode, automatic mode with confirm/correct. You also lead the dry run.

**Your files:** `app/ui.py`, `app/session.py`, new tests `tests/test_app_*.py`.
Not yours: `app/main.py` (Daryn joins the parts there; ask him for a hook you need).

**Signatures (AGENTS.md):** `Session(out_dir, child, age_months, video).save_try(tr, *, task,
leg, guessed_task, guessed_leg, task_confidence, corrected, score, confidence, clues, reasons,
raw26=None) -> Path`; `load_record(path)`; `LOG_FIELDS`.

**You need from others (use the stand-in until it lands):**
| What | From | Stand-in now |
|---|---|---|
| live points | Louise (`live/pose.py`) | `live.pose.FakePoseStream` |
| start/end events | Theresia (`live/segment.py`) | `Segmenter` placeholder (works on fake data) |
| readiness messages | Theresia (`live/quality.py`) | `check()` placeholder |
| task guess, score, reasons | Marty, Revallo | stubs in `model/`, `features/` |

**Who waits for you:** Daryn (integration Nov 22), everyone (dry run Nov 25).

## Tasks (write `Done YYYY-MM-DD` in Status in the same PR)
| Due | Task | Level | Done when | Status |
|---|---|---|---|---|
| Nov 8 | Milestone 1 (`python scripts/summarize_group_B.py`) | Easy | First line `files: 4661 \| children: 106 \| attempts: 1555` | |
| Nov 11 | GUI skeleton (`app/ui.py`, `session.py`) with fake points: keys 1/2/3, L/R, Space, Enter, Backspace, N, V; state machine ready / recording / confirm | Medium | All keys work and the state is correct | |
| Nov 13 | One JSON per try (`K03_task2_try4_L.json`: time stamps, 26 raw points, 25 converted points) and `log.csv` with `LOG_FIELDS` | Easy | 5 fake tries → 5 JSON files and 5 log rows | |
| Nov 15 | Video saving in its own thread, only when V is on; file name carries the child code | Medium | fps doesn't drop compared with V off | |
| Nov 19 | Testing mode: no score on screen; enter child code and age in months; show the readiness check before the first child; all log fields filled | Easy | Screen never shows a score; readiness problems in plain words | |
| Nov 21 | Automatic mode: show ready / recording / guessed task and leg; Enter confirms, 1/2/3 and L/R correct, Backspace cancels; manual Start/Stop stays as fallback | Medium | Both modes write the same JSON and log rows on fake and recorded clips | |
| Nov 22 | Crash-safe: every try saved immediately, no double key presses; help Daryn join the parts | Medium | Kill the app mid-session and earlier tries are safe | |
| Nov 25 | Lead the dry run: 3 people × 3 tasks × 3 tries, automatic mode with the operator correcting | Medium | 27 JSON + 27 log rows; count of tries automatic mode caught | |
| Optional | Demo mode: score and stick figure on screen | Easy | Only after CP6 (Nov 27) | |

**Watch out:** the live laptop is probably yours (NVIDIA); keep the GUI light so pose keeps
15+ fps. Draw the mirrored picture for display only. Never show a score in testing mode.
