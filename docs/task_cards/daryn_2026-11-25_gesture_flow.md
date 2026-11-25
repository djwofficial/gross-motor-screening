# Task card: Daryn, 2026-11-25: gesture flow in the score window

**Goal:** In the window that `Open GUI.command` / `Open GUI.bat` opens (`app.ui --show-score`), recording
starts with an open palm and a 3-2-1 countdown, one recording can hold several tasks in a row
(a card per attempt: task, score, time), and crossed arms (X) stop it. Without the palm, attempts are
still detected and shown, but not saved.

**Input → output:** live BODY_25 points of the tracked child (unmirrored pixels) + the camera frame →
gesture events (`live/gesture.py`), attempts (`live/bouts.py`), one saved try per attempt (same JSON +
`log.csv` as before, `meta.mode = "gesture"`, `meta.attempt`, shared `video_file`).

**Files:** new `live/gesture.py`, `live/bouts.py`, `tests/test_live_gesture.py`,
`tests/test_gesture_flow.py`; changed `app/demo.py`, `app/ui.py` (Controller: `gesture` mode,
`countdown` state, `Runner._write_try`), `app/screen.py`, `app/main.py`, `tests/test_main.py`,
`AGENTS.md`, `README.md`. Testing mode without a score is unchanged; the old flow is `--mode auto`.

**Thresholds (all GUESSES, tune on the live demo):**
| where | name | value |
|---|---|---|
| `live/gesture.py` | `PALM_MIN_HEIGHT` (wrist vs its shoulder, + wrist above elbow) | −0.15 torso (was 0.5: above the head, never fired on the first live try) |
| | `PALM_HOLD_S` / `CROSS_HOLD_S` | 0.7 s / 0.8 s |
| | `HAND_MIN_SCORE` (per finger, PIP and tip) / `FINGER_RATIO` | 0.3 (was 0.4 mean) / 1.15 |
| | `SHOULDER_TO_TORSO` (ruler when the hips are out of the picture) | 1.8 |
| `live/bouts.py` | `READY_STILL_S` / `END_STILL_S` | 0.6 s / 1.0 s (both feet down) |
| | `STAND_BEFORE_HOP_S` | 2.0 s |
| `app/demo.py` | `DONE_CLUE` (an attempt counts as a task) | T1 `one_foot_s` ≥ 1.0, T2 `hops` ≥ 2, T3 `air_s` > 0 |
| | `COUNTDOWN_S` / `INTERMEZZO_S` / `SUMMARY_S` | 3.0 / 2.5 / 6.0 s |

**Done when / evidence (2026-11-25, Mac, CPU):**
- `pytest`: 356 passed, 2 skipped (after the first-live-try fixes); `ruff format .` + `ruff check .` clean.
- `tests/test_gesture_flow.py`: synthetic recording, palm → countdown → stand → hop → jump → X:
  3 tries saved with tasks [1, 2, 3], 3 rows in `log.csv`, 3 attempt cards, summary "3 tries".
  Without the palm: 1 attempt shown as "not recorded", 0 saved. Backspace cancels the countdown.
  Testing-mode render shows no score in any gesture state.
- `live.bouts` on fake tries 1-2-3 in a row: 3 attempts, each around its own movement.
  Stand straight into hops (no foot down) is cut into 2 attempts at the first hop.
- Hand model: `python -m live.gesture --download` fetched
  `rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256` (48.9 MB) into `~/.cache/rtmlib/hub/checkpoints`;
  one check of a 120×120 crop on the Mac CPU takes 10.2 ms (median of 30), at most 10 per second
  and only while a hand is raised; a noise crop scores 0.14-0.26 and is rejected (not open).
- [ ] **Live check (to do):** double-click `Open GUI.command`, palm → 3-2-1 → stand 3 s → hop 3 s →
  jump → X. Note here: palm fires (yes/no, how many seconds), 3 cards, summary, 3 rows in
  `runs/<date>/log.csv`, fps in the top bar while idle with a raised hand. If the ring stays amber
  ("Open your hand"), lower `HAND_MIN_SCORE`.
- [ ] Run `python -m live.gesture --download` on the Windows + NVIDIA laptop too (needs internet).

**First live try (2026-11-25):** the palm did not fire. Fixes: the palm also works on the readiness
screen, a hand at face or chest height counts, feet/hips out of the picture no longer block, the
finger check is per finger (0.3). The ring now says what is missing ("Open your hand · 2/4 fingers").

**Watch out:** gestures are read only from the tracked child (ChildTracker); a teacher outside the
lock uses Space. The X and the palm do not fire while one foot is clearly up (one leg, hopping). An attempt below `DONE_CLUE` (e.g. walking back) is not shown and not saved.
