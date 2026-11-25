# Task card: Marty, 2026-11-26: cyber HUD look for the score window

Daryn agreed that Marty changes his files for this (app/screen.py, model/predict.py, two tests).
There is no GitHub issue for it.

**Goal:** restyle the `app.ui --show-score` / Open GUI window (`app/screen.py`, on top of Daryn's
gesture flow f0729f5) as a space-cyber game HUD. Every behaviour stays the same: keys, clickable
buttons, palm/Space start, Backspace cancel, X stop, saving, log.csv.

**Input → output:** the Runner's state (unchanged) → the window picture. A new helper
`model.predict.predict_proba(task, clues) -> (p_bad, p_good, p_perfect)` feeds the evaluation bar
through `live["proba"]` (app/demo.py).

**Signature (don't change):** `predict(task, clues)`, `render(runner, view, show_score)`,
`buttons`, `hit`, `run_window`.

**Files changed:** `app/screen.py`, `model/predict.py` (new function only), `app/demo.py`
(`proba` in the live dict), `app/fonts/` (Orbitron + Rajdhani, SIL OFL 1.1, from
github.com/google/fonts), `scripts/preview_cyber_ui.py`, `tests/test_app_cyber_hud.py`, and the
two tests below that checked the old look.

**Phases:** floor plate (fake-3D trapezoid, tilt constants `PLATE_*` are GUESSES) → READY fill
(both ankles ≥ 0.3 on the plate, visual only) → glitching 3-2-1 in a segmented ring → GO →
task panel (top right; glitch swap when the task changes) + evaluation bar (left, eased live
probabilities) → result box (opens from the centre, then flies into the sidebar) → sidebar rows
(newest on top, flash on landing). Testing mode: same look but no bar, score, confidence or sidebar.
Animations run on `time.monotonic` (`screen._clock`) or on the Runner's stream seconds, never on
frame counts.

**Tests changed (they checked the old look, not behaviour):**
- `tests/test_app_demo_auto.py::test_detecting_chip_only_with_show_score`: the "Detecting: …"
  chip is now the task panel ("TASK 3", "STANDING LONG JUMP", "Detected · 87%"). The testing-mode
  check (`"detecting" not in hidden`) is kept, and "87%" is now checked too.
- `tests/test_framing_and_demo_ux.py::test_floating_shoulder_score_badge_…` became
  `test_eval_bar_shows_live_probabilities_and_stays_off_in_testing_mode`, because the badge was
  removed on purpose. The testing-mode no-score check is kept and extended.

**Done when / evidence:** see the PR. `ruff format .`, `ruff check .` and `pytest` all pass.
`python -m app.main --fake` runs. `python scripts/preview_cyber_ui.py` writes every phase and 3
frames of each transition to `runs/ui_preview/` (git-ignored).

**Evidence (2026-11-26, Windows, .venv Python 3.12):**
- `ruff format .`, `ruff check .`: clean. `pytest`: all tests pass, including 9 new ones in
  `tests/test_app_cyber_hud.py` (the predict_proba argmax matches predict for tasks 1-3).
- `python -m app.main --fake`: 1 try saved, log.csv written.
- Render time of one picture at 1920x1080, camera view, recording (30 renders): **49 ms** new vs
  **63 ms** on main. The vignette and scan lines are one cached uint8 multiply, the floor plate is
  one mask, and there is no wasted background fill.
- `python scripts/preview_cyber_ui.py`: 60 PNGs in `runs/ui_preview/` (show_score and `t_*`
  testing mode). They were checked by eye.
