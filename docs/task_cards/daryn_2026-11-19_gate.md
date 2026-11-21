# Task card: Daryn, 2026-11-19: model gate, random forest per task beats the baseline (#9)

Paste this whole card into your AI tool together with AGENTS.md. You can commit and push directly to `main` (or use a branch/PR) with `Closes #9`.

**Goal:** one random forest per task, trained on the full group B clue table, that beats the
most-common-score baseline on macro-F1 and quadratic weighted kappa (GroupKFold by child). Save the
models so `model.predict()` gives the live app a score and a confidence.

**Input → output:**
- In: `runs/clue_table.csv` (one row per group B file, built by Louise's
  `scripts/make_clue_table.py` with Marty's `pick_child`; not in git).
- Out: console report per task, plus `models/task{1,2,3}_rf.joblib` =
  `{"model", "features"}` with `features = CLUE_NAMES[n]` without `duration_s`.

**Signature (don't change):** `predict(task, clues) -> (score 0/1/2, confidence 0..1)`; model file
format above.

**Files you may change:** `model/train_rf.py`, `model/predict.py`, `models/`, this card.

**Steps:**
1. Rebuild the clue table with the current clue code:
   `python scripts/make_clue_table.py --tasks 1 2 3 --out runs/clue_table.csv`
2. Gate and save: `python -m model.train_rf --table runs/clue_table.csv --save`
3. Check `predict` on a fake try per task, `python -m app.main --fake`, `rescore.py`, `pytest`.

**Done when:** macro-F1 and kappa above the baseline for all 3 tasks; .joblib files exist; predict
returns a score and a confidence.

## Evidence (2026-11-19, main @ 132ef20, which includes Louise's hop rule 43c0e33)

Clue table: 3,356 rows (3 files under 30 frames left out). Compared with the 22:42 table, only task 2
changed (`hops` in 387 rows, `longest_pause_s` in 151 rows), from the new rule that a hop needs the
lifted foot 0.3 torso lengths up.

`python -m model.train_rf --table runs/clue_table.csv --save`, GroupKFold(5) by child, child leak
check PASS for all three tasks:

| Task | Files | Children | Random forest macro-F1 | kappa | Most common score macro-F1 | kappa | Beats baseline |
|---|---|---|---|---|---|---|---|
| 1 one leg | 1,205 | 98 | 0.62 ± 0.06 | 0.58 ± 0.09 | 0.19 ± 0.02 | 0.00 | yes |
| 2 hop | 1,055 | 96 | 0.66 ± 0.00 | 0.66 ± 0.06 | 0.20 ± 0.03 | 0.00 | yes |
| 3 long jump | 1,096 | 103 | 0.36 ± 0.05 | 0.19 ± 0.11 | 0.26 ± 0.02 | 0.00 | yes |

Task 2 per fold: macro-F1 0.66 / 0.65 / 0.66 / 0.66 / 0.65 (the ± 0.00 is real, not a bug).
Task 2 before the new hop rule: 0.60 ± 0.04 / 0.59 ± 0.05.

Model files: task1 1.2 MB, task2 0.6 MB, task3 1.0 MB (all under 10 MB). Features: task 1
`one_foot_s, touchdowns, sway, wobble_s`; task 2 `hops, longest_pause_s, touchdowns`; task 3
`feet_gap_takeoff, feet_gap_landing, air_s, jump_size`. No `duration_s`. Task 1 and 3 files are
byte-identical to the 22:42 ones (same table rows, fixed seed); only task 2 was retrained.

`predict` on `make_fake_try` per task: (2, 0.84), (0, 0.86), (2, 0.82).
`python -m app.main --fake --task {1,2,3}`: 3 JSON files + 3 log rows; `python rescore.py` on that
folder: 3 of 3 match the live score. `ruff format --check`, `ruff check`: clean. `pytest`: 213 passed.

**Per-second forest for task 2: not tried (decision, Daryn 2026-11-19).** The first rough test
(2026-11-09, one-person files only, old clues, `docs/project.md`) gave 0.65 for the per-second
forest. The per-file forest on the new clues now gives 0.66 ± 0.00, so the per-second forest is
unlikely to win by enough to justify the time before integration (#10, Nov 22). Revisit after CP6
if there is time; `model/train_rf.py` keeps the note.

**Watch out:**
- From now on the models are frozen for the dry run. Any retrain (new clue rule, new table) means
  every saved live try must be rescored, otherwise #11 (27/27 match) fails. Tries saved before
  2026-11-19 23:39 used the old task 2 model or no model at all.
- Task 3 stays weak (kappa 0.19), as the risk table expected; below `CHECK_BY_TEACHER_BELOW` the
  log says "check by teacher".

## Evidence (2026-11-20, Phase 1 + Phase 2 clue update & re-freeze before Nov 25 dry run)

Clue changes (`features/clues.py`), keeping `make_forest()` unchanged (`min_samples_leaf=3, max_depth=None, class_weight="balanced"`):
- **Task 3 (`Hybrid-C` + unified frontal/side `jump_size`):** `_JUMP_WINDOW_S=0.45`, `_MAX_FRAME_GAP_S=0.25`, `_MIN_AIR_S=0.06`, `(L_HIP + R_HIP)/2` fallback + short interior NaN gap interpolation (`<= 0.15 s`), chord airborne membership (`>= 0.04` ankles, `>= 0.08` hip) validated by peak clearance above both chord (`>= 0.06` ankles) and strict max (`>= 0.03` ankles, `>= 0.12` hip); unified `jump_size = hypot(dx, 1.5 * max(0, dy) + 2.5 * max(0, scale_gain))` over `0.35 s` grounded windows. All-zero tries drop from `22%` (`Perfect 19%`) to `8.6%` (`94/1,096`, `Perfect 4.3%`).
- **Task 2 (`Fix A`):** bounded hop troughs (`HALF_HOP_S=0.25`), `PAUSE_GAP_S=0.45`, and between-hop `touchdowns` (`_BOTH` events between first hop end and last hop start).
- **Task 1 (`+4` core stagger clues):** appended `hip_speed`, `neck_speed`, `lift_std`, `wrist_speed` to `CLUE_NAMES[1]` (measured in the trimmed core of the longest one-foot stretch).

Clue table rebuilt (`scripts/make_clue_table.py --tasks 1 2 3 --out runs/clue_table.csv`): `3,356` rows (`3` under 30 frames left out, `0` without a torso).

`python -m model.train_rf --table runs/clue_table.csv --save` (seed 0, `GroupKFold(5)` by child, child leak check **PASS** for all 3 tasks):

| Task | Files | Children | Seed-0 macro-F1 | Seed-0 kappa | Baseline macro-F1 | Beats baseline |
|---|---|---|---|---|---|---|
| 1 one leg | 1,205 | 98 | 0.64 ± 0.07 | 0.60 ± 0.08 | 0.19 ± 0.02 | yes |
| 2 hop | 1,055 | 96 | 0.68 ± 0.01 | 0.70 ± 0.06 | 0.20 ± 0.03 | yes |
| 3 long jump | 1,096 | 103 | 0.51 ± 0.02 | 0.46 ± 0.05 | 0.26 ± 0.02 | yes |

5-seed mean (`seeds 0..4`, fold-averaged `GroupKFold(5)` by child) on `runs/clue_table.csv`:

| Task | 5-Seed macro-F1 | 5-Seed kappa | Good Recall | Cam 1 F1 | Cam 1 kappa | `cov60` / `acc60` | Cam 1 Perfect Recall |
|---|---:|---:|---:|---:|---:|---|---:|
| 1 one leg | 0.641 | 0.600 | 62.6% | 0.633 | 0.599 | 0.60 / 0.73 | 75.5% |
| 2 hop | 0.678 | 0.709 | 53.1% | 0.688 | 0.738 | 0.82 / 0.76 | 78.7% |
| 3 long jump | 0.521 | 0.465 | 28.9% | 0.526 | 0.477 | 0.57 / 0.71 | 71.6% |

Task classifier (`python -m model.task_classifier --save`, `3,356` files, `106` children, `16` features, leak check **PASS**):
- Overall accuracy `0.8728` (`0.8728 ± 0.0386`), overall macro-F1 `0.8724` (`0.8712 ± 0.0386`).

Re-frozen model files (all `< 10 MB`):
- `models/task1_rf.joblib` (`1.16 MB`): `one_foot_s, touchdowns, sway, wobble_s, hip_speed, neck_speed, lift_std, wrist_speed`
- `models/task2_rf.joblib` (`0.52 MB`): `hops, longest_pause_s, touchdowns`
- `models/task3_rf.joblib` (`1.06 MB`): `feet_gap_takeoff, feet_gap_landing, air_s, jump_size`
- `models/task_classifier.joblib` (`2.03 MB`): 16 features (unchanged)

Verification checks:
- By-eye sheet (`docs/eye_check/by_eye_sheet.csv`): Task 1 `3/3` (`0.00 / 2.30 / 3.93 s`), Task 2 `6/7` (`B213` stays `0` vs `1`), Task 3 `5/5` (`2 / 1 / 1 / 1 / 1` air events).
- Adult team clips: `9/9` `Move_3` clips in `docs/task_cards/team_clips_27_try.zip` detected cleanly (including `2, 2, 3` episodes on Revallo's 3 clips, zero phantom crouch jumps); non-zero pauses on `3/3` `runs/team_clips_video_test/T03_task2_try{1,2,3}_L.json` (`0.567 / 0.733 / 0.500 s`).
- `predict` on `make_fake_try` per task: `(2, 0.54)`, `(0, 0.81)`, `(1, 0.36)`.
- `python -m app.main --fake --task {1,2,3}` + `python rescore.py`: `3` of `3` match.
- `python -m model.baseline --matrices` (both with and without `--table runs/clue_table.csv`): PASS, cold load + predict `~27–31 ms` (`< 1 s`).

**Re-freeze note (2026-11-20):** All 4 models (`task1_rf.joblib`, `task2_rf.joblib`, `task3_rf.joblib`, `task_classifier.joblib`) are re-frozen before the Nov 25 dry run (`#11/#43`). Any JSON runs saved before `2026-11-20` used the old clues/models and must be rescored or treated as pre-update.

**Tests:** `tests/contracts/test_model.py`, `tests/contracts/test_clues.py`, `tests/test_train_rf.py`, `tests/test_clues_task1.py`, `tests/test_clues_task2.py`, `tests/test_clues_task3_frontal.py`, `tests/test_live_long_jump.py`, and `tests/test_model_marty.py` stay green.
