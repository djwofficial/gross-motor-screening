# Task card: Marty, Nov 23: task 3 experiments (missed jumps and extra clues)

**Goal:** find out why the task 3 (standing long jump) model scores lowest, and test cheap fixes.
**Experiment only: no change to `features/`, `models/` or any existing file.** Proposals at the end
go to Revallo (task 3 clues), Daryn (model) and Theresia/Freya (dry run).

**Input → output:** group B task 3 files (`data/raw/B_skel.zip`, CC0), loaded and picked like
`scripts/make_clue_table.py` (1096 files; 2 under 30 frames left out; 103 children) → printed
tables. Every number below comes from a script in `scripts/experiments/` run on 2026-11-23.
All results: random forest or other model, **GroupKFold(5) by child, seeds 0-4 = 25 runs**,
macro-F1 and quadratic kappa computed like `model.train_rf` (`_macro_f1`, `_kappa`).
Only group B aggregate numbers here, no file-by-file results.

**Files added:** `scripts/experiments/exp_t3_model_swap.py`, `exp_t3_stagger.py`,
`t3_more_clues.py`, `exp_t3_best.py`, `t3_feet_out.py`, `exp_t3_feet.py` (section 9), this card.

## 1. Model swap (`exp_t3_model_swap.py`)
Current 4 clues (`feet_gap_takeoff`, `feet_gap_landing`, `air_s`, `jump_size`).

| model | macro-F1 | kappa |
|---|---|---|
| random forest (`make_forest`) | **0.522** | **0.464** |
| HistGradientBoosting, `class_weight="balanced"` | 0.498 | 0.424 |
| same, `learning_rate=0.05, max_depth=3` | 0.493 | 0.440 |
| StandardScaler + LogisticRegression (`balanced`, `max_iter=2000`) | 0.487 | 0.427 |

The forest stays best: the limit is the clues, not the model. (Boosting gives the same result for
every seed here: no early stopping below 10 000 rows, so the seed changes nothing.)

## 2. Stagger clue (`exp_t3_stagger.py`)
Idea: "feet together" may mean time, not space: `takeoff_stagger_s` / `landing_stagger_s` = time
between the two ankles leaving / touching the ground.
- Macro-F1 0.521 ± 0.007 (4 clues) → 0.534 ± 0.004 (4 + 2 stagger), mean over seeds 0-4.
- But half the take-off staggers (477 of 944) and most landing staggers (772 of 973) are under
  one frame (0.033 s at 30 fps), and the medians per label barely differ (take-off 0.026 / 0.032
  / 0.035 s for Bad / Good / Perfect). At 30 fps this clue mostly measures noise. Not proposed.

## 3. 13 extra clues "A" (`t3_more_clues.py`, fixed copy in `exp_t3_best.py`)
`n_air, best_jump_size, dx, dy, scale_gain, crouch, hip_peak, arm_up, arm_range, land_squat,
land_move, land_step, hands_low` (torso lengths and seconds, from the first jump). In
`exp_t3_best.py` bug 1 is fixed: the pre-jump windows (`pre`, `crouch`) stop before the first air
frame (`t < a0`, was `t <= a0`). `arm_range` is left as is (it runs to a0 + 0.2 s on purpose).
Missing values filled with -9.

## 4. Knee fallback "B" (`exp_t3_best.py`)
When an ankle is missing, its height is filled from the same side's knee: knee height − (that
leg's median knee-ankle height / torso from this file's standing frames) × this frame's torso.
The copied `_task3_events` gives the same timeline as `features.clues` on all 1096 files.
Guesses (not tuned): "standing" = hip-ankle height ≥ 0.9 × its 90th percentile in the file;
fewer than 5 standing frames on a side → no fallback (11 sides); only the height (y) is filled.

- No-jump files (no air event; same with `air_s == 0`): **26 of 44 Good/Perfect now get a jump,
  41 of 50 Bad stay no-jump** (9 Bad files gain a jump).
- By-eye task 3 clips (`docs/eye_check/by_eye_sheet.csv`): jump counts stay 2, 1, 1, 1, 1.

## 5. Result table (`exp_t3_best.py`)

| version | macro-F1 | kappa | Good recall | better than current in (F1 / kappa) |
|---|---|---|---|---|
| current 4 clues | 0.522 | 0.464 | 0.28 | – |
| 4 clues with B | 0.518 | 0.458 | 0.28 | 11/25 / 9/25 |
| **4 + A** | **0.558** | **0.529** | **0.30** | 14/25 / 21/25 |
| 4 + A with B | 0.534 | 0.521 | 0.26 | 15/25 / 25/25 |

**Recommendation: 4 + A without B.** Best mean on all three numbers. B finds missed jumps but
also gives 9 Bad files a jump and does not raise F1. Caution: A beats current in F1 in only 14/25
runs (the kappa gain is steadier, 21/25), so the gain is real but small.

## 6. Finding: the feet leave the picture (`t3_feet_out.py`)
Why does the jump finder miss jumps of Good/Perfect children?

| group | files | median share of frames with an ankle missing (neck + hip seen) | files that lose an ankle right after it was near the bottom edge | median lowest ankle (share of picture height) |
|---|---|---|---|---|
| jump found | 1002 | 0.01 | 414 | 0.96 |
| no jump, Bad | 50 | 0.00 | 17 | 0.86 |
| no jump, Good/Perfect | 44 | **0.41** | **31** | **1.00** |

Guess: "near the bottom edge" = ankle last seen below 90 % of the picture height. In most missed
Good/Perfect jumps the child jumps toward the bottom of the picture and the feet leave it: a
camera-view problem, not a movement problem. It happens on all 3 cameras (164/366, 145/364,
153/366 files lose an ankle at the bottom edge).

**This matters for the Nov 25 dry run:** live, a long jump toward the laptop can carry the feet
out of the picture, and then the jump finder sees no jump even for a good jump. Check it on the
day (are both feet visible from start line to landing in every task 3 try?).

## 7. Proposals (no code changed yet)
1. **Dry run check that the feet stay in the picture** (Theresia `live/quality.py`, Freya dry
   run): for task 3, the readiness check / dry run confirms both feet are in the picture at the
   start line *and* at the landing spot (tape the landing area inside the picture).
2. **"Check by teacher" when the feet left the picture** (Daryn `app/main.py`, Revallo clues):
   if an ankle is missing in a large share of the task 3 try (threshold to be set from the dry
   run; a guess), log "check by teacher" instead of a score, like low confidence does now.
3. **Add the 13 A clues to task 3** (Revallo, `features/clues.py` + `CLUE_NAMES[3]`, tell Daryn):
   4 + A without B, after checking each clue on the by-eye clips.
4. **Untested next idea:** B only when the current finder finds no jump, so the other files stay
   unchanged.

## 8. Ported into `features/clues.py` (2026-11-25, branch `marty/t3-clues-a`)
Proposal 3 done: the 13 A clues (4 + A without B) are appended to `CLUE_NAMES[3]` (first 5 names
unchanged) and computed by `_task3_extra_clues` in `compute(3, ...)`. It uses BODY_25 points
only, on the normalized pose (torso lengths), and puts -9 where a clue can't be measured, so
`compute` always returns finite values. `explain(3, ...)` still uses only the old clues.
`model/task_classifier.py` is unchanged.

**One deliberate difference from `exp_t3_best.py`:** the A windows use `_TIME_EPS` at their edges,
like the rest of the task 3 code. Edges such as a1 + 1.2 s fall exactly on 30 fps frames, so
without it, floating-point rounding decided whether that edge frame was in. That broke
`tests/test_clues_task3_frontal.py::test_exact_20fps_015s_nan_gap_is_clock_offset_invariant`
(`hands_low` changed with the clock offset). Effect, checked on all 1096 group B task 3 files:
- without the eps, the port gives exactly the experiment's values on every file (0 differ)
- with the eps, 370 files have at least one A value that differs (crouch 289, land_squat 285,
  land_move 66, land_step 24, arm_range 20, arm_up 9, hands_low 9)

**Result** on the rebuilt `runs/clue_table.csv` (`python scripts/make_clue_table.py --tasks 1 2 3`;
values rounded to 4 decimals). Same CV as `exp_t3_best.py`: `make_forest`, GroupKFold(5) by child,
seeds 0-4 = 25 runs, 1096 rows, 103 children:

| version | macro-F1 | kappa | Good recall | new better in (F1 / kappa) |
|---|---|---|---|---|
| card §5, current 4 clues | 0.522 | 0.464 | 0.28 | – |
| card §5, 4 + A (no eps) | 0.558 | 0.529 | 0.30 | 14/25 / 21/25 |
| table, old 4 clues | 0.521 ± 0.028 | 0.466 ± 0.052 | 0.28 | – |
| **table, 17 clues (pipeline)** | **0.558 ± 0.057** | **0.528 ± 0.079** | **0.30** | **18/25 / 23/25** |

Gain: macro-F1 **+0.037**, kappa +0.062, which clears the +0.02 gate, so the model was saved.

Checks:
- Task 1 and 2 rows: the metadata columns and every `CLUE_NAMES[1]`/`[2]` column are
  string-identical to the table built on `origin/main` (1205 and 1055 rows). The first 5 task 3
  columns are identical too (1096 rows). The only change is 13 new columns.
- `python -m model.task_classifier`: printed accuracy, macro-F1 and confusion matrix CSV are
  identical before and after (10 clue columns: accuracy 0.8504, macro-F1 0.8484; all features:
  accuracy 0.8746, macro-F1 0.8730).
- `python -m model.train_rf --table runs/clue_table.csv --task 3 --save` (seed 0): child leak
  check PASS, macro-F1 0.56 ± 0.06, kappa 0.51 ± 0.09 (most common score 0.26 / 0.00). Saved
  `models/task3_rf.joblib` (0.9 MB, 17 features).
- `python -m app.main --fake`, `ruff format .`, `ruff check .`, `pytest`: all pass.
  `tests/contracts` unchanged. New `tests/test_clues_task3_extra.py`: 18 names, finite on
  `make_fake_try`, -9 with no pose. `tests/test_model_marty.py` now checks the classifier's own
  `TASK_FEATURES`, not every `CLUE_NAMES[task]` name.

## 9. Feet-in-picture clues (`exp_t3_feet.py`, 2026-11-25, experiment only)
Can the forest tell "didn't jump" from "jumped out of the picture"? Feet clues per file from
`eval.dry_run_count.feet_in_picture` (1920x1080, point score > 0.3): `ankle_missing`,
`lost_at_bottom`, `reached_edge`. Same 1096 rows; no jump found (`n_air == 0`): 94 = 50 Bad +
44 Good/Perfect. Rule: a variant counts only if mean F1 gain >= +0.02 and F1 wins >= 18/25.

| variant | macro-F1 | kappa | Good recall | F1 / kappa wins vs V0 | counts |
|---|---|---|---|---|---|
| V0 = current 17 features | 0.557 | 0.527 | 0.30 | – | – |
| V1 = V0 + ankle_missing | 0.553 | 0.540 | 0.31 | 11/25 / 19/25 | no |
| V2 = V1 + lost_at_bottom + reached_edge | 0.552 | 0.531 | 0.31 | 13/25 / 13/25 | no |

On the 94 no-jump files (seed 0, out of fold) V0 calls all 94 Bad; V1 moves 14 of 44
Good/Perfect to Good but also 7 of 50 Bad to Good, never Perfect. Median `ankle_missing`, no jump:
Bad 0.00, Good 0.48, Perfect 0.43 (jump found: 0.03-0.04 for every label).

As a flag outside the model (no-jump files only, flagged = `ankle_missing` > cutoff; cutoffs are
guesses to compare, none chosen):

| cutoff | Bad flagged | Good+Perfect flagged |
|---|---|---|
| 0.05 | 20/50 | 38/44 |
| 0.10 | 19/50 | 34/44 |
| 0.20 | 13/50 | 31/44 |
| 0.30 | 12/50 | 27/44 |

**Result:** no feet clue goes into the forest (no F1 gain). As a "check by teacher" flag when no
jump is found, `ankle_missing` > 0.20 catches 31/44 Good/Perfect at the cost of 13/50 Bad: a
team decision (Daryn), nothing ported. `features/`, `models/`, `app/` unchanged.

## Done when
The numbers above are reproduced by the scripts in `scripts/experiments/` (each prints them;
`exp_t3_best.py` and `exp_t3_stagger.py` take about 2-3 minutes; `exp_t3_stagger.py` needs
`runs/clue_table.csv` from `scripts/make_clue_table.py --tasks 3`). `ruff format .`,
`ruff check .`, `pytest` pass.
