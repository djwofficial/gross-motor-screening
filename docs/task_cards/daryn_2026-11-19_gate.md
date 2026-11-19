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

**Tests:** `tests/contracts/test_model.py` and `tests/test_train_rf.py` stay green.
