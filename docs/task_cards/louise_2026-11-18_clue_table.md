# Task card: Louise, Nov 18 (done Nov 19): clue table for all 3 tasks (#22)

**Goal:** one row per group B file for tasks 1-3 with the clues of its task, so Daryn can train
and gate the random forests and Marty can train the task classifier.

**Input → output:** `data/raw/B_skel.zip` → `runs/clue_table.csv` (git-ignored; sent to Daryn and
Marty). Columns: `file, child, task, attempt, camera, label, n_frames, max_people, picked`, then
the clues in `features.clues.CLUE_NAMES[task]`.

**Run:** `python scripts/make_clue_table.py --tasks 1 2 3 --out runs/clue_table.csv` (about 6 s).

**Files changed:** `scripts/make_clue_table.py` (#22, earlier commit), this card, the header note
in `features/clues.py`.

## Done when: no empty values, or every filled-in value is written down

Evidence (2026-11-19, `main` at 28d5f85 plus the header-note fix):

- 3,356 rows, 106 children: task 1 1,205 · task 2 1,055 · task 3 1,096.
- 0 empty and 0 non-finite values in each task's own clue columns.
- Blank cells are only in the other tasks' clue columns (a task 1 row has no `hops`); this is
  the format `model/train_rf.py` reads ("each task only needs its own clue columns").
- Left out, written down here: 3 clips under 30 frames (Marty's loader rule, #14):
  `B113_GMS_2_1_3` (26 frames), `B159_GMS_3_1_2` (23), `B159_GMS_3_3_2` (17). So the table has
  3,356 rows, not 3,359.
- `picked = temp` for 943 files with 2+ people (task 1 299 · task 2 272 · task 3 372): the child
  was chosen by the temporary picker in `scripts/make_clue_table.py` (nearest the centre, then
  nearest neck). **Re-run the table when Marty's `pick_child` (#16) lands.**

**Update 2026-11-19 (later):** Daryn switched the script to Marty's `pick_child` (fc7d33a), so
`picked` is now `single` (2,413) or `pick_child` (943). Re-run on that version: still 3,356 rows;
random forest macro-F1 / kappa 0.62 / 0.58 (task 1), 0.60 / 0.59 (task 2), 0.36 / 0.19 (task 3).

## Random forest gate (`python -m model.train_rf --table runs/clue_table.csv --task N`)

GroupKFold(5) by child, child leak check PASS for all three.

| Task | Files | Random forest macro-F1 | kappa | Most common score macro-F1 | kappa | Beats baseline |
|---|---|---|---|---|---|---|
| 1 one leg | 1,205 | 0.61 ± 0.07 | 0.57 ± 0.08 | 0.19 ± 0.02 | 0.00 | yes |
| 2 hop | 1,055 | 0.60 ± 0.04 | 0.59 ± 0.04 | 0.20 ± 0.03 | 0.00 | yes |
| 3 long jump | 1,096 | 0.36 ± 0.03 | 0.20 ± 0.11 | 0.26 ± 0.02 | 0.00 | yes |

Task 3 (Revallo's clues) beats the baseline but stays weak, as the plan expected (risk table).

## Known weak points (next)

- `longest_pause_s` (task 2) has median 0 for every label.
- Bad hop clip `B210_GMS_2_2_2`: 1 hop found, 0 by eye.
- Touch-downs, wobble and pauses not yet checked by eye.
- Temporary picker not checked by eye on multi-person files.
