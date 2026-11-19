# Task card: Louise, Nov 25 (started Nov 19): evaluation report (#25)

**Goal:** one report covering all 3 tasks: confusion matrix per task, per-fold numbers, and the
mix-ups compared with the scoring rules, for the final report and slides.

**Input → output:** the clue table (`scripts/make_clue_table.py`) → `runs/report.md` (git-ignored)
and the same text on screen.

**Run:** `python -m eval.report --table runs/clue_table.csv` (about 20 s).

**Files:** `eval/report.py`, `tests/test_eval_report.py`, this card. It reuses `model/train_rf.py`
(same folds, same forest, same seed), so its fold numbers equal Daryn's gate.

**Done when:** one report covering all 3 tasks. Done 2026-11-19 on the clue table built from
`main` at 43c0e33 (`pick_child`, task 2 `HOP_LIFT` 0.3). Re-run it whenever the clue table or the
forest settings change.

## Evidence (2026-11-19)

| Task | Tries | Forest macro-F1 | Forest kappa | Baseline macro-F1 | Correct per expert score (Bad / Good / Perfect) |
|---|---|---|---|---|---|
| 1 one leg | 1,205 | 0.62 ± 0.06 | 0.58 ± 0.09 | 0.19 | 63% / 57% / 75% |
| 2 hop | 1,055 | 0.66 ± 0.00 | 0.66 ± 0.06 | 0.20 | 80% / 49% / 74% |
| 3 long jump | 1,096 | 0.36 ± 0.05 | 0.19 ± 0.11 | 0.26 | 54% / 16% / 52% |

## What the mix-ups say (for the report)

- **Good is the hardest score in every task** (57%, 49%, 16% correct). It sits between two
  rules that differ by a judgement word: "staggers" (task 1), "pauses a little" (task 2),
  "a long distance" (task 3).
- **Task 1, Good predicted as Perfect (127 tries):** median 3.47 s on one foot and 0.27 s of
  wobble, against 0.90 s for Good predicted right. These children stood over 3 s with little
  wobble we could measure; "staggers" is the part the clues miss.
- **Task 2, Good predicted as Perfect (76 tries):** median 4 hops, 1 touch-down. Pauses do not
  help: `longest_pause_s` is 0 in the median for every group (three definitions tried, none
  separated Good from Perfect).
- **Task 3, Perfect predicted as Bad (206 tries):** every clue is 0, i.e. no jump was found.
  No jump is found in 22% of task 3 tries, including 19% of Perfect ones, on all three cameras.
  Reported to Revallo (task 3 owner) together with the camera 1 `jump_size` issue.
- Tries where nothing was measured: task 1 4% (mostly Bad, as expected: no lift), task 2 17%
  (mostly Bad: no hop), task 3 22% (all scores: a detection miss).
