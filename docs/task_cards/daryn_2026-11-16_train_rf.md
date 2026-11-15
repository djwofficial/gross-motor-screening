# Task card: Daryn, 2026-11-16: random forest training script, task 1 first (#8)

Paste this whole card into your AI tool together with AGENTS.md. One card = one branch = one PR.

**Goal:** have `model/train_rf.py` ready before Louise's clue table lands, so the Nov 19 gate (#9)
is only "run it on the real table". It trains one random forest per task on the clue table, checks
it with GroupKFold by child, and prints macro-F1 and quadratic weighted kappa next to the
most-common-score baseline.

**Input → output:**
- In: the clue table, a CSV with one row per dataset file: `child` (e.g. B010), `task` (1/2/3),
  `label` (0 Bad, 1 Good, 2 Perfect; the words Bad/Good/Perfect or a `label_index` column also
  work) and the clue columns `CLUE_NAMES[task]`. Other columns (`file`, `attempt`, `camera`) are
  allowed and ignored.
- Out: a console report per task: macro-F1 and kappa per fold and as mean ± sd, for the random
  forest and for always guessing the most common score, plus the child-leak check (PASS/FAIL).
  With `--save`: `models/task{n}_rf.joblib`.

**Signature (don't change):** the model file read by `model/predict.py`:
`{"model": fitted classifier, "features": CLUE_NAMES[n] without duration_s}`.

**Files you may change:** `model/train_rf.py`, `tests/test_train_rf.py`, this card.

**Steps:**
1. `load_table(path)` / `check_table(table)`: read the CSV; clear errors for missing columns,
   empty clue values or labels outside 0/1/2.
2. `features_for(task)`: `CLUE_NAMES[task]` without `duration_s`.
3. `cross_validate(table, task)`: GroupKFold(5) by child; random forest (200 trees,
   `class_weight="balanced"`, fixed seed) and `DummyClassifier("most_frequent")`; macro-F1 and
   quadratic weighted kappa per fold; child-leak check.
4. `fit_and_save(table, task)`: fit on every row of the task, save the bundle (compressed).
5. `make_fake_table()`: a clearly fake table (children F000, F001, ...) so the script and the tests
   run before the real table exists.
6. CLI: `python -m model.train_rf --table clue_table.csv [--task 1] [--save]` and
   `python -m model.train_rf --fake`.

**Done when:** it runs without errors and prints F1 and kappa. Now on the fake table; on Louise's
task 1 clues as soon as they land.

**Watch out:**
- `duration_s` is never a feature: live, the length of a try comes from our own start and end.
- Each attempt has up to 3 camera files. Splitting by child keeps them in the same fold.
- The labels are unbalanced (task 1 attempts: 61 Bad, 169 Good, 173 Perfect), hence
  `class_weight="balanced"` and macro-F1.
- Never save a model trained on the fake table (`--fake --save` is refused). Fake numbers only
  show that the code runs; real numbers come from the real clue table.
- Keep the saved file under 10 MB so it can be committed (predict.py).

**Tests:** `tests/contracts/test_model.py` stays green (no model file is committed, so `predict`
keeps its stub behaviour). New behaviour tests in `tests/test_train_rf.py`.
