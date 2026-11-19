# Task card: Marty, 2026-11-22: Confusion matrices for Louise (#18)

**Issue:** [#18](https://github.com/djwofficial/gross-motor-screening/issues/18).

**Goal:** Produce per-task and per-fold confusion matrices (CSV and PNG files) from Daryn's random forest (`model.train_rf`) with `GroupKFold(5)` by child using the provided `clue_table.csv` (Task 1: `macro-F1 0.61 ± 0.07`, `kappa 0.57 ± 0.08`) and Group B tries to hand off to Louise, and verify that `model.predict.predict` takes under 1 s.

**Input → output:** `clue_table.csv` (`1,205` Task 1 rows) + Group B tasks 2–3 clues + `model.train_rf.make_forest` → 37 CSV and PNG confusion matrix files in `runs/marty_eval/matrices/` (`task{1..3}_cm.{csv,png}`, `task{1..3}_fold{1..5}_cm.{csv,png}`, and `rf_confusion_matrices_summary.csv`) + measured `model.predict.predict` latency (`< 1 s`).

**Signature (don't change):** No changes to `model/train_rf.py` or `model/predict.py`.

**Files you may change:** `model/baseline.py`, `tests/test_model_marty.py`, `docs/task_cards/marty_2026-11-22_confusion_matrices.md`.

**Steps:**
1. Load the provided Task 1 clue table (`runs/marty_eval/clue_table_task1.csv` / `/Users/dyn/Downloads/clue_table.csv`) and Group B task 2–3 clues via `python -m model.baseline --matrices`.
2. Run `GroupKFold(n_splits=5)` grouped by `child` using `model.train_rf.make_forest` and `features_for(task)` (excluding `duration_s`).
3. Export confusion matrices per task and per fold as CSV and PNG files under `runs/marty_eval/matrices/`.
4. Benchmark `model.predict.predict(task, clues)` (including cold load from `.joblib` and warm calls) and verify it finishes in under `1.0 s`.

**Done when:** CSV and PNG confusion matrix files exist per task and per fold, and `predict` latency is verified under `1 s`.

**Watch out:** Split by child (`GroupKFold`), never by file or attempt. Never include `duration_s` as a model feature. Do not leave uncommitted `.joblib` files in Daryn's `models/` directory.

**Tests:** `tests/test_model_marty.py` (`test_generate_rf_confusion_matrices_and_predict_under_1s`).

**Evidence (2026-11-19):**

Command: `.venv/bin/python -m model. train_rf --table runs/marty_eval/clue_table_task1.csv` and `.venv/bin/python -m model.baseline --matrices --out-dir runs/marty_eval/matrices`

```text
Task 1: 1205 files, 98 children, GroupKFold(5) by child
  child leak check: PASS
  fold    forest F1  forest kappa   base F1  base kappa
  1            0.69          0.65      0.19        0.00
  2            0.57          0.52      0.19        0.00
  3            0.53          0.44      0.20        0.00
  4            0.58          0.57      0.19        0.00
  5            0.68          0.65      0.15        0.00
  random forest     macro-F1 0.61 ± 0.07   kappa 0.57 ± 0.08
  most common score macro-F1 0.19 ± 0.02   kappa 0.00 ± 0.00
  beats the baseline (both numbers): yes

Generated 37 CSV/PNG matrix files in runs/marty_eval/matrices:
  Task 1 overall confusion matrix (rows=true 0/1/2, cols=pred 0/1/2):
    true 0: [114, 53, 15]
    true 1: [91, 287, 126]
    true 2: [24, 113, 382]
  Task 2 overall confusion matrix (rows=true 0/1/2, cols=pred 0/1/2):
    true 0: [266, 118, 86]
    true 1: [76, 115, 79]
    true 2: [70, 86, 159]
  Task 3 overall confusion matrix (rows=true 0/1/2, cols=pred 0/1/2):
    true 0: [61, 37, 81]
    true 1: [40, 57, 115]
    true 2: [119, 164, 422]

Predict latency check (must be < 1.0 s):
  Task 1 cold load + predict: 30.01 ms (< 1000 ms: PASS)
  Task 2 cold load + predict: 28.31 ms (< 1000 ms: PASS)
  Task 3 cold load + predict: 30.29 ms (< 1000 ms: PASS)
```
