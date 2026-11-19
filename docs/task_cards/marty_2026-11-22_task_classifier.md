# Task card: Marty, 2026-11-22: Guess the task and the leg (#17)

**Issue:** [#17](https://github.com/djwofficial/gross-motor-screening/issues/17).

**Goal:** Implement `model.task_classifier.predict_task(points, t)` to guess which task (`1`, `2`, or `3`) and support leg (`"L"` or `"R"`) a try belongs to, and evaluate task classification across Group B with `GroupKFold(5)` by child using the Task 1 clue table (`clue_table.csv`) and Group B tries.

**Input → output:**
- Runtime: `points: float32 (T, 25, 3)` in pixels, `t: float64 (T,)` in seconds → `(task: int 1/2/3, leg: str "L"/"R", confidence: float 0..1)`.
- Offline CLI (`python -m model.task_classifier`): Group B clue table (`clue_table.csv` + `B_skel.zip` or `--table`) → `GroupKFold(5)` accuracy, macro-F1, child-leak check, and 3×3 confusion matrix CSV/PNG.

**Signature (don't change):** `predict_task(points: np.ndarray, t: np.ndarray) -> tuple[int, str, float]`.

**Files you may change:** `model/task_classifier.py`, `tests/test_model_marty.py`, `docs/task_cards/marty_2026-11-22_task_classifier.md`.

**Steps:**
1. Normalize points with `common.geometry.normalize(points)` (origin at first valid neck, scale by torso length, `y` points up).
2. Determine the support leg (`"L"` or `"R"`) from the lower mean ankle/foot height in normalized coordinates.
3. Merge the provided Task 1 clue table (`1,205` files: `one_foot_s`, `touchdowns`, `sway`, `wobble_s`) with extracted movement clues (`hops`, `longest_pause_s`, `feet_gap_takeoff`, `feet_gap_landing`, `air_s`, `jump_size`) and kinematic descriptors (`one_foot_frac`, `foot_diff_med`, `foot_diff_p75`, `sup_vy_std`, `hip_vy_std`, `sup_y_range`).
4. Evaluate `GroupKFold(5)` by child across all 3,356 valid Group B files and save the 3×3 confusion matrix as CSV and PNG.

**Done when:** Accuracy and confusion matrix reported with `GroupKFold(5)` by child (`child leak check: PASS`), and `tests/contracts/test_model.py` + behaviour tests pass.

**Watch out:** Never use `duration_s` as a model feature. Always split by child (`GroupKFold`), never by file or attempt.

**Tests:** `tests/contracts/test_model.py` and `tests/test_model_marty.py`.

**Evidence (2026-11-19):**

Command: `.venv/bin/python -m model.task_classifier`

```text
Task classifier (10 clue columns): 3356 files, 106 children, GroupKFold(5) by child
  features (10): one_foot_s, touchdowns, sway, wobble_s, hops, longest_pause_s, feet_gap_takeoff, feet_gap_landing, air_s, jump_size
  child leak check: PASS
  fold    accuracy  macro-F1
  1         0.8646    0.8641
  2         0.7738    0.7671
  3         0.8497    0.8433
  4         0.8390    0.8360
  5         0.8176    0.8115
  mean ± sd accuracy: 0.8290 ± 0.0315 (overall 0.8290)
  mean ± sd macro-F1: 0.8244 ± 0.0332 (overall 0.8253)
  confusion matrix (rows = true task 1/2/3, cols = predicted task 1/2/3):
    task 1: [1084, 68, 53]
    task 2: [105, 783, 167]
    task 3: [23, 158, 915]

Task classifier (clues + kinematic features): 3356 files, 106 children, GroupKFold(5) by child
  features (16): one_foot_s, touchdowns, sway, wobble_s, hops, longest_pause_s, feet_gap_takeoff, feet_gap_landing, air_s, jump_size, one_foot_frac, foot_diff_med, foot_diff_p75, sup_vy_std, hip_vy_std, sup_y_range
  child leak check: PASS
  fold    accuracy  macro-F1
  1         0.9360    0.9354
  2         0.8363    0.8325
  3         0.9003    0.8981
  4         0.8838    0.8815
  5         0.9283    0.9259
  mean ± sd accuracy: 0.8969 ± 0.0357 (overall 0.8969)
  mean ± sd macro-F1: 0.8947 ± 0.0366 (overall 0.8953)
  confusion matrix (rows = true task 1/2/3, cols = predicted task 1/2/3):
    task 1: [1116, 60, 29]
    task 2: [44, 905, 106]
    task 3: [13, 94, 989]

Support leg distribution (L/R) across Group B files:
  task 1: L=580, R=625
  task 2: L=498, R=557
  task 3: L=528, R=568

Saved confusion matrix CSV: runs/marty_eval/task_classifier/task_classifier_cm.csv
Saved confusion matrix PNG: runs/marty_eval/task_classifier/task_classifier_cm.png
```
