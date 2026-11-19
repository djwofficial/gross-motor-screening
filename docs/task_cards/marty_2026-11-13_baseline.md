# Task card: Marty, 2026-11-13: Baseline per task (#15)

**Issue:** [#15](https://github.com/djwofficial/gross-motor-screening/issues/15).

**Goal:** Implement `model/baseline.py` to evaluate the most-common-score baseline (`DummyClassifier(strategy="most_frequent")`) per task with `GroupKFold(5)` by child on Group B, verifying macro-F1, quadratic weighted kappa, and the child-leak check.

**Input → output:** Group B task 1–3 labels and child IDs from `data/raw/B_skel.zip` (or `--table`) → console report of per-task macro-F1 (`mean ± sd`), quadratic weighted kappa (`mean ± sd`), and child-leak check (`PASS`).

**Signature (don't change):** `main()` in `model/baseline.py`.

**Files you may change:** `model/baseline.py`, `tests/test_model_marty.py`, `docs/task_cards/marty_2026-11-13_baseline.md`.

**Steps:**
1. Load Group B file metadata (`child`, `task`, `label`) for tasks 1, 2, and 3.
2. Run `GroupKFold(n_splits=5)` grouped by `child` with `DummyClassifier(strategy="most_frequent")`.
3. Verify `set(groups[train]) & set(groups[test]) == set()` in every fold (`Child-leak check: PASS`).
4. Compute macro-F1 (`f1_score(..., average="macro", zero_division=0)`) and quadratic weighted kappa (`cohen_kappa_score(..., weights="quadratic")`).

**Done when:** Numbers match Card 1 (`macro-F1 0.19 / 0.20 / 0.26`, `kappa 0.00`) and the child-leak check prints `PASS`.

**Watch out:** Split by child (`GroupKFold`), never by file or attempt. Group B only.

**Tests:** `tests/test_model_marty.py`.

**Evidence (2026-11-19):**

Command: `.venv/bin/python -m model.baseline`

```text
Child-leak check: PASS
task    files  children    macro-F1 (mean ± sd)    kappa (mean ± sd)    leak
1        1205        98    0.19 ± 0.02 (0.1851)          0.00 ± 0.00    PASS
2        1056        96    0.20 ± 0.03 (0.2033)          0.00 ± 0.00    PASS
3        1098       103    0.26 ± 0.02 (0.2598)          0.00 ± 0.00    PASS
```
