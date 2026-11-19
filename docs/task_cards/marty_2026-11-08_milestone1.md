# Task card: Marty, 2026-11-08: Milestone 1 (#13)

**Issue:** [#13](https://github.com/djwofficial/gross-motor-screening/issues/13).

**Goal:** Verify the local Python 3.12 setup and the official group B dataset archive (`data/raw/B_skel.zip`) by running the Milestone 1 summary script.

**Input → output:** The official CC0 `B_skel.zip` archive in `data/raw/` → the console output of `python scripts/summarize_group_B.py`.

**Signature (don't change):** No public interface changes.

**Files you may change:** This task card (`docs/task_cards/marty_2026-11-08_milestone1.md`). Keep `data/raw/B_skel.zip` gitignored.

**Steps:**
1. Verify Python 3.12 virtual environment and pinned dependencies.
2. Point `data/raw/B_skel.zip` at the official GigaDB `B_skel.zip` archive (DOI `10.5524/102382`).
3. Run `.venv/bin/python scripts/summarize_group_B.py`.
4. Run `pytest` including `tests/test_dataset.py`.

**Done when:** First output line is `files: 4661 | children: 106 | attempts: 1555`.

**Watch out:** Use group B only. Never commit `B_skel.zip` or raw points.

**Tests:** `tests/test_dataset.py` and `tests/contracts/`.

**Evidence (2026-11-19):**

- Interpreter: Python 3.12.13 (`.venv/bin/python`)
- Command: `.venv/bin/python scripts/summarize_group_B.py`
- Actual output:

```text
files: 4661 | children: 106 | attempts: 1555

Tasks 1-3: files 3359 | attempts 1121 | children 106
task    files  attempts  children   Bad / Good / Perfect (attempts)
1        1205       403        98   61 / 169 / 173
2        1056       352        96   157 / 90 / 105
3        1098       366       103   60 / 71 / 235

Frames per file: min 17, median 164, max 600; under 30: 3; over 300: 141
Files with 2+ people in some frame: 944 (28%)
Attempts whose cameras disagree on the label: 0
Children who did all 3 tasks: 91
```
