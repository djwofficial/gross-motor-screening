# Task card: Revallo, 2026-11-08: Milestone 1

**Issue:** [#26](https://github.com/djwofficial/gross-motor-screening/issues/26).

**Goal:** Verify the local Python 3.12 setup against the official group B dataset.
This task verifies existing code; it does not implement the live pose module.

**Input -> output:** The official CC0 `B_skel.zip` archive in `data/raw/` ->
the output of `python scripts/summarize_group_B.py`.

**Signature (don't change):** No public interface changes.

**Files you may change:** This task card and its verification evidence. Python,
the virtual environment, and the dataset are local setup; keep data ignored.
Do not change shared code, contract tests, or pinned requirements.

**Steps:**
1. Use branch `revallo/milestone1` and Python 3.12 with the pinned dependencies.
2. Download group B only from the official GigaDB record, DOI `10.5524/102382`.
3. Keep the archive intact; run `python scripts/summarize_group_B.py`.
4. Run the existing dataset tests and the required repository checks.
5. Record actual results before opening the PR with `Closes #26`.

**Done when:** The script's first output line is exactly
`files: 4661 | children: 106 | attempts: 1555`, as required by the README and
the team workspace's Coding plan to Nov 27. Record the command, interpreter,
dataset source, and actual output as evidence; do not report an expected value
as a measured result.

**Watch out:** Use only the public group B dataset. Never commit the archive or
raw points. Never substitute dependency versions without Daryn's approval.

**Tests:** Existing `tests/test_dataset.py`, contract tests, and fake-pipeline
tests. No new tests or contract changes are needed for this setup task.

**Evidence (2026-11-12):** Completed locally on branch `revallo/milestone1`.

- Interpreter: Python 3.12.13; NumPy 2.5.4, pytest 9.2.0, Ruff 0.17.4.
  This verifies Milestone 1; the full live GPU environment is a separate task.
- Archive: 211,959,553 bytes. MD5
  `c749b5497306e4d18205c3e46be9f574` matches the official object metadata;
  `python -m zipfile -t data/raw/B_skel.zip.download` passed before the
  verified download was renamed to `data/raw/B_skel.zip`.
- Command: `.venv\Scripts\python.exe scripts\summarize_group_B.py`.
  Actual output:

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

- `.venv\Scripts\python.exe -m pytest`: **49 passed in 32.26s**;
  the dataset-dependent test ran and passed.
- Ruff format/check and `git diff --check` passed. Source code and pinned
  requirements are unchanged; the archive is ignored by Git.

**Dataset sources:**
- [GigaDB file metadata](https://gigadb.org/gigadb/api/dataset/list_dataset_files/?dataset_id=102382)
  provides the group B URL and size.
- [Official archive](https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/102001_103000/102382/B_skel.zip)
  supplies `x-amz-meta-md5chksum: x0m1SXMG5NGCBcPka+n1dA==`, the MD5 above.
- [DataCite record](https://api.datacite.org/dois/10.5524/102382) confirms
  DOI `10.5524/102382`, GigaScience, and the CC0-1.0 license.
