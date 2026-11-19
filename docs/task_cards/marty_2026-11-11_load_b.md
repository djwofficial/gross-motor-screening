# Task card: Marty, 2026-11-11: Group B dataset loader (#14)

**Issue:** [#14](https://github.com/djwofficial/gross-motor-screening/issues/14).

**Goal:** Turn `B_skel.zip` entries into the shared `common.format.Try` format (`load_try`), picking the child with `common.pick.pick_child`, dropping clips under 30 frames, keeping clips over 300 frames uncut, and verifying normalisation (`common.geometry.normalize`).

**Input → output:** `z: zipfile.ZipFile` (`data/raw/B_skel.zip`) and `name: str` (`B/{child}_GMS_{task}_{attempt}_{camera}.json`) → validated `common.format.Try` (`points` float32 `(T, 25, 3)` in pixels, `t` float64 `(T,)` in seconds at 30 fps, `meta` dict).

**Signature (don't change):** `load_try(z: zipfile.ZipFile, name: str, min_frames: int = MIN_FRAMES) -> Try`.

**Files you may change:** `data/load_b.py`, `tests/test_load_b.py`, `docs/task_cards/marty_2026-11-11_load_b.md`.

**Steps:**
1. Parse file metadata with `parse_name(name)` and read multi-person frames with `read_file(z, name)`.
2. Reject clips with fewer than 30 frames (`ValueError`).
3. Keep full clip length (never cut at 300 frames).
4. Pick the child track via `common.pick.pick_child(frames)`, clip scores to `[0, 1]`, compute `t = np.arange(T, dtype=np.float64) / 30.0`, and validate with `common.format.validate`.
5. Verify on all 3,359 task 1–3 files in `data/raw/B_skel.zip`, including `common.geometry.normalize` (centre on first valid neck, scale by torso length).

**Done when:** 3,359 files for tasks 1–3 (1,205 / 1,056 / 1,098) are scanned, the 3 clips under 30 frames are dropped (`3,356` valid `Try` objects loaded, `141` clips over 300 frames kept uncut), and `common.geometry.normalize` succeeds on all 3,356 loaded tries.

**Watch out:** Never write a duplicate normaliser; `Try.points` stores raw pixels as required by `common.format.validate`, and clues call `common.geometry.normalize`.

**Tests:** `tests/contracts/test_format.py`, `tests/test_dataset.py`, and `tests/test_load_b.py`.

**Evidence (2026-11-19):**

Command: `.venv/bin/python -m data.load_b`

```text
Tasks 1-3 files: 3359 (1205 / 1056 / 1098)
Loaded (>= 30 frames): 3356 (1205 / 1055 / 1096)
Dropped (< 30 frames): 3 -> [('B/B113_GMS_2_1_3.json', 26), ('B/B159_GMS_3_1_2.json', 23), ('B/B159_GMS_3_3_2.json', 17)]
Kept full length (> 300 frames, not cut): 141
Normalized (common.geometry.normalize) check: 3356/3356 OK
```
