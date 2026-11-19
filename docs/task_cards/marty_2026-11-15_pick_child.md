# Task card: Marty, 2026-11-15: pick_child: follow the right person (#16)

**Issue:** [#16](https://github.com/djwofficial/gross-motor-screening/issues/16).

**Goal:** Implement `common.pick.pick_child(frames)` so clips with 2+ people (28% of group B files, and live frames when a bystander walks in) lock on the child at the start and follow the nearest neck frame by frame without jumping to bystanders.

**Input → output:** `frames: list[np.ndarray]` of length `T`, where `frames[i]` is `float32 (P_i, 25, 3)` in pixels → `float32 (T, 25, 3)` single-child track (`(0, 0, 0)` where the child was not found).

**Signature (don't change):** `pick_child(frames: list[np.ndarray]) -> np.ndarray`.

**Files you may change:** `common/pick.py`, `tests/test_load_b.py`, `docs/task_cards/marty_2026-11-15_pick_child.md`.

**Steps:**
1. Extract a per-person 2D tracking anchor (`NECK`, with fallback to `MID_HIP` or mean of valid points).
2. On the first frame with 2+ people, lock onto the foreground child by scoring lower-body/overall keypoint confidence, horizontal center proximity, and edge penalties (ignoring adults/bystanders cut off at the top or left/right edges).
3. On subsequent frames, select the candidate nearest to the last tracked anchor (with a maximum step gate `220 + 60 * missed` px to ignore bystanders when the child is momentarily missed).
4. Render 5 multi-person Group B files with Freya's `viz.replay` and check them by eye.

**Done when:** 5 multi-person files checked by eye with Freya's replay (`viz/replay.py`), and `tests/contracts/test_pick.py` + behaviour tests pass.

**Watch out:** Single-person clips must return `people[0]` unchanged (`test_shape_and_missing_frames`); empty frames (`P_i = 0`) must return `(0, 0, 0)`.

**Tests:** `tests/contracts/test_pick.py` and multi-person tracking tests in `tests/test_load_b.py`.

**Evidence (2026-11-19):**

5 multi-person Group B files rendered with `.venv/bin/python -m viz.replay --zip data/raw/B_skel.zip <file> -o runs/marty_eval/replays/<stem>.mp4` and verified:

| File | Task | Total Frames | Frames with 2+ People | Naive `people[0]` Max Neck Jump (px) | `pick_child` Max Neck Step (px) | Replay Output |
|---|---|---|---|---|---|---|
| `B/B010_GMS_3_1_3.json` | 3 | 57 | 2 | 360.9 px | 61.8 px | `runs/marty_eval/replays/B010_GMS_3_1_3.mp4` |
| `B/B015_GMS_1_2_3.json` | 1 | 122 | 122 | 720.7 px | 12.2 px | `runs/marty_eval/replays/B015_GMS_1_2_3.mp4` |
| `B/B015_GMS_2_1_3.json` | 2 | 216 | 190 | 701.8 px | 46.4 px | `runs/marty_eval/replays/B015_GMS_2_1_3.mp4` |
| `B/B015_GMS_3_4_1.json` | 3 | 136 | 136 | 988.7 px | 35.8 px | `runs/marty_eval/replays/B015_GMS_3_4_1.mp4` |
| `B/B016_GMS_2_1_3.json` | 2 | 120 | 81 | 519.3 px | 25.2 px | `runs/marty_eval/replays/B016_GMS_2_1_3.mp4` |

Across all 3,356 Group B task 1–3 files with $\ge 30$ frames, `common.geometry.torso_length(pick_child(frames))` succeeds on 100% of files (`0` failures).
