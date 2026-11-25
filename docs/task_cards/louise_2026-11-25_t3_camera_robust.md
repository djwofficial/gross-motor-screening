# Task card: Louise, Nov 25: task 3 clues that agree across cameras

**Goal:** lower the measurement noise in the task 3 clues. Group B films every try with 3 cameras
and one expert score, yet the forest gives the 3 cameras different scores in 37% of task 3 tries.

**Finding (group B, 1,096 task 3 files):** "camera noise" = share of a clue's variance that is
camera-to-camera disagreement on the same try (0 = all cameras agree). The task 3 clues are
43-61% camera noise (`jump_size` 0.43, `dy` 0.52, `dx` 0.97), against 0.13 for task 2 `hops`.
Cause: distances are in one torso length for the whole file, but the child grows ~30% on screen
while jumping toward a camera. Averaging the 3 cameras per try would give macro-F1 0.605 (not usable
live, one camera), so the noise costs about 0.04.

**Change:** two new clues in `features/clues.py` (`_task3_extra_raw`), appended to `CLUE_NAMES[3]`:
- `jump_size_local`: same formula as `jump_size`, but dx and dy are divided by the torso size at
  that moment (geometric mean of the median torso 0.6 s before take-off and 0.6 s after landing),
  and the size change is `log(torso_after / torso_before)`.
- `hip_peak_local`: `hip_peak` divided by the torso 0.6 s before take-off.

The old clues stay as they are: `model/task_classifier.py` uses `jump_size` with hand-set weights.

**Result:** random forest, GroupKFold(5) by child, seeds 0-4, `runs/clue_table.csv` rebuilt.

| | camera noise | | macro-F1 | kappa | Good recall |
|---|---|---|---|---|---|
| `jump_size` → `_local` | 0.43 → 0.18 | before | 0.558 ± 0.056 | 0.528 | 0.30 |
| `hip_peak` → `_local` | 0.26 → 0.16 | after | **0.568 ± 0.058** | **0.538** | **0.32** |

Better F1 in 16 of 25 runs: the gain is real but small. `models/task3_rf.joblib` retrained (19
features).

**Tried, not kept:** 0.35 s windows (noise 0.20, smaller gain); replacing `dx`/`dy`/`scale_gain`
with local versions or dropping them (F1 0.563-0.567, kappa lower); model swaps, class weights and
an ordinal forest on all tasks (no gain); training without labels the forest confidently disagrees
with (no gain: the labels are not the problem); feet-out-of-picture clues (F1 0.547).

**Next:** the same check for task 1 `sway`, `lift_std`, `touchdowns` (40-44% camera noise).
