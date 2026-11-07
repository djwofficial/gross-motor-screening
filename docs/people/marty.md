# Marty: dataset loader, baseline, models

**Your part:** turn group B into training data, build the baseline and the random forest per
task, the task + leg classifier, and `pick_child` for files with 2+ people.

**Your files:** `data/load_b.py`, `common/pick.py`, `model/` (`baseline.py`, `train_rf.py`,
`predict.py`, `task_classifier.py`), `models/`, new tests `tests/test_model_*.py`,
`tests/test_load_*.py`.

**Signatures (AGENTS.md):** `pick_child(frames: list[(P,25,3)]) -> (T,25,3)`;
`predict(task, clues) -> (score 0/1/2, confidence 0..1)`;
`predict_task(points, t) -> (task 1/2/3, leg "L"/"R", confidence)`;
model file `models/task{n}_rf.joblib` = `{"model", "features"}`.

**You need from others:**
| What | From | Stand-in now |
|---|---|---|
| clue table for all 3,359 files | Revallo (Nov 18) | duration only from `features.clues.compute`; build your own small table for task 1 |
| replay to check pick_child by eye | Theresia (`viz/replay.py`, Nov 10) | plot a few frames with matplotlib |

**Who waits for you:** Revallo (loader, Nov 11), Louise (`pick_child`, Nov 15),
Freya/Daryn (`predict`, `predict_task`).

## Tasks (write `Done YYYY-MM-DD` in Status in the same PR)
| Due | Task | Level | Done when | Status |
|---|---|---|---|---|
| Nov 8 | Milestone 1 | Easy | First line `files: 4661 \| children: 106 \| attempts: 1555` | |
| Nov 11 | `data/load_b.py`: B_skel.zip → shared format; drop clips under 30 frames; normalize with `common.geometry`; don't cut at 300 frames | Medium | 3,359 files for tasks 1–3 (1,205 / 1,056 / 1,098) | |
| Nov 13 | `model/baseline.py` per task, GroupKFold by child | Easy | Close to Card 1 (macro-F1 0.19 / 0.20 / 0.26, kappa 0.00); child-leak check prints PASS | |
| Nov 15 | `common/pick.py`: `pick_child()` for files with 2+ people (28% of B files): lock on the person at the start, then follow the nearest neck | Hard | 5 multi-person files checked by eye with Theresia's replay | |
| Nov 16 | `model/train_rf.py` with task 1 clues first | Medium | Runs and prints F1 and kappa | |
| Nov 19 | Gate: random forest per task beats the baseline; save models; write `model.predict()` | Medium | Macro-F1 and kappa above baseline for all 3 tasks; .joblib files exist; predict returns score + confidence | |
| Nov 22 | `model/task_classifier.py`: guess task (1/2/3) and leg from the clue table, GroupKFold by child | Medium | Accuracy and confusion matrix reported; if low, the app uses keys only | |
| Nov 22 | Confusion matrices per task and per fold to Revallo; `predict` under 1 s | Easy | CSV and PNG files exist | |

**Watch out:** split by child, never by file or attempt. Group B only. Every number in a PR
comes from a script you ran. `.joblib` files are models, not child data, but check their size.
