# Marty: dataset loader, baseline, pick_child, task classifier

**Your part:** turn group B into training data, build the baseline, follow the right person in files
with 2+ people, and guess the task and leg of a try.

**Your files:** `data/load_b.py`, `common/pick.py`, `model/baseline.py`, `model/task_classifier.py`,
new tests `tests/test_load_*.py`, `tests/test_model_*.py`.
Not yours: `model/train_rf.py`, `model/predict.py`, `models/` (Daryn).

**Signatures (AGENTS.md):** `pick_child(frames: list[(P,25,3)]) -> (T,25,3)`;
`predict_task(points, t) -> (task 1/2/3, leg "L"/"R", confidence)`.

**You need from others:**
| What | From | Stand-in now |
|---|---|---|
| clue table for all 3,359 files | Louise (Nov 18) | build a small table for task 1 yourself |
| replay to check pick_child by eye | Freya (`viz/replay.py`, Nov 10) | plot a few frames with matplotlib |
| forest predictions for the matrices | Daryn (Nov 19) | baseline predictions |

**Who waits for you:** Louise and Daryn (loader, Nov 11), Revallo (`pick_child`, Nov 15),
Freya and Daryn (`predict_task`).

## Tasks (write `Done YYYY-MM-DD` in Status in the same PR)
| Due | Task | Level | Done when | Status |
|---|---|---|---|---|
| Nov 8 | Milestone 1 | Easy | First output line: files: 4661 \| children: 106 \| attempts: 1555 |  |
| Nov 11 | data/load_b.py: B_skel.zip → shared format; drop clips under 30 frames; center on the neck, scale by torso length; don't cut at 300 frames | Medium | 3,359 files for tasks 1–3 (1,205 / 1,056 / 1,098) |  |
| Nov 13 | model/baseline.py per task, GroupKFold by child | Easy | Numbers close to Card 1 (macro-F1 0.19 / 0.20 / 0.26, kappa 0.00) and the child-leak check prints PASS |  |
| Nov 15 | common/pick.py: pick_child() for files with 2+ people (28% of B files): lock on the person at the start, then follow the nearest neck. Revallo reuses the same function for live | Hard | 5 multi-person files checked by eye with Freya's replay |  |
| Nov 22 | model/task_classifier.py: guess the task (1/2/3) and the leg from the clue table, GroupKFold by child | Medium | Accuracy and confusion matrix reported; if accuracy is low, the app uses keys only |  |
| Nov 22 | Confusion matrices per task and per fold from Daryn's random forest, handed to Louise; check that predict takes under 1 s | Easy | CSV and PNG files exist |  |

**Watch out:** split by child, never by file or attempt. Group B only. Every number in a PR
comes from a script you ran.
