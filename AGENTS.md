# AGENTS.md: rules for every AI coding tool (and every teammate) in this repo

Read this fully before writing code. Claude Code reads it through CLAUDE.md, Codex reads it
directly. With ChatGPT on the web, paste this file at the start of the chat.

## The project in one paragraph
A live laptop-webcam **screening aid** that helps preschool teachers notice children (36-53
months) whose gross motor performance is below their age group. It is **never a diagnosis**:
never write that it detects "abnormal development" or predicts the future. One child does one
task at a time: 1 stand on one leg, 2 hop on one leg, 3 standing long jump. RTMPose (rtmlib
`BodyWithFeet`, Halpe-26) finds the body points, we convert them to the dataset's 25 BODY_25
points, build a timeline of moments (foot down at 2.0 s) and movement clues from it, and a random
forest gives a score 0 Bad / 1 Good / 2 Perfect. The timeline records what happened: never write
that a child failed.
Training data: Korean multiview child motor development dataset, **group B only** (Kim et al.
2023, GigaScience 12, giad039). Real children are used only to test, never to retrain.

Background (research question, data, how we judge success, timeline): `docs/project.md`.

## Start of every session: who am I working with?
When the user says who they are ("I'm Freya", "aku Marty"):
1. Read `docs/people/<name>.md` (daryn, marty, revallo, louise, freya, theresia). It holds their
   part, their files, what they need from others, and their tasks with due dates and issue numbers.
2. Find their open tasks: `gh issue list --repo djwofficial/gross-motor-screening --label
   owner:<name> --state open --limit 50 --json number,title,milestone,body`. Their next task is the
   open issue with the earliest Due (in its body). Without `gh`, ask them to open the board
   (https://github.com/users/djwofficial/projects/3) and paste the issue. Check today's date and `git log --oneline -15` /
   `git branch -a`. Say which task it is, its due date, and whether it is late or early (starting
   early is fine).
3. If `docs/task_cards/<name>_<date>_*.md` exists for it, follow that card. If not, draft one from
   the row with `docs/task_card_template.md` and show it before writing code.
4. Work on a branch `<name>/<short-task>`; stay inside their files. When the task's "Done when"
   check passes, open the PR with `Closes #<issue>` and the evidence. Merging it closes the issue
   and moves the card to Done.
If the user doesn't say who they are, ask before changing any file.

## Hard rules
1. **Units: seconds and torso lengths, never raw pixels.** Use `common.geometry.normalize`
   (origin = neck in the first frame, scale = torso length, **y points up**). Dataset points are
   1920x1080 OpenPose, live points are 1280x720 RTMPose; the same clue code must work on both.
2. **Group B only.** Never base a decision on group C numbers.
3. **Split by child** (GroupKFold with the child as group). Never by file or attempt.
4. **Every number needs a source**: a script that actually ran, the paper, or a cited link.
   Otherwise label it as a guess. Don't invent API calls: check rtmlib/OpenCV/sklearn source.
5. **Children's data never goes into an AI, into git, or into GitHub issues.** Points, videos, logs, child codes and
   ages from testing day stay on the encrypted drive (consent form; deleted by 2026-12-25).
   Debug with group B data (CC0), `common.format.make_fake_try`, or team clips.
6. **Pose detection runs on the unmirrored picture.** Mirror only for display.
7. **Stay in your folder.** Change only the files of the task you were given. `common/`,
   `tests/contracts/`, `app/main.py`, `requirements*.txt`, `.github/` change only through Daryn.
   Don't edit or weaken a contract test to make it pass; ask Daryn if a signature must change.
8. **Python 3.12**, versions pinned in `requirements.txt`. Don't add a dependency without
   asking Daryn (it must install on Mac and on Windows + NVIDIA).

## The shared format (common/format.py)
```
Try.points  float32 (T, 25, 3)   x, y, score per frame, BODY_25 order (common/body25.py), pixels
Try.t       float64 (T,)         seconds since the first frame, strictly increasing
Try.meta    dict (JSON)          source, width, height, fps, task, leg, child, ...
missing point = (0, 0, 0)        same as the dataset
leg           "L" / "R"          the SUPPORT leg (the one the child stands or hops on)
```

## Who owns what, and the signatures that must not change
| Owner | Files | Interface |
|---|---|---|
| Daryn | `common/format.py`, `common/body25.py`, `common/geometry.py`, `app/main.py`, `rescore.py`, `model/train_rf.py`, `model/predict.py`, `models/`, tests, CI | `Try`, `validate`, `save`, `load`, `make_fake_try(task, leg, ...)`; `torso_length(points)`, `normalize(points)`; `predict(task, clues) -> (score 0/1/2, confidence 0..1)`; model file `models/task{n}_rf.joblib` = `{"model", "features"}` (`duration_s` is not a model feature) |
| Marty | `data/load_b.py`, `common/pick.py`, `model/baseline.py`, `model/task_classifier.py` | `pick_child(frames: list[(P,25,3)]) -> (T,25,3)`; `predict_task(points, t) -> (task 1/2/3, leg "L"/"R", confidence)` |
| Louise | `features/clues.py` (tasks 1 and 2), `eval/report.py` | `timeline(task, points, t) -> list[{"start_s", "end_s", "state"}]`; `compute(task, points, t) -> dict` with exactly `CLUE_NAMES[task]`; `explain(task, clues, events=None) -> str` |
| Revallo | `live/pose.py`, `live/convert.py`, `live/capture.py`, `live/bench_fps.py`, `features/clues.py` (task 3, in its own functions) | `PoseStream(source, mode, device).read() -> (t, keypoints (P,26,2), scores (P,26), frame)` or `None`; `halpe_to_body25(keypoints, scores) -> (...,25,3)`; `HALPE_TO_BODY25` |
| Freya | `app/ui.py`, `app/session.py`, `viz/replay.py` | `Session(out_dir, child, age_months, video).save_try(tr, *, task, leg, guessed_task, guessed_leg, task_confidence, corrected, score, confidence, clues, reasons, raw26=None, events=None) -> Path`; `load_record(path)`; `LOG_FIELDS`; `render(tr, out_path, ..., events=None)` |
| Theresia | `viz/blur.py`, `live/segment.py`, `live/quality.py`; new: `live/start_area.py`, `eval/agreement.py`, `eval/dry_run_count.py` | `Segmenter().update(points (25,3) or None, t) -> "idle"/"ready"/"start"/"recording"/"end"`; `check(points, fps, frame_size, start_area) -> list[str]` |

Product flow (`app/main.py`): pose stream -> `halpe_to_body25` -> `pick_child` -> `Segmenter`
(automatic start/end; manual Start/Stop as fallback) -> `predict_task` -> operator confirms or
corrects -> `timeline` -> `compute` -> `predict` -> `explain` -> `Session.save_try` (JSON + `log.csv`).
Below `CHECK_BY_TEACHER_BELOW` confidence the log says "check by teacher" instead of a score.
No score and no timeline on screen in testing mode.

## How to work
- One task = one branch (`marty/load-b`, `louise/clues-task1`, ...) = one pull request into `main`.
- Before pushing: `ruff format .`, `ruff check .`, `pytest`. CI runs the same on every PR.
- Put the "Done when" evidence from the task card (number, plot, screenshot) in the PR text.
- Add your own behaviour tests in `tests/` (not in `tests/contracts/`).
- Stubs raise `NotImplementedError("Name, date: ...")` or return safe dummy values; replace them,
  keep the signature. `python -m app.main --fake` must keep working after your change.
- Task cards: `docs/task_cards/`. Template: `docs/task_card_template.md`.

## Versions (requirements.txt, checked on PyPI 2026-11-08)
Python 3.12 · numpy 2.5.4 · pandas 3.1.1 · scikit-learn 1.9.1 · joblib 1.6.0 · matplotlib 3.11.2 ·
opencv-contrib-python 5.0.0.93 · rtmlib 0.0.16 (installed with `--no-deps`) · onnxruntime 1.31.1
(Mac) / onnxruntime-gpu[cuda,cudnn] 1.31.1 (Windows + NVIDIA, CUDA 13) · pytest 9.2.0 · ruff 0.17.4.
rtmlib `BodyWithFeet(mode="balanced")` = YOLOX-m detector + RTMPose-m Halpe-26 (256x192);
output `keypoints (P, 26, 2)`, `scores (P, 26)` (scores can exceed 1; `halpe_to_body25` clips).
