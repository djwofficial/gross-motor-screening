# Daryn: repo, shared format, model training, integration, rescore (team lead)

**Your part:** the shared contract (`common/`, contract tests, CI), training the scoring model and
passing the Nov 19 gate, joining the parts in `app/main.py`, `rescore.py`, the code freeze. You also
review changes to shared files.

**Your files:** `common/format.py`, `common/body25.py`, `common/geometry.py`, `app/main.py`,
`rescore.py`, `model/train_rf.py`, `model/predict.py`, `models/`, `tests/`, `.github/`,
`requirements*.txt`, `AGENTS.md`, `docs/`.

**Your laptop is a Mac:** the random forest needs no GPU. Integrate with recorded files and
`--video`; test live on Freya's laptop (joint sessions Nov 15, 19, 22).

**You need from others:** the clue table (Louise, Nov 18), the loader (Marty, Nov 11), every part for
Nov 22. After Nov 19 keep your time for integration.

**Admin still open:** `docs/admin.md` (invite the 5 teammates, CODEOWNERS usernames,
branch protection after GitHub Student Pack).

## Tasks (status is on the [project board](https://github.com/users/djwofficial/projects/3); `Closes #<issue>` in your PR closes the task)
| Due | Task | Level | Done when | Issue |
|---|---|---|---|---|
| Nov 8 | GitHub repo (private), folder layout, requirements.txt with exact versions (rtmlib, onnxruntime-gpu, numpy, pandas, scikit-learn, opencv), README with install steps, plus scripts/summarize_group_B.py for Milestone 1 | Easy | A teammate clones, installs and gets the Milestone 1 first line without asking anything | #1 |
| Nov 8 | common/format.py: Try dataclass, save/load, validate, fake-try generator, plus examples/fake_try.npz | Easy | validate(load(fake)) passes | #2 |
| Nov 8 | AGENTS.md and CLAUDE.md in the repo root, plus a task-card template (see Working with AI coding tools) | Easy | A teammate pastes one card into their AI and the AI explains the repo rules correctly | #3 |
| Nov 9 | Stubs for every function (including live/segment.py, live/quality.py, model/task_classifier.py) and app/main.py --fake running start to finish in automatic and manual mode; contract tests in tests/contracts/, smoke test, GitHub Actions CI and CODEOWNERS | Medium | python -m app.main --fake writes 1 JSON and 1 log row; tests pass on the stubs; CI is green on main | #4 |
| Nov 9 | Timeline: stub and contract test for features.clues.timeline; events saved in the JSON of each try | Easy | Tests are green; the JSON from --fake has events | #5 |
| Nov 10 | rescore.py first version: folder of JSON files → clues → model → CSV of scores (moved up from Nov 19) | Medium | Runs on JSON files from app/main.py --fake | #6 |
| Nov 15 | Swap each stub for the real module as it lands; keep the smoke test green | Medium | Test is green after every merge | #7 |
| Nov 16 | model/train_rf.py set up with task 1 clues first | Medium | Runs without errors and prints F1 and kappa | #8 |
| Nov 19 | Gate: random forest per task on the full clue table beats the baseline. Clip length (duration_s) stays out of the model. For task 2 also try a per-second forest and keep whichever scores higher. Save the models and write model.predict() | Medium | Macro-F1 and kappa above the baseline for all 3 tasks; .joblib files exist; predict returns a score and a confidence | #9 |
| Nov 22 | Join everything in app/main.py on the live laptop: live → clues → model → log (helped by Revallo and Freya) | Hard | A score appears in the log within a few seconds of each try, for all 3 tasks | #10 |
| Nov 25 | Compare rescore with the live scores on the 27 dry-run tries | Medium | 27/27 match | #11 |
| Nov 27 | Code freeze, tag v1.0, fresh install from scratch on the live laptop | Easy | Clean install, then a mini dry run passes | #12 |
