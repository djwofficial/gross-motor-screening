# Task card: Daryn, Nov 15 (#7): Swap stubs for the real modules

**Goal:** Remove the remaining placeholder stubs owned by Daryn now that real modules (`ChildTracker`, frozen RF models in `models/`, task classifier, clues for tasks 1–3, and GUI runners) have landed, keeping all CI contract and smoke checks green.

**Input → output:** `FakePoseStream` / `PoseStream` frames `(P, 26, 2)` and `(P, 26)` -> BODY_25 `(25, 3)` tracked child via `ChildTracker` -> clues -> frozen RF `predict` -> JSON try + `log.csv`.

**Signature (don't change):** `score_try(tr, task)`, `finish_try(...)`, `run_fake(args)`, `main(argv=None)` in `app/main.py`; `predict(task, clues)` in `model/predict.py`.

**Files you may change:** `app/main.py`, `model/predict.py`, `README.md`, `tests/test_main.py`, `docs/task_cards/daryn_2026-11-15_swap_stubs.md`. (Note: `viz/blur.py` is owned by Theresia under `#52`.)

**Steps:**
1. Replace `pick_child([people25])[0]` and hardcoded `[0]` `raw26` selection in `app/main.py:run_fake` with `live.track.ChildTracker((1280, 720))` so multi-person frames track one child consistently.
2. Remove the obsolete live-mode exit-2 stub in `app/main.py:main` and the stale `STUB:` note in `model/predict.py`.
3. Add `tests/test_main.py` verifying headless lazy imports (no `cv2`/`rtmlib` in `--fake` mode) and multi-person `ChildTracker` selection + `rescore`.

**Done when:** No Daryn stubs remain in `app/main.py` or `model/predict.py`, and `ruff format --check .`, `ruff check .`, and `pytest` pass.

Evidence recorded on macOS (`Python 3.12.13`):
- `pytest tests/test_main.py -v`: `4 passed in 2.18s`.
- `--fake` 3-task CLI run (`task 1`: `0.101s`, `task 2`: `0.065s`, `task 3`: `0.062s`) + `rescore`: `3/3`.

**Watch out:** Keep `cv2` and `rtmlib` lazily imported only inside live execution paths so `--fake` and GitHub Actions CI (`requirements-ci.txt`, which omits `rtmlib` and `onnxruntime`) stay fast and clean.

**Tests:** Contract suite `tests/contracts/*` stays unchanged and green; behaviour tests added in `tests/test_main.py`.
