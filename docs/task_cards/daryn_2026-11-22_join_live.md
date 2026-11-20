# Task card: Daryn, Nov 22 (#10): Join everything on the live laptop

**Goal:** Wire `python -m app.main` cleanly to the live webcam/video pipeline (`PoseStream` -> `run_gui` / `run_demo` -> `ChildTracker` -> `Segmenter` -> `predict_task` -> `timeline`/`compute`/`predict`/`explain` -> `Session.save_try`) and verify scoring/logging across all 3 tasks.

**Input → output:** Webcam index (`--camera 0`) or video file (`--video <path>`) -> live GUI (testing mode via `app.ui.run_gui`, or `--show-score` via `app.demo.run_demo`) -> saved JSON try + appended row in `<out>/log.csv`.

**Signature (don't change):** `main(argv=None) -> int`, `score_try(tr, task)` in `app/main.py`; `run_gui` in `app/ui.py`; `run_demo` in `app/demo.py`; `PoseStream` in `live/pose.py`.

**Files you may change:** `app/main.py`, `README.md`, `tests/test_main.py`, `docs/task_cards/daryn_2026-11-22_join_live.md`.

**Steps:**
1. Implement `run_live(args)` and `_frame_size(stream)` in `app/main.py`, defaulting `--device` and `--pose-mode` by OS (`cpu`/`lightweight` on macOS, `cuda`/`balanced` elsewhere) and routing to `run_gui` (default `--view stick`) or `run_demo` when `--show-score` is passed (default `--view camera`), always closing `stream` in `finally`.
2. Update `README.md` run instructions to reflect `python -m app.main` and `python -m app.main --show-score`.
3. Add mocked live dispatch tests and 3-task scoring speed + `rescore` checks in `tests/test_main.py`.
4. Run a physical live-webcam check on Freya's Windows + NVIDIA laptop (`python -m app.main --show-score` or `python -m app.main`) across tasks 1, 2, and 3 to verify real camera capture and live timing in `<out>/log.csv`.

**Done when:** A score appears in the log within a few seconds of each try, for all 3 tasks, and `ruff format --check .`, `ruff check .`, and `pytest` pass.

Evidence recorded so far on macOS (`Python 3.12.13`):
- `tests/test_main.py` (`4 passed in 2.18s`): verifies `app.main` live dispatch (`run_gui` default testing mode and `--video`/`--show-score` -> `run_demo`, closing `PoseStream` even on exception) and 3-task scoring + `rescore`.
- End-to-end `--fake` + real clue/model/session pipeline across all 3 tasks:
  - Task 1 (`K00_task1_try1_L.json`): `0.101s`, exit `0`
  - Task 2 (`K00_task2_try1_L.json`): `0.065s`, exit `0`
  - Task 3 (`K00_task3_try1_L.json`): `0.062s`, exit `0`
  - `rescore(out)`: `3/3` matching.
- Remaining live hardware step: run `python -m app.main` (testing mode) and `python -m app.main --show-score` across tasks 1–3 on Freya's Windows + NVIDIA laptop with Revallo and Freya (kept open under `Refs #10` until verified on the live laptop).

**Watch out:** Live mode requires a physical webcam/video session to test OpenCV/RTMPose interactively; automated CI runs headless without `rtmlib` or webcam hardware.

**Tests:** Existing `tests/contracts/*`, `tests/test_pipeline.py`, `tests/test_app_ui.py`, `tests/test_app_screen.py` stay green; new tests in `tests/test_main.py`.
