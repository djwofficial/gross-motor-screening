# Task card: Revallo, 2026-11-10: live CUDA FPS benchmark (#27)

**Goal:** give the team a repeatable check of RTMPose BodyWithFeet balanced mode on Freya's live laptop, with a person in the unmirrored webcam picture.

**Input → output:** webcam index, device, mode, warmup duration and measurement duration → console report with GPU name, ONNXRuntime providers of both actual model sessions, warmup and measured frame counts, elapsed seconds and mean FPS. No images or points are saved.

**Signature (don't change):** use `PoseStream(source, mode, device).read()` and `.close()`; preserve `python -m live.bench_fps --device cuda --mode balanced --seconds 30`.

**Files you may change:** `live/bench_fps.py`, `tests/test_live_bench_fps.py`, this task card. Shared files, dependencies, contracts and `live/pose.py` stay under their existing ownership.

**Steps:**
1. Check the official rtmlib model/session API and ONNXRuntime provider API.
2. Implement bounded warmup and measurement, checking both real sessions for CUDA and preventing silent runtime fallback.
3. Print separate warmup/measurement statistics and actionable errors; close the stream on failure or interruption.
4. Test with deterministic fake streams and clocks, without GPU, rtmlib or webcam dependencies.
5. Run the official command on Freya's laptop with a person present, and attach its actual output to the PR.

**Done when:** CUDA is used and mean FPS is at least 15 on Freya's laptop (30 is the target). Thresholds come from issue #27 and `docs/people/revallo.md`. Unit tests alone do not complete the hardware check.

**Watch out:** keep detector input unmirrored; an empty scene can skip pose inference and give misleading FPS. Model loading and warmup are excluded from measured FPS. Registered session providers do not prove individual kernel placement. Turn off Windows Studio Effects automatic framing before the real run.

**Tests:** deterministic timing/counts, empty scene, CPU fallback rejection, read failure, cleanup on error/Ctrl+C, CLI input validation. Existing conversion contracts and fake pipeline must remain green.

**Evidence (2026-11-12):** Python 3.12: 17 deterministic benchmark cases passed; full suite 65 passed, 1 skipped (group B ZIP absent in this worktree). Ruff check and formatting passed; CLI `--help` and `git diff --check` passed. Official live-laptop/person measurement is pending; no hardware performance result is claimed here. No camera was opened during these checks.

**Reading the report:** mean FPS includes every completed read. Person FPS divides frames containing a detected person by the same measured elapsed time, so an empty scene cannot inflate the passing result. The gate requires CUDA first in both sessions and at least 15 person frames/s; CPU runs are diagnostic and return a non-passing status. Exit codes: 0 pass, 1 below the gate/CPU, 2 invalid run, 130 interrupted.

**Sources inspected:**
- [rtmlib BodyWithFeet](https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/solution/body_with_feet.py)
- [rtmlib BaseTool session setup](https://github.com/Tau-J/rtmlib/blob/main/rtmlib/tools/base.py)
- [ONNXRuntime session/provider API](https://onnxruntime.ai/docs/api/python/api_summary.html)
- [NVIDIA SMI GPU inventory API](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
