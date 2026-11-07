# Task card: Louise, Nov 10: RTMPose fps on every laptop, pick the live laptop

Paste this whole card into your AI tool together with AGENTS.md. One card = one branch = one PR.

**Goal:** know how fast rtmlib `BodyWithFeet` runs on each NVIDIA laptop, so we pick the live
laptop for the dry run, testing day and demo.

**Input → output:** webcam (or a recorded clip) → a printed line per run:
`laptop, GPU, device, mode, frames, mean fps, mean fps with video saving on`.

**Signature (don't change):** `live.pose.PoseStream(source, mode, device).read()` returns
`(t, keypoints (P,26,2), scores (P,26), frame)` or `None`. Use it; don't call rtmlib elsewhere.

**Files you may change:** `live/bench_fps.py`, `live/pose.py`, `live/capture.py`, new tests in
`tests/test_live_*.py`.

**Steps:**
1. On your Mac: `python -m live.bench_fps --device cpu --mode balanced --seconds 30` runs.
   rtmlib downloads the ONNX models on the first run (YOLOX-m + RTMPose-m Halpe-26).
2. Turn off auto-framing (Mac: Center Stage; Windows: Studio Effects "Automatic framing").
3. Ask Marty, Revallo, Freya and Theresia to run it with `--device cuda` on their laptops
   (after `nvidia-smi` shows CUDA 13.x or higher). Also try `--mode lightweight`.
4. Put the table in the PR and in the Doc; pick the live laptop (target 30 fps, minimum 15).

**Done when:** fps table for the 4 NVIDIA laptops, live laptop chosen.

**Watch out:** measure with a person in the picture (no person = faster). Time stamps come
from `time.perf_counter()` per frame. Run the detector on the unmirrored picture.

**Tests:** `tests/contracts/test_convert.py` stays green. CI has no webcam or rtmlib, so don't
import rtmlib at module level.
