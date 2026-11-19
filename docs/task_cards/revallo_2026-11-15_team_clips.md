# Task card: Revallo, 2026-11-15: process the adult team clips

**Issue:** [#31](https://github.com/djwofficial/gross-motor-screening/issues/31).
**Branch:** `revallo/process-team-clips`.

**Download:** [27 adult team Try files and manifest](team_clips_27_try.zip).

**Goal:** Convert the team's 27 adult recordings into valid BODY_25 Try files,
with a clear per-file manifest for handoff to Louise and Theresia.

**Input -> output:** Adult team videos from the operator-provided Drive folder
-> one validated `.npz` Try per recording, plus an aggregate report and ZIP for
handoff. Keep the original recordings in team storage.

**Signatures (don't change):** Reuse `PoseStream`, `halpe_to_body25`,
`pick_child`, `capture_try`, and `common.format.save/load/validate`.
The recording pipeline is in PR #59; record its exact commit in the evidence.

**Files you may change:** This task card, its handoff ZIP, and local ignored processing artifacts.
Fix any discovered pipeline defect on its own existing capture branch, without
changing shared modules, contract tests, or dependency pins.

**Steps:**
1. Inspect the Drive listing, count the clips, and confirm task/support-leg
   labels from filenames or the operator. Do not invent labels.
2. Install the already-pinned live runtime and model files locally.
3. Process unmirrored videos through the real RTMPose detector, preserving
   media timestamps and the pipeline's 30 fps resampling.
4. Validate every saved Try and record source identity, run parameters,
   output path, frame count, duration, and any processing errors locally.
5. Package the successful outputs for handoff and record aggregate evidence.

**Done when:** 27 valid Try files exist and the operator has a clear handoff
bundle for Louise and Theresia. Do not claim a missing, failed, or artificially
inferred clip is a successful real-detector result.

**Watch out:** This input is adult team practice footage. Testing-day child
videos/points are never handled in this workflow or sent to an AI. Task 1/2
leg labels refer to the support leg. Revallo authorized the derived skeleton
ZIP for team handoff. Keep source media and testing-day data out of Git.

**Tests:** Existing capture/pose contracts and behavior tests; a real input
video must produce a file accepted by `common.format.load` and `validate`.

**Evidence recorded on 2026-11-19.** The local CUDA run on 2026-11-15
completed all 27 adult recordings with zero extraction errors. The capture
commit was `ea62542bc0bd3858b17ceca946be2dddfbbc2723` (PR #59). RTMLib
0.0.16 with BodyWithFeet balanced and ONNX Runtime GPU 1.31.1 processed
3,726 source frames and produced 3,873 BODY_25 frames on the 30 FPS grid.
The observed source frame count matched the container count for each clip.
CUDA was the first provider registered in both sessions after inference.

The user confirmed Move_1 = task 1, Move_2 = task 2, Move_3 = task 3.
The relabel pass reloaded and validated all 27 files without running inference
again. The 18 clips for tasks 1 and 2 retain pending support-leg labels as
requested. Ten clips contain multiple detections and need person selection
review because the shared picker takes the first detection. Five of the
nine task 3 clips have no such flag. These are format and detection checks.

The handoff ZIP passed its archive integrity check and contains the 27
validated Try files, a CSV manifest, and usage notes. It preserves the points,
timestamps and per-clip review flags. Local paths and Drive file IDs are
omitted. The manifest keeps the team video folder and filename for matching
each Try to its source.
