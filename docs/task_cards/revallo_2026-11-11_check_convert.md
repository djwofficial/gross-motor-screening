# Task card: Revallo, 2026-11-11: check Halpe-26 to BODY_25 (#28)

**Goal:** verify the joint order against official MMPose metadata and produce a
reproducible labelled calibration plot. The plot uses synthetic points; a real
unmirrored RTMPose frame remains a separate live check.

**Input -> output:** synthetic Halpe-26 coordinates `(26, 2)` in image pixels and
scores `(26,)` become BODY_25 points `(25, 3)`, plus a labelled PNG showing both
orders and anatomical left/right. The right wrist is raised asymmetrically.

**Signature (don't change):** `halpe_to_body25(keypoints, scores) -> (..., 25, 3)`;
`HALPE_TO_BODY25` and `HALPE26_NAMES` keep their current meaning.

**Files you may change:** `live/convert.py`, this card, and the generated
`docs/task_cards/revallo_2026-11-11_check_convert.png`.

**Steps:**
1. Compare all Halpe names and all BODY_25 indices against official MMPose
   `configs/_base_/datasets/halpe26.py`; record the immutable source below.
2. Add an optional `--plot PATH` CLI using lazily imported, already-pinned
   matplotlib. Construct the calibration frame by anatomical name, independently
   of the conversion index list, and call the real conversion function.
3. Render both index/name orders without mirroring, highlighting Halpe right
   wrist 10 becoming BODY_25 right wrist 4. Run the existing conversion contracts.

**Done when:** the source comparison agrees and a labelled, unmirrored real
webcam/RTMPose frame confirms anatomical left/right, with the right wrist at
BODY_25 point 4. The synthetic plot is an intermediate calibration check;
it does not complete the real-frame requirement.

**Watch out:** anatomical right is on the viewer's left for this front-facing,
unmirrored drawing. Point 17 (`head`) is dropped; no output point is invented.
Coordinates in the diagram are illustrative pixels, not movement clues.

**Tests:** keep `tests/contracts/test_convert.py` unchanged and passing. Check the
CLI help, render the PNG with a headless backend, and inspect the result.

## Evidence and reproduction

- Official metadata, checked 2026-11-12:
  [MMPose halpe26.py at 759b39c](https://github.com/open-mmlab/mmpose/blob/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/configs/_base_/datasets/halpe26.py).
- Read-only source comparison: all 26 names and 25 mapped indices agreed;
  BODY_25 right wrist 4 takes Halpe right wrist 10.
- Render command (from the repository):
  `python -m live.convert --plot docs/task_cards/revallo_2026-11-11_check_convert.png`
- Contract command: `python -m pytest tests/contracts/test_convert.py -q`.
- Image generated and visually inspected: [synthetic calibration PNG](revallo_2026-11-11_check_convert.png).
- Validation completed 2026-11-12: existing conversion contracts 2 passed;
  scoped Ruff check/format passed; CLI help and headless PNG generation passed.
- Full local suite with the verified group B archive: **49 passed in 32.77s**;
  repository-wide Ruff format/check and `git diff --check` passed.
- Status: source and synthetic checks complete; live-frame evidence pending.
- Live limitation: no camera, child data, or RTMPose output is used in this
  synthetic calibration. The original real-frame check remains outstanding.
