# Task card: Theresia, 2026-11-17: start area (live/start_area.py), issue #49

**Goal:** the operator marks the taped start area once on the camera picture; it is saved, the
readiness check says when the feet are not in it, and automatic start only becomes "ready" while
the child stands in it.

**Input → output:** a camera picture → the operator drags a box on the mirrored picture →
`runs/start_area.json` with `(x0, y0, x1, y1)` in unmirrored pixels and the picture size.

**Signatures (don't change):** `check(points, fps, frame_size, start_area)`;
`Segmenter().update(points, t)`. New, optional: `Segmenter(start_area=None)`, so `Segmenter()` works
as before.

**Files you may change:** `live/start_area.py` (new), `live/segment.py`, `live/quality.py`,
`tests/test_live_start_area.py`, `tests/test_live_quality.py`.

**Steps:**
1. `python -m live.start_area --camera 0`: grab a picture, show it mirrored, drag a box, Enter
   saves, flipped back to unmirrored pixels. `--show` draws the saved box on the live picture.
2. `load(frame_size=...)`: rescales when the camera size differs; `None` when nothing saved.
3. `feet_inside(points, area)`: one rule for both users (80% of the found foot points inside;
   `None` when no foot point is found).
4. `live/quality.py` and `live/segment.py` use it.

**Done when:** the area is saved and the readiness check uses it (tests); the Segmenter only becomes
ready inside the area.

**Result (2026-11-22):** `tests/test_live_start_area.py` passes: save/load (with rescaling to a
new camera size), refusing an empty or out-of-picture box, the mirror flip (a box dragged at x
100-300 on the mirrored 1280-wide picture is saved as x 980-1180), drawing it back on the mirrored
picture, `feet_inside` true/false/unknown, the readiness check with a saved area (feet in: `[]`;
area elsewhere: "The feet are not in the start area"), and the Segmenter: with an area around the
feet it gives the same events as before for all 6 fake tries (tasks 1-3, legs L/R) up to the end;
with an area elsewhere it never becomes ready. The marking window (`cv2.selectROI`) is tested
with a stand-in for the mouse drag; check it on the live laptop webcam.

**Still to hook up (not Theresia's files):** `app/ui.py` (Freya) and `app/main.py` (Daryn) create
`Segmenter()` and call `check(...)` without the area. Each needs
`area = live.start_area.load(frame_size=frame_size)`, `Segmenter(start_area=area)` and
`check(..., start_area=area)`.

**Watch out:** mirrored display vs unmirrored points (AGENTS.md rule 6). The area is in pixels on
purpose: it is a place in the picture. After a long jump the child lands outside the area, so the
next "ready" waits until the child is back on the tape (intended). `app/ui.py` is Freya's: she adds
the 3 lines that load the area and pass it on.

**Tests:** `tests/contracts/test_segment_quality.py` stays green; new `tests/test_live_start_area.py`.
