# Task card: Theresia, 2026-11-17: readiness check (live/quality.py), issue #48

**Goal:** before each child, tell the teacher in plain words what to fix in the setup: nobody in
the picture, camera too close or too far, too dark, frame rate too low, feet outside the tape.

**Input → output:** points `(T, 25, 3)` (or `(25, 3)`) of the picked person over the last ~1.5 s,
measured fps, picture size, optional start area `(x0, y0, x1, y1)` in pixels → list of messages,
`[]` = ready.

**Signature (don't change):** `check(points, fps, frame_size=(1280, 720), start_area=None) ->
list[str]`. Contract strings stay: `"fps"`, `"whole body"`, `"Nobody is in the picture."`.

**Files you may change:** `live/quality.py`, `tests/test_live_quality.py`.

**Steps:**
1. Separate checks: body cut off at an edge or parts not found (too close), body shorter than
   25% of the picture height (too far, guess), big-joint confidence below 0.5 (dark, guess),
   0 < fps < 15 (fps 0 = not measured yet, no message), feet outside the start area (only foot
   points that were found count).
2. One message per problem, saying what the teacher can do. When the picture is dark, lost body
   parts are explained by the light message, not a second "whole body" message.
3. Tests on the 27 team clips and on 3 bad setups made from them.

**Done when:** each of the 3 bad setups gets the right message; normal team clips stay `[]`
(list the ones that don't).

**Result (2026-11-19):** first 1.5 s of each of the 27 team clips (1080x1920 portrait):
- normal setup: 23 `[]`; 4 "feet are cut off at the bottom" (folder1_Move_2_clip2,
  folder1_Move_3_clip3, folder2_Move_2_clip2, folder2_Move_2_clip3: the feet really touch the
  bottom edge of the picture).
- on the 23 clean clips: zoomed in 1.6x → "whole body" 23/23; confidence x0.45 → "Points are
  unsure" 23/23; start area beside the feet → "feet are not in the start area" 23/23; zoomed out
  0.2x → "looks small" 23/23. Each case gets exactly one message.

**Watch out:** the bad setups are simulated from the points, not filmed; a real dark room may lower
confidence differently. MIN_BODY_FRAC (25%) and MIN_MEAN_SCORE (0.5) are guesses; check them on the
live laptop (720p) and on the dry run. Pixels are used only for the picture edges and the start
area, which are picture coordinates.

**Tests:** `tests/contracts/test_segment_quality.py` stays green; new `tests/test_live_quality.py`.
