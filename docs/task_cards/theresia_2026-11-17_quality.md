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

## Added 2026-11-24: task 3 landing room (`check(..., task=3)`, `landing_room`)
Why: Marty's task 3 experiments (`docs/task_cards/marty_2026-11-23_t3_experiments.md`): in 31 of
44 missed Good/Perfect jumps the feet leave the bottom of the picture.

Measured on group B task 3 (1,091 files, scratch script, 2026-11-24). Drop = how far the lowest
foot point moves down during the try, in start torso lengths, for files whose feet stay in:

| Camera | Files | Feet reach bottom edge | Drop median | Drop p95 | Edge files with >= 1.5 torso room at start |
|---|---|---|---|---|---|
| 1 | 366 | 186 | 0.89 | 1.77 | 24 of 186 |
| 2 | 363 | 153 | 0.72 | 1.45 | 6 of 153 |
| 3 | 362 | 158 | 0.47 | 1.12 | 3 of 158 |

`ROOM_T3 = 1.5`: with at least 1.5 torso lengths of picture below the feet at the start line, files
whose feet reach the edge drop from 497 to 33. Team clips (adults, phone): the 9 task 3 clips
have 0.0-0.6 torso lengths of room, and 2 of them lost the feet at the bottom, so all 9 would get
the message. The setup needs the floor in front of the start line in the picture.

Optional keyword `task=None`: old calls are unchanged. The app passes `task=3` only when the
operator has chosen task 3 (Freya, `app/ui.py`, preset task). Tests:
`tests/test_live_quality_landing.py`.

Does enough room help the score? Task 3 random forest (`make_forest`, current 4 clues),
out-of-fold predictions, GroupKFold(5) by child, seeds 0-4, group B (1,084 files with a start
room; scratch script 2026-11-24). All files: F1 0.519, kappa 0.468, matching Daryn's 5-seed 0.521 /
0.465.

| Files | n | Macro-F1 | Kappa | Good recall |
|---|---|---|---|---|
| room >= 1.5 at start (check passes) | 265 | 0.565 | 0.572 | 0.28 |
| room < 1.5 at start (check warns) | 819 | 0.502 | 0.433 | 0.26 |
| feet stay in the picture | 593 | 0.559 | 0.534 | 0.30 |
| feet reach the bottom edge | 491 | 0.457 | 0.375 | 0.21 |

This compares different files (cameras and children differ), so it's a link, not proof. Good
recall stays low either way: the landing room doesn't fix the Good/Perfect mix-up (that needs
better clues).
