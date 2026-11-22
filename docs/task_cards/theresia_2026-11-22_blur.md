# Task card: Theresia, 2026-11-22: face blurring (viz/blur.py), issue #52

**Goal:** a blurred copy of every level-3 try video in which no face can be seen, checked frame by
frame, so only the blurred copy is ever shown.

**Input → output:** the app's try video (`runs/<date>/<child>_<token>.avi`, unmirrored, MJPG) →
`<video>_blurred.avi` + `<video>_blurred.check.csv` (one row per frame: people found, heads placed,
status `ok` / `whole_nobody` / `whole_unsure`, and `eye_check` filled in by the eye check).

**Signature:** `blur_video(video_path, points_json, out_path)` (kept; `points_json` is optional and
only checks that the JSON's `meta.video_file` names this video).

**Files you may change:** `viz/blur.py`, `tests/test_viz_blur.py`.

**How it works:**
- RTMPose runs again on the video's own frames and blurs EVERY person found, not only the picked
  child. The JSON points are not used for the places: they start 1 s before the video (pre-roll),
  hold only the picked child, and the recorder may drop frames.
- One ellipse per person from nose, eyes and ears (BODY_25 0, 15-18). Its size is the largest of
  the head-point spread, shoulder width and torso length, with a margin (sizes are guesses). With no
  head point found, the ellipse goes above the neck.
- Each frame also gets the ellipses of the 2 frames before and after it.
- Fail-safe: a frame with nobody found, or with a person whose head can't be placed, is blurred
  whole.

**Steps:**
1. `python -m viz.blur runs/2026-11-25/T01_xxxx.avi --device cpu` (about 1 s per frame on this
   laptop's CPU in balanced mode; `--device cuda` on the NVIDIA laptop).
2. `python -m viz.blur runs/2026-11-25/T01_xxxx_blurred.avi --check`: step through every frame
   (D / right arrow = next, A / left = back, F = a face can be seen here, Q = save and quit).
3. `python -m viz.blur runs/2026-11-25/T01_xxxx_blurred.avi --summary` prints "PASSED" only when
   every frame was seen and none is marked F.

**Done when:** 3 team videos: no frame shows a face.

**Result so far (2026-11-22):**
- Tests: `tests/test_viz_blur.py` (9 tests, synthetic frames). They cover the ellipse covering all
  head points, the ellipse above the neck, the frame statuses, neighbour frames, that only the
  inside of the ellipse changes, the whole-frame fail-safe, video + CSV writing, a JSON that names
  another video, and the eye-check pass rule.
- Run on one group B dataset clip (CC0, `B356_GMS_1_2_2.mp4`, 254 frames, CPU): 169 frames got face
  ellipses, 74 were blurred whole (nobody found), 11 were blurred whole (unsure). About 3 min,
  including the one-time model download. Not eye-checked; this clip only tests the pipeline.

**Eye check on 3 team videos (fill in):**
| Video | Frames | Seen | Face visible | Passed |
|---|---|---|---|---|
| team 1 | | | | |
| team 2 | | | | |
| team 3 | | | | |

**Watch out:**
- Testing-day videos (Nov 29) stay on the encrypted drive: blur and check them on the live laptop,
  never in an AI tool.
- Keep the unblurred original only as long as the consent form allows.
- If a frame shows a face, note its number; the ellipse margin may need to grow.
