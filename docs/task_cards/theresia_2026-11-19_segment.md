# Task card: Theresia, 2026-11-19: automatic start and end (live/segment.py), issue #50

**Goal:** automatic start and end of a try that work on real camera points, not only on fake tries:
still about 1 s at the start line = ready, movement = start, still about 1.5 s or leaving = end.

**Input → output:** one frame of the picked child's points `(25, 3)` (or `None`) and its time in
seconds → `"idle" | "ready" | "start" | "recording" | "end"`.

**Signature (don't change):** `Segmenter().update(points, t)`, `seg.reset()`.
`tests/contracts/test_segment_quality.py` stays green.

**Files you may change:** `live/segment.py`, `tests/test_segment_team_clips.py`.

**Steps:**
1. Run the Segmenter on the 27 team clips (`docs/task_cards/team_clips_27_try.zip`).
2. Start found = exactly one start, within 0.5 s of the first lift, hop or jump in
   `features.clues.timeline` (Louise's code, the closest thing to a by-eye start time we have).
3. End: the clips stop 0.1-1 s after the movement, so the end is tested on a SIMULATED 2 s tail:
   the last pose held still ("hold"), or nobody in the picture ("leave").
4. Tune, list the misses with a reason.

**Done when:** start and end found for most of the 27 team clips; misses listed.

**Changes:**
- `still_speed` 0.35 → 0.5 torso lengths per second. Standing still on the phone clips measures
  0.2-0.5 (sway plus detector noise), so 0.35 was never "still" in 5 clips. Grid on the 27 clips:
  0.35 → 20/27 starts, 0.45 → 22, 0.5 → 23, 0.6 → 24 (0.6 is close to move_speed 0.8, not used).
  `ready_still_s` stays 1.0 (the agreed "about 1 s"; 0.8 would add one clip).
- A foot coming off the floor (ankle height difference > `feet_tol`, 0.15 torso lengths) now starts
  the try. Before, a slow lift stayed under move_speed and the ready pose followed it, so the start
  came late or never (folder1_Move_1_clip3, folder3_Move_1_clip1). 23 → 24/27.
- `python -m live.segment --team-clips` prints the table below.

**Result (2026-11-22, `python -m live.segment --team-clips`):**
- start found **24/27**
- end found in the clip itself 1/27 (the clips are cut right after the movement)
- end with a SIMULATED 2 s tail: leaving **25/27**, holding still **19/27**
- the 6 fake tries (tasks 1-3, legs L/R) still give one start and one end each

**Misses:**
| Clip | Miss | Reason |
|---|---|---|
| folder2_Move_2_clip2 | no start, no end | never ready: the hop starts 1.47 s into the clip, after too little standing still |
| folder3_Move_1_clip2 | no start, no end | the foot goes up at 0.67 s, before 1 s of standing is possible (in the app the child stands on the tape first) |
| folder1_Move_2_clip2 | start 0.64 s early | the foot comes up briefly at 1.8 s before the first hop at 2.37 s; the whole try is still recorded |
| folder1_Move_1_clip1, folder2_Move_2_clip1, folder2_Move_2_clip3, folder3_Move_2_clip2, folder3_Move_2_clip3 | no end when holding still | the clip stops with one foot up, so the try is not over (correct); leaving ends them |
| folder1_Move_3_clip3 | no end when holding still | the last pose (after landing) differs from the ready pose by 0.6 torso lengths; leaving ends it |

**Watch out:** the tails are simulated, not filmed; recording a few team tries with 2 s of
standing after the movement would test the end for real. The clips are adults on a phone in
portrait; children on the 720p webcam may sway more. Check the misses again on the Nov 25 dry run
(`eval/dry_run_count.py`, #53).

**Tests:** `tests/contracts/test_segment_quality.py` (unchanged), new
`tests/test_segment_team_clips.py` (24/27 starts, ≥25 leaving ends, ≥19 still ends, a slow lift
starts the try).
