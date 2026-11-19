# Task card: Louise, Nov 14 (late, started Nov 19): task 1 timeline and clues (#20)

Paste this whole card into your AI tool together with AGENTS.md. One card = one branch = one PR.

**Goal:** turn one stand-on-one-leg try into a timeline of moments (both feet / steady / wobble)
and the task 1 clues the random forest scores on. Daryn's model gate (CP3, Nov 19) and the clue
table (#22) wait on this.

**Input → output:** `points` float32 (T, 25, 3) BODY_25 pixels (dataset 1920x1080 OpenPose or live
1280x720 RTMPose), `t` float64 (T,) seconds →
- `timeline(1, points, t)`: `[{"start_s", "end_s", "state"}]`, state in `STATES[1]` = both_feet,
  steady, wobble (or `UNKNOWN` where the ankles are missing), in time order, no gaps or overlaps.
- `compute(1, points, t)`: exactly `CLUE_NAMES[1]`, all finite floats:
  `duration_s` (s), `one_foot_s` (longest one-foot stretch, s), `touchdowns` (times the lifted
  foot came down and went up again = one-foot stretches − 1; putting it down at the end doesn't count), `sway` (sideways range of the mid-hip while
  on one foot, torso lengths), `wobble_s` (total wobble time, s).

**Signature (don't change):** `timeline(task, points, t)`, `compute(task, points, t)` in
`features/clues.py`.

**Files you may change:** `features/clues.py` (task 1 parts only; task 3 is Revallo's),
`tests/test_clues_task1.py` (new), `scripts/` helper for the by-eye check (new file only).

**Steps:**
1. `normalize(points)` (torso lengths, y up). Per frame: ankle height gap = |y_L_ankle − y_R_ankle|;
   missing ankle (score < MIN_SCORE) → UNKNOWN frame.
2. On one foot = gap > 0.15 torso lengths (start value from the 2026-11-09 rough test; tune).
   Merge flickers shorter than 0.2 s into their neighbours.
3. Inside one-foot stretches: wobble = mid-hip sideways speed (or support-ankle shift) above a
   threshold for ≥ 0.2 s; otherwise steady. Start threshold is a guess; tune by eye.
4. Build the timeline as runs of states; compute every clue from the timeline (one source).
5. Pick 3 group B task 1 clips (one Bad, one Good, one Perfect, one person each), render them
   with `viz/replay.py` (timeline bar on), write by-eye counts: seconds on one foot, second the
   foot comes down, touch-downs. Use Theresia's by-eye sheet if it exists.
6. Compare and tune 0.15 / 0.2 s / wobble threshold; write the final values as constants with a
   comment saying where they came from.

**Done when:** 3 clips match the by-eye counts, including the second a foot comes down (within
about 0.2 s). Paste the comparison table and the 3 replay screenshots in the PR. Also print the
median `one_foot_s` per label over all task 1 files (expect it to rise Bad → Good → Perfect).

**Watch out:** units are seconds and torso lengths only; y points up after normalize; `leg` is the
support leg; 84 one-leg files are > 300 frames, don't cut; 28% of files have 2+ people (use
`pick_child` when Marty's lands; for now pick one-person clips); never say a child failed;
`duration_s` stays out of the model.

**Tests:** `tests/contracts/test_clues.py` stays green. New `tests/test_clues_task1.py`: a fake
try that lifts one foot for exactly 2.0 s gives `one_foot_s ≈ 2.0` and `touchdowns == 0` (two lifts → 1); a
try with both feet down gives `one_foot_s == 0`; the timeline covers t[0]..t[-1] with no gaps.
