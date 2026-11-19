# Task card: Freya, Nov 19: follow the child live (#33, handed over by Revallo)

**Goal:** the live window keeps following the same child when a parent or another child walks
into the picture, so their points never get into a try.

**Problem found:** `app/ui.py` called `common.pick.pick_child([people25])` on one frame at a time.
pick_child keeps its lock only inside one call, so live it chose the best-looking start-line
candidate again in every frame and could jump to the other person. The raw 26 points saved in
each JSON were also always the first detected person, not the picked child.

**Input → output:** everyone found in a frame, `(P, 25, 3)` BODY_25 → the index of the child, or
`None` (not found this frame).

**Files:** new `live/track.py` (`ChildTracker`), `app/ui.py` (`Runner.feed` uses it; reset for a
new child, when readiness is accepted, and when a clip starts again), `tests/test_live_track.py`.
`common/pick.py` is unchanged (its rules and constants are reused).

**Rules:** lock = Marty's start score (legs and feet well detected, near the centre, not at the
edge); then the nearest neck (or mid-hip) within Marty's step limit, scaled to the picture width
(220 px + 60 px per missed frame for 1920 px). Live the step grows for at most 2 missed frames
(pick_child: 5), because with 5 a person about 250 px away took over after 3 frames in which the
child was hidden. Lost for more than 2 s → lock again. 2 frames and 2 s are guesses: tune them
at the dry run.

**Evidence (2026-11-19, Freya's laptop, Python 3.12):**
- All 944 group B task 1-3 files with 2+ people in some frame, a jump = the picked neck moves
  more than 20% of the picture width from the last frame it was found:

  | Method | Jumps | Files with a jump |
  |---|---|---|
  | Old: pick_child on one frame at a time | 310 | 63 |
  | New: ChildTracker frame by frame | 1 | 1 |
  | pick_child on the whole clip (offline) | 3 | 2 |

  ChildTracker gives the same person as the offline pick_child on 161,114 / 162,265 frames (99.3%).
- `tests/test_live_track.py`: the old per-frame pick jumps to a second person who walks to the
  centre, the tracker doesn't; someone passing 160 px beside the child is ignored; a child hidden
  for 0.5 s is not replaced; the lock is taken again after the child left; the window saves the
  child's points and raw points, not the other person's; 80 dataset files agree ≥ 95% with the
  offline pick.

**Done when (#33):** "Test with 2 people in the picture: the track doesn't jump to the other
person". Shown on the dataset and in tests. **Still to do:** one live check on the live laptop with
2 team members (Revallo), e.g. at the dry run (notes sheet: "someone walks behind the performer").

**Watch out:** two people crossing right in front of each other can still swap (nearest neck);
keep helpers and parents outside the picture (run sheet).
