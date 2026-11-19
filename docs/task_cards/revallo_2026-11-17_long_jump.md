# Task card: Revallo, 2026-11-17: standing long jump

**Goal:** Extract the standing long jump timeline and movement clues from
BODY_25 points for [#32](https://github.com/djwofficial/gross-motor-screening/issues/32).

**Input and output.** BODY_25 points and timestamps produce before, air,
after, or unknown intervals and the existing task 3 clue names.

**Signatures and files.** Keep `timeline` and `compute` unchanged. Add task 3
functions in `features/clues.py` and behavior tests in `tests/test_live_long_jump.py`.

**Approach.** Normalize by the shared torso ruler. Look for simultaneous
ankle lift and hip rise between nearby lower poses, using timestamps to
measure the interval. Repeated jumps can produce multiple air intervals.
The first complete interval supplies ankle gaps divided by hip width at
takeoff and landing, air time in seconds, and projected horizontal hip travel
in torso lengths. Missing pose data can yield unknown intervals and zero clues
in the existing numeric schema.

The [team plan, Job Card 2](https://docs.google.com/document/d/1QlTnea4xj9TfZC4iJMadVhvg4ThUVe4_Gz5CyzK38gY/edit)
also describes a torso-size ratio for frontal camera 1. The shared clue
interface receives points and timestamps without a camera ID. This branch
uses projected horizontal travel for every clip. Camera-specific distance
needs agreement on how to pass the viewpoint through the shared interface.

**Done when.** The existing contracts pass and three adult team recordings
agree with by-eye counts. Save only aggregate evidence. The thresholds are
initial engineering guesses and do not define a scoring standard.

**Evidence on 2026-11-19.** Visual inspection of decoded frames with media
timestamps and the measured timeline agreed on the airborne episode counts
for three single-person adult clips. The first episode outputs rounded to
milliseconds were 2.500–2.767 s, 3.033–3.300 s, and 1.700–1.867 s. Visual
boundaries were approximate because of the frontal view and floor shadows.
The three clips contained 2, 2, and 3 airborne episodes respectively. No
direct forward distance measurement was available. The local full suite had
82 passes on Python 3.12, including the Group B check, the existing clue
contracts, and the new behavior tests. Ruff format and lint passed.
