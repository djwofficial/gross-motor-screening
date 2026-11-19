# By-eye sheet (#46, taken over by Louise from Theresia on 2026-11-19)

`by_eye_sheet.csv`: 9 group B clips, 3 per task (one Bad, one Good, one Perfect), counted by eye in
Freya's replay (`viz/replay.py`). It is the reference the clue code is checked against (#20, #21,
#23, Revallo's #32). Only values someone actually counted are filled in; a blank means not counted
yet. Never fill a cell from the program's output.

| Column | Meaning | Tasks |
|---|---|---|
| seconds_on_one_foot | longest time standing on one foot, s | 1 |
| foot_down_s | second the lifted foot first comes down (after the first lift) | 1, 2 |
| touchdowns | times the lifted foot touches down and goes up again (`1+` = yes, counted as yes/no only) | 1, 2 |
| hops | number of hops on one foot (a touch-down is not a hop) | 2 |
| jumps | number of jumps with both feet | 3 |
| feet_together_takeoff / _landing | yes / no | 3 |

Videos (stick figure plus timeline bar; the `_slow_` ones play at 10 fps, a third of real speed) are
made locally into `runs/eye_check/` (git-ignored). Camera 2 is the right 45-degree view, which
shows lifts, hops and jumps more clearly than the front camera.
