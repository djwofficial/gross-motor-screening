# Task card: Louise, Nov 22 (started Nov 20): clues and timeline on the team clips (#24)

**Goal:** check that the clues, tuned on the dataset's OpenPose points, still work on RTMPose
points from a real camera (the live app's detector) before testing day on Nov 29; tune
thresholds where they don't.

**Input → output:** Revallo's 27 adult team Try files (#31,
`docs/task_cards/team_clips_27_try.zip`, unzipped into `runs/team_clips/`) and Theresia's by-eye
sheet (#47, `eval/By-Eye Coding Sheet.csv`) → `runs/team_clips_check.csv` and a printed summary.

**Run:**
```bash
unzip -oq docs/task_cards/team_clips_27_try.zip -d runs/team_clips
python scripts/check_team_clips.py
```

**Matching (assumed, to confirm with Theresia):** folderN = P0N, clipN = Video N, Move_k = Task k.

**Done when:** most clips match the by-eye counts; clips that differ are listed.

## Results (2026-11-20, `main` at 1cb6ba5)

| Check | Result |
|---|---|
| Task 2 hops vs by eye | **7 / 9 match** |
| Task 1 seconds on one foot vs by eye | not comparable yet (see below) |
| Task 3 (Revallo's clues) | `jump_size` 0.00-0.03 on all 9 clips |
| App score vs Theresia's score | 8 / 27 (not meaningful yet: adults, task 1 times unclear, task 3 jump size 0) |

### Clips that differ

| Clip | Task | By eye | Program | Why |
|---|---|---|---|---|
| P01 Video 1 | 2 | 2 hops | 1 | not looked into yet |
| P02 Video 2 | 2 | 2 hops | 3 | not looked into yet |
| P01 Video 1-3 | 1 | 6 / 6 / 5 s | 0.6 / 0.5 / 1.0 s | low foot lift (below) |
| P02 Video 2 | 1 | 5 s | 0.7 s | not looked into yet |
| P02 Video 1, 3; P03 Video 1-3 | 1 | 6 s | 2.8-4.8 s | the by-eye time is about the whole clip (below) |

### Task 1: the by-eye times can't be used yet

Several by-eye times are longer than the clip itself: P01 Video 1 6.0 s (clip 5.5 s), P01 Video 2
6.0 s (5.2 s), P03 Video 2 6.0 s (4.9 s). All times are whole seconds. They look like the length of
the whole try (or of the original video), not the seconds on one foot. Example: P03 Video 1 stands
on both feet until 1.4 s and on one foot until the clip ends at 6.2 s; the program says 4.8 s,
the sheet says 6 s. **Asked Theresia how she timed task 1.**

### Task 1: very low foot lifts are missed (known limit)

P01 lifts the foot only about 0.15-0.3 torso lengths (90th percentile of the ankle gap 0.17-0.31;
P03: 0.65-0.77; dataset children's real lifts 0.6-0.9). `ONE_FOOT_GAP` (0.15) flickers and
`LIFT_PEAK` (0.3) then drops the stretch as a step toward the camera.

Tried on 2026-11-20 and **not adopted**: measuring each ankle's rise above its own standing height
(set when both feet are level) instead of the gap between the ankles.

| Check | Current rule | Rise rule (level < 0.1) | Rise rule (level < 0.05) |
|---|---|---|---|
| B040 Bad, cameras 1/2/3 (eye 0 s) | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0.2 |
| B249 Good (eye 2.3 s) | 2.30 | 1.73 | 2.30 |
| B210 Perfect (eye 4 s) | 3.93 | 3.90 | 3.97 |
| P01 Video 1 / 2 / 3 | 0.6 / 0.5 / 1.0 | 0.6 / 0.3 / 1.1 | 1.9 / 0.6 / 1.1 |
| Task 1 forest macro-F1 / kappa | **0.61 / 0.58** | 0.61 / 0.50 | 0.56 / 0.40 |

It helps one adult clip a little and costs the model a lot on the dataset it learns from, so the
current rule stays. Write it in the report as a limit: lifts under about 0.3 torso lengths are not
counted as standing on one foot (seen in 1 of 3 team members). On testing day KEE scores every try
on paper, and unsure tries say "check by teacher".

### Other notes

- The team clips are portrait phone videos (1080 x 1920), not the laptop webcam. Clues are in
  torso lengths, so the size doesn't matter, but the camera angle and distance differ from testing day.
- 10 clips are flagged `review_person_selection`: the files keep only the first detected person,
  so they can't be re-picked with `pick_child`. Results for those clips are marked in the CSV.
- Support leg is still pending for tasks 1 and 2 (not in Theresia's sheet). The clues don't need it.

## Still open

1. Theresia: how task 1 was timed; confirm the folder/clip ↔ participant/video matching; support legs.
2. Look into the 2 task 2 clips that differ (P01 Video 1, P02 Video 2).
3. Re-run once the timing is clear, then close #24.
