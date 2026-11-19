# Dry run notes sheet: Wed Nov 25 (#43)

Print 2 copies: one for the note-taker, one for the scorer. Team members only (we agreed);
codes T01-T03, age 48, consent level 3. Checklist: `docs/task_cards/freya_2026-11-25_dry_run.md`.

Time: ______  Room: ______  Live laptop: ______  Operator: Freya
Performers: T01 ______  T02 ______  T03 ______  Scorer: ______  Notes: ______

**Each try:** stand still at the tape line **about 2 s before** and **about 2 s after** the
movement, otherwise automatic start/end can't be judged.
**Auto start / Auto end:** ✓ = the app found it by itself, ✗ = missed, M = operator used Space.
**Paper score:** the scorer writes 0 / 1 / 2 (Bad / Good / Perfect, paper Table 3) without
looking at the screen.

| # | Code | Task | Try | Leg | Auto start | Auto end | Guess right (task/leg) | Paper score | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 1 | T01 | 1 one leg | 1 | L | | | | | |
| 2 | T01 | 1 one leg | 2 | R | | | | | |
| 3 | T01 | 1 one leg | 3 | L | | | | | |
| 4 | T01 | 2 hop | 1 | L | | | | | |
| 5 | T01 | 2 hop | 2 | R | | | | | |
| 6 | T01 | 2 hop | 3 | L | | | | | |
| 7 | T01 | 3 jump | 1 | – | | | | | |
| 8 | T01 | 3 jump | 2 | – | | | | | |
| 9 | T01 | 3 jump | 3 | – | | | | | |
| 10 | T02 | 1 one leg | 1 | L | | | | | |
| 11 | T02 | 1 one leg | 2 | R | | | | | |
| 12 | T02 | 1 one leg | 3 | L | | | | | |
| 13 | T02 | 2 hop | 1 | L | | | | | |
| 14 | T02 | 2 hop | 2 | R | | | | | |
| 15 | T02 | 2 hop | 3 | L | | | | | |
| 16 | T02 | 3 jump | 1 | – | | | | | |
| 17 | T02 | 3 jump | 2 | – | | | | | |
| 18 | T02 | 3 jump | 3 | – | | | | | |
| 19 | T03 | 1 one leg | 1 | L | | | | | |
| 20 | T03 | 1 one leg | 2 | R | | | | | |
| 21 | T03 | 1 one leg | 3 | L | | | | | |
| 22 | T03 | 2 hop | 1 | L | | | | | |
| 23 | T03 | 2 hop | 2 | R | | | | | |
| 24 | T03 | 2 hop | 3 | L | | | | | |
| 25 | T03 | 3 jump | 1 | – | | | | | |
| 26 | T03 | 3 jump | 2 | – | | | | | |
| 27 | T03 | 3 jump | 3 | – | | | | | |

## Planned on-purpose tests (do each once, write the result)

| Test | Expected | Result |
|---|---|---|
| Backspace during RECORDING | Try not saved, back to IDLE | |
| Backspace at CONFIRM | Try not saved | |
| V on for T01, one try, Enter | `.avi` clip in `runs/2026-11-25/`, log `video` = 1 | |
| V with consent level 1 (extra test child T09) | "Video only for consent level 3" | |
| N and Q during a try | Refused until Enter or Backspace | |
| Someone walks behind the performer | Stick figure stays on the performer | |
| Close the window during a try, restart, same code | Earlier tries still there, numbering continues | |

## After the run (Done when)

- [ ] 27 JSON files and 27 rows in `runs/2026-11-25/log.csv` (cancelled tries not counted): ____ / ____
- [ ] `python rescore.py runs/2026-11-25` → ____ / 27 match (Daryn, #11)
- [ ] Automatic mode caught start ____ / 27 and end ____ / 27 → mode for Nov 29: auto / manual (Theresia, #53)
- [ ] Replays of all 27 tries checked by eye (`python -m viz.replay runs/2026-11-25/<file>.json`)
- [ ] Every bug found has a GitHub issue with its owner
