# Task card: Freya, Nov 25: lead the dry run (#43)

**Goal:** prove the whole app works on ourselves before testing day (Nov 29): 3 people × 3 tasks
× 3 tries in automatic mode, with the operator confirming or correcting each try. The count of
tries that automatic mode caught decides the mode for Nov 29 (Theresia, #53).

**Input → output:** the live laptop (NVIDIA, CUDA) + webcam → `runs/2026-11-25/`: 27 JSON files,
27 rows in `log.csv`, then `rescore.csv` (Daryn, #11) and 27 replays (Theresia, #55).

**Files you may change:** none for the run itself. Bugs found → one issue each, owner fixes.

## Before the day (by Nov 24)
- [ ] The live laptop is chosen and has the NVIDIA setup: `nvidia-smi` shows CUDA 13+, the
      install from README (onnxruntime-gpu, `pip install --no-deps rtmlib==0.0.16`), Python 3.12.
- [ ] `git pull` on `main` after the Nov 22 integration; `pytest` is green on that laptop.
- [ ] Revallo's fps check with video saving on (#34): 15 fps or more.
- [ ] Automatic framing (Windows Studio Effects) is off.
- [ ] Room: tape a start line where a standing person fills about half the picture height
      (Card 3 step 5); clear landing space toward the laptop for the jump.

## Roles on the day
| Role | Who | Job |
|---|---|---|
| Operator | Freya | runs the window, enters codes, confirms or corrects each try |
| Performers | 3 teammates (codes T01, T02, T03; consent level 3 so V can be tested) | do the tasks |
| Scorer | 1 teammate | writes 0/1/2 per try on paper, without looking at the screen (as KEE will) |
| Notes | 1 teammate | per try: did automatic start/end work? anything odd? (sheet below) |

## Run
1. `python -m app.ui --camera 0 --device cuda --out runs/2026-11-25` (the joined `app.main` if
   Daryn has replaced this entry point by then).
2. For each performer: type code, age (use 48), consent level 3 → readiness check → fix what it
   says → Enter.
3. Order for every performer: task 1 ×3, task 2 ×3, task 3 ×3. Alternate legs in tasks 1 and 2
   (L, R, L). Before each try press the task key (1/2/3) and the leg key so the confirm step
   starts from the right answer.
4. Automatic mode: wait for READY, the performer moves, the app ends the try. At the confirm
   step check the task and leg, then Enter. If automatic mode misses a try: Space to start and
   stop by hand, and write it on the notes sheet.
5. Test on purpose once each: Backspace during recording (try not saved), V on and off,
   N in the middle of a try (must refuse), Q in the middle of a try (must refuse), and one try
   with someone walking through the back of the picture.
6. Kill test once: close the window with the X during a recording, restart, enter the same
   code: earlier tries must still be there and numbering must continue.

## Notes sheet (one row per try)
| Performer | Task | Try | Leg | Auto start OK | Auto end OK | Manual used | Paper score | Notes |
|---|---|---|---|---|---|---|---|---|

## Done when
- [ ] 27 JSON files and 27 rows in `log.csv` (cancelled tries not counted).
- [ ] `python rescore.py runs/2026-11-25` → 27/27 match (Daryn, #11).
- [ ] Theresia's count of automatic-mode catches per task (#53) → decide auto or manual for Nov 29.
- [ ] Replays of all 27 tries (`python -m viz.replay runs/2026-11-25/<file>.json`) checked by eye.
- [ ] Every bug found has an issue with its owner.

**Watch out:** this is team data (we agree to it), not children's data, but keep `runs/` out of
git anyway (.gitignore). Never paste testing-day (Nov 29) data into an AI tool.
