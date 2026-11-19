# Operator card: the app window on Nov 25 (dry run) and Nov 29 (testing day)

Print this page and keep it next to the laptop. Owner: Freya. The app is a **screening aid,
not a diagnosis**: the screen never shows a score, and we never tell a parent or child a score.

## Start

1. Live laptop (NVIDIA), plugged in. Windows Studio Effects → **Automatic framing off**.
2. In the repo folder, on the latest `main`:
   ```
   python -m app.ui --camera 0 --device cuda --mode auto --view stick --out runs/2026-11-29
   ```
   Use the day's date in `--out`. `--view` (stick / camera / neutral) is still to be agreed with
   KEE. On a laptop without NVIDIA add `--device cpu --pose-mode lightweight` (practice only).
3. Only the Revallo dashboard shows scores: **never open it on testing day** while KEE,
   parents or children can see the screen.

## Screens

| Screen (top left) | What it means | What you do |
|---|---|---|
| **SETUP** | New child | Type the child code (K01…), Enter; age in months, Enter; consent level 1/2/3 from the signed form, Enter |
| **READINESS** | Checks the picture | Fix what it says (see below), then Enter. Enter also works with problems left: write them on the notes sheet |
| **IDLE** | Waiting | The child walks to the tape line and stands still |
| **READY** | Child stands still | The next movement starts the try |
| **RECORDING** | Try is being recorded | Wait. The try ends when the child stands still again (about 1.5 s) |
| **CONFIRM** | Shows the guessed task and leg | Check them: Enter saves; 1/2/3 or L/R correct first; Backspace drops the try |

## Keys

| Key | Does |
|---|---|
| **1 / 2 / 3** | Task: 1 stand on one leg · 2 hop on one leg · 3 jump with feet together. Press before a try (or at CONFIRM to correct) |
| **L / R** | Support leg: the leg the child **stands or hops on** (not the lifted one) |
| **Space** | Start / stop a try by hand (use it when automatic start or end doesn't come) |
| **Enter** | Confirm and save the try |
| **Backspace** | Drop the try (not saved): child stopped, fell, someone walked in, practice try |
| **N** | Next child (only between tries) |
| **V** | Video on / off. Works only for **consent level 3** |
| **Q** | Quit (only between tries) |

## Per child (same order every time)

1. **N** (or the first SETUP) → code, age in months, consent level from the form.
2. Consent level 3 and the parent agreed to video → **V** (status bar: video ON).
3. READINESS → fix → Enter.
4. Before each try press the **task** key and the **leg** key, so CONFIRM starts from the right answer.
   Task 1 ×4–5, task 2 ×4–5, task 3 ×4–5. Alternate legs in tasks 1 and 2 (L, R, L, R …).
5. Each try: READY → child moves → RECORDING → child stands still → CONFIRM → Enter.
6. After the last try: copy the new files in `runs/<date>/` to the encrypted drive.

## When something goes wrong

| You see | Do |
|---|---|
| "Nobody is in the picture" | Child in the picture? Light on? Lens shutter open? |
| "The whole body is not in the picture" | Child to the tape line, or move the laptop back. Feet must be visible |
| "Points are unsure … too dark" | More light; nobody standing behind the child |
| "Frame rate is … fps" | Close other programs; laptop plugged in; more light |
| READY never comes | Child must stand still with both feet down for about 1 s |
| RECORDING doesn't end | The child must stand still for about 1.5 s. Or press **Space** |
| Automatic start missed the try | **Space** at the start and **Space** at the end next time |
| Wrong task or leg at CONFIRM | Press the right **1/2/3** and **L/R**, then Enter |
| "Video only for consent level 3" | Correct: no video for levels 1 and 2 |
| "Video not saved: …" | The points are saved; carry on, tell Revallo/Freya afterwards |
| "Age … is outside 36-53" | Carry on: the child is tested, the log marks it |
| The window crashed or closed | Earlier tries are safe. Start it again, type **the same code**: numbering continues |
| N or Q does nothing | Finish the open try first: Enter or Backspace |

## Never

- Never show or say a score to a child or parent; the parent summary says 僅供參考，不是診斷.
- Never put testing-day files (points, videos, log, codes, ages) into git, a GitHub issue or an AI tool.
- Never leave `runs/` on the laptop after the day: it goes to the encrypted drive and is deleted by 2026-12-25.
