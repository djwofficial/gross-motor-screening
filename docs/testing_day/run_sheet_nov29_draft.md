# Testing-day run sheet, Sun Nov 29, 新竹市東區親子館 (DRAFT to agree with KEE by Nov 25)

Based on the team workspace (Card 5 and "Testing with real children"). Everything marked
**[ask KEE]** is still open. The app is a **screening aid, not a diagnosis**: it never says a
child failed and never predicts the future.

## Still to agree with KEE

| # | Question | Answer |
|---|---|---|
| 1 | **Hours** on Nov 29 (arrive / first child / last child / leave) | |
| 2 | **How many children** aged 3-5 are coming, and is there a sign-up list with times? | |
| 3 | **Who scores** each try on paper (2 KEE scorers, workspace open question 5) | |
| 4 | How to score the **gaps in the rules**: standing on one leg 1-3 s, what counts as "a long distance" jump, does a practice try count (open question 9) | |
| 5 | **Tries per task** a 3-4-year-old can do comfortably (plan: 4-5, open question 6) | |
| 6 | What the **laptop screen shows** during testing: stick figure, camera picture or a neutral picture (linked to the consent form) | |
| 7 | The **gift** for each child (consent form: 尚未決定) | |

## Before the day

- [ ] Print **2 consent forms per child** plus spares (consent_form1.docx, with the date Nov 29 filled in)
- [ ] Print **KEE scoring sheets** (one row per try: code, task, try, leg, score 0/1/2) and pens
- [ ] Print the **operator card** (`docs/testing_day/operator_card.md`)
- [ ] Bring: live laptop + charger, extension cord, **tape** for the start line, the **encrypted drive**, the gift
- [ ] Day before: app runs on the live laptop from the latest `main`; one practice try with V on

## Room

- Laptop on a table facing the child. Tape a start line where a standing child fills about half
  the picture height (workspace Card 3, step 5).
- The child jumps **toward the laptop**: keep the landing area and the space up to the table clear.
- Mark where KEE and helpers stand, **outside the picture**. Parents stand behind the laptop.
- Check the light: the readiness check must not say "too dark" or a low frame rate.

## Roles

| Role | Who | Job |
|---|---|---|
| Explain and guide | KEE | Explain the form to parents; show each movement; guide the child |
| Paper scoring | 2 KEE scorers | Score every try 0/1/2 with the dataset rules (paper Table 3), separately, without looking at the screen |
| Operator | Freya | Run the app: enter code, age in months, consent level; confirm or correct each try |
| Helpers | team | Waiting children and parents outside the picture; time per child; copy files after each child |

## Per child (about 10-15 min, a guess to check on Nov 25)

1. **Consent:** KEE explains the form. The parent signs **2 copies** and ticks **one** of the 3 levels:
   1 points only, nothing shown in class · 2 the stick figure may be shown in class ·
   3 video kept and may be shown with only the child's face blurred.
2. **Code:** give the child a code (K01, K02 …) and write it **only on our copy** of the form.
   Write down the age in years and months.
3. **Assent:** ask the child if they want to play. If not, stop there.
4. **Operator:** N → code, age in months, consent level. Level 3: V on.
5. **Tasks**, same order for every child: 1 stand on one leg → 2 hop on one leg → 3 jump with
   feet together. KEE shows each movement first. 4-5 tries each **[ask KEE]**, alternate legs in
   tasks 1 and 2, rest whenever the child wants.
6. **Scoring:** the 2 KEE scorers write 0/1/2 for every try on paper; the operator confirms each
   try in the app. Nobody tells the child or parent a score.
7. **After the child:** copy the new files to the encrypted drive.

## After the day

- [ ] Same count: point files = log rows = paper scores (per child and task)
- [ ] Type the paper scores into a sheet; a second person checks the typing
- [ ] Back up to the encrypted drive; delete `runs/` from the laptop
- [ ] All data deleted by **2026-12-25** (consent form)

## Data rules (consent form)

- Testing-day data (points, videos, log, codes, ages) stays on the encrypted drive.
- Never in git, a GitHub issue, a chat, or any AI tool.
- Video only for consent level 3; faces are blurred later when editing, not in the app.
