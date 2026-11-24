# Task card: Theresia, 2026-11-25: agreement tables (eval/agreement.py), issue #54

**Goal:** the testing-day tables (app vs KEE, KEE vs KEE) and KEE's practice-clip check (20 group
B clips vs the answer key), ready before Nov 29. Built and tested on FAKE data only.

**Input → output:**
- the app's `log.csv` + the KEE paper sheets typed in (`code, task, try, kee1, kee2`) → per task
  and for all tasks: n, exact agreement, within one point, quadratic weighted kappa, 3x3 counts;
  for app vs KEE 1, app vs KEE 2, app vs KEE where both scorers agree, and KEE 1 vs KEE 2.
- KEE practice scores (`clip, [scorer], score`) + the answer key (`clip, task, score`, from #51)
  → the same table per scorer.
- "check by teacher" tries have no app score: they are counted, not compared.

**Files you may change:** `eval/agreement.py`, `tests/test_eval_agreement.py`.

**Steps:**
1. `python -m eval.agreement --fake runs/fake_agreement` shows the tables on made-up scores.
2. After KEE scores the practice clips: `python -m eval.agreement --practice kee_practice.csv
   --key answer_key.csv`.
3. Testing day, on the live laptop only: `python -m eval.agreement --log runs/2026-11-29/log.csv
   --kee kee_scores.csv -o runs/2026-11-29/agreement.md`. The output stays on the encrypted drive.

**Done when:** tables come out of a fake score file.

**Result (2026-11-24, FAKE scores from `make_fake_files`, 12 made-up codes × 3 tasks × 4 tries):**
`tests/test_eval_agreement.py` passes (14 tests). `python -m eval.agreement --fake ...` prints all
6 tables. For example, app vs KEE scorer 1 on the fake scores (the app was made to agree about 70%
of the time, so these numbers say nothing about the real app):

| Task | n | Exact | Within 1 | Kappa (quadratic) |
|---|---|---|---|---|
| 1 stand on one leg | 43 | 72% | 100% | 0.76 |
| 2 hop on one leg | 46 | 63% | 98% | 0.69 |
| 3 long jump | 45 | 73% | 100% | 0.78 |
| all | 134 | 69% | 99% | 0.74 |

**Watch out:**
- Testing-day scores, child codes and the log never go into an AI tool, git or GitHub issues.
  Paste only fake tables here.
- Small n per task: read the counts, not only the kappa. Kappa is "n/a" when it is not defined.
- Kappa: `sklearn.metrics.cohen_kappa_score(weights="quadratic")`, the same as `model/baseline.py`.
