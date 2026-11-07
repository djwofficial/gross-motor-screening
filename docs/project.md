# The research project (background for people and AI tools)

Read this once to understand *why* the code exists. The rules are in [AGENTS.md](../AGENTS.md);
your own tasks are in [docs/people/](people/).

## Who and what
NTHU machine learning final project, Group 11, together with KEE (the preschool partner who
runs the testing day). Team: Daryn (lead), Marty, Revallo, Louise, Freya, Theresia.

**Question:** can a laptop webcam plus a small model score three gross motor tasks for
children aged 36-53 months about as consistently as a trained adult does?

**What we build:** a screening aid that helps a teacher notice which children may need a closer
look. It is never a diagnosis and never predicts the future.

## The three tasks and the score
| Task | Name | Clues we measure (features/clues.py) |
|---|---|---|
| 1 | Stand on one leg | time on one foot (s), touch-downs, sway (torso lengths) |
| 2 | Hop on one leg | hop count, longest pause (s), touch-downs |
| 3 | Standing long jump | feet together at take-off and landing, time in the air (s), jump size (torso lengths) |

Score per try: 0 Bad, 1 Good, 2 Perfect (the dataset's expert labels).

## Data
- **Training:** Korean multiview child motor development dataset (Kim et al. 2023,
  GigaScience 12, giad039; GigaDB 10.5524/102382, CC0). **Group B only.**
- Verified by `scripts/summarize_group_B.py` (2026-11-08): 4661 files, 106 children,
  1555 attempts; tasks 1-3: 3359 files, 1121 attempts; labels Bad/Good/Perfect
  61/169/173 (task 1), 157/90/105 (task 2), 60/71/235 (task 3). 28% of files have 2+ people.
- **Testing day:** Sunday 2026-11-29 at 新竹市東區親子館, about 15 children. Their data is only
  for testing, never for training, never in an AI tool or in git, deleted by 2026-12-25.

## Pipeline
webcam → RTMPose (rtmlib `BodyWithFeet`, 26 points) → convert to 25 BODY_25 points →
pick the child → automatic start/end → guess task and leg (operator confirms) → clues →
random forest → score + confidence + reasons → JSON per try + `log.csv`.
Below the confidence threshold the log says "check by teacher" instead of a score.
The screen shows no score in testing mode.

## How we judge success
1. **Offline, group B:** random forest vs baseline per task, GroupKFold by child,
   macro-F1 and quadratic weighted kappa. Gate on Nov 19: the forest beats the baseline.
2. **Product:** dry run on Nov 25 (3 people × 3 tasks × 3 tries = 27 tries), rescore matches
   live 27/27, and we count how many tries automatic start/end caught.
3. **Testing day:** agreement tables app vs KEE and KEE vs KEE.

## Timeline (checkpoints)
| CP | Date | What must exist |
|---|---|---|
| CP0 | Sun Nov 8 | Repo, format and stubs; everyone installed and ran Milestone 1 |
| CP1 | Wed Nov 11 | Every part has a first version that runs on its own |
| CP2 | Sun Nov 15 | Every part produces real output (not fake) |
| CP3 | Thu Nov 19 | Random forest beats the baseline; all parts complete |
| CP4 | Sun Nov 22 | App runs end to end on ourselves (integration) |
| CP5 | Wed Nov 25 | Dry run: 27 tries pass, rescore matches |
| CP6 | Fri Nov 27 | Code freeze, tag v1.0: product ready |
| — | Sun Nov 29 | Testing day at 親子館 |

Full plan: [docs/plan/task_split_nov27.md](plan/task_split_nov27.md) (same as the Google Doc tab
*Coding plan to Nov 27*).
