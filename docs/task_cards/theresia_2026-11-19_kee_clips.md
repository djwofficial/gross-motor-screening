# Task card: Theresia, 2026-11-19: KEE practice clips (viz/kee_clips.py), issue #51

**Goal:** about 20 group B stick-figure clips KEE scores before testing day, so we can see how
KEE's scores compare with the dataset's expert scores (`eval.agreement --practice ... --key ...`).

**Input → output:** `data/raw/B_skel.zip` (group B, CC0) → `runs/kee_practice/clips/clip_01..20.mp4`
(for KEE), `runs/kee_practice/answer_key.csv` (clip, task, score, label, file: keep it away from KEE)
and a blank `runs/kee_practice/kee_practice.csv` for KEE's scores.

**Files you may change:** `viz/kee_clips.py` (new), `tests/test_viz_kee_clips.py` (new).

**How the clips are picked:** tasks 1-3, at least 60 frames (2 s, a guess), 2 clips per task ×
score (18) plus 2 more from the biggest cells, at most one clip per child, one camera per attempt,
fixed seed. Clip numbers are shuffled. The clip shows only the stick figure, the time and the
task number: no child code, file name or score (`viz.replay.render` with a stripped meta).

**Steps:**
1. Put `B_skel.zip` in `data/raw/` (GigaDB 10.5524/102382, or copy it from Marty; never commit it).
2. `python -m viz.kee_clips --zip data/raw/B_skel.zip --out runs/kee_practice`
3. Watch 2-3 clips: whole movement, no score on screen.
4. Give `clips/` and `kee_practice.csv` to KEE. Keep `answer_key.csv` until KEE has scored.

**Done when:** 20 mp4 files and a separate answer key.

**Result (2026-11-24):** `tests/test_viz_kee_clips.py` (3 tests) builds a fake zip in the group B
format. It checks 20 clips from 20 different children covering all 9 task × score cells, the 2
extras going to different tasks, the shuffled clip names, that the answer key matches the clips
and sits outside `clips/`, and that the clip meta has no label or child.

Real run: `B_skel.zip` downloaded from the official GigaDB link (211,959,553 bytes,
`python -m zipfile -t` passed). `python -m viz.kee_clips --zip data/raw/B_skel.zip --out
runs/kee_practice` wrote 20 mp4 files (81-298 frames, 2.7-9.9 s), `answer_key.csv` and a blank
`kee_practice.csv`. Spread of the 20 (from the script's output; which clip has which score stays
in the answer key):

| Task | 0 Bad | 1 Good | 2 Perfect |
|---|---|---|---|
| 1 stand on one leg | 2 | 2 | 3 |
| 2 hop on one leg | 2 | 2 | 2 |
| 3 long jump | 2 | 2 | 3 |

**Watch out:** `*.mp4` is gitignored; share the clips outside git. Group B only.
