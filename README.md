# Gross motor screening (NTHU ML final project, Group 11 × KEE)

A live laptop-webcam **screening aid** for preschool teachers: one child does one task
(stand on one leg, hop on one leg, standing long jump), RTMPose finds the body points, and a
model gives a score 0 Bad / 1 Good / 2 Perfect a few seconds after each try.
**It is not a diagnosis.**

- Plan, tasks and checkpoints: team Google Doc, tab *Coding plan to Nov 27*.
- Who is doing what right now: [project board](https://github.com/users/djwofficial/projects/3).
- Background: [docs/project.md](docs/project.md). Your own tasks: [docs/people/](docs/people/) (tell your AI "I'm <name>").
- Rules for people and AI tools: [AGENTS.md](AGENTS.md). Read it before your first PR.

## 1. Install (Python 3.12)

**Windows + NVIDIA laptop** (Marty, Revallo, Freya, Theresia; also the live laptop)
1. Check the driver: `nvidia-smi` must show `CUDA Version: 13.x` or higher. If lower, update the
   NVIDIA driver. You do **not** need to install CUDA or cuDNN yourself: pip brings them.
2. Install Python 3.12 from python.org (tick "Add python.exe to PATH").
3. In the repo folder:
   ```bat
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt -r requirements-dev.txt
   pip install --no-deps rtmlib==0.0.16
   python -c "import onnxruntime as o; print(o.get_available_providers())"
   ```
   The last line must list `CUDAExecutionProvider`.

**Mac (Apple silicon)** (Daryn, Louise): rtmlib runs on the CPU here.
```bash
uv python install 3.12          # or install Python 3.12 from python.org
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements.txt -r requirements-dev.txt
uv pip install --no-deps rtmlib==0.0.16
```

Why `--no-deps` for rtmlib: it always pulls the CPU `onnxruntime`, which breaks
`onnxruntime-gpu`. Its real dependencies are already pinned in `requirements.txt`.

## 2. Dataset (group B only, never committed)
Download `B_skel.zip` (202 MB) from GigaDB, doi [10.5524/102382](https://doi.org/10.5524/102382)
(CC0), and put it in `data/raw/B_skel.zip`. Don't unzip it; the code reads the zip directly.

**Milestone 1:**
```bash
python scripts/summarize_group_B.py
```
The first line must be `files: 4661 | children: 106 | attempts: 1555`.

## 3. Run
```bash
python -m app.main --fake                    # whole pipeline on a fake try (auto start/end)
python -m app.main --fake --task 1 --mode manual
python rescore.py runs/<date>                # scores the saved tries again; must match
pytest                                       # contract tests + smoke test (CI runs the same)
ruff format . && ruff check .                # before every commit
```
The live webcam mode is joined on Nov 22. Until then every part works on fake data
(`common.format.make_fake_try`) or the dataset.

## 4. Repo layout
```
common/    shared format, BODY_25 names, torso-length geometry, pick_child   (Daryn; pick: Marty)
data/      group B loader, raw/ (ignored)                                   (Marty)
model/     baseline, task classifier (Marty); random forest, predict, models/ (Daryn)
features/  moment timeline, movement clues, reason text        (Louise; task 3: Revallo)
eval/      evaluation report (Louise), agreement tables (Theresia)
live/      webcam + RTMPose, 26->25 conversion, fps check                   (Revallo)
live/      segment.py (auto start/end), quality.py (readiness check)        (Theresia)
app/       main.py (joins the parts, Daryn), ui.py + session.py (Freya)
viz/       stick-figure replay (Freya), face blur (Theresia)
scripts/   summarize_group_B.py (Milestone 1)
tests/     contracts/ (the fence for every part) + test_pipeline.py (smoke test)
docs/      task card template and task cards
runs/      app output (ignored; testing-day data goes to the encrypted drive, never git)
```

## 5. Working together
One task = one branch = one pull request. CI must be green before merging. Change only your
own folder; shared files (`common/`, `tests/contracts/`, `app/main.py`, requirements) change
through Daryn. Details: [AGENTS.md](AGENTS.md).

## References
- Dataset paper: Kim H. H. et al. (2023), GigaScience 12, giad039,
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10220505
- Dataset authors' code (reference only, not used): https://github.com/DigitalHealthcareLab/22ActionRecognitionTool (MIT)
- rtmlib: https://github.com/Tau-J/rtmlib
