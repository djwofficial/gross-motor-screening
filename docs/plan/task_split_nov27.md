# Coding plan to Nov 27 (product only)

Updated 2026-11-09 (Daryn). The product deadline moves from Dec 6 to Nov 27. This plan covers coding only; slides, report, video and testing-day preparation come later. Testing day: Sunday Nov 29 at 新竹市東區親子館, two days after the code freeze.

Ready on Nov 27 means: one GUI window on the live laptop (Freya's, NVIDIA) runs the readiness check, finds the start and end of each try, guesses the task and leg, lets the operator confirm or correct, and writes the score with its reasons to log.csv, for all 3 tasks. Testing mode shows no score and no timeline. Video is saved only when V is on. rescore.py gives the same scores as live. A dry run of 3 people × 3 tasks × 3 tries passes (27 JSON files and 27 log rows).

Not in the Nov 27 target: TCN (dropped), demo mode (only if time is left), report and slides.

## Product flow

One GUI window on the live laptop, started with python -m app.main. No installer, web app or phone app.

Before the first child: a readiness check shows whether the whole body is in the picture, the fps is 15 or more, the light is good enough (point confidence) and the feet are in the start area, and tells the operator what to fix.

Each try: the child stands still at the start line for about 1 s (ready); movement starts the try; standing still for about 1.5 s or leaving the area ends it. The app then guesses the task and leg, the operator presses Enter to confirm or corrects it (1/2/3, L/R, Backspace cancels the try), and the scoring model gives 0, 1 or 2. Manual Start/Stop stays as a fallback; the dry run decides which mode we use on testing day.

Moments: each try also gets a timeline measured from the points, for example foot down at 2.0 s, wobble at 4.1 s. The dataset has one expert score per try and none per moment, so the timeline is measured, not learned. The official score stays 0, 1 or 2, the same scale KEE uses. The timeline is saved in the JSON of the try and named in the reasons; it records what happened and never says a child failed.

One child at a time. KEE, parents and waiting children stay out of the picture; pick_child ignores anyone else who walks in.

Saved per try: the points file (always) and one row in log.csv with child code, age in months, task, try, leg, guessed task, corrected or not, score, confidence and the reasons (clues, for example 4.2 s on one foot, 0 touch-downs). When the model isn't confident, the row says check by teacher instead of a score. Video only if the parent ticked level 3 and V is on.

Screen: no score while testing. Whether the large area shows the camera picture, the stick figure or a neutral picture is still open (ask KEE; linked to the consent form). Demo mode for class may show everything.

Results: per try, log.csv. After testing day, agreement tables app vs KEE and KEE vs KEE. Offline, the random forest against the baselines on group B.

We don't use the 22ActionRecognitionTool code: it has no released weights, and its recognition model answers which task, not how well. We train our own small task classifier and scoring model on group B; the repo is only a reference.

## How we work

One repo, one shared format: a try is an array (frames × 25 × 3: x, y, score) in BODY_25 order, plus time stamps in seconds. Everyone uses fake data until the part they need is ready, so nobody waits.

Stubs on day 1: Daryn writes every function with its final signature and a dummy body, so everyone can import and test right away: live.pose.PoseStream.read() -> (t, kp26); live.convert.halpe_to_body25(kp26) -> (25, 3); common.pick.pick_child(frames) -> track; features.clues.timeline(task, points, t) -> moments; features.clues.compute(task, points, t) -> dict; model.predict(task, clues) -> (score, confidence); model.task_classifier.predict_task(points, t) -> (task, leg, confidence); live.segment.Segmenter.update(points, t) -> event; live.quality.check(points, fps) -> problems; app.session.save_try(...).

Checkpoints: a 15-minute check on each date below. Anything not done gets a named helper and a new date.

Levels: Hard is on the critical path or difficult. Medium is normal work. Easy is small or can slip 1–2 days.

Laptops: Daryn and Louise have Macs (no CUDA), and their work (model training, clues) needs no GPU. Marty, Revallo, Freya and Theresia have NVIDIA. The live laptop is Freya's. Revallo develops the live module on an NVIDIA laptop and does a 10-minute check on Freya's laptop on Nov 10. Daryn integrates on a Mac with recorded files, then tests on the live laptop.

requirements.txt uses platform markers so it installs on Mac, Windows and Linux: onnxruntime-gpu; sys_platform != "darwin" and onnxruntime; sys_platform == "darwin".

## Checkpoints

| CP | Date | What must exist |
|---|---|---|
| CP0 | Sun Nov 8 | Setup: repo, format and stubs ready; everyone can install and has run Milestone 1 |
| CP1 | Wed Nov 11 | Every part has a first version that runs on its own |
| CP2 | Sun Nov 15 | Every part produces real output (not fake) |
| CP3 | Thu Nov 19 | Model gate: random forest beats the baseline (moved up from Nov 22); all parts complete |
| CP4 | Sun Nov 22 | App runs end to end on ourselves (moved up from Nov 29) |
| CP5 | Wed Nov 25 | Dry run: 27 tries pass, rescore matches, and we count how many tries automatic mode caught correctly (moved up from Dec 3) |
| CP6 | Fri Nov 27 | Bugs fixed, code freeze, tag v1.0: product ready |

## Tasks per person

### Daryn: repo, format, model training, integration

| Due | Task | Level | Done when | Status |
|---|---|---|---|---|
| Nov 8 | GitHub repo (private), folder layout, requirements.txt with exact versions (rtmlib, onnxruntime-gpu, numpy, pandas, scikit-learn, opencv), README with install steps, plus scripts/summarize_group_B.py for Milestone 1 | Easy | A teammate clones, installs and gets the Milestone 1 first line without asking anything | Done 2026-11-08 |
| Nov 8 | common/format.py: Try dataclass, save/load, validate, fake-try generator, plus examples/fake_try.npz | Easy | validate(load(fake)) passes | Done 2026-11-08 |
| Nov 8 | AGENTS.md and CLAUDE.md in the repo root, plus a task-card template (see Working with AI coding tools) | Easy | A teammate pastes one card into their AI and the AI explains the repo rules correctly | Done 2026-11-08 |
| Nov 9 | Stubs for every function (including live/segment.py, live/quality.py, model/task_classifier.py) and app/main.py --fake running start to finish in automatic and manual mode; contract tests in tests/contracts/, smoke test, GitHub Actions CI and CODEOWNERS | Medium | python -m app.main --fake writes 1 JSON and 1 log row; tests pass on the stubs; CI is green on main | Done 2026-11-08 |
| Nov 9 | Timeline: stub and contract test for features.clues.timeline; events saved in the JSON of each try | Easy | Tests are green; the JSON from --fake has events | Done 2026-11-09 |
| Nov 10 | rescore.py first version: folder of JSON files → clues → model → CSV of scores (moved up from Nov 19) | Medium | Runs on JSON files from app/main.py --fake | Done 2026-11-08 |
| Nov 15 | Swap each stub for the real module as it lands; keep the smoke test green | Medium | Test is green after every merge |  |
| Nov 16 | model/train_rf.py set up with task 1 clues first | Medium | Runs without errors and prints F1 and kappa |  |
| Nov 19 | Gate: random forest per task on the full clue table beats the baseline. Clip length (duration_s) stays out of the model. For task 2 also try a per-second forest and keep whichever scores higher. Save the models and write model.predict() | Medium | Macro-F1 and kappa above the baseline for all 3 tasks; .joblib files exist; predict returns a score and a confidence |  |
| Nov 22 | Join everything in app/main.py on the live laptop: live → clues → model → log (helped by Revallo and Freya) | Hard | A score appears in the log within a few seconds of each try, for all 3 tasks |  |
| Nov 25 | Compare rescore with the live scores on the 27 dry-run tries | Medium | 27/27 match |  |
| Nov 27 | Code freeze, tag v1.0, fresh install from scratch on the live laptop | Easy | Clean install, then a mini dry run passes |  |

### Marty: dataset, baseline, task classifier

| Due | Task | Level | Done when | Status |
|---|---|---|---|---|
| Nov 8 | Milestone 1 | Easy | First output line: files: 4661 \| children: 106 \| attempts: 1555 |  |
| Nov 11 | data/load_b.py: B_skel.zip → shared format; drop clips under 30 frames; center on the neck, scale by torso length; don't cut at 300 frames | Medium | 3,359 files for tasks 1–3 (1,205 / 1,056 / 1,098) |  |
| Nov 13 | model/baseline.py per task, GroupKFold by child | Easy | Numbers close to Card 1 (macro-F1 0.19 / 0.20 / 0.26, kappa 0.00) and the child-leak check prints PASS |  |
| Nov 15 | common/pick.py: pick_child() for files with 2+ people (28% of B files): lock on the person at the start, then follow the nearest neck. Revallo reuses the same function for live | Hard | 5 multi-person files checked by eye with Freya's replay |  |
| Nov 22 | model/task_classifier.py: guess the task (1/2/3) and the leg from the clue table, GroupKFold by child | Medium | Accuracy and confusion matrix reported; if accuracy is low, the app uses keys only |  |
| Nov 22 | Confusion matrices per task and per fold from Daryn's random forest, handed to Louise; check that predict takes under 1 s | Easy | CSV and PNG files exist |  |

### Louise: moment timeline, movement clues, evaluation

| Due | Task | Level | Done when | Status |
|---|---|---|---|---|
| Nov 8 | Milestone 1 | Easy | Same first line |  |
| Nov 14 | Task 1 timeline and clues in features/clues.py: moments (both feet, steady, wobble), longest time on one foot (s), sway (torso lengths), touch-downs, wobble time. Build the timeline first and compute the clues from it | Medium | 3 clips match the by-eye counts, including the second a foot comes down |  |
| Nov 17 | Task 2 timeline and clues: moments (hop, pause, both feet), hop count, longest pause | Hard | 3 clips match the by-eye counts |  |
| Nov 18 | Clue table for all 3,359 files, one row per file (task 3 columns from Revallo), sent to Daryn and Marty | Medium | No empty values, or every filled-in value is written down |  |
| Nov 19 | features/clues.py: explain(task, clues, events) → short reason text for the log, for example 4.2 s on one foot, foot down at 2.0 s | Easy | Reason text for 3 example clips matches the by-eye sheet |  |
| Nov 22 | Check the clues and timeline on our own team clips (Theresia's 27 recordings, through Revallo's --video mode); tune thresholds | Medium | Most clips match the by-eye counts; clips that differ are listed |  |
| Nov 25 | eval/report.py: confusion matrix per task, per-fold numbers, mix-ups compared with the scoring rules | Medium | One report covering all 3 tasks |  |

### Revallo: live pose module (critical path), long-jump clues

| Due | Task | Level | Done when | Status |
|---|---|---|---|---|
| Nov 8 | Milestone 1 | Easy | Same first line |  |
| Nov 10 | 10-minute check on Freya's laptop (the live laptop): rtmlib BodyWithFeet (balanced) with CUDA and a person in the picture (live/bench_fps.py) | Easy | CUDA is used and fps is 15 or more |  |
| Nov 11 | live/convert.py: check HALPE_TO_BODY25 against mmpose halpe26.py; plot 1 labelled frame to check left and right | Easy | Plot is correct (right wrist = point 4) |  |
| Nov 13 | live/capture.py and pose.py: 720p30 capture thread, auto-framing off, unmirrored picture, time stamp per frame, resample to 30 fps, output a Try; plus a --video file mode for recorded clips (moved up from Nov 15) | Hard | Recorded clip → valid Try file |  |
| Nov 15 | Video saving in its own thread, only when V is on; file name carries the child code | Medium | fps doesn't drop compared with V off |  |
| Nov 15 | Run the 27 team clips through --video mode and hand the points to Louise and Theresia | Easy | 27 valid Try files |  |
| Nov 17 | Task 3 timeline and clues, in their own functions in features/clues.py: moments (before, air, after), feet together, time in the air, jump size. Task 3 was the weakest in the first test, so it comes first after capture | Medium | 3 clips match the by-eye counts |  |
| Nov 19 | Pick the child live: use Marty's pick_child(), lock on the person at the start line, hold the track when someone else walks into the picture | Hard | Test with 2 people in the picture: the track doesn't jump to the other person |  |
| Nov 22 | Performance on the live laptop (with Freya): fps with video saving on, a dim-room test if time allows; help Daryn join the parts | Medium | 15 fps or more with everything on (ideally 30) |  |

### Freya: replay, app window, testing mode

| Due | Task | Level | Done when | Status |
|---|---|---|---|---|
| Nov 8 | Milestone 1 | Easy | Same first line |  |
| Nov 10 | viz/replay.py: Try file → stick-figure mp4 with point numbers (used by Louise, Marty and Theresia for checking by eye) | Easy | Plays 1 dataset file and 1 fake file |  |
| Nov 11 | GUI skeleton (app/ui.py, session.py) with fake points: keys 1/2/3, L/R, Space, Enter, Backspace, N, V; state machine for ready / recording / confirm | Medium | All keys work and the state is correct |  |
| Nov 13 | One JSON per try (K03_task2_try4_L.json: time stamps, 26 raw points, 25 converted points, events) and log.csv (child, age, task, try, leg, guessed task, corrected, score, confidence, reasons) | Easy | 5 fake tries → 5 JSON files and 5 log rows |  |
| Nov 13 | Replay: coloured bar under the stick figure showing the timeline, one colour per state, for checking moments by eye | Easy | A try with hand-made events shows the right colours at the right seconds |  |
| Nov 19 | Testing mode: no score and no timeline on screen; enter child code and age in months; show Theresia's readiness check before the first child; all log fields filled | Easy | The screen never shows a score or timeline; readiness problems are shown in plain words |  |
| Nov 21 | Automatic mode: show ready / recording / guessed task and leg; Enter confirms, 1/2/3 and L/R correct, Backspace cancels the try; manual Start/Stop stays as fallback | Medium | Both modes write the same JSON and log rows on fake and recorded clips |  |
| Nov 22 | Crash-safe: every try is saved immediately, no double key presses; help Daryn join the parts | Medium | Kill the app mid-session and earlier tries are safe |  |
| Nov 25 | Lead the dry run: 3 people × 3 tasks × 3 tries, in automatic mode with the operator correcting | Medium | 27 JSON files and 27 log rows |  |
| Optional | Demo mode: score, stick figure and timeline on screen | Easy | Only after CP6 |  |

### Theresia: test clips, readiness, start and end, privacy, testing-day tables

| Due | Task | Level | Done when | Status |
|---|---|---|---|---|
| Nov 8 | Milestone 1 | Easy | Same first line |  |
| Nov 11 | By-eye sheet for 9 dataset clips (3 per task) using Freya's replay: seconds on one foot, the second a foot comes down, number of hops, feet together or not | Easy | A 9-row table exists |  |
| Nov 13 | Record team clips (720p30, auto-framing off): 3 people × 3 tasks × 3 tries, some with another person walking past, plus a by-eye count sheet. This is the test data for Revallo and Louise | Medium | 27 videos and 1 by-eye count table |  |
| Nov 17 | live/quality.py: readiness check from the points (whole body in the picture, fps, light from point confidence, feet in the start area) | Easy | 3 deliberately bad setups from team clips each get the right message |  |
| Nov 17 | live/start_area.py: the operator marks the start area on the camera picture; the readiness check and automatic start use it | Easy | The area is saved and the readiness check uses it |  |
| Nov 19 | live/segment.py: automatic start and end from the points (still about 1 s at the start line = ready, movement = start, still about 1.5 s or leaving = end); tune on the 27 team clips | Medium | Start and end found for most of the 27 team clips; misses listed |  |
| Nov 19 | About 20 group B stick-figure clips for KEE calibration, spread over the 3 tasks and 3 scores, expert score hidden | Easy | 20 mp4 files and a separate answer key |  |
| Nov 22 | viz/blur.py: blur faces using RTMPose head points (nose, eyes, ears), then check frame by frame | Hard | 3 team videos: no frame shows a face |  |
| Nov 25 | eval/dry_run_count.py: count how many of the 27 dry-run tries automatic mode caught correctly. The result decides the mode for Nov 29 | Easy | One count per task |  |
| Nov 25 | eval/agreement.py: app vs KEE and KEE vs KEE tables, and KEE's 20 practice clips vs the answer key. Built and tested on fake data only | Medium | Tables come out of a fake score file |  |
| Nov 25 | Replays from the dry-run JSON files (to spot bugs by eye) | Easy | 27 replays made |  |

## Working with AI coding tools

Everyone uses AI tools such as Claude, ChatGPT and Codex. Without shared rules, each AI guesses its own format, edits other people's files or invents APIs, and the parts won't fit together at integration on Nov 22. These rules prevent that.

| Rule | What to do |
|---|---|
| One context file for every AI | AGENTS.md in the repo root holds the shared format, the function signatures, a folder → owner table, the library versions and the project rules (seconds and torso lengths, never pixels; group B only; never a diagnosis; pose detection on the unmirrored picture). CLAUDE.md contains only the line @AGENTS.md. For ChatGPT on the web, paste AGENTS.md at the start of the chat or put it in the Project instructions. |
| A task card is the prompt | Each task row above becomes one card: goal, input → output, signature, the files it may change, the Done when check and things to watch out for. Paste the card straight into the AI instead of writing a prompt from scratch. |
| Contract tests are the fence | The AI's job is to make tests/contracts/test_<part>.py pass without changing the test. If a test or signature really needs to change, ask Daryn; don't let the AI change it quietly. |
| Folder boundaries | Each person and their AI change only their own folder. common/ and tests/contracts/ change only through Daryn. On GitHub, CODEOWNERS and branch protection make a change to someone else's folder need that owner's review. |
| Small PRs | One task = one branch = one PR. Paste the Done when evidence (number, plot or screenshot) in the PR description. Merge only after the smoke test is green. |
| Don't trust numbers or APIs from the AI | Every number must come from a script that actually ran. Check rtmlib's API and the Halpe indices against the real source (the rtmlib repo, mmpose halpe26.py). Tell the AI the library versions in requirements.txt. |
| Owners explain their code | At each checkpoint, each owner explains how their part works in 2–3 minutes. If they can't, the part isn't counted as Done. This also prepares for questions at the final presentation. |
| Children's data never goes into an AI | Points, videos, logs, child codes and ages from testing day are never pasted or uploaded into any AI chat, because the consent form says the data stays on an encrypted drive and is deleted by Dec 25. For debugging, use group B data (CC0), fake data or our own team clips. |
| Same formatter | Run ruff format before each commit, so AI diffs aren't full of whitespace changes. |

## Load per person

| Person | Hard | Medium | Easy |
|---|---|---|---|
| Daryn | 1 | 6 | 5 |
| Marty | 1 | 2 | 3 |
| Louise | 1 | 4 | 2 |
| Revallo | 2 | 3 | 4 |
| Freya | 0 | 4 | 6 |
| Theresia | 1 | 3 | 7 |

Revallo is on the critical path. Three things lighten that load: Marty writes pick_child(), Theresia records the test clips, and Theresia owns the readiness check and automatic start and end, tuned on those clips.

## Main risks and fallbacks

| Risk | Fallback |
|---|---|
| Daryn can't test live on a Mac | Joint sessions on the live laptop on Nov 15, 19 and 22. The --video file mode lets most testing happen without a GPU. |
| rtmlib won't run on CUDA (CP1, Nov 10) | Try another onnxruntime version first. If it still fails, fall back to MediaPipe with the Card 3 mapping. |
| Task 2 hop count is poor (CP3) | The random forest still runs on the clues we have. Louise tunes again on Nov 22 with the team clips. |
| Random forest doesn't beat the baseline on Nov 19 | Write down the result as it is and keep integrating; the app still runs. Daryn and Marty try other features or settings until Nov 22. |
| Task 3 stays weak (first rough test: macro-F1 0.40, against 0.27 for always guessing the most common score) | Revallo works on it first after capture. Report the result as it is; tries the model isn't sure about say check by teacher. |
| Timeline moments don't match what we see by eye (live RTMPose points differ from the dataset's OpenPose points) | Check on the 9 by-eye clips and the 27 team clips and tune. If a state is unreliable, it stays out of the reasons. |
| Automatic start and end miss tries in a busy room | Use manual Start/Stop on testing day; the dry run count decides. The operator can always cancel or correct a try. |
| Task classifier guesses wrong too often | The operator confirms every guess; if accuracy is low, the app goes back to keys 1/2/3 and L/R. |
| Under 15 fps with video on | Lower the saved-video resolution, or use lightweight mode. Measure both first. |

## Final check on Nov 27

Each checkpoint is checked against the Done when column, and the owner writes Done and the date in Status. Final check: clean install on the live laptop, 27-try dry run, rescore.py matches 27/27, then tag v1.0.

