# Daryn: repo, shared format, integration, rescore (team lead)

**Your part:** the shared contract (`common/`, contract tests, CI), joining the parts in
`app/main.py`, `rescore.py`, the code freeze. You also review changes to shared files.

**Your files:** `common/format.py`, `common/body25.py`, `common/geometry.py`, `app/main.py`,
`rescore.py`, `tests/`, `.github/`, `requirements*.txt`, `AGENTS.md`, `docs/`.

**Your laptop is a Mac:** integrate with recorded files and `--video`; test live on the live
laptop (joint sessions Nov 15, 19, 22).

**Admin still open:** `docs/admin.md` (invite the 5 teammates, CODEOWNERS usernames,
branch protection after GitHub Student Pack).

## Tasks
| Due | Task | Level | Done when | Status |
|---|---|---|---|---|
| Nov 8 | Repo, layout, pinned requirements, README, `scripts/summarize_group_B.py` | Easy | Teammate clones, installs, gets the Milestone 1 line | Done 2026-11-08 |
| Nov 8 | `common/format.py` + `examples/fake_try.npz` | Easy | `validate(load(fake))` passes | Done 2026-11-08 |
| Nov 8 | AGENTS.md, CLAUDE.md, task-card template | Easy | AI explains the repo rules correctly | Done 2026-11-08 |
| Nov 9 | Stubs for every function, `app/main.py --fake` (auto + manual), contract tests, smoke test, CI, CODEOWNERS | Medium | 1 JSON + 1 log row; tests pass; CI green | Done 2026-11-08 |
| Nov 10 | `rescore.py` first version | Medium | Runs on `--fake` output | Done 2026-11-08 |
| Nov 15 | Swap each stub for the real module as it lands; keep the smoke test green | Medium | Green after every merge | |
| Nov 22 | Join everything in `app/main.py` on the live laptop (with Louise and Freya) | Hard | Score in the log within a few seconds of each try, all 3 tasks | |
| Nov 25 | Compare rescore with live scores on the 27 dry-run tries | Medium | 27/27 match | |
| Nov 27 | Code freeze, tag v1.0, fresh install on the live laptop | Easy | Clean install, mini dry run passes | |
