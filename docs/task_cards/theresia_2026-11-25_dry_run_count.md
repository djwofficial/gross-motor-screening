# Task card: Theresia, 2026-11-25: dry run count (eval/dry_run_count.py), issue #53

**Goal:** one count per task of how many of the 27 dry-run tries automatic mode caught by itself,
so the team can choose automatic or manual mode for testing day (Nov 29).

**Input → output:** `runs/2026-11-25/*.json` (one per saved try) + the notes sheet typed into a
CSV → per task: caught / tries, auto starts, auto ends, and the tries that needed Space.

**Caught** = automatic mode started AND ended the try (agreed with Theresia, 2026-11-22):
- start: JSON `meta.mode == "auto"` (`"manual"` = the operator pressed Space to start);
- end: the notes sheet's "Auto end" ✓. The JSON can't show it: a try the app started and the
  operator stopped with Space is saved as mode "auto" too. If `meta.stop` ("auto"/"manual") is
  ever added (Freya, `app/ui.py`), the script uses it and the notes are not needed for the end.
- cancelled tries (Backspace) are not saved; the notes sheet lists them.

**Files you may change:** `eval/dry_run_count.py`, `tests/test_eval_dry_run_count.py`.

**Steps:**
1. Before Nov 25: `python -m eval.dry_run_count --template runs/2026-11-25/notes.csv` makes the
   27-row CSV in the order of `docs/testing_day/dry_run_notes_sheet.md`.
2. After the run: type the paper "Auto start" / "Auto end" marks into it (✓, ✗ or M).
3. `python -m eval.dry_run_count runs/2026-11-25 --notes runs/2026-11-25/notes.csv`.
4. Paste the table into this card and into #53; the team decides the mode from it.

**Done when:** one count per task.

**Result so far (2026-11-22, fake tries only, the real run is Nov 25):**
`tests/test_eval_dry_run_count.py` builds a fake dry run with `app.session.Session` (3 people × 3
tasks × 3 tries) with 2 starts and 2 ends by hand, and gets 8/9 caught for each task (24/27);
without notes every end is "not known" and counted as not caught; `meta.stop` is used when present;
a notes row that disagrees with the JSON on the start is flagged.

**Dry run result (fill in on Nov 25):**
| Task | Caught | Auto start | Auto end |
|---|---|---|---|
| 1 stand on one leg | __ / 9 | __ / 9 | __ / 9 |
| 2 hop on one leg | __ / 9 | __ / 9 | __ / 9 |
| 3 long jump | __ / 9 | __ / 9 | __ / 9 |

**Watch out:** no auto/manual threshold has been agreed, so the script only counts. Dry-run files
are team data (runs/ is gitignored). Testing-day files (Nov 29) never go into an AI tool: run the
script on the live laptop only.

**Tests:** new `tests/test_eval_dry_run_count.py`.
