"""Dry run count: how many tries automatic mode caught by itself, per task. Owner: Theresia (#53).

The count decides automatic or manual mode for testing day (Nov 29). It only counts; the team
decides (no threshold has been agreed, so none is built in).

    python -m eval.dry_run_count runs/2026-11-25 --notes dry_run_notes.csv
    python -m eval.dry_run_count --template dry_run_notes.csv     # blank 27-row notes CSV

A try is CAUGHT when automatic mode both started and ended it:
- start: the JSON meta says mode "auto" (the operator pressed Space to start = "manual");
- end:   the notes sheet's "Auto end" column says ✓. The JSON can't show this: a try the app
         started and the operator stopped with Space is still saved as mode "auto". If the JSON
         meta has "stop" ("auto"/"manual"), that is used instead of the notes.
Cancelled tries (Backspace) are not saved, so they are not counted; the notes sheet lists them.

Notes CSV (the printed docs/testing_day/dry_run_notes_sheet.md, typed in): one row per try with
columns code, task, try, auto_start, auto_end (more columns are kept but not used).
Marks: ✓ / y / yes / 1 = worked by itself; ✗ / x / n / no / 0 = missed; M / Space = operator.

Team data only (dry run, Nov 25). Testing-day files (Nov 29) stay on the encrypted drive: run this
script there, never paste the files or the output with child codes into an AI tool.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

TASK_NAMES = {1: "stand on one leg", 2: "hop on one leg", 3: "long jump"}
NOTES_FIELDS = ["n", "code", "task", "try", "leg", "auto_start", "auto_end", "notes"]
_YES = {"✓", "✔", "v", "y", "yes", "1", "ok", "true"}
_NO = {"✗", "✘", "x", "n", "no", "0", "false", "m", "space", "manual"}


def mark(value) -> bool | None:
    """✓ -> True, ✗ or M -> False, blank or unknown -> None."""
    v = str(value or "").strip().lower()
    if v in _YES:
        return True
    if v in _NO:
        return False
    return None


def read_tries(run_dir) -> list[dict]:
    """One row per saved try JSON in run_dir: code, task, try, leg, start (auto?), stop."""
    rows = []
    for p in sorted(Path(run_dir).glob("*_task*_try*_*.json")):
        meta = json.loads(p.read_text(encoding="utf-8"))["meta"]
        stop = meta.get("stop")
        rows.append(
            {
                "file": p.name,
                "code": str(meta.get("child")),
                "task": int(meta["task"]),
                "try": int(meta["try"]),
                "leg": meta.get("leg"),
                "auto_start": meta.get("mode") == "auto",
                "auto_end": None if stop is None else stop == "auto",
            }
        )
    return rows


def read_notes(path) -> dict[tuple, dict]:
    """Notes CSV -> {(code, task, try): {"auto_start": bool|None, "auto_end": bool|None, ...}}."""
    out = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if not (r.get("code") or "").strip() or not (r.get("task") or "").strip():
                continue
            task = int(str(r["task"]).strip()[0])  # "1", "1 one leg" -> 1
            key = (r["code"].strip(), task, int(r["try"]))
            out[key] = {
                "auto_start": mark(r.get("auto_start")),
                "auto_end": mark(r.get("auto_end")),
                "notes": r.get("notes", ""),
            }
    return out


def count(run_dir, notes_path=None) -> dict:
    """Per task: tries, caught (start and end automatic), auto starts, auto ends, misses.

    Returns {"tasks": {1: {...}, 2: {...}, 3: {...}}, "rows": [...], "unknown_end": [...]}.
    """
    rows = read_tries(run_dir)
    notes = read_notes(notes_path) if notes_path else {}
    for r in rows:
        n = notes.get((r["code"], r["task"], r["try"]))
        if r["auto_end"] is None and n is not None:
            r["auto_end"] = n["auto_end"]
        if n is not None and n["auto_start"] is not None and n["auto_start"] != r["auto_start"]:
            r["check"] = "notes and JSON disagree on the start"
        r["caught"] = bool(r["auto_start"] and r["auto_end"])
    tasks = {}
    for task in (1, 2, 3):
        rs = [r for r in rows if r["task"] == task]
        tasks[task] = {
            "tries": len(rs),
            "caught": sum(r["caught"] for r in rs),
            "auto_start": sum(r["auto_start"] for r in rs),
            "auto_end": sum(r["auto_end"] is True for r in rs),
            "end_unknown": sum(r["auto_end"] is None and r["auto_start"] for r in rs),
            "missed_start": [r["file"] for r in rs if not r["auto_start"]],
            "missed_end": [r["file"] for r in rs if r["auto_start"] and r["auto_end"] is False],
        }
    unknown = [r["file"] for r in rows if r["auto_start"] and r["auto_end"] is None]
    return {"tasks": tasks, "rows": rows, "unknown_end": unknown}


def report(result: dict) -> str:
    lines = ["task                   caught  auto start  auto end  (of tries)"]
    total = {"tries": 0, "caught": 0}
    for task, c in result["tasks"].items():
        lines.append(
            f"{task} {TASK_NAMES[task]:<18}  {c['caught']:>2}/{c['tries']:<3}  "
            f"{c['auto_start']:>4}/{c['tries']:<5}  {c['auto_end']:>3}/{c['tries']}"
        )
        total["tries"] += c["tries"]
        total["caught"] += c["caught"]
    lines.append(f"all                     {total['caught']:>2}/{total['tries']}")
    for task, c in result["tasks"].items():
        if c["missed_start"]:
            lines.append(f"task {task} start by hand (Space): {', '.join(c['missed_start'])}")
        if c["missed_end"]:
            lines.append(f"task {task} end by hand: {', '.join(c['missed_end'])}")
    if result["unknown_end"]:
        lines.append(
            f"end not known for {len(result['unknown_end'])} tries (no notes row, counted as "
            f"not caught): {', '.join(result['unknown_end'])}"
        )
    for r in result["rows"]:
        if r.get("check"):
            lines.append(f"check {r['file']}: {r['check']}")
    return "\n".join(lines)


def write_template(path, codes=("T01", "T02", "T03")) -> Path:
    """Blank notes CSV in the order of the printed sheet: task 1, 2, 3 x 3 tries, legs L R L."""
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=NOTES_FIELDS)
        w.writeheader()
        n = 0
        for code in codes:
            for task in (1, 2, 3):
                for k, leg in enumerate(("L", "R", "L"), start=1):
                    n += 1
                    w.writerow(
                        {
                            "n": n,
                            "code": code,
                            "task": task,
                            "try": k,
                            "leg": "" if task == 3 else leg,
                        }
                    )
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Count dry-run tries caught by automatic mode")
    ap.add_argument("run_dir", nargs="?", help="folder with the try JSON files (runs/2026-11-25)")
    ap.add_argument("--notes", help="notes sheet typed in as CSV (auto_end column)")
    ap.add_argument("--template", help="write a blank notes CSV here and stop")
    args = ap.parse_args(argv)
    if args.template:
        print(f"wrote {write_template(args.template)}")
        return 0
    if not args.run_dir:
        ap.error("give the run folder, or --template")
    result = count(args.run_dir, args.notes)
    if not result["rows"]:
        print(f"no try JSON files in {args.run_dir}")
        return 1
    print(report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
