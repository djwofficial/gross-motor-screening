"""#24 (Louise): our clues and timeline on the 27 adult team clips vs Theresia's by-eye sheet.

The team clips are RTMPose points from real videos (Revallo, #31), the same detector as the live
app, so this checks that clues tuned on the dataset's OpenPose points still hold on RTMPose.

Inputs
- runs/team_clips/points/*.npz   unzip docs/task_cards/team_clips_27_try.zip into runs/team_clips
- eval/By-Eye Coding Sheet.csv   Theresia's counts (#47)
Matching (ASSUMED until Theresia confirms): folderN = P0N, clipN = Video N, Move_k = Task k.

Run: python scripts/check_team_clips.py [--out runs/team_clips_check.csv]
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.format import load  # noqa: E402
from features.clues import compute, explain, timeline  # noqa: E402
from model.predict import predict  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
POINTS = ROOT / "runs" / "team_clips" / "points"
MANIFEST = ROOT / "runs" / "team_clips" / "manifest.csv"
SHEET = ROOT / "eval" / "By-Eye Coding Sheet.csv"
LABELS = {"bad": 0, "good": 1, "perfect": 2}
NAME = re.compile(r"folder(\d)_Move_(\d)_clip(\d)\.npz$")
# One-foot time within this many seconds of the by-eye time counts as a match (Theresia wrote
# whole seconds, so +-0.5 s is rounding plus the 0.2 s flicker rule).
TIME_TOL_S = 0.7


def _num(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def read_sheet() -> dict[tuple[int, int, int], dict]:
    out = {}
    with SHEET.open(encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            vals = list(r.values())
            person = int(r["Participant"].strip()[1:])
            task = int(r["Task"].split()[-1])
            video = int(r["Video"].split()[-1])
            out[(person, task, video)] = {
                "eye_hops": _num(vals[3]),
                "eye_one_foot_s": _num(vals[4]),
                "eye_condition": vals[5].strip(),
                "eye_feet_together": vals[6].strip(),
                "eye_score": vals[7].strip(),
            }
    return out


def read_multi() -> dict[str, bool]:
    if not MANIFEST.exists():
        return {}
    with MANIFEST.open(encoding="utf-8-sig") as f:
        return {
            Path(r["try_file"]).name: r["review_person_selection"] == "True"
            for r in csv.DictReader(f)
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/team_clips_check.csv")
    a = ap.parse_args()
    sheet, multi = read_sheet(), read_multi()
    files = sorted(POINTS.glob("*.npz"))
    if not files:
        sys.exit(f"no point files in {POINTS}: unzip docs/task_cards/team_clips_27_try.zip there")

    rows = []
    for path in files:
        m = NAME.search(path.name)
        person, task, video = (int(g) for g in m.groups())
        tr = load(path)
        ev = timeline(task, tr.points, tr.t)
        c = compute(task, tr.points, tr.t)
        score, conf = predict(task, c)
        eye = sheet.get((person, task, video), {})
        row = {
            "clip": path.stem,
            "person": f"P0{person}",
            "task": task,
            "video": video,
            "two_people": multi.get(path.name, False),
            **eye,
            "app_score": ["Bad", "Good", "Perfect"][score],
            "app_confidence": round(conf, 2),
            **{k: round(v, 2) for k, v in c.items()},
            "reasons": explain(task, c, ev),
        }
        # verdicts: only where the sheet has a value to compare with
        if task == 1 and eye.get("eye_one_foot_s") is not None:
            row["one_foot_match"] = abs(c["one_foot_s"] - eye["eye_one_foot_s"]) <= TIME_TOL_S
        if task == 2 and eye.get("eye_hops") is not None:
            row["hops_match"] = int(c["hops"]) == int(eye["eye_hops"])
        if eye.get("eye_score"):
            row["score_match"] = LABELS.get(eye["eye_score"].lower()) == score
        rows.append(row)

    cols: list[str] = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    for task, name in ((1, "one leg"), (2, "hop"), (3, "long jump")):
        print(f"\nTask {task} ({name})")
        for r in (r for r in rows if r["task"] == task):
            if task == 1:
                got = f"eye {r.get('eye_one_foot_s')} s | app {r['one_foot_s']:.1f} s"
            elif task == 2:
                got = f"eye {r.get('eye_hops')} hops | app {int(r['hops'])} hops"
            else:
                got = (
                    f"eye feet together {r.get('eye_feet_together')} | app gap "
                    f"{r['feet_gap_takeoff']:.2f}/{r['feet_gap_landing']:.2f}, "
                    f"jump {r['jump_size']:.2f}, air {r['air_s']:.2f} s"
                )
            flag = " [2+ people: check picker]" if r["two_people"] else ""
            print(
                f"  {r['person']} v{r['video']}: {got} | score eye {r.get('eye_score')} "
                f"app {r['app_score']} ({r['app_confidence']:.2f}){flag}"
            )
    for key, label in (
        ("one_foot_match", "task 1 one-foot time"),
        ("hops_match", "task 2 hops"),
        ("score_match", "score (app vs Theresia)"),
    ):
        v = [r[key] for r in rows if key in r]
        if v:
            print(f"{label}: {sum(v)}/{len(v)} match")
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
