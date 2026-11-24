"""About 20 group B stick-figure clips for KEE calibration, expert score hidden. Owner: Theresia (#51).

    python -m viz.kee_clips --zip data/raw/B_skel.zip --out runs/kee_practice

writes
    runs/kee_practice/clips/clip_01.mp4 ... clip_20.mp4   (give these to KEE)
    runs/kee_practice/answer_key.csv                       (keep this away from KEE until the end)

- Group B only (CC0 dataset, Kim et al. 2023): no child from our testing day is ever in here.
- Spread over the 3 tasks and the 3 scores: 2 clips per task x score (18), plus 2 more so there
  are 20 (each from a different task, the most common score of that task). One camera per
  attempt, at most one clip per child.
- The clip shows only the stick figure, the time and the task number: no child code, no file
  name, no score. Clip numbers are shuffled, so the order says nothing about the score.
- answer_key.csv: clip, task, score (0/1/2), label, file. It is what eval.agreement --key reads.
"""

from __future__ import annotations

import argparse
import csv
import sys
import zipfile
from pathlib import Path

import numpy as np

from common.format import Try

N_CLIPS = 20
PER_CELL = 2
MIN_FRAMES = 60  # at least 2 s, so KEE sees the whole movement (guess)
KEY_FIELDS = ["clip", "task", "score", "label", "file"]


def _labels(z: zipfile.ZipFile, names: list[str]):
    """Yield (name, label_index) reading only the label (no points)."""
    import json

    for name in names:
        d = json.loads(z.read(name))
        if len(d.get("data", [])) >= MIN_FRAMES:
            yield name, int(d["label_index"])


def choose(zip_path, n: int = N_CLIPS, per_cell: int = PER_CELL, seed: int = 0) -> list[str]:
    """Pick n file names: per_cell per (task, score), then fill up; one per child and attempt."""
    from data.load_b import parse_name

    rng = np.random.default_rng(seed)
    with zipfile.ZipFile(zip_path) as z:
        names = [
            m
            for m in z.namelist()
            if m.endswith(".json") and Path(m).name.startswith("B") and "_GMS_" in m
        ]
        names = [m for m in names if parse_name(m)["task"] in (1, 2, 3)]
        names = [names[i] for i in rng.permutation(len(names))]
        cells: dict[tuple, list[str]] = {}
        for name, score in _labels(z, names):
            cells.setdefault((parse_name(name)["task"], score), []).append(name)
    picked, children = [], set()

    def take(cell) -> bool:
        for name in cells.get(cell, []):
            child = parse_name(name)["child"]
            if child not in children and name not in picked:
                picked.append(name)
                children.add(child)
                return True
        return False

    for _ in range(per_cell):
        for task in (1, 2, 3):
            for score in (0, 1, 2):
                take((task, score))
    # fill up one clip at a time: first the biggest cell of each task (so the extras go to
    # different tasks), then the other cells, biggest first; repeat until n
    by_size = sorted(cells, key=lambda c: -len(cells[c]))
    firsts = [
        next(c for c in by_size if c[0] == t) for t in (1, 2, 3) if any(c[0] == t for c in by_size)
    ]
    order = sorted(firsts, key=lambda c: -len(cells[c])) + [c for c in by_size if c not in firsts]
    while len(picked) < n:
        added = False
        for c in order:
            if len(picked) >= n:
                break
            added = take(c) or added
        if not added:
            break
    return picked[:n]


def hidden(tr: Try) -> Try:
    """Same points, meta with only what the clip may show (task) and the picture size."""
    m = tr.meta
    keep = {"task": m.get("task"), "width": m.get("width"), "height": m.get("height")}
    return Try(points=tr.points, t=tr.t, meta={k: v for k, v in keep.items() if v is not None})


def make_clips(zip_path, out_dir, n: int = N_CLIPS, seed: int = 0) -> dict:
    from data.load_b import load_try
    from viz.replay import render

    out = Path(out_dir)
    clips = out / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    names = choose(zip_path, n=n, seed=seed)
    order = np.random.default_rng(seed + 1).permutation(len(names))
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for k, i in enumerate(order, start=1):
            tr = load_try(z, names[i], min_frames=MIN_FRAMES)
            clip = f"clip_{k:02d}"
            render(hidden(tr), clips / f"{clip}.mp4", labels=False)
            rows.append(
                {
                    "clip": clip,
                    "task": tr.meta["task"],
                    "score": tr.meta["label_index"],
                    "label": tr.meta["label"],
                    "file": tr.meta["file"],
                }
            )
    key = out / "answer_key.csv"
    with open(key, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=KEY_FIELDS)
        w.writeheader()
        w.writerows(rows)
    sheet = out / "kee_practice.csv"  # blank sheet for KEE's scores (eval.agreement --practice)
    with open(sheet, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "scorer", "score"])
        w.writeheader()
        for r in rows:
            w.writerow({"clip": r["clip"], "scorer": "", "score": ""})
    return {"clips": clips, "key": key, "sheet": sheet, "rows": rows}


def spread(rows) -> str:
    """Count table task x score (from the answer key; don't show it to KEE)."""
    lines = ["task   0 Bad  1 Good  2 Perfect"]
    for task in (1, 2, 3):
        c = [sum(r["task"] == task and r["score"] == s for r in rows) for s in (0, 1, 2)]
        lines.append(f"{task}      {c[0]:>5}  {c[1]:>6}  {c[2]:>9}")
    return "\n".join(lines)


def main(argv=None) -> int:
    from data.load_b import DEFAULT_ZIP

    ap = argparse.ArgumentParser(description="Group B stick-figure clips for KEE calibration")
    ap.add_argument("--zip", default=str(DEFAULT_ZIP), help="group B skeleton zip (B_skel.zip)")
    ap.add_argument("--out", default="runs/kee_practice")
    ap.add_argument("-n", type=int, default=N_CLIPS)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if not Path(args.zip).exists():
        print(f"{args.zip} not found: put the group B skeleton zip there (GigaDB 10.5524/102382)")
        return 1
    res = make_clips(args.zip, args.out, n=args.n, seed=args.seed)
    print(f"wrote {len(res['rows'])} clips to {res['clips']}")
    print(f"answer key (keep away from KEE): {res['key']}")
    print(f"blank KEE sheet: {res['sheet']}")
    print(spread(res["rows"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
