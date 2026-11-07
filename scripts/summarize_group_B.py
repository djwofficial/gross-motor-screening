"""Milestone 1: summarise dataset group B straight from B_skel.zip.

    python scripts/summarize_group_B.py [path/to/B_skel.zip]      (default: data/raw/B_skel.zip)

The first line must be:  files: 4661 | children: 106 | attempts: 1555
The task 1-3 numbers should match the team Doc (Info tab, 2026-11-04): 3,359 files, 1,121
attempts, Bad/Good/Perfect attempts 61/169/173 (task 1), 157/90/105 (task 2), 60/71/235 (task 3).
Reading all 4,661 JSON files takes about 5 seconds (M4 Mac, 2026-11-08).
"""

from __future__ import annotations

import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.load_b import DEFAULT_ZIP, parse_name, read_file  # noqa: E402

LABELS = ["Bad", "Good", "Perfect"]


def main(zip_path: Path) -> None:
    if not zip_path.exists():
        sys.exit(
            f"Not found: {zip_path}\nDownload B_skel.zip from GigaDB (doi 10.5524/102382) and put it in data/raw/."
        )
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".json"))
        info = [parse_name(n) for n in names]
        children = {i["child"] for i in info}
        attempts = {(i["child"], i["task"], i["attempt"]) for i in info}
        print(f"files: {len(names)} | children: {len(children)} | attempts: {len(attempts)}")

        label_of: dict = {}
        label_mismatch = 0
        frames_per_file, multi_person = [], 0
        for n, i in zip(names, info, strict=True):
            if i["task"] not in (1, 2, 3):
                continue
            frames, label_index, _ = read_file(z, n)
            key = (i["child"], i["task"], i["attempt"])
            if key in label_of and label_of[key] != label_index:
                label_mismatch += 1
            label_of.setdefault(key, label_index)
            frames_per_file.append(len(frames))
            multi_person += any(len(p) >= 2 for p in frames)

    sub = [i for i in info if i["task"] in (1, 2, 3)]
    print(
        f"\nTasks 1-3: files {len(sub)} | attempts {len(label_of)} | children {len({i['child'] for i in sub})}"
    )
    print(
        f"{'task':<6}{'files':>7}{'attempts':>10}{'children':>10}   Bad / Good / Perfect (attempts)"
    )
    for task in (1, 2, 3):
        files = [i for i in sub if i["task"] == task]
        keys = {k for k in label_of if k[1] == task}
        counts = Counter(label_of[k] for k in keys)
        kids = {i["child"] for i in files}
        bgp = " / ".join(str(counts.get(c, 0)) for c in range(3))
        print(f"{task:<6}{len(files):>7}{len(keys):>10}{len(kids):>10}   {bgp}")
    f = np.array(frames_per_file)
    print(
        f"\nFrames per file: min {f.min()}, median {int(np.median(f))}, max {f.max()}; under 30: {(f < 30).sum()}; over 300: {(f > 300).sum()}"
    )
    print(
        f"Files with 2+ people in some frame: {multi_person} ({100 * multi_person / len(f):.0f}%)"
    )
    print(f"Attempts whose cameras disagree on the label: {label_mismatch}")
    per_child = defaultdict(set)
    for c, t, _a in label_of:
        per_child[c].add(t)
    print(f"Children who did all 3 tasks: {sum(len(v) == 3 for v in per_child.values())}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ZIP)
