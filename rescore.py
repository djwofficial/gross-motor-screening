"""Score saved tries again offline; the scores must match the live ones. Owner: Daryn.

    python rescore.py runs/2026-11-08            # writes runs/2026-11-08/rescore.csv

Reads every {child}_task{n}_try{k}_{leg}.json in the folder, recomputes clues and score with
the current code and model files, and compares with the score saved live.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

from app.main import score_try
from app.session import load_record


def rescore(folder: Path) -> tuple[int, int]:
    rows, match = [], 0
    files = sorted(p for p in folder.glob("*_task*_try*_*.json"))
    for p in files:
        rec = load_record(p)
        task = int(rec["meta"]["task"])
        _clues, score, confidence, reasons = score_try(rec["try"], task)
        same = score == rec["score"] and abs(confidence - rec["confidence"]) < 1e-6
        match += same
        rows.append(
            {
                "json_file": p.name,
                "task": task,
                "live_score": rec["score"],
                "rescore": score,
                "live_confidence": f"{rec['confidence']:.4f}",
                "confidence": f"{confidence:.4f}",
                "same": int(same),
                "reasons": reasons,
            }
        )
    with (folder / "rescore.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "json_file",
                "task",
                "live_score",
                "rescore",
                "live_confidence",
                "confidence",
                "same",
                "reasons",
            ],
        )
        w.writeheader()
        w.writerows(rows)
    return len(files), match


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python rescore.py RUN_FOLDER")
    n, m = rescore(Path(sys.argv[1]))
    print(f"rescored {n} tries: {m} match the live score")
    sys.exit(0 if n == m else 1)
