"""One testing session: one JSON file per try and one row per try in log.csv. Owner: Freya.

Shared interface (final, change only through Daryn):

    s = Session(out_dir, child="K03", age_months=47, video=False)
    path = s.save_try(tr, raw26=None, task=2, leg="L", guessed_task=2, guessed_leg="L",
                      task_confidence=0.8, corrected=False, score=1, confidence=0.71,
                      clues={...}, reasons="3 hops, ...", events=[...])
    record = load_record(path)       # dict with "try" (common.format.Try) and everything saved

Files:  {out_dir}/{child}_task{task}_try{n}_{leg}.json  (n counts per child and task, from 1)
        {out_dir}/log.csv  (LOG_FIELDS)
Every try is written as soon as it ends, so a crash never loses earlier tries.
events is the moment timeline (features.clues.timeline); it goes in the JSON file only.
Video saving (only when the parent ticked level 3 and V is on) is Revallo's Nov 15 task.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from common.format import Try, validate
from model.predict import CHECK_BY_TEACHER_BELOW

LOG_FIELDS = [
    "time",
    "child",
    "age_months",
    "task",
    "try",
    "leg",
    "guessed_task",
    "guessed_leg",
    "task_confidence",
    "corrected",
    "score",
    "confidence",
    "check_by_teacher",
    "reasons",
    "n_frames",
    "duration_s",
    "video",
    "json_file",
]


class Session:
    def __init__(self, out_dir, child: str, age_months: int, video: bool = False):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.child, self.age_months, self.video = child, int(age_months), bool(video)
        self.log_path = self.out_dir / "log.csv"
        self._counts: dict[int, int] = {}
        for p in self.out_dir.glob(
            f"{child}_task*_try*_*.json"
        ):  # continue numbering after a restart
            task, n = int(p.name.split("_task")[1][0]), int(p.name.split("_try")[1].split("_")[0])
            self._counts[task] = max(self._counts.get(task, 0), n)

    def save_try(
        self,
        tr: Try,
        *,
        task: int,
        leg: str,
        guessed_task: int,
        guessed_leg: str,
        task_confidence: float,
        corrected: bool,
        score: int,
        confidence: float,
        clues: dict,
        reasons: str,
        raw26: np.ndarray | None = None,
        events: list[dict] | None = None,
    ) -> Path:
        validate(tr)
        n = self._counts.get(task, 0) + 1
        self._counts[task] = n
        check = confidence < CHECK_BY_TEACHER_BELOW
        name = f"{self.child}_task{task}_try{n}_{leg}.json"
        rec = {
            "meta": {
                **tr.meta,
                "child": self.child,
                "age_months": self.age_months,
                "task": task,
                "try": n,
                "leg": leg,
            },
            "t": tr.t.tolist(),
            "points25": tr.points.tolist(),
            "points26": None if raw26 is None else np.asarray(raw26, dtype=np.float32).tolist(),
            "guessed_task": guessed_task,
            "guessed_leg": guessed_leg,
            "task_confidence": task_confidence,
            "corrected": corrected,
            "clues": clues,
            "score": score,
            "confidence": confidence,
            "check_by_teacher": check,
            "reasons": reasons,
            "events": events,
        }
        path = self.out_dir / name
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rec))
        tmp.replace(path)  # atomic: a half-written file never appears

        new = not self.log_path.exists()
        with self.log_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
            if new:
                w.writeheader()
            w.writerow(
                {
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "child": self.child,
                    "age_months": self.age_months,
                    "task": task,
                    "try": n,
                    "leg": leg,
                    "guessed_task": guessed_task,
                    "guessed_leg": guessed_leg,
                    "task_confidence": f"{task_confidence:.2f}",
                    "corrected": int(corrected),
                    "score": "" if check else score,
                    "confidence": f"{confidence:.2f}",
                    "check_by_teacher": int(check),
                    "reasons": reasons,
                    "n_frames": tr.n_frames,
                    "duration_s": f"{tr.duration:.2f}",
                    "video": int(self.video),
                    "json_file": name,
                }
            )
        return path


def load_record(path) -> dict:
    rec = json.loads(Path(path).read_text())
    rec["try"] = validate(
        Try(
            points=np.asarray(rec["points25"], dtype=np.float32),
            t=np.asarray(rec["t"], dtype=np.float64),
            meta=rec["meta"],
        )
    )
    return rec
