"""Score one try from its clues. Owner: Daryn (real model by Nov 19).

Shared interface (final, change only through Daryn):

    predict(task, clues) -> (score, confidence)
        task:  1, 2 or 3
        clues: dict from features.clues.compute(task, ...)
        score: int 0 (Bad), 1 (Good) or 2 (Perfect)
        confidence: float 0..1 (e.g. the random forest's probability of the chosen score)

The app writes "check by teacher" instead of a score when confidence < CHECK_BY_TEACHER_BELOW.

Model files: models/task{n}_rf.joblib, a dict {"model": fitted sklearn classifier,
"features": list of clue names in order}. Committed under 10 MB; if a model
file is missing it falls back to (1, 0.0) so rows say "check by teacher".
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
CHECK_BY_TEACHER_BELOW = 0.6


@lru_cache(maxsize=None)
def _load(task: int):
    path = MODELS_DIR / f"task{task}_rf.joblib"
    if not path.exists():
        return None
    import joblib

    return joblib.load(path)


def predict(task: int, clues: dict[str, float]) -> tuple[int, float]:
    if task not in (1, 2, 3):
        raise ValueError("task must be 1, 2 or 3")
    bundle = _load(task)
    if bundle is None:
        return 1, 0.0
    x = np.array([[clues[name] for name in bundle["features"]]], dtype=np.float64)
    proba = bundle["model"].predict_proba(x)[0]
    k = int(np.argmax(proba))
    return int(bundle["model"].classes_[k]), float(proba[k])


def predict_proba(task: int, clues: dict[str, float]) -> tuple[float, float, float]:
    """(p_bad, p_good, p_perfect) of one try, from the same forest as predict().

    Ordered by score via model.classes_ (a score the forest never saw gets 0). Without a model
    file it returns (0.0, 0.0, 0.0). Used by the live evaluation bar in app/screen.py.
    """
    if task not in (1, 2, 3):
        raise ValueError("task must be 1, 2 or 3")
    bundle = _load(task)
    if bundle is None:
        return 0.0, 0.0, 0.0
    x = np.array([[clues[name] for name in bundle["features"]]], dtype=np.float64)
    proba = bundle["model"].predict_proba(x)[0]
    out = [0.0, 0.0, 0.0]
    for cls, p in zip(bundle["model"].classes_, proba, strict=True):
        if int(cls) in (0, 1, 2):
            out[int(cls)] = float(p)
    return out[0], out[1], out[2]
