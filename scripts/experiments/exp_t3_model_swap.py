"""EXPERIMENT (Marty, 2026-11-23): does another model beat the random forest on the 4 task 3 clues?

Same rows as scripts/make_clue_table.py (group B task 3, clips under 30 frames left out, child
picked with common.pick.pick_child), clues from features.clues.compute, features from
model.train_rf.features_for(3). GroupKFold(5) by child, seeds 0-4 (25 runs per model), macro-F1
and quadratic kappa computed like model.train_rf (_macro_f1, _kappa). Writes no files.

Run:  .venv\\Scripts\\python.exe scripts/experiments/exp_t3_model_swap.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.pick import pick_child  # noqa: E402
from data.load_b import DEFAULT_ZIP, FPS, list_files, parse_name, read_file  # noqa: E402
from features.clues import compute  # noqa: E402
from model.train_rf import _kappa, _macro_f1, features_for, make_forest  # noqa: E402

MIN_FRAMES = 30  # same as scripts/make_clue_table.py

MODELS = {
    "random forest (make_forest)": make_forest,
    "boosting (defaults)": lambda s: HistGradientBoostingClassifier(
        class_weight="balanced", random_state=s
    ),
    "boosting (lr 0.05, depth 3)": lambda s: HistGradientBoostingClassifier(
        class_weight="balanced", learning_rate=0.05, max_depth=3, random_state=s
    ),
    "logistic regression": lambda s: make_pipeline(
        StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)
    ),
}


def load() -> pd.DataFrame:
    rows = []
    with zipfile.ZipFile(DEFAULT_ZIP) as z:
        for name in list_files():
            m = parse_name(name)
            if m["task"] != 3:
                continue
            frames, label, _ = read_file(z, name)
            if len(frames) < MIN_FRAMES:
                continue
            if max(len(fr) for fr in frames) <= 1:
                pts = np.stack(
                    [fr[0] if len(fr) else np.zeros((25, 3), np.float32) for fr in frames]
                )
            else:
                pts = pick_child(frames)
            t = np.arange(len(pts)) / FPS
            try:
                clues = compute(3, pts, t)
            except ValueError:  # no frame with both neck and mid-hip
                continue
            rows.append({"child": str(m["child"]), "label": label, **clues})
    return pd.DataFrame(rows)


def main() -> None:
    d = load()
    x = d[features_for(3)].to_numpy(dtype=np.float64)
    y = d["label"].to_numpy(dtype=int)
    g = d["child"].to_numpy()
    folds = list(GroupKFold(n_splits=5).split(x, y, g))
    print(f"task 3: {len(d)} rows, {len(set(g))} children, features {features_for(3)}")
    print(f"  {'model':<28} macro-F1 (seed 0)  kappa (seed 0)   [25 runs]")
    for title, make in MODELS.items():
        f1 = np.zeros((5, 5))
        kp = np.zeros((5, 5))
        for seed in range(5):
            for k, (tr, te) in enumerate(folds):
                pr = make(seed).fit(x[tr], y[tr]).predict(x[te])
                f1[seed, k], kp[seed, k] = _macro_f1(y[te], pr), _kappa(y[te], pr)
        print(
            f"  {title:<28} {f1.mean():.3f} ({f1[0].mean():.3f})     "
            f"{kp.mean():.3f} ({kp[0].mean():.3f})"
        )


if __name__ == "__main__":
    main()
