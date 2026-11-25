"""Random forest per task on the clue table, GroupKFold by child. Owner: Daryn (Nov 16-19).

Saves models/task{n}_rf.joblib as {"model": fitted classifier, "features": CLUE_NAMES[n] without
duration_s} (the format model/predict.py loads). Gate on Nov 19: macro-F1 and kappa above the
baseline. For task 2 also try a per-second forest (one-second windows labelled with the score of
their try, probabilities averaged over the try) and keep whichever scores higher (not tried for
the Nov 19 gate: see docs/task_cards/daryn_2026-11-19_gate.md).

Clue table (Louise, #22): a CSV with one row per dataset file and the columns
    child   e.g. "B010": the group for GroupKFold, so no child is in train and test at once
    task    1, 2 or 3
    label   0 Bad, 1 Good, 2 Perfect (the words Bad/Good/Perfect, or a column called
            label_index, also work)
    + the clue names in features.clues.CLUE_NAMES[task]
Other columns (file, attempt, camera, ...) are allowed and ignored. A table may hold several
tasks; each task only needs its own clue columns.

Usage:
    python -m model.train_rf --table clue_table.csv                  # every task in the table
    python -m model.train_rf --table clue_table.csv --task 1 --save  # also writes the model file
    python -m model.train_rf --fake                                  # fake table: checks the code only

duration_s is never a feature: live, the length of a try comes from our own automatic start and
end, so the model must not learn from it.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import GroupKFold

from features.clues import CLUE_NAMES

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
N_SPLITS = 5
SEED = 0
LABELS = (0, 1, 2)
LABEL_WORDS = {"Bad": 0, "Good": 1, "Perfect": 2}
NOT_FEATURES = ("duration_s",)
MAX_MODEL_MB = 10.0  # predict.py: commit the model file only if it is under 10 MB


def features_for(task: int) -> list[str]:
    """The model's features for one task, in order: CLUE_NAMES[task] without duration_s."""
    if task not in CLUE_NAMES:
        raise ValueError("task must be 1, 2 or 3")
    return [name for name in CLUE_NAMES[task] if name not in NOT_FEATURES]


def make_forest(seed: int = SEED) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=3,
        class_weight="balanced",  # the labels are unbalanced (task 3: mostly Perfect)
        random_state=seed,
        n_jobs=-1,
    )


# ---------------------------------------------------------------------------------------------
# The clue table


def check_table(table: pd.DataFrame, task: int | None = None) -> None:
    """Raise ValueError with a clear message if the table can't be used for training."""
    missing = {"child", "task", "label"} - set(table.columns)
    if missing:
        raise ValueError(f"clue table is missing the columns {sorted(missing)}")
    if not table["label"].isin(LABELS).all():
        bad = sorted(set(table.loc[~table["label"].isin(LABELS), "label"].astype(str)))
        raise ValueError(f"labels must be 0, 1 or 2 (Bad, Good, Perfect); found {bad[:5]}")
    tasks = [task] if task is not None else sorted(int(t) for t in table["task"].unique())
    for t in tasks:
        rows = table[table["task"] == t]
        if rows.empty:
            raise ValueError(f"the clue table has no rows for task {t}")
        cols = features_for(t)
        missing = [c for c in cols if c not in table.columns]
        if missing:
            raise ValueError(f"task {t}: the clue table is missing the clue columns {missing}")
        values = rows[cols].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            n = int((~np.isfinite(values)).any(axis=1).sum())
            raise ValueError(
                f"task {t}: {n} rows have an empty or infinite clue value; fill them in "
                "(and write down how) before training"
            )


def load_table(path: str | Path) -> pd.DataFrame:
    """Read the clue table CSV, accept label words or label_index, and check it."""
    table = pd.read_csv(path, dtype={"child": str})
    if "label" not in table.columns and "label_index" in table.columns:
        table = table.rename(columns={"label_index": "label"})
    if "label" in table.columns:
        table["label"] = table["label"].map(lambda v: LABEL_WORDS.get(v, v))
        table["label"] = pd.to_numeric(table["label"], errors="coerce")
    check_table(table)
    table["label"] = table["label"].astype(int)
    table["task"] = table["task"].astype(int)
    return table


def _task_arrays(table: pd.DataFrame, task: int):
    check_table(table, task)
    rows = table[table["task"] == task]
    x = rows[features_for(task)].to_numpy(dtype=np.float64)
    y = rows["label"].to_numpy(dtype=int)
    groups = rows["child"].astype(str).to_numpy()
    return x, y, groups


# ---------------------------------------------------------------------------------------------
# Cross-validation and the model file


def _kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    k = cohen_kappa_score(y_true, y_pred, labels=list(LABELS), weights="quadratic")
    return 0.0 if np.isnan(k) else float(k)  # undefined when both sides are one score only


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=list(LABELS), average="macro", zero_division=0))


@dataclass
class CVResult:
    task: int
    n_rows: int
    n_children: int
    leak_free: bool = True
    f1: list[float] = field(default_factory=list)  # random forest, one value per fold
    kappa: list[float] = field(default_factory=list)
    base_f1: list[float] = field(default_factory=list)  # always the most common score
    base_kappa: list[float] = field(default_factory=list)

    @property
    def beats_baseline(self) -> bool:
        """The Nov 19 gate: mean macro-F1 and mean kappa both above the baseline."""
        return bool(
            np.mean(self.f1) > np.mean(self.base_f1)
            and np.mean(self.kappa) > np.mean(self.base_kappa)
        )


def cross_validate(
    table: pd.DataFrame, task: int, n_splits: int = N_SPLITS, seed: int = SEED
) -> CVResult:
    """GroupKFold by child: random forest vs always guessing the most common score."""
    x, y, groups = _task_arrays(table, task)
    n_children = len(np.unique(groups))
    if n_children < n_splits:
        raise ValueError(f"task {task}: {n_children} children, need at least {n_splits}")
    res = CVResult(task=task, n_rows=len(y), n_children=n_children)
    for train, test in GroupKFold(n_splits=n_splits).split(x, y, groups):
        if set(groups[train]) & set(groups[test]):
            res.leak_free = False
        forest = make_forest(seed).fit(x[train], y[train])
        dummy = DummyClassifier(strategy="most_frequent").fit(x[train], y[train])
        for model, f1s, kappas in (
            (forest, res.f1, res.kappa),
            (dummy, res.base_f1, res.base_kappa),
        ):
            pred = model.predict(x[test])
            f1s.append(_macro_f1(y[test], pred))
            kappas.append(_kappa(y[test], pred))
    return res


def fit_and_save(
    table: pd.DataFrame, task: int, models_dir: str | Path = MODELS_DIR, seed: int = SEED
) -> Path:
    """Fit on every row of the task and write models/task{n}_rf.joblib for model/predict.py."""
    x, y, _ = _task_arrays(table, task)
    model = make_forest(seed).fit(x, y)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / f"task{task}_rf.joblib"
    joblib.dump({"model": model, "features": features_for(task)}, path, compress=3)
    return path


def report(res: CVResult) -> str:
    def ms(values: list[float]) -> str:
        return f"{np.mean(values):.2f} ± {np.std(values):.2f}"

    lines = [
        f"Task {res.task}: {res.n_rows} files, {res.n_children} children, "
        f"GroupKFold({len(res.f1)}) by child",
        f"  child leak check: {'PASS' if res.leak_free else 'FAIL'}",
        f"  {'fold':<6}{'forest F1':>11}{'forest kappa':>14}{'base F1':>10}{'base kappa':>12}",
    ]
    folds = zip(res.f1, res.kappa, res.base_f1, res.base_kappa, strict=True)
    for i, (f1, k, bf1, bk) in enumerate(folds, start=1):
        lines.append(f"  {i:<6}{f1:>11.2f}{k:>14.2f}{bf1:>10.2f}{bk:>12.2f}")
    lines += [
        f"  random forest     macro-F1 {ms(res.f1)}   kappa {ms(res.kappa)}",
        f"  most common score macro-F1 {ms(res.base_f1)}   kappa {ms(res.base_kappa)}",
        f"  beats the baseline (both numbers): {'yes' if res.beats_baseline else 'no'}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# A fake table, so the code and the tests run before Louise's table exists


def _fake_clues(task: int, level: float, rng: np.random.Generator) -> dict[str, float]:
    def noise(scale: float) -> float:
        return float(rng.normal(scale=scale))

    duration = float(rng.uniform(4.0, 8.0))
    if task == 1:
        clues = {
            "one_foot_s": float(np.clip(2.5 + 2.0 * level + noise(0.8), 0.0, duration)),
            "touchdowns": float(max(0, round(1.0 - level + noise(0.5)))),
            "sway": float(abs(0.30 - 0.08 * level + noise(0.05))),
            "wobble_s": float(max(0.0, 1.0 - 0.5 * level + noise(0.3))),
            "hip_speed": float(max(0.0, 0.20 - 0.06 * level + noise(0.04))),
            "neck_speed": float(max(0.0, 0.25 - 0.07 * level + noise(0.05))),
            "lift_std": float(max(0.0, 0.12 - 0.04 * level + noise(0.03))),
            "wrist_speed": float(max(0.0, 0.80 - 0.25 * level + noise(0.15))),
        }
    elif task == 2:
        clues = {
            "hops": float(max(0, round(2.0 + 1.5 * level + noise(0.7)))),
            "longest_pause_s": float(max(0.0, 1.0 - 0.4 * level + noise(0.3))),
            "touchdowns": float(max(0, round(1.0 - level + noise(0.5)))),
        }
    else:
        clues = {
            "feet_gap_takeoff": float(abs(0.40 - 0.15 * level + noise(0.10))),
            "feet_gap_landing": float(abs(0.45 - 0.15 * level + noise(0.10))),
            "air_s": float(max(0.05, 0.25 + 0.05 * level + noise(0.03))),
            "jump_size": float(max(0.0, 1.0 + 0.4 * level + noise(0.20))),
            "n_air": float(max(1, round(1.0 + noise(0.3)))),
            "best_jump_size": float(max(0.0, 1.1 + 0.4 * level + noise(0.20))),
            "dx": float(abs(0.8 + 0.3 * level + noise(0.15))),
            "dy": float(0.3 + 0.1 * level + noise(0.08)),
            "scale_gain": float(0.05 + 0.03 * level + noise(0.03)),
            "crouch": float(0.25 + 0.05 * level + noise(0.05)),
            "hip_peak": float(max(0.0, 0.3 + 0.08 * level + noise(0.05))),
            "arm_up": float(0.2 + 0.2 * level + noise(0.15)),
            "arm_range": float(max(0.0, 1.0 + 0.3 * level + noise(0.2))),
            "land_squat": float(0.3 + 0.05 * level + noise(0.05)),
            "land_move": float(max(0.0, 0.3 - 0.1 * level + noise(0.08))),
            "land_step": float(max(0.0, 0.2 - 0.08 * level + noise(0.06))),
            "hands_low": float(0.3 + 0.1 * level + noise(0.1)),
        }
    clues["duration_s"] = duration
    assert set(clues) == set(CLUE_NAMES[task]), "update _fake_clues when CLUE_NAMES changes"
    return clues


def make_fake_table(
    n_children: int = 40, attempts: int = 3, cameras: int = 3, seed: int = SEED
) -> pd.DataFrame:
    """A FAKE clue table for all 3 tasks (children F000, F001, ...). The clues follow a hidden
    level that also sets the label, so a working forest should beat the baseline on it. Its
    numbers only show that the code runs: never report them, never save a model from them."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_children):
        child = f"F{c:03d}"  # F for fake: never a real group B code
        skill = float(rng.normal())  # one hidden ability per child
        for task in (1, 2, 3):
            for attempt in range(1, attempts + 1):
                level = skill + float(rng.normal(scale=0.7))
                label = int(np.digitize(level, [-0.5, 0.5]))  # 0 Bad, 1 Good, 2 Perfect
                for camera in range(1, cameras + 1):
                    rows.append(
                        {
                            "child": child,
                            "task": task,
                            "attempt": attempt,
                            "camera": camera,
                            "label": label,
                            **_fake_clues(task, level, rng),
                        }
                    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--table", type=Path, help="clue table CSV (one row per dataset file)")
    src.add_argument("--fake", action="store_true", help="use a fake table to check the code")
    ap.add_argument("--task", type=int, choices=(1, 2, 3), help="only this task (default: all)")
    ap.add_argument("--save", action="store_true", help="also write models/task{n}_rf.joblib")
    ap.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = ap.parse_args(argv)

    if args.fake and args.save:
        print("Refusing to save a model trained on the fake table.")
        return 2
    if args.fake:
        table = make_fake_table()
        print("FAKE TABLE: these numbers only show that the code runs. Don't report them.\n")
    else:
        table = load_table(args.table)

    tasks = [args.task] if args.task else sorted(int(t) for t in table["task"].unique())
    status = 0
    for task in tasks:
        res = cross_validate(table, task)
        print(report(res))
        if not res.leak_free:
            status = 1
        if args.save:
            path = fit_and_save(table, task, args.models_dir)
            mb = path.stat().st_size / 1e6
            note = "" if mb < MAX_MODEL_MB else f" (over {MAX_MODEL_MB:.0f} MB: don't commit it)"
            print(f"  saved {path} ({mb:.1f} MB){note}")
        print()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
