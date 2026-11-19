"""Guess which task the child did and on which leg. Owner: Marty (Nov 22).

Shared interface (final, change only through Daryn):

    predict_task(points, t) -> (task, leg, confidence)
        points: float32 (T, 25, 3) of the child, pixels; t: float64 (T,) seconds
        task:   1, 2 or 3
        leg:    "L" or "R" = the support leg (the one the child stands or hops on);
                for task 3 (both feet) it is still returned but the app ignores it
        confidence: float 0..1 for the task guess

The operator always confirms (Enter) or corrects (1/2/3, L/R), so a wrong guess costs one key.
If accuracy on group B (GroupKFold by child) is low, the app uses keys only.
When models/task_classifier.joblib is present, predict_task uses the fitted RandomForest on the
extracted clues; without a model file on disk, it returns (1, leg, 0.0) so keys-only mode and
existing UI tests work identically to model/predict.py.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from common import body25 as b
from common.geometry import normalize
from data.load_b import DEFAULT_ZIP, iter_tries
from features.clues import compute as compute_clues

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_FILENAME = "task_classifier.joblib"
TASKS = (1, 2, 3)
N_SPLITS = 5
SEED = 0

CLUE_COLUMNS = [
    "one_foot_s",
    "touchdowns",
    "sway",
    "wobble_s",
    "hops",
    "longest_pause_s",
    "feet_gap_takeoff",
    "feet_gap_landing",
    "air_s",
    "jump_size",
]

EXTRA_COLUMNS = [
    "one_foot_frac",
    "foot_diff_med",
    "foot_diff_p75",
    "sup_vy_std",
    "hip_vy_std",
    "sup_y_range",
]

TASK_FEATURES = CLUE_COLUMNS + EXTRA_COLUMNS


def _interp_1d(arr: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float64, copy=True)
    ok = np.isfinite(out)
    if not np.any(ok):
        return np.zeros_like(out)
    if not np.all(ok):
        idx = np.arange(len(out))
        out[~ok] = np.interp(idx[~ok], idx[ok], out[ok])
    return out


def _smooth_1d(arr: np.ndarray, width: int = 5) -> np.ndarray:
    if len(arr) < width or width <= 1:
        return arr.copy()
    pad = width // 2
    padded = np.pad(arr, pad, mode="edge")
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.convolve(padded, kernel, mode="valid")


def support_leg_from_norm(n: np.ndarray) -> str:
    """Return 'L' or 'R' for the lower (support) ankle/foot in normalized coordinates (y points up)."""
    l_pts = n[:, b.LEFT_FOOT, 1]
    r_pts = n[:, b.RIGHT_FOOT, 1]
    l_ank = n[:, b.L_ANKLE, 1]
    r_ank = n[:, b.R_ANKLE, 1]
    ly = (
        float(np.nanmean(l_ank))
        if np.isfinite(l_ank).any()
        else (float(np.nanmean(l_pts)) if np.isfinite(l_pts).any() else np.nan)
    )
    ry = (
        float(np.nanmean(r_ank))
        if np.isfinite(r_ank).any()
        else (float(np.nanmean(r_pts)) if np.isfinite(r_pts).any() else np.nan)
    )
    return "L" if (np.isnan(ry) or (not np.isnan(ly) and ly <= ry)) else "R"


def extract_task_clues(points: np.ndarray, t: np.ndarray) -> dict[str, float]:
    """Compute movement clues from Louise & Revallo (features.clues.compute) plus task kinematics."""
    n = normalize(points)
    n_frames = len(t)
    dt = float((t[-1] - t[0]) / max(n_frames - 1, 1)) if n_frames > 1 else 1.0 / 30.0
    duration_s = float(t[-1] - t[0]) if n_frames > 1 else 0.0

    ly = _interp_1d(n[:, b.L_ANKLE, 1])
    ry = _interp_1d(n[:, b.R_ANKLE, 1])
    hy = _interp_1d(n[:, b.MID_HIP, 1])

    t1_clues = compute_clues(1, points, t)
    t2_clues = compute_clues(2, points, t)
    t3_clues = compute_clues(3, points, t)

    foot_diff = np.abs(ly - ry)
    up_mask = foot_diff > 0.12
    one_foot_frac = float(up_mask.mean()) if n_frames > 0 else 0.0
    foot_diff_med = float(np.median(foot_diff)) if n_frames > 0 else 0.0
    foot_diff_p75 = float(np.percentile(foot_diff, 75)) if n_frames > 0 else 0.0

    sup_y = np.minimum(ly, ry)
    sup_y_smooth = _smooth_1d(sup_y, width=5)
    base_y = float(np.percentile(sup_y_smooth, 20)) if n_frames > 0 else 0.0
    hop_mask = up_mask & ((sup_y_smooth - base_y) > 0.07)
    fallback_hops = float(
        np.sum((~hop_mask[:-1]) & hop_mask[1:]) + (1 if (len(hop_mask) > 0 and hop_mask[0]) else 0)
    )
    hops = float(t2_clues["hops"]) if float(t2_clues["hops"]) > 0 else fallback_hops
    sup_vy_std = float(np.std(np.diff(sup_y_smooth) / max(dt, 1e-3))) if n_frames > 1 else 0.0
    hip_vy_std = float(np.std(np.diff(hy) / max(dt, 1e-3))) if n_frames > 1 else 0.0
    sup_y_range = (
        float(np.percentile(sup_y_smooth, 95) - np.percentile(sup_y_smooth, 5))
        if n_frames > 0
        else 0.0
    )

    return {
        "duration_s": duration_s,
        "one_foot_s": float(t1_clues["one_foot_s"]),
        "touchdowns": float(t1_clues["touchdowns"]),
        "sway": float(t1_clues["sway"]),
        "wobble_s": float(t1_clues["wobble_s"]),
        "hops": hops,
        "longest_pause_s": float(t2_clues["longest_pause_s"]),
        "feet_gap_takeoff": float(t3_clues["feet_gap_takeoff"]),
        "feet_gap_landing": float(t3_clues["feet_gap_landing"]),
        "air_s": float(t3_clues["air_s"]),
        "jump_size": float(t3_clues["jump_size"]),
        "one_foot_frac": one_foot_frac,
        "foot_diff_med": foot_diff_med,
        "foot_diff_p75": foot_diff_p75,
        "sup_vy_std": sup_vy_std,
        "hip_vy_std": hip_vy_std,
        "sup_y_range": sup_y_range,
    }


def predict_task_from_clues(points: np.ndarray, t: np.ndarray) -> tuple[int, str, float]:
    """Guess (task 1/2/3, support leg 'L'/'R', confidence 0..1) directly from normalized clues."""
    n = normalize(points)
    leg = support_leg_from_norm(n)
    f = extract_task_clues(points, t)

    s1 = (
        2.6 * f["one_foot_frac"]
        + 0.45 * min(f["one_foot_s"], 5.0)
        - 1.1 * f["hops"]
        - 4.5 * f["sup_y_range"]
        - 1.8 * f["jump_size"]
    )
    s2 = (
        1.6 * f["one_foot_frac"]
        + 1.15 * min(f["hops"], 8.0)
        + 4.5 * f["sup_y_range"]
        - 0.6 * max(0.0, f["jump_size"] - 1.0)
    )
    s3 = (
        3.2 * (1.0 - f["one_foot_frac"])
        + 2.4 * f["jump_size"]
        + 1.5 * f["air_s"]
        - 4.0 * f["foot_diff_med"]
        - 0.8 * f["hops"]
    )

    logits = np.array([s1, s2, s3], dtype=np.float64)
    logits -= logits.max()
    probs = np.exp(logits)
    probs /= probs.sum()
    best_idx = int(np.argmax(probs))
    return int(TASKS[best_idx]), leg, float(np.clip(probs[best_idx], 0.0, 1.0))


@lru_cache(maxsize=None)
def _load():
    path = MODELS_DIR / MODEL_FILENAME
    if not path.exists():
        return None
    return joblib.load(path)


def fit_and_save_task_model(
    table: pd.DataFrame,
    models_dir: str | Path = MODELS_DIR,
    feature_cols: list[str] | None = None,
    seed: int = SEED,
) -> Path:
    """Fit a task classifier RandomForest on table and save task_classifier.joblib."""
    if feature_cols is None:
        feature_cols = [c for c in TASK_FEATURES if c in table.columns]
    x = table[feature_cols].to_numpy(dtype=np.float64)
    y = table["task"].to_numpy(dtype=int)
    clf = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    ).fit(x, y)
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    path = models_dir / MODEL_FILENAME
    joblib.dump({"model": clf, "features": list(feature_cols)}, path, compress=3)
    return path


def predict_task(points: np.ndarray, t: np.ndarray) -> tuple[int, str, float]:
    """Guess (task, leg, confidence). Uses models/task_classifier.joblib if present; else keys-only (1, leg, 0.0)."""
    n = normalize(points)
    leg = support_leg_from_norm(n)
    bundle = _load()
    if bundle is None:
        return 1, leg, 0.0
    clues = extract_task_clues(points, t)
    x = np.array([[clues[name] for name in bundle["features"]]], dtype=np.float64)
    proba = bundle["model"].predict_proba(x)[0]
    k = int(np.argmax(proba))
    return int(bundle["model"].classes_[k]), leg, float(proba[k])


def build_group_b_clue_table(
    zip_path: str | Path = DEFAULT_ZIP,
    task1_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Build the clue table across all valid Group B task 1-3 files.

    By default every clue comes from extract_task_clues (features.clues.compute), exactly like
    the live app, so the saved model sees the same numbers in training and live. task1_csv is
    only for experiments: if given, its Task 1 columns replace the computed ones.
    """
    t1_map: dict[str, dict[str, float]] = {}
    if task1_csv is not None and Path(task1_csv).exists():
        t1_df = pd.read_csv(task1_csv)
        cols = [
            c
            for c in ("duration_s", "one_foot_s", "touchdowns", "sway", "wobble_s")
            if c in t1_df.columns
        ]
        for _, row in t1_df.iterrows():
            t1_map[str(row["file"])] = {c: float(row[c]) for c in cols}

    rows = []
    for _name, tr in iter_tries(zip_path=zip_path, tasks=TASKS):
        m = tr.meta
        n = normalize(tr.points)
        clues = extract_task_clues(tr.points, tr.t)
        if int(m["task"]) == 1 and m["file"] in t1_map:
            clues.update(t1_map[m["file"]])
        rows.append(
            {
                "file": m["file"],
                "child": m["child"],
                "task": int(m["task"]),
                "attempt": int(m["attempt"]),
                "camera": int(m["camera"]),
                "label": int(m["label_index"]),
                "leg": support_leg_from_norm(n),
                **clues,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class TaskCVResult:
    n_rows: int
    n_children: int
    feature_names: list[str]
    leak_free: bool = True
    fold_acc: list[float] = field(default_factory=list)
    fold_f1: list[float] = field(default_factory=list)
    overall_acc: float = 0.0
    overall_f1: float = 0.0
    matrix: np.ndarray = field(default_factory=lambda: np.zeros((3, 3), dtype=int))


def cross_validate_task_classifier(
    table: pd.DataFrame,
    feature_cols: list[str] | None = None,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> TaskCVResult:
    """GroupKFold(5) by child for predicting task (1/2/3) from the clue table."""
    if feature_cols is None:
        feature_cols = [c for c in TASK_FEATURES if c in table.columns]
    x = table[feature_cols].to_numpy(dtype=np.float64)
    y = table["task"].to_numpy(dtype=int)
    groups = table["child"].astype(str).to_numpy()
    n_children = len(np.unique(groups))
    if n_children < n_splits:
        raise ValueError(f"need at least {n_splits} children, got {n_children}")

    res = TaskCVResult(n_rows=len(y), n_children=n_children, feature_names=list(feature_cols))
    oof_pred = np.zeros_like(y)

    for train, test in GroupKFold(n_splits=n_splits).split(x, y, groups):
        if set(groups[train]) & set(groups[test]):
            res.leak_free = False
        clf = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(x[train], y[train])
        pred = clf.predict(x[test])
        oof_pred[test] = pred
        res.fold_acc.append(float(accuracy_score(y[test], pred)))
        res.fold_f1.append(
            float(f1_score(y[test], pred, labels=list(TASKS), average="macro", zero_division=0))
        )

    res.overall_acc = float(accuracy_score(y, oof_pred))
    res.overall_f1 = float(
        f1_score(y, oof_pred, labels=list(TASKS), average="macro", zero_division=0)
    )
    res.matrix = confusion_matrix(y, oof_pred, labels=list(TASKS))
    return res


def save_task_confusion_matrix(
    matrix: np.ndarray, out_dir: Path, prefix: str = "task_classifier"
) -> tuple[Path, Path]:
    """Save the 3x3 task confusion matrix as CSV and PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = ["Task 1 (stand)", "Task 2 (hop)", "Task 3 (jump)"]
    df = pd.DataFrame(
        matrix,
        index=[f"true_{i}" for i in TASKS],
        columns=[f"pred_{i}" for i in TASKS],
    )
    csv_path = out_dir / f"{prefix}_cm.csv"
    df.to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(5.2, 4.4), dpi=140)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(3), labels=labels, rotation=15, ha="right")
    ax.set_yticks(range(3), labels=labels)
    ax.set_xlabel("Predicted task")
    ax.set_ylabel("True task")
    ax.set_title("Task Classifier Confusion Matrix (GroupKFold by child)")
    thresh = matrix.max() / 2.0 if matrix.size else 0.0
    for i in range(3):
        for j in range(3):
            val = int(matrix[i, j])
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontweight="bold",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    png_path = out_dir / f"{prefix}_cm.png"
    fig.savefig(png_path)
    plt.close(fig)
    return csv_path, png_path


def report_task_cv(res: TaskCVResult, title: str = "Task classifier") -> str:
    lines = [
        f"{title}: {res.n_rows} files, {res.n_children} children, GroupKFold({len(res.fold_acc)}) by child",
        f"  features ({len(res.feature_names)}): {', '.join(res.feature_names)}",
        f"  child leak check: {'PASS' if res.leak_free else 'FAIL'}",
        f"  {'fold':<6}{'accuracy':>10}{'macro-F1':>10}",
    ]
    for i, (acc, f1) in enumerate(zip(res.fold_acc, res.fold_f1, strict=True), start=1):
        lines.append(f"  {i:<6}{acc:>10.4f}{f1:>10.4f}")
    lines += [
        f"  mean ± sd accuracy: {np.mean(res.fold_acc):.4f} ± {np.std(res.fold_acc):.4f} (overall {res.overall_acc:.4f})",
        f"  mean ± sd macro-F1: {np.mean(res.fold_f1):.4f} ± {np.std(res.fold_f1):.4f} (overall {res.overall_f1:.4f})",
        "  confusion matrix (rows = true task 1/2/3, cols = predicted task 1/2/3):",
        f"    task 1: {res.matrix[0].tolist()}",
        f"    task 2: {res.matrix[1].tolist()}",
        f"    task 3: {res.matrix[2].tolist()}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Task (1/2/3) and support-leg classifier evaluation")
    ap.add_argument(
        "--table", type=Path, help="optional clue table CSV (default: compute from B_skel.zip)"
    )
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="path to B_skel.zip")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/marty_eval/task_classifier"),
        help="directory for confusion matrix CSV and PNG",
    )
    ap.add_argument(
        "--save",
        action="store_true",
        help="train on all files and save models/task_classifier.joblib",
    )
    args = ap.parse_args(argv)

    if args.table:
        table = pd.read_csv(args.table, dtype={"child": str})
    else:
        table = build_group_b_clue_table(args.zip)

    res_clues = cross_validate_task_classifier(table, feature_cols=CLUE_COLUMNS)
    print(report_task_cv(res_clues, title="Task classifier (10 clue columns)"))
    print()

    avail_all = [c for c in TASK_FEATURES if c in table.columns]
    res_all = cross_validate_task_classifier(table, feature_cols=avail_all)
    print(report_task_cv(res_all, title="Task classifier (clues + kinematic features)"))

    if "leg" in table.columns:
        leg_counts = table.groupby(["task", "leg"]).size().unstack(fill_value=0)
        print("\nSupport leg distribution (L/R) across Group B files:")
        for task in TASKS:
            if task in leg_counts.index:
                l_cnt = int(leg_counts.loc[task].get("L", 0))
                r_cnt = int(leg_counts.loc[task].get("R", 0))
                print(f"  task {task}: L={l_cnt}, R={r_cnt}")

    csv_path, png_path = save_task_confusion_matrix(res_all.matrix, args.out_dir)
    print(f"\nSaved confusion matrix CSV: {csv_path}")
    print(f"Saved confusion matrix PNG: {png_path}")

    if args.save:
        model_path = fit_and_save_task_model(table, feature_cols=avail_all)
        print(f"Saved task model: {model_path}")
    return 0 if (res_clues.leak_free and res_all.leak_free) else 1


if __name__ == "__main__":
    raise SystemExit(main())
