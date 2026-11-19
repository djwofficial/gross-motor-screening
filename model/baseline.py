"""Most-common-score baseline per task, GroupKFold by child, plus RF confusion matrices.

Owner: Marty (Nov 13 baseline #15; Nov 22 confusion matrices & predict latency check #18).

Expected (Job Card 1, our earlier check with scikit-learn 1.9.1): macro-F1 0.19 / 0.20 / 0.26
for tasks 1 / 2 / 3, quadratic weighted kappa 0.00, and the child-leak check prints PASS.
Tools: DummyClassifier(strategy="most_frequent"), GroupKFold(5), f1_score(average="macro"),
cohen_kappa_score(weights="quadratic").

Usage:
    python -m model.baseline
    python -m model.baseline --matrices --out-dir runs/marty_eval/matrices
"""

from __future__ import annotations

import argparse
import importlib
import json
import time
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold

from data.load_b import DEFAULT_ZIP, list_files, parse_name
from features.clues import CLUE_NAMES
from model.task_classifier import build_group_b_clue_table
from model.train_rf import (
    LABELS,
    N_SPLITS,
    SEED,
    features_for,
    fit_and_save,
    load_table,
    make_forest,
)

predict_mod = importlib.import_module("model.predict")

SCORE_NAMES = ("Bad (0)", "Good (1)", "Perfect (2)")


@dataclass
class BaselineResult:
    task: int
    n_rows: int
    n_children: int
    leak_free: bool = True
    f1: list[float] = field(default_factory=list)
    kappa: list[float] = field(default_factory=list)


def load_group_b_labels(zip_path: str | Path = DEFAULT_ZIP, min_frames: int = 0) -> pd.DataFrame:
    """Load (file, child, task, attempt, camera, label) from B_skel.zip without full skeleton decode."""
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for name in list_files(zip_path):
            info = parse_name(name)
            if info["task"] not in (1, 2, 3):
                continue
            d = json.loads(z.read(name))
            if len(d["data"]) < min_frames:
                continue
            rows.append(
                {
                    "file": Path(name).name,
                    "child": info["child"],
                    "task": int(info["task"]),
                    "attempt": int(info["attempt"]),
                    "camera": int(info["camera"]),
                    "label": int(d["label_index"]),
                }
            )
    return pd.DataFrame(rows)


def evaluate_baseline(table: pd.DataFrame, task: int, n_splits: int = N_SPLITS) -> BaselineResult:
    """GroupKFold(5) by child with DummyClassifier(strategy='most_frequent')."""
    rows = table[table["task"] == task]
    y = rows["label"].to_numpy(dtype=int)
    groups = rows["child"].astype(str).to_numpy()
    x = np.zeros((len(y), 1), dtype=np.float64)
    n_children = len(np.unique(groups))
    if n_children < n_splits:
        raise ValueError(f"task {task}: {n_children} children, need at least {n_splits}")

    res = BaselineResult(task=task, n_rows=len(y), n_children=n_children)
    for train, test in GroupKFold(n_splits=n_splits).split(x, y, groups):
        if set(groups[train]) & set(groups[test]):
            res.leak_free = False
        pred = DummyClassifier(strategy="most_frequent").fit(x[train], y[train]).predict(x[test])
        res.f1.append(
            float(f1_score(y[test], pred, labels=list(LABELS), average="macro", zero_division=0))
        )
        k = cohen_kappa_score(y[test], pred, labels=list(LABELS), weights="quadratic")
        res.kappa.append(0.0 if np.isnan(k) else float(k))
    return res


def format_baseline_report(results: list[BaselineResult]) -> str:
    all_pass = all(r.leak_free for r in results)
    lines = [
        f"Child-leak check: {'PASS' if all_pass else 'FAIL'}",
        f"{'task':<6}{'files':>7}{'children':>10}{'macro-F1 (mean ± sd)':>24}{'kappa (mean ± sd)':>21}{'leak':>8}",
    ]
    for r in results:
        f1_str = f"{np.mean(r.f1):.2f} ± {np.std(r.f1):.2f} ({np.mean(r.f1):.4f})"
        k_str = f"{np.mean(r.kappa):.2f} ± {np.std(r.kappa):.2f}"
        leak_str = "PASS" if r.leak_free else "FAIL"
        lines.append(
            f"{r.task:<6}{r.n_rows:>7}{r.n_children:>10}{f1_str:>24}{k_str:>21}{leak_str:>8}"
        )
    return "\n".join(lines)


def _save_cm_csv_and_png(matrix: np.ndarray, csv_path: Path, png_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        matrix,
        index=["true_Bad_0", "true_Good_1", "true_Perfect_2"],
        columns=["pred_Bad_0", "pred_Good_1", "pred_Perfect_2"],
    )
    df.to_csv(csv_path)

    fig, ax = plt.subplots(figsize=(4.8, 4.2), dpi=140)
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(3), labels=list(SCORE_NAMES), rotation=15, ha="right")
    ax.set_yticks(range(3), labels=list(SCORE_NAMES))
    ax.set_xlabel("Predicted score")
    ax.set_ylabel("True expert score")
    ax.set_title(title)
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
    fig.savefig(png_path)
    plt.close(fig)


def generate_rf_confusion_matrices(
    table: pd.DataFrame,
    out_dir: str | Path,
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> dict:
    """Compute per-task and per-fold RF confusion matrices (CSV + PNG) and check predict < 1 s."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    created_files: list[Path] = []
    task_matrices: dict[int, np.ndarray] = {}
    fold_matrices: dict[tuple[int, int], np.ndarray] = {}

    tasks = sorted(int(t) for t in table["task"].unique())
    for task in tasks:
        rows = table[table["task"] == task]
        cols = features_for(task)
        x = rows[cols].to_numpy(dtype=np.float64)
        y = rows["label"].to_numpy(dtype=int)
        groups = rows["child"].astype(str).to_numpy()

        total_cm = np.zeros((3, 3), dtype=int)
        for fold_idx, (train, test) in enumerate(
            GroupKFold(n_splits=n_splits).split(x, y, groups), start=1
        ):
            forest = make_forest(seed).fit(x[train], y[train])
            pred = forest.predict(x[test])
            cm = confusion_matrix(y[test], pred, labels=list(LABELS))
            total_cm += cm
            fold_matrices[(task, fold_idx)] = cm

            f_csv = out_dir / f"task{task}_fold{fold_idx}_cm.csv"
            f_png = out_dir / f"task{task}_fold{fold_idx}_cm.png"
            _save_cm_csv_and_png(
                cm,
                f_csv,
                f_png,
                title=f"Task {task} RF Confusion Matrix — Fold {fold_idx}",
            )
            created_files.extend([f_csv, f_png])

            f1 = float(
                f1_score(y[test], pred, labels=list(LABELS), average="macro", zero_division=0)
            )
            k = cohen_kappa_score(y[test], pred, labels=list(LABELS), weights="quadratic")
            kappa = 0.0 if np.isnan(k) else float(k)
            summary_rows.append(
                {
                    "task": task,
                    "fold": str(fold_idx),
                    "n_samples": int(len(test)),
                    "macro_f1": round(f1, 4),
                    "kappa": round(kappa, 4),
                    **{f"cm_{i}_{j}": int(cm[i, j]) for i in range(3) for j in range(3)},
                }
            )

        task_matrices[task] = total_cm
        t_csv = out_dir / f"task{task}_cm.csv"
        t_png = out_dir / f"task{task}_cm.png"
        _save_cm_csv_and_png(
            total_cm,
            t_csv,
            t_png,
            title=f"Task {task} RF Confusion Matrix (All {n_splits} Folds)",
        )
        created_files.extend([t_csv, t_png])
        summary_rows.append(
            {
                "task": task,
                "fold": "all",
                "n_samples": int(total_cm.sum()),
                "macro_f1": round(
                    float(
                        np.mean(
                            [
                                r["macro_f1"]
                                for r in summary_rows
                                if r["task"] == task and r["fold"] != "all"
                            ]
                        )
                    ),
                    4,
                ),
                "kappa": round(
                    float(
                        np.mean(
                            [
                                r["kappa"]
                                for r in summary_rows
                                if r["task"] == task and r["fold"] != "all"
                            ]
                        )
                    ),
                    4,
                ),
                **{f"cm_{i}_{j}": int(total_cm[i, j]) for i in range(3) for j in range(3)},
            }
        )

    summary_csv = out_dir / "rf_confusion_matrices_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    created_files.append(summary_csv)

    # Measure model.predict.predict latency (< 1 s requirement) using temporary model bundles
    models_tmp = out_dir / "tmp_models"
    for task in tasks:
        fit_and_save(table, task, models_dir=models_tmp, seed=seed)

    old_dir = predict_mod.MODELS_DIR
    latencies_s: dict[int, float] = {}
    max_single_s = 0.0
    try:
        predict_mod.MODELS_DIR = models_tmp
        predict_mod._load.cache_clear()
        for task in tasks:
            sample_row = table[table["task"] == task].iloc[0]
            clues = {name: float(sample_row[name]) for name in CLUE_NAMES[task]}
            # Cold call (includes joblib load)
            t0 = time.perf_counter()
            score, conf = predict_mod.predict(task, clues)
            cold_s = time.perf_counter() - t0
            assert score in (0, 1, 2) and 0.0 <= conf <= 1.0
            # Warm calls
            t1 = time.perf_counter()
            for _ in range(20):
                predict_mod.predict(task, clues)
            warm_s = (time.perf_counter() - t1) / 20.0
            latencies_s[task] = cold_s
            max_single_s = max(max_single_s, cold_s, warm_s)
    finally:
        predict_mod.MODELS_DIR = old_dir
        predict_mod._load.cache_clear()

    return {
        "task_matrices": task_matrices,
        "fold_matrices": fold_matrices,
        "summary_csv": summary_csv,
        "created_files": created_files,
        "latencies_s": latencies_s,
        "max_single_s": max_single_s,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Per-task baseline and RF confusion matrices")
    ap.add_argument("--table", type=Path, help="optional clue table CSV")
    ap.add_argument("--zip", type=Path, default=DEFAULT_ZIP, help="path to B_skel.zip")
    ap.add_argument(
        "--matrices",
        action="store_true",
        help="also generate RF confusion matrices per task and per fold (CSV + PNG) and check predict < 1 s (#18)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/marty_eval/matrices"),
        help="output folder for CSV and PNG confusion matrices",
    )
    args = ap.parse_args(argv)

    if args.table:
        label_df = load_table(args.table)
    else:
        label_df = load_group_b_labels(args.zip, min_frames=0)

    tasks = sorted(int(t) for t in label_df["task"].unique())
    results = [evaluate_baseline(label_df, task) for task in tasks]
    print(format_baseline_report(results))

    if args.matrices:
        clue_df = load_table(args.table) if args.table else build_group_b_clue_table(args.zip)
        info = generate_rf_confusion_matrices(clue_df, args.out_dir)
        print(f"\nGenerated {len(info['created_files'])} CSV/PNG matrix files in {args.out_dir}:")
        for task in sorted(info["task_matrices"]):
            cm = info["task_matrices"][task]
            print(f"  Task {task} overall confusion matrix (rows=true 0/1/2, cols=pred 0/1/2):")
            for r_i, row in enumerate(cm):
                print(f"    true {r_i}: {row.tolist()}")
        print("\nPredict latency check (must be < 1.0 s):")
        for task, lat in info["latencies_s"].items():
            print(f"  Task {task} cold load + predict: {lat * 1000:.2f} ms (< 1000 ms: PASS)")
        if info["max_single_s"] >= 1.0:
            return 1

    return 0 if all(r.leak_free for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
