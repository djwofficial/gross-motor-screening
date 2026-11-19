"""Evaluation report. Owner: Louise (#25, Nov 25).

Per task, on the clue table, with GroupKFold by child (the same folds and forest as
model/train_rf.py, so the numbers match Daryn's gate):
- macro-F1 and quadratic weighted kappa per fold, random forest vs always the most common score
- the confusion matrix of the out-of-fold predictions (every try is predicted by a forest that
  never saw that child)
- the most common mix-ups, with the clues of those tries next to the scoring rule (paper Table 3)
- which clues the forests lean on (mean importance over the folds)
The app vs KEE and KEE vs KEE tables are Theresia's eval/agreement.py.

Run:  python -m eval.report --table runs/clue_table.csv [--out runs/report.md]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GroupKFold

from model.train_rf import (
    N_SPLITS,
    SEED,
    _kappa,
    _macro_f1,
    check_table,
    features_for,
    load_table,
    make_forest,
)

LABELS = ["Bad", "Good", "Perfect"]
TASKS = {1: "stand on one leg", 2: "hop on one leg", 3: "standing long jump"}
# Scoring criteria, group B tasks 1-3 (Kim et al. 2023, Table 3; team doc Info > 3)
RULES = {
    1: [
        "Cannot stand on one foot even for a moment.",
        "Can stand on one foot for more than 3 s but staggers.",
        "Can stand on one foot for more than 3 s without staggering.",
    ],
    2: [
        "Cannot hop even once.",
        "Can hop 2-3 steps but pauses a little.",
        "Can hop 2-3 steps without difficulty.",
    ],
    3: [
        "Cannot jump with feet together.",
        "Can jump with feet together.",
        "Can jump a long distance with feet together.",
    ],
}
TOP_MIXUPS = 2


def evaluate(table: pd.DataFrame, task: int, n_splits: int = N_SPLITS, seed: int = SEED) -> dict:
    """Out-of-fold predictions and per-fold scores for one task."""
    check_table(table, task)
    rows = table[table["task"] == task].reset_index(drop=True)
    feats = features_for(task)
    x = rows[feats].to_numpy(dtype=np.float64)
    y = rows["label"].to_numpy(dtype=int)
    groups = rows["child"].astype(str).to_numpy()

    pred = np.full(len(y), -1)
    folds, importances, leak_free = [], [], True
    for k, (train, test) in enumerate(GroupKFold(n_splits=n_splits).split(x, y, groups), 1):
        leak_free &= not (set(groups[train]) & set(groups[test]))
        forest = make_forest(seed).fit(x[train], y[train])
        dummy = DummyClassifier(strategy="most_frequent").fit(x[train], y[train])
        p, d = forest.predict(x[test]), dummy.predict(x[test])
        pred[test] = p
        importances.append(forest.feature_importances_)
        folds.append(
            {
                "fold": k,
                "children": len(set(groups[test])),
                "tries": len(test),
                "forest_f1": _macro_f1(y[test], p),
                "forest_kappa": _kappa(y[test], p),
                "base_f1": _macro_f1(y[test], d),
                "base_kappa": _kappa(y[test], d),
                "base_guess": LABELS[int(d[0])],
            }
        )
    cm = np.zeros((3, 3), dtype=int)
    for t_, p_ in zip(y, pred, strict=True):
        cm[t_, p_] += 1
    return {
        "task": task,
        "rows": rows,
        "features": feats,
        "y": y,
        "pred": pred,
        "folds": pd.DataFrame(folds),
        "cm": cm,
        "importance": pd.Series(np.mean(importances, axis=0), index=feats).sort_values(
            ascending=False
        ),
        "children": len(set(groups)),
        "leak_free": leak_free,
    }


def _fmt(v: float) -> str:
    return f"{v:.2f}"


def _md_table(header: list[str], body: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in body]
    return out


def task_section(res: dict) -> list[str]:
    task, f = res["task"], res["folds"]
    y, pred, cm, rows, feats = res["y"], res["pred"], res["cm"], res["rows"], res["features"]
    lines = [
        f"## Task {task}: {TASKS[task]}",
        "",
        f"{len(y)} tries, {res['children']} children, GroupKFold({len(f)}) by child; "
        f"child leak check: {'PASS' if res['leak_free'] else 'FAIL'}.",
        "",
        "### Scores per fold",
        "",
    ]
    body = [
        [
            str(r.fold),
            str(r.children),
            str(r.tries),
            _fmt(r.forest_f1),
            _fmt(r.forest_kappa),
            _fmt(r.base_f1),
            _fmt(r.base_kappa),
            r.base_guess,
        ]
        for r in f.itertuples()
    ]
    body.append(
        [
            "**mean ± sd**",
            "",
            "",
            f"**{f.forest_f1.mean():.2f} ± {f.forest_f1.std(ddof=0):.2f}**",
            f"**{f.forest_kappa.mean():.2f} ± {f.forest_kappa.std(ddof=0):.2f}**",
            f"{f.base_f1.mean():.2f} ± {f.base_f1.std(ddof=0):.2f}",
            f"{f.base_kappa.mean():.2f} ± {f.base_kappa.std(ddof=0):.2f}",
            "",
        ]
    )
    lines += _md_table(
        [
            "Fold",
            "Children",
            "Tries",
            "Forest macro-F1",
            "Forest kappa",
            "Baseline macro-F1",
            "Baseline kappa",
            "Baseline guess",
        ],
        body,
    )
    beats = f.forest_f1.mean() > f.base_f1.mean() and f.forest_kappa.mean() > f.base_kappa.mean()
    lines += ["", f"Beats the baseline (both numbers): {'yes' if beats else 'no'}.", ""]

    # tries where the clue code measured nothing (every clue except duration is 0)
    measured = [c for c in feats if c != "duration_s"]
    empty = (rows[measured] == 0).all(axis=1).to_numpy()
    per_label = ", ".join(
        f"{LABELS[i]} {empty[y == i].mean():.0%}" for i in range(3) if (y == i).any()
    )
    lines += [
        f"Tries where nothing was measured (every clue 0): {int(empty.sum())} of {len(y)} "
        f"({empty.mean():.0%}); per expert score: {per_label}.",
        "",
    ]

    lines += [
        "### Confusion matrix (out-of-fold predictions)",
        "",
        "Rows: expert score. Columns: forest prediction. In brackets: share of the row.",
        "",
    ]
    body = []
    for i in range(3):
        n = cm[i].sum()
        cells = [f"{cm[i, j]} ({cm[i, j] / n:.0%})" if n else "0" for j in range(3)]
        body.append([f"**{LABELS[i]}** (n={n})", *cells])
    lines += _md_table(["Expert \\ forest", *LABELS], body)
    recall = [cm[i, i] / cm[i].sum() if cm[i].sum() else 0.0 for i in range(3)]
    lines += [
        "",
        "Correct per expert score: "
        + ", ".join(f"{LABELS[i]} {recall[i]:.0%}" for i in range(3))
        + ".",
        "",
    ]

    lines += ["### Most common mix-ups, next to the scoring rule", ""]
    off = sorted(
        ((cm[i, j], i, j) for i in range(3) for j in range(3) if i != j and cm[i, j]),
        reverse=True,
    )[:TOP_MIXUPS]
    shown = [c for c in feats if c != "duration_s"]
    for n, i, j in off:
        mixed = rows[(y == i) & (pred == j)][shown].median()
        right_i = rows[(y == i) & (pred == i)][shown].median()
        right_j = rows[(y == j) & (pred == j)][shown].median()
        lines += [
            f"**{LABELS[i]} predicted as {LABELS[j]}: {n} tries.**",
            f"Rule for {LABELS[i]}: {RULES[task][i]} Rule for {LABELS[j]}: {RULES[task][j]}",
            "",
        ]
        lines += _md_table(
            [
                "Clue (median)",
                f"{LABELS[i]} predicted as {LABELS[j]}",
                f"{LABELS[i]}, predicted right",
                f"{LABELS[j]}, predicted right",
            ],
            [[c, _fmt(mixed[c]), _fmt(right_i[c]), _fmt(right_j[c])] for c in shown],
        )
        lines.append("")

    lines += ["### Clues the forests lean on (mean importance over folds)", ""]
    lines += _md_table(["Clue", "Importance"], [[c, _fmt(v)] for c, v in res["importance"].items()])
    lines.append("")
    return lines


def build_report(table: pd.DataFrame, tasks: list[int]) -> str:
    results = [evaluate(table, t) for t in tasks]
    lines = [
        "# Evaluation report: random forest per task (group B)",
        "",
        "Screening aid, not a diagnosis. Every number below comes from out-of-fold predictions: "
        "each try is scored by a forest trained without that child.",
        "",
        "## Summary",
        "",
    ]
    body = []
    for r in results:
        f = r["folds"]
        body.append(
            [
                f"{r['task']} {TASKS[r['task']]}",
                str(len(r["y"])),
                f"{f.forest_f1.mean():.2f} ± {f.forest_f1.std(ddof=0):.2f}",
                f"{f.forest_kappa.mean():.2f} ± {f.forest_kappa.std(ddof=0):.2f}",
                f"{f.base_f1.mean():.2f}",
                f"{f.base_kappa.mean():.2f}",
            ]
        )
    lines += _md_table(
        [
            "Task",
            "Tries",
            "Forest macro-F1",
            "Forest kappa",
            "Baseline macro-F1",
            "Baseline kappa",
        ],
        body,
    )
    lines.append("")
    for r in results:
        lines += task_section(r)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--table", required=True, help="clue table CSV (scripts/make_clue_table.py)")
    ap.add_argument("--tasks", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--out", default="runs/report.md")
    a = ap.parse_args(argv)
    text = build_report(load_table(a.table), a.tasks)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
