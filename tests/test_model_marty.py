"""Behaviour tests for model/baseline.py and model/task_classifier.py (owner Marty)."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest

from common.format import make_fake_try
from model.baseline import evaluate_baseline, generate_rf_confusion_matrices
from model.task_classifier import (
    CLUE_COLUMNS,
    cross_validate_task_classifier,
    extract_task_clues,
    fit_and_save_task_model,
    predict_task,
    predict_task_from_clues,
    save_task_confusion_matrix,
)
from model.train_rf import make_fake_table

tc_mod = importlib.import_module("model.task_classifier")


@pytest.mark.parametrize("task", [1, 2, 3])
def test_baseline_group_kfold_leak_free_and_zero_kappa(task: int):
    table = make_fake_table(n_children=25)
    res = evaluate_baseline(table, task=task, n_splits=5)
    assert res.leak_free
    assert len(res.f1) == 5 and len(res.kappa) == 5
    assert all(k == 0.0 for k in res.kappa)
    assert all(0.0 <= f <= 1.0 for f in res.f1)


@pytest.mark.parametrize("leg", ["L", "R"])
@pytest.mark.parametrize("task", [1, 2, 3])
def test_predict_task_from_clues_distinguishes_all_three_fake_tasks(task: int, leg: str):
    tr = make_fake_try(task=task, leg=leg)
    guessed_task, guessed_leg, conf = predict_task_from_clues(tr.points, tr.t)
    assert guessed_task == task
    assert 0.0 < conf <= 1.0
    if task in (1, 2):
        assert guessed_leg == leg


def test_predict_task_with_saved_model_bundle(tmp_path: Path):
    table = make_fake_table(n_children=15)
    path = fit_and_save_task_model(table, models_dir=tmp_path, feature_cols=CLUE_COLUMNS)
    assert path.exists()
    old_dir = tc_mod.MODELS_DIR
    try:
        tc_mod.MODELS_DIR = tmp_path
        tc_mod._load.cache_clear()
        tr = make_fake_try(task=2, leg="R")
        gt, gl, conf = predict_task(tr.points, tr.t)
        assert gt in (1, 2, 3) and gl == "R" and 0.0 < conf <= 1.0
    finally:
        tc_mod.MODELS_DIR = old_dir
        tc_mod._load.cache_clear()


def test_extract_task_clues_returns_all_finite_values():
    from features.clues import CLUE_NAMES

    for task in (1, 2, 3):
        tr = make_fake_try(task=task, leg="L")
        clues = extract_task_clues(tr.points, tr.t)
        for col in (*CLUE_COLUMNS, *CLUE_NAMES[task]):
            assert col in clues
            assert np.isfinite(clues[col])


def test_task_classifier_group_kfold_and_confusion_matrix(tmp_path: Path):
    table = make_fake_table(n_children=20)
    res = cross_validate_task_classifier(table, feature_cols=CLUE_COLUMNS, n_splits=5)
    assert res.leak_free
    assert res.matrix.shape == (3, 3)
    assert int(res.matrix.sum()) == len(table)
    csv_path, png_path = save_task_confusion_matrix(res.matrix, tmp_path)
    assert csv_path.exists() and png_path.exists()


def test_generate_rf_confusion_matrices_and_predict_under_1s(tmp_path: Path):
    table = make_fake_table(n_children=15)
    info = generate_rf_confusion_matrices(table, out_dir=tmp_path, n_splits=5)
    assert info["summary_csv"].exists()
    for task in (1, 2, 3):
        assert (tmp_path / f"task{task}_cm.csv").exists()
        assert (tmp_path / f"task{task}_cm.png").exists()
        for fold in range(1, 6):
            assert (tmp_path / f"task{task}_fold{fold}_cm.csv").exists()
            assert (tmp_path / f"task{task}_fold{fold}_cm.png").exists()
        assert info["latencies_s"][task] < 1.0
    assert info["max_single_s"] < 1.0


def test_generate_rf_confusion_matrices_without_duration_column(tmp_path: Path):
    table = make_fake_table(n_children=15).drop(columns=["duration_s"])
    info = generate_rf_confusion_matrices(table, out_dir=tmp_path, n_splits=5)
    assert info["summary_csv"].exists()
    assert info["max_single_s"] < 1.0

    # Also verify a present-but-NaN non-feature column does not leak NaN into predict().
    table_nan = make_fake_table(n_children=15)
    table_nan["duration_s"] = np.nan
    info_nan = generate_rf_confusion_matrices(table_nan, out_dir=tmp_path / "nan", n_splits=5)
    assert info_nan["summary_csv"].exists()
    assert info_nan["max_single_s"] < 1.0
