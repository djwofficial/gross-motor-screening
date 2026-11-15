"""Behaviour tests for model/train_rf.py (owner Daryn). They use the fake clue table only."""

import joblib
import numpy as np
import pytest

from features.clues import CLUE_NAMES
from model.train_rf import (
    check_table,
    cross_validate,
    features_for,
    fit_and_save,
    load_table,
    main,
    make_fake_table,
)


def test_duration_is_never_a_feature():
    for task in (1, 2, 3):
        assert "duration_s" not in features_for(task)
        assert features_for(task) == [n for n in CLUE_NAMES[task] if n != "duration_s"]


@pytest.mark.parametrize("task", [1, 2, 3])
def test_cross_validation_on_the_fake_table(task):
    res = cross_validate(make_fake_table(), task)
    assert res.leak_free and len(res.f1) == len(res.base_f1) == 5
    assert all(0.0 <= v <= 1.0 for v in res.f1 + res.base_f1)
    assert all(-1.0 <= v <= 1.0 for v in res.kappa + res.base_kappa)
    # the fake clues follow the label, so a working forest must beat the most common score
    assert res.beats_baseline


def test_saved_file_has_the_format_predict_loads(tmp_path):
    path = fit_and_save(make_fake_table(), 1, models_dir=tmp_path)
    assert path.name == "task1_rf.joblib"
    bundle = joblib.load(path)
    assert set(bundle) == {"model", "features"} and bundle["features"] == features_for(1)
    proba = bundle["model"].predict_proba(np.zeros((1, len(bundle["features"]))))[0]
    assert list(bundle["model"].classes_) == [0, 1, 2]
    assert abs(proba.sum() - 1.0) < 1e-9


def test_csv_with_label_words_loads(tmp_path):
    table = make_fake_table(n_children=10)
    table["label"] = table["label"].map({0: "Bad", 1: "Good", 2: "Perfect"})
    path = tmp_path / "clues.csv"
    table.to_csv(path, index=False)
    back = load_table(path)
    assert back["label"].isin([0, 1, 2]).all()
    assert back["child"].str.startswith("F").all()


def test_label_index_column_is_accepted(tmp_path):
    table = make_fake_table(n_children=10).rename(columns={"label": "label_index"})
    path = tmp_path / "clues.csv"
    table.to_csv(path, index=False)
    assert load_table(path)["label"].isin([0, 1, 2]).all()


def test_missing_clue_column_is_a_clear_error():
    table = make_fake_table(n_children=10).drop(columns=["one_foot_s"])
    with pytest.raises(ValueError, match="one_foot_s"):
        check_table(table)


def test_empty_clue_value_is_a_clear_error():
    table = make_fake_table(n_children=10)
    first = table.index[table["task"] == 2][0]
    table.loc[first, "hops"] = np.nan
    with pytest.raises(ValueError, match="task 2"):
        check_table(table)


def test_cli_runs_on_the_fake_table_and_refuses_to_save_it(capsys):
    assert main(["--fake", "--task", "1"]) == 0
    out = capsys.readouterr().out
    assert "FAKE TABLE" in out and "child leak check: PASS" in out and "kappa" in out
    assert main(["--fake", "--save"]) == 2
