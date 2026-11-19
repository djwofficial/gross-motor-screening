"""Behaviour tests for eval/report.py (Louise, #25). Fake clue table only."""

from eval.report import build_report, evaluate
from model.train_rf import make_fake_table


def test_every_try_is_predicted_once_by_a_forest_without_its_child():
    table = make_fake_table()
    for task in (1, 2, 3):
        res = evaluate(table, task)
        assert res["leak_free"]
        assert (res["pred"] >= 0).all()
        assert res["cm"].sum() == len(res["y"]) == int((table["task"] == task).sum())
        assert len(res["folds"]) == 5


def test_report_has_every_section():
    text = build_report(make_fake_table(), [1, 2, 3])
    for part in ("## Summary", "Scores per fold", "Confusion matrix", "mix-ups", "lean on"):
        assert part in text
    for task in (1, 2, 3):
        assert f"## Task {task}" in text
    assert "not a diagnosis" in text
