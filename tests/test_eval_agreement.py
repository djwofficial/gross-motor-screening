"""Behaviour tests for eval/agreement.py (#54), FAKE data only (testing-day scores never go here)."""

import csv

import numpy as np
import pytest

from eval import agreement as ag


def _write(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)
    return path


@pytest.mark.parametrize(
    "value, expected",
    [("0", 0), ("2", 2), (" 1 ", 1), ("2.0", 2), ("", None), ("3", None), ("check", None)],
)
def test_score(value, expected):
    assert ag.score(value) == expected


def test_compare_counts_and_kappa():
    a = [0, 1, 2, 2, None, 1]
    b = [0, 1, 1, 2, 2, None]
    c = ag.compare(a, b)
    assert c["n"] == 4
    assert c["exact"] == pytest.approx(0.75)
    assert c["within_1"] == 1.0
    assert c["table"].sum() == 4 and c["table"][2, 1] == 1
    assert 0 < c["kappa"] < 1
    assert ag.compare([1, 2, 0], [1, 2, 0])["kappa"] == pytest.approx(1.0)


def test_kappa_not_defined_when_every_score_is_the_same():
    c = ag.compare([1, 1, 1], [1, 1, 1])
    assert c["exact"] == 1.0 and c["kappa"] is None
    assert ag.compare([], [])["n"] == 0


def test_join_skips_check_by_teacher_and_lists_unmatched(tmp_path):
    log = _write(
        tmp_path / "log.csv",
        ["child", "task", "try", "score", "check_by_teacher"],
        [
            {"child": "F01", "task": 1, "try": 1, "score": 2, "check_by_teacher": 0},
            {"child": "F01", "task": 1, "try": 2, "score": "", "check_by_teacher": 1},
            {"child": "F01", "task": 2, "try": 1, "score": 0, "check_by_teacher": 0},
            {"child": "F02", "task": 3, "try": 1, "score": 1, "check_by_teacher": 0},
        ],
    )
    kee = _write(
        tmp_path / "kee.csv",
        ag.KEE_FIELDS,
        [
            {"code": "F01", "task": 1, "try": 1, "kee1": 2, "kee2": 1},
            {"code": "F01", "task": 1, "try": 2, "kee1": 1, "kee2": 1},
            {"code": "F01", "task": 2, "try": 1, "kee1": 0, "kee2": 0},
            {"code": "F09", "task": 1, "try": 1, "kee1": 0, "kee2": 0},
        ],
    )
    j = ag.join(log, kee)
    assert len(j["rows"]) == 3
    assert j["app_only"] == [("F02", 3, 1)] and j["kee_only"] == [("F09", 1, 1)]
    checked = [r for r in j["rows"] if r["check"]]
    assert len(checked) == 1 and checked[0]["app"] is None
    t = ag.by_task(j["rows"], "app", "kee1")
    assert t["all"]["n"] == 2 and t[1]["n"] == 1 and t[3]["n"] == 0
    text = ag.testing_day_report(j)
    assert "1 were 'check by teacher'" in text
    assert "1 tries only in the app log, 1 only on the KEE sheet" in text


def test_tables_come_out_of_fake_files(tmp_path, capsys):
    """Done when (#54): tables come out of a fake score file."""
    assert ag.main(["--fake", str(tmp_path), "-o", str(tmp_path / "tables.md")]) == 0
    out = capsys.readouterr().out
    assert out.startswith("**FAKE DATA")
    for title in (
        "App vs KEE scorer 1",
        "App vs KEE scorer 2",
        "KEE scorer 1 vs KEE scorer 2",
        "KEE 1 vs answer key",
        "KEE 2 vs answer key",
    ):
        assert title in out
    assert (tmp_path / "tables.md").read_text(encoding="utf-8").count("| all |") == 6


def test_fake_kee_agrees_more_with_itself_than_with_the_app(tmp_path):
    p = ag.make_fake_files(tmp_path, n_children=40, seed=1)
    rows = ag.join(p["log"], p["kee"])["rows"]
    app_kee = ag.by_task(rows, "app", "kee1")["all"]["kappa"]
    kee_kee = ag.by_task(rows, "kee1", "kee2")["all"]["kappa"]
    assert kee_kee > app_kee > 0


def test_practice_vs_answer_key(tmp_path):
    key = _write(
        tmp_path / "key.csv",
        ["clip", "task", "score"],
        [{"clip": f"c{i}", "task": 1 + i % 3, "score": i % 3} for i in range(6)],
    )
    prac = _write(
        tmp_path / "prac.csv",
        ["clip", "score"],
        [{"clip": f"c{i}", "score": i % 3} for i in range(6)] + [{"clip": "zz", "score": 1}],
    )
    p = ag.practice(prac, key)
    assert p["missing"] == ["zz"]
    res = p["tables"]["KEE"]["all"]
    assert res["n"] == 6 and res["exact"] == 1.0
    assert np.trace(res["table"]) == 6
    assert "Clips not in the answer key: zz" in ag.practice_report(p)


def test_cli_needs_pairs(tmp_path):
    with pytest.raises(SystemExit):
        ag.main(["--log", str(tmp_path / "log.csv")])
