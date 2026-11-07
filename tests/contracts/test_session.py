"""Contract: app/session.py (owner Freya). One JSON per try and one log row per try."""

import csv

import numpy as np

from app.session import LOG_FIELDS, Session, load_record
from common.format import make_fake_try


def _save(s, task=2, conf=0.9):
    tr = make_fake_try(task=task)
    return s.save_try(
        tr,
        task=task,
        leg="L",
        guessed_task=task,
        guessed_leg="L",
        task_confidence=0.8,
        corrected=False,
        score=1,
        confidence=conf,
        clues={"duration_s": tr.duration},
        reasons="test",
    )


def test_files_log_and_round_trip(tmp_path):
    s = Session(tmp_path, child="K03", age_months=47)
    p1 = _save(s)
    p2 = _save(s, conf=0.1)
    assert p1.name == "K03_task2_try1_L.json" and p2.name == "K03_task2_try2_L.json"
    rows = list(csv.DictReader((tmp_path / "log.csv").open()))
    assert list(rows[0]) == LOG_FIELDS and len(rows) == 2
    assert rows[0]["score"] == "1" and rows[1]["score"] == "" and rows[1]["check_by_teacher"] == "1"
    rec = load_record(p1)
    np.testing.assert_allclose(rec["try"].points, make_fake_try(task=2).points)


def test_numbering_continues_after_restart(tmp_path):
    _save(Session(tmp_path, child="K04", age_months=50))
    assert _save(Session(tmp_path, child="K04", age_months=50)).name == "K04_task2_try2_L.json"
