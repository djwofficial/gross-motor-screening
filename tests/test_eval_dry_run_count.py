"""Behaviour tests for eval/dry_run_count.py (#53), on fake tries only (real run: Nov 25)."""

import csv

import pytest

from app.session import Session
from common.format import make_fake_try
from eval import dry_run_count as drc

CODES = ("T01", "T02", "T03")


def _fake_run(out, manual_start=(), stop=None):
    """27 tries like the dry run: 3 people x tasks 1-3 x 3 tries, legs L R L.

    manual_start: set of (code, task, try) the operator started with Space.
    stop: optional {(code, task, try): "auto"/"manual"} written as meta["stop"].
    """
    for code in CODES:
        s = Session(out, child=code, age_months=48, consent_level=3)
        for task in (1, 2, 3):
            for k, leg in enumerate(("L", "R", "L"), start=1):
                tr = make_fake_try(task=task, leg=leg)
                key = (code, task, k)
                tr.meta["mode"] = "manual" if key in manual_start else "auto"
                if stop and key in stop:
                    tr.meta["stop"] = stop[key]
                s.save_try(
                    tr,
                    task=task,
                    leg=leg,
                    guessed_task=task,
                    guessed_leg=leg,
                    task_confidence=0.9,
                    corrected=False,
                    score=1,
                    confidence=0.9,
                    clues={},
                    reasons="fake",
                )


def _notes(path, missed_end=(), start_mark=None):
    drc.write_template(path, CODES)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        key = (r["code"], int(r["task"]), int(r["try"]))
        r["auto_start"] = (start_mark or {}).get(key, "✓")
        r["auto_end"] = "M" if key in missed_end else "✓"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=drc.NOTES_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def test_template_matches_the_printed_sheet(tmp_path):
    p = drc.write_template(tmp_path / "notes.csv")
    with open(p, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 27
    assert [r["task"] for r in rows[:9]] == ["1"] * 3 + ["2"] * 3 + ["3"] * 3
    assert [r["leg"] for r in rows[:9]] == ["L", "R", "L"] * 2 + [""] * 3


def test_one_count_per_task(tmp_path):
    run = tmp_path / "run"
    _fake_run(run, manual_start={("T02", 2, 1), ("T03", 3, 3)})
    notes = _notes(
        tmp_path / "notes.csv",
        missed_end={("T01", 1, 2), ("T02", 2, 1)},
        start_mark={("T02", 2, 1): "M", ("T03", 3, 3): "M"},
    )
    res = drc.count(run, notes)
    c = res["tasks"]
    assert [c[k]["tries"] for k in (1, 2, 3)] == [9, 9, 9]
    assert [c[k]["auto_start"] for k in (1, 2, 3)] == [9, 8, 8]
    assert [c[k]["caught"] for k in (1, 2, 3)] == [8, 8, 8]
    assert c[1]["missed_end"] == ["T01_task1_try2_R.json"]
    assert c[2]["missed_start"] == ["T02_task2_try1_L.json"]
    assert not res["unknown_end"]
    assert not any(r.get("check") for r in res["rows"])
    text = drc.report(res)
    assert "all                     24/27" in text


def test_without_notes_the_end_is_unknown_and_not_caught(tmp_path):
    _fake_run(tmp_path)
    res = drc.count(tmp_path)
    assert all(c["caught"] == 0 and c["end_unknown"] == 9 for c in res["tasks"].values())
    assert "end not known for 27 tries" in drc.report(res)


def test_meta_stop_is_used_without_notes(tmp_path):
    stop = {(c, t, k): "auto" for c in CODES for t in (1, 2, 3) for k in (1, 2, 3)}
    stop[("T03", 1, 3)] = "manual"
    _fake_run(tmp_path, stop=stop)
    c = drc.count(tmp_path)["tasks"]
    assert [c[k]["caught"] for k in (1, 2, 3)] == [8, 9, 9]


def test_notes_and_json_disagree_is_flagged(tmp_path):
    run = tmp_path / "run"
    _fake_run(run)
    notes = _notes(tmp_path / "notes.csv", start_mark={("T01", 3, 1): "M"})
    res = drc.count(run, notes)
    flagged = [r["file"] for r in res["rows"] if r.get("check")]
    assert flagged == ["T01_task3_try1_L.json"]


@pytest.mark.parametrize(
    "value, expected",
    [("✓", True), ("y", True), ("1", True), ("✗", False), ("M", False), ("no", False), ("", None)],
)
def test_marks(value, expected):
    assert drc.mark(value) is expected


def test_cli(tmp_path, capsys):
    _fake_run(tmp_path / "run")
    notes = _notes(tmp_path / "notes.csv")
    assert drc.main([str(tmp_path / "run"), "--notes", str(notes)]) == 0
    assert "27/27" in capsys.readouterr().out
    assert drc.main([str(tmp_path / "empty")]) == 1


def test_task3_feet_in_picture(tmp_path):
    """Task 3 tries get the feet measure; feet pushed off the bottom are flagged."""
    import numpy as np

    from common import body25 as b

    tr = make_fake_try(task=3)
    size = (tr.meta.get("width") or 1280, tr.meta.get("height") or 720)
    ok = drc.feet_in_picture(tr.points, size)
    assert ok["ankle_missing"] == 0 and not ok["reached_edge"] and not ok["lost_at_bottom"]
    assert ok["start_room"] is not None

    p = tr.points.copy()
    h = size[1]
    half = len(p) // 2
    feet = [b.R_ANKLE, b.L_ANKLE, b.R_HEEL, b.L_HEEL, b.R_BIG_TOE, b.L_BIG_TOE]
    p[half - 1, feet, 1] = 0.99 * h  # feet go down to the bottom edge ...
    p[half:, feet] = 0  # ... and leave the picture
    out = drc.feet_in_picture(p, size)
    assert out["lost_at_bottom"] and out["reached_edge"]
    assert 0.4 < out["ankle_missing"] < 0.6
    assert np.isfinite(out["start_room"])

    _fake_run(tmp_path / "run")
    res = drc.count(tmp_path / "run")
    t3 = [r for r in res["rows"] if r["task"] == 3]
    assert len(t3) == 9 and all("feet" in r for r in t3)
    assert not any("feet" in r for r in res["rows"] if r["task"] != 3)
    assert "task 3: feet in the picture" in drc.report(res)
