"""Smoke test (owner Daryn): the whole app on fake tries, then rescore must match."""

import csv

import pytest

from app.main import main
from app.session import load_record
from rescore import rescore


@pytest.mark.parametrize("mode", ["auto", "manual"])
def test_app_fake_end_to_end(tmp_path, mode):
    for task in (1, 2, 3):
        assert main(["--fake", "--mode", mode, "--task", str(task), "--out", str(tmp_path)]) == 0
    rows = list(csv.DictReader((tmp_path / "log.csv").open()))
    assert [int(r["task"]) for r in rows] == [1, 2, 3]
    assert len(list(tmp_path.glob("K00_task*_try1_L.json"))) == 3
    assert all(load_record(p)["events"] for p in tmp_path.glob("K00_task*_try1_L.json"))
    n, same = rescore(tmp_path)
    assert n == 3 and same == 3
