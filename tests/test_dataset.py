"""Dataset helpers. The Milestone 1 check runs only where data/raw/B_skel.zip exists (not in CI)."""

import subprocess
import sys

import pytest

from data.load_b import DEFAULT_ZIP, parse_name


def test_parse_name():
    assert parse_name("B/B010_GMS_1_2_3.json") == {
        "child": "B010",
        "task": 1,
        "attempt": 2,
        "camera": 3,
    }
    with pytest.raises(ValueError):
        parse_name("B/readme.txt")


@pytest.mark.skipif(not DEFAULT_ZIP.exists(), reason="data/raw/B_skel.zip not present")
def test_milestone_1_first_line():
    out = subprocess.run(
        [sys.executable, "scripts/summarize_group_B.py"], capture_output=True, text=True, check=True
    )
    assert out.stdout.splitlines()[0] == "files: 4661 | children: 106 | attempts: 1555"
