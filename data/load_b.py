"""Group B dataset -> shared format. Owner: Marty (Nov 11).

Dataset: Kim et al. 2023, GigaScience 12, giad039; GigaDB doi 10.5524/102382 (CC0).
Put B_skel.zip in data/raw/ (never commit it). We read the zip directly, no unzipping.

File name: B/{child}_GMS_{task}_{attempt}_{camera}.json, e.g. B010_GMS_1_1_1.json
Inside:    {"data": [{"frame_index": i, "skeleton": [{"pose": [x0, y0, ... x24, y24],
            "score": [c0 .. c24]}, ...]}], "label": "Bad|Good|Perfect", "label_index": 0|1|2}
Points:    BODY_25, pixels of a 1920x1080 frame; a missing point is (0, 0) with score 0.
Frame rate: 30 fps (paper).

Real here (used by scripts/summarize_group_B.py): parse_name, list_files, read_file.
TODO Marty: load_try = pick the child (common.pick.pick_child), drop clips under 30 frames,
don't cut at 300 frames, and return a common.format.Try with t = frame / 30.
Normalisation (centre on the neck, scale by torso length) is common.geometry.normalize;
call it from the clues, don't write a second version.
"""

from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np

from common.format import Try

DEFAULT_ZIP = Path(__file__).resolve().parent / "raw" / "B_skel.zip"
FPS = 30.0
_NAME = re.compile(r"(?P<child>[A-C]\d{3})_GMS_(?P<task>\d)_(?P<attempt>\d+)_(?P<camera>\d)\.json$")


def parse_name(name: str) -> dict:
    """'B/B010_GMS_1_2_3.json' -> {'child': 'B010', 'task': 1, 'attempt': 2, 'camera': 3}."""
    m = _NAME.search(name)
    if not m:
        raise ValueError(f"not a dataset file name: {name}")
    d = m.groupdict()
    return {
        "child": d["child"],
        "task": int(d["task"]),
        "attempt": int(d["attempt"]),
        "camera": int(d["camera"]),
    }


def list_files(zip_path: str | Path = DEFAULT_ZIP) -> list[str]:
    with zipfile.ZipFile(zip_path) as z:
        return sorted(n for n in z.namelist() if n.endswith(".json"))


def read_file(z: zipfile.ZipFile, name: str) -> tuple[list[np.ndarray], int, str]:
    """-> (frames, label_index, label). frames[i] is float32 (P_i, 25, 3), every person in frame i."""
    d = json.loads(z.read(name))
    frames = []
    for fr in d["data"]:
        people = []
        for sk in fr.get("skeleton", []):
            xy = np.asarray(sk["pose"], dtype=np.float32).reshape(25, 2)
            sc = np.asarray(sk["score"], dtype=np.float32).reshape(25, 1)
            people.append(np.concatenate([xy, sc], axis=1))
        frames.append(np.stack(people) if people else np.zeros((0, 25, 3), dtype=np.float32))
    return frames, int(d["label_index"]), str(d["label"])


def load_try(z: zipfile.ZipFile, name: str) -> Try:
    raise NotImplementedError("Marty, Nov 11: data/load_b.py load_try")
