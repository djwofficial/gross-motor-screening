"""Group B dataset -> shared format. Owner: Marty (Nov 11).

Dataset: Kim et al. 2023, GigaScience 12, giad039; GigaDB doi 10.5524/102382 (CC0).
Put B_skel.zip in data/raw/ (never commit it). We read the zip directly, no unzipping.

File name: B/{child}_GMS_{task}_{attempt}_{camera}.json, e.g. B010_GMS_1_1_1.json
Inside:    {"data": [{"frame_index": i, "skeleton": [{"pose": [x0, y0, ... x24, y24],
            "score": [c0 .. c24]}, ...]}], "label": "Bad|Good|Perfect", "label_index": 0|1|2}
Points:    BODY_25, pixels of a 1920x1080 frame; a missing point is (0, 0) with score 0.
Frame rate: 30 fps (paper).

Real here (used by scripts/summarize_group_B.py): parse_name, list_files, read_file, load_try.
load_try picks the child (common.pick.pick_child), drops clips under 30 frames, does not cut at
300 frames, and returns a common.format.Try with t = frame / 30.
Normalisation (centre on the neck, scale by torso length) is common.geometry.normalize;
call it from the clues, don't write a second version.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from common.format import Try, validate
from common.geometry import normalize
from common.pick import pick_child

DEFAULT_ZIP = Path(__file__).resolve().parent / "raw" / "B_skel.zip"
FPS = 30.0
MIN_FRAMES = 30
WIDTH = 1920
HEIGHT = 1080
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


def load_try(z: zipfile.ZipFile, name: str, min_frames: int = MIN_FRAMES) -> Try:
    """Read one group B JSON file from an open zip archive into the shared Try format."""
    info = parse_name(name)
    frames, label_index, label = read_file(z, name)
    if len(frames) < min_frames:
        raise ValueError(f"{name}: {len(frames)} frames is under the {min_frames}-frame minimum")
    points = pick_child(frames)
    points[..., 2] = np.clip(points[..., 2], 0.0, 1.0)
    t = np.arange(len(frames), dtype=np.float64) / FPS
    meta = {
        "source": "dataset",
        "file": Path(name).name,
        "width": WIDTH,
        "height": HEIGHT,
        "fps": FPS,
        "label": label,
        "label_index": label_index,
        **info,
    }
    return validate(Try(points=points.astype(np.float32), t=t, meta=meta))


def iter_tries(
    zip_path: str | Path = DEFAULT_ZIP,
    tasks: tuple[int, ...] = (1, 2, 3),
    min_frames: int = MIN_FRAMES,
) -> Iterator[tuple[str, Try]]:
    """Yield (name, Try) for every file in tasks with at least min_frames frames."""
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".json"))
        for name in names:
            info = parse_name(name)
            if info["task"] not in tasks:
                continue
            try:
                yield name, load_try(z, name, min_frames=min_frames)
            except ValueError:
                continue


def main(zip_path: Path = DEFAULT_ZIP) -> int:
    total_by_task: Counter[int] = Counter()
    loaded_by_task: Counter[int] = Counter()
    dropped_short: list[tuple[str, int]] = []
    over_300 = 0
    norm_ok = 0

    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if n.endswith(".json"))
        for name in names:
            info = parse_name(name)
            task = info["task"]
            if task not in (1, 2, 3):
                continue
            total_by_task[task] += 1
            try:
                tr = load_try(z, name)
            except ValueError:
                frames, _, _ = read_file(z, name)
                dropped_short.append((name, len(frames)))
                continue
            loaded_by_task[task] += 1
            if tr.n_frames > 300:
                over_300 += 1
            _ = normalize(tr.points)
            norm_ok += 1

    total = sum(total_by_task.values())
    loaded = sum(loaded_by_task.values())
    print(
        f"Tasks 1-3 files: {total} ({total_by_task[1]} / {total_by_task[2]} / {total_by_task[3]})"
    )
    print(
        f"Loaded (>= {MIN_FRAMES} frames): {loaded} "
        f"({loaded_by_task[1]} / {loaded_by_task[2]} / {loaded_by_task[3]})"
    )
    print(f"Dropped (< {MIN_FRAMES} frames): {len(dropped_short)} -> {dropped_short}")
    print(f"Kept full length (> 300 frames, not cut): {over_300}")
    print(f"Normalized (common.geometry.normalize) check: {norm_ok}/{loaded} OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
