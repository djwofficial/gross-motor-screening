"""Halpe-26 (RTMPose / rtmlib BodyWithFeet) -> BODY_25 (dataset order). Owner: Revallo.

Drop Halpe point 17 (head) and re-order. No point is made up.

HALPE26_NAMES is copied from rtmlib 0.0.16, rtmlib/visualization/skeleton/halpe26.py (checked
2026-11-08 by printing it from the installed package). tests/contracts/test_convert.py checks the
mapping name by name against common/body25.py.

Checked all names/indices against official MMPose halpe26.py on 2026-11-12:
https://github.com/open-mmlab/mmpose/blob/759b39c13fea6ba094afc1fa932f51dc1b11cbf9/configs/_base_/datasets/halpe26.py
The optional --plot CLI draws a synthetic, unmirrored calibration frame. It does not
replace the separate live RTMPose/webcam check of anatomical left and right.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HALPE26_NAMES = [
    "nose",  # 0
    "left_eye",  # 1
    "right_eye",  # 2
    "left_ear",  # 3
    "right_ear",  # 4
    "left_shoulder",  # 5
    "right_shoulder",  # 6
    "left_elbow",  # 7
    "right_elbow",  # 8
    "left_wrist",  # 9
    "right_wrist",  # 10
    "left_hip",  # 11
    "right_hip",  # 12
    "left_knee",  # 13
    "right_knee",  # 14
    "left_ankle",  # 15
    "right_ankle",  # 16
    "head",  # 17  (dropped)
    "neck",  # 18
    "hip",  # 19  (= BODY_25 mid_hip)
    "left_big_toe",  # 20
    "right_big_toe",  # 21
    "left_small_toe",  # 22
    "right_small_toe",  # 23
    "left_heel",  # 24
    "right_heel",  # 25
]

# BODY_25 index i takes Halpe-26 index HALPE_TO_BODY25[i].
HALPE_TO_BODY25 = [
    0,
    18,
    6,
    8,
    10,
    5,
    7,
    9,
    19,
    12,
    14,
    16,
    11,
    13,
    15,
    2,
    1,
    4,
    3,
    20,
    22,
    24,
    21,
    23,
    25,
]


def halpe_to_body25(keypoints: np.ndarray, scores: np.ndarray) -> np.ndarray:
    """keypoints (..., 26, 2) in pixels, scores (..., 26) -> float32 (..., 25, 3) as x, y, score.

    Works for one person (26, 2), several people (P, 26, 2) or a clip (T, 26, 2).
    Scores are clipped to 0..1 (RTMPose SimCC scores are not guaranteed to stay in that range).
    """
    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if keypoints.shape[-2:] != (26, 2) or scores.shape != keypoints.shape[:-1]:
        raise ValueError(
            f"expected keypoints (...,26,2) and scores (...,26), got {keypoints.shape} and {scores.shape}"
        )
    idx = np.array(HALPE_TO_BODY25)
    out = np.concatenate(
        [keypoints[..., idx, :], np.clip(scores[..., idx], 0.0, 1.0)[..., None]], axis=-1
    )
    return out.astype(np.float32)


def _synthetic_halpe_frame() -> tuple[np.ndarray, np.ndarray]:
    """Illustrative image coordinates, assigned by anatomy rather than the index map."""
    coordinates = {
        "nose": (320, 135),
        "left_eye": (330, 125),
        "right_eye": (310, 125),
        "left_ear": (345, 138),
        "right_ear": (295, 138),
        "left_shoulder": (375, 195),
        "right_shoulder": (265, 195),
        "left_elbow": (410, 265),
        "right_elbow": (235, 135),
        "left_wrist": (425, 335),
        "right_wrist": (250, 60),
        "left_hip": (355, 340),
        "right_hip": (285, 340),
        "left_knee": (365, 445),
        "right_knee": (275, 450),
        "left_ankle": (375, 555),
        "right_ankle": (270, 550),
        "head": (320, 100),
        "neck": (320, 185),
        "hip": (320, 340),
        "left_big_toe": (390, 575),
        "right_big_toe": (250, 575),
        "left_small_toe": (405, 565),
        "right_small_toe": (235, 565),
        "left_heel": (365, 568),
        "right_heel": (280, 568),
    }
    return (
        np.array([coordinates[name] for name in HALPE26_NAMES], dtype=np.float32),
        np.ones(26, dtype=np.float32),
    )


def _plot_mapping(out_path: str | Path) -> Path:
    """Plot synthetic anatomy; this does not verify actual RTMPose inference."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from common import body25 as b

    def anatomical_name(name):
        if name == "mid_hip":
            return "hip"
        if name.startswith("l_"):
            return f"left_{name[2:]}"
        if name.startswith("r_"):
            return f"right_{name[2:]}"
        return name

    def color(name):
        if name.startswith(("right_", "r_")):
            return "#2563eb"
        if name.startswith(("left_", "l_")):
            return "#c2410c"
        return "#475569"

    def draw(ax, xy, names, bones, title):
        for first, second in bones:
            ax.plot(*xy[[first, second]].T, color="#94a3b8", linewidth=2, zorder=1)
        for i, name in enumerate(names):
            ax.scatter(*xy[i], s=35, color=color(name), zorder=3)
        # Put dense face/foot labels in spaced columns with leader lines.
        for on_right in (False, True):
            indices = [
                i
                for i, name in enumerate(names)
                if (
                    anatomical_name(name).startswith("left_")
                    or name in ("nose", "neck", "hip", "mid_hip")
                )
                == on_right
            ]
            indices.sort(key=lambda i: (float(xy[i, 1]), i))
            positions = []
            for i in indices:
                y = float(xy[i, 1])
                positions.append(max(y, positions[-1] + 22) if positions else y)
            overflow = max(0.0, positions[-1] - 625)
            for i, y in zip(indices, positions, strict=True):
                name = names[i]
                wrist = anatomical_name(name) == "right_wrist"
                label = f"{i:02d} {name}" + (" (raised)" if wrist else "")
                ax.annotate(
                    label,
                    xy=xy[i],
                    xytext=(465 if on_right else 190, y - overflow),
                    ha="left" if on_right else "right",
                    va="center",
                    fontsize=8.5,
                    color=color(name),
                    fontweight="bold" if wrist else "normal",
                    arrowprops={"arrowstyle": "-", "color": color(name), "alpha": 0.4},
                    zorder=4,
                )
                if wrist:
                    ax.scatter(*xy[i], s=180, facecolors="none", edgecolors="#111827", zorder=5)
        ax.set(xlim=(0, 660), ylim=(650, 15), aspect="equal", title=title)
        ax.set_axis_off()

    keypoints, scores = _synthetic_halpe_frame()
    converted = halpe_to_body25(keypoints, scores)
    halpe_indices = {name: i for i, name in enumerate(HALPE26_NAMES)}
    halpe_bones = [
        (halpe_indices[anatomical_name(b.NAMES[a])], halpe_indices[anatomical_name(b.NAMES[z])])
        for a, z in b.BONES
    ]
    halpe_bones.append((halpe_indices["head"], halpe_indices["neck"]))

    figure = Figure(figsize=(18, 10), facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    figure.subplots_adjust(left=0.03, right=0.97, top=0.82, bottom=0.12, wspace=0.06)
    draw(axes[0], keypoints, HALPE26_NAMES, halpe_bones, "Halpe-26 input: index + name")
    draw(axes[1], converted[:, :2], b.NAMES, b.BONES, "BODY_25 output: index + name")
    figure.suptitle(
        "SYNTHETIC CALIBRATION | unmirrored, front-facing", fontsize=17, fontweight="bold"
    )
    figure.text(
        0.5,
        0.92,
        "Anatomical RIGHT: blue, viewer's left  |  Anatomical LEFT: orange, viewer's right",
        ha="center",
        fontsize=12,
    )
    figure.text(
        0.5,
        0.87,
        "Raised right wrist: Halpe 10 -> BODY_25 4.  Head 17 is omitted.",
        ha="center",
        fontsize=12,
    )
    figure.text(
        0.5,
        0.065,
        "Joint-order source: MMPose configs/_base_/datasets/halpe26.py @ 759b39c (see task card)",
        ha="center",
        fontsize=10,
    )
    figure.text(
        0.5,
        0.035,
        "Synthetic illustration only: no webcam, child data, or RTMPose inference.",
        ha="center",
        fontsize=11,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=150)
    return out_path


def _main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plot", type=Path, metavar="PATH", help="save a labelled synthetic calibration PNG"
    )
    args = parser.parse_args(argv)
    if args.plot is None:
        parser.print_help()
        return 0
    print(f"Saved synthetic, unmirrored mapping calibration: {_plot_mapping(args.plot)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
