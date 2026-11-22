"""Start area: the operator marks where the child stands before a try. Owner: Theresia (Nov 17).

Once per room (or after the laptop moves), the operator drags a box around the taped start area
on the camera picture:

    python -m live.start_area --camera 0            # mark and save (Enter = save, Esc = cancel)
    python -m live.start_area --camera 0 --show     # show the saved area on the live picture
    python -m live.start_area --image frame.png     # mark on a saved picture instead

The picture is shown MIRRORED, as in the GUI, so left/right feel natural to the operator. The box
is flipped back and saved in UNMIRRORED pixels, the same coordinates the pose points use
(AGENTS.md rule 6). Saved to runs/start_area.json (gitignored) together with the picture size.

Used by:
    area = load(frame_size=(1280, 720))         # (x0, y0, x1, y1) in pixels, or None
    live.quality.check(points, fps, frame_size, start_area=area)
    live.segment.Segmenter(start_area=area)     # "ready" only while the feet are in the area

The area is in picture pixels on purpose: it is a place in the picture, not a body measure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from common import body25 as b

DEFAULT_PATH = Path("runs") / "start_area.json"
FEET = b.LEFT_FOOT + b.RIGHT_FOOT  # ankles, toes, heels
MIN_SCORE = 0.3  # a foot point with a lower score counts as not found
MIN_IN_AREA = 0.8  # at least 80% of the found foot points must be inside the area
WINDOW = "Start area: drag a box around the tape, Enter = save, Esc = cancel"

Area = tuple[float, float, float, float]


def tidy(area) -> Area:
    """(x0, y0, x1, y1) with x0 <= x1 and y0 <= y1, as floats."""
    x0, y0, x1, y1 = (float(v) for v in area)
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def unmirror(area, width: float) -> Area:
    """A box marked on the mirrored picture -> the same place in the unmirrored picture."""
    x0, y0, x1, y1 = tidy(area)
    return tidy((width - x1, y0, width - x0, y1))


def rescale(area, from_size, to_size) -> Area:
    """Move a box from one picture size to another (e.g. marked at 1920x1080, used at 1280x720)."""
    sx = float(to_size[0]) / float(from_size[0])
    sy = float(to_size[1]) / float(from_size[1])
    x0, y0, x1, y1 = tidy(area)
    return x0 * sx, y0 * sy, x1 * sx, y1 * sy


def save(area, frame_size, path=DEFAULT_PATH) -> Path:
    """Write the area (unmirrored pixels) and the picture size it belongs to as JSON."""
    x0, y0, x1, y1 = tidy(area)
    w, h = int(frame_size[0]), int(frame_size[1])
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError(f"start area {area} is empty or outside the {w}x{h} picture")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"area": [x0, y0, x1, y1], "frame_size": [w, h], "units": "pixels, unmirrored"}
    path.write_text(json.dumps(data, indent=1))
    return path


def load(path=DEFAULT_PATH, frame_size=None) -> Area | None:
    """The saved area, rescaled to frame_size if given; None if nothing was saved yet."""
    path = Path(path)
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    area = tidy(data["area"])
    if frame_size is not None and tuple(data["frame_size"]) != tuple(frame_size):
        area = rescale(area, data["frame_size"], frame_size)
    return area


def feet_inside(points, area, min_score: float = MIN_SCORE) -> bool | None:
    """Are the feet in the area? points (25, 3) or (T, 25, 3) in pixels.

    True when at least MIN_IN_AREA of the found foot points (ankles, toes, heels, all frames) are
    inside; False when fewer are; None when no foot point was found (then nobody can tell).
    """
    p = np.asarray(points, dtype=np.float32)
    p = p[None] if p.ndim == 2 else p
    x0, y0, x1, y1 = tidy(area)
    f = p[:, FEET]
    seen = f[..., 2] > min_score
    if not np.any(seen):
        return None
    inside = (f[..., 0] >= x0) & (f[..., 0] <= x1) & (f[..., 1] >= y0) & (f[..., 1] <= y1)
    return bool(inside[seen].mean() >= MIN_IN_AREA)


def draw(img, area, mirrored: bool = True, colour=(0, 220, 255), thick: int = 2):
    """Draw the area on img (BGR, changed in place). mirrored=True when img is shown mirrored."""
    import cv2

    if area is None:
        return img
    x0, y0, x1, y1 = unmirror(area, img.shape[1]) if mirrored else tidy(area)
    cv2.rectangle(img, (int(x0), int(y0)), (int(x1), int(y1)), colour, thick, cv2.LINE_AA)
    cv2.putText(
        img,
        "start",
        (int(x0) + 4, int(y0) - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        colour,
        2,
        cv2.LINE_AA,
    )
    return img


# ---------------------------------------------------------------- operator marking (needs a screen)


def mark(frame) -> Area | None:
    """Show the frame mirrored, let the operator drag a box; return it UNMIRRORED (or None)."""
    import cv2

    shown = cv2.flip(frame, 1)
    x, y, w, h = cv2.selectROI(WINDOW, shown, showCrosshair=False, fromCenter=False)
    cv2.destroyWindow(WINDOW)
    if w <= 0 or h <= 0:
        return None
    return unmirror((x, y, x + w, y + h), frame.shape[1])


def _grab(camera: int):
    """One frame from the webcam at the size the app uses (1280x720), after a short warm-up."""
    import cv2

    cap = cv2.VideoCapture(camera)
    try:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        frame = None
        for _ in range(15):  # the first frames are often dark while the camera adjusts
            ok, f = cap.read()
            if ok:
                frame = f
        if frame is None:
            raise RuntimeError(f"camera {camera} gave no picture")
        return frame
    finally:
        cap.release()


def _show(camera: int, path) -> None:
    import cv2

    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            area = load(path, (frame.shape[1], frame.shape[0]))
            img = draw(cv2.flip(frame, 1), area, mirrored=True)
            cv2.imshow("Start area (Esc = close)", img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Mark the start area on the camera picture")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--camera", type=int, default=0, help="webcam index (default 0)")
    src.add_argument("--image", type=Path, help="mark on this picture instead of the webcam")
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH, help=f"default {DEFAULT_PATH}")
    ap.add_argument("--show", action="store_true", help="show the saved area on the live picture")
    args = ap.parse_args(argv)

    if args.show:
        _show(args.camera, args.out)
        return 0
    import cv2

    frame = cv2.imread(str(args.image)) if args.image else _grab(args.camera)
    if frame is None:
        ap.error(f"could not read {args.image}")
    area = mark(frame)
    if area is None:
        print("cancelled, nothing saved")
        return 1
    path = save(area, (frame.shape[1], frame.shape[0]), args.out)
    x0, y0, x1, y1 = area
    print(f"saved {path}: x {x0:.0f}-{x1:.0f}, y {y0:.0f}-{y1:.0f} (unmirrored pixels)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
