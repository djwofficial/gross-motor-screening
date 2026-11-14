"""Try -> stick-figure mp4 with point numbers, for checking things by eye. Owner: Freya.

    render(tr, out_path, fps=30, size=None, labels=True, events=None) -> Path

Draws common.body25.BONES on a black picture, one video frame per row of tr.points.
- Right side of the body is orange, left side is blue, the middle line is white, so left/right
  mix-ups show at once. Points with score 0 (not found) are not drawn and get no lines.
- labels=True writes the point number next to each point (0 nose, 1 neck, ... 24 right heel).
- The top line shows the time in seconds (from tr.t), the frame number and meta (source, task, leg).
- events (features.clues.timeline, or hand-made): a coloured bar under the figure with one colour
  per state, a white cursor at the current time and the current state written next to it.

Command line (from the repo root):

    python -m viz.replay runs/2026-11-14/K00_task2_try1_L.json      # a saved try (+ its events)
    python -m viz.replay examples/fake_try.npz -o fake.mp4           # a common.format .npz file
    python -m viz.replay B010_GMS_1_1_1.json                          # one dataset file
    python -m viz.replay --zip data/raw/B_skel.zip B/B010_GMS_1_1_1.json
    python -m viz.replay --fake 2 --events demo                      # fake try + hand-made events

Dataset files can hold 2+ people; they go through common.pick.pick_child first (t = frame / 30).
The picture is drawn as the points are (not mirrored): the child's right side is on the left of
the picture when the child faces the camera.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

from common import body25 as b
from common.format import Try, make_fake_try, validate

DATASET_FPS = 30.0  # paper; see data/load_b.py
MIN_SCORE = 0.05  # a point with a lower score counts as not found
BAR_H = 36  # pixels of the timeline bar
TOP_H = 28  # pixels of the text line at the top

# BGR colours
WHITE = (255, 255, 255)
GREY = (110, 110, 110)
RIGHT = (0, 140, 255)  # orange
LEFT = (255, 160, 40)  # blue
MIDDLE = (235, 235, 235)

_RIGHT_PTS = {b.R_SHOULDER, b.R_ELBOW, b.R_WRIST, b.R_HIP, b.R_KNEE, b.R_ANKLE, b.R_EYE, b.R_EAR}
_RIGHT_PTS |= set(b.RIGHT_FOOT)
_LEFT_PTS = {b.L_SHOULDER, b.L_ELBOW, b.L_WRIST, b.L_HIP, b.L_KNEE, b.L_ANKLE, b.L_EYE, b.L_EAR}
_LEFT_PTS |= set(b.LEFT_FOOT)

# One colour per timeline state (features.clues.STATES); unknown states get grey.
STATE_COLOURS = {
    "unknown": (80, 80, 80),
    # task 1
    "both_feet": (200, 200, 200),
    "steady": (80, 200, 80),
    "wobble": (70, 60, 235),  # red
    # task 2
    "hop": (230, 120, 60),
    "pause": (60, 220, 230),
    # task 3
    "before": (200, 200, 200),
    "air": (230, 120, 60),
    "after": (80, 200, 80),
}
_EXTRA = [(180, 80, 200), (60, 60, 220), (160, 160, 40), (120, 200, 160)]


def side_colour(i: int) -> tuple[int, int, int]:
    if i in _RIGHT_PTS:
        return RIGHT
    if i in _LEFT_PTS:
        return LEFT
    return MIDDLE


def state_colour(state: str, _cache: dict = {}) -> tuple[int, int, int]:  # noqa: B006
    if state in STATE_COLOURS:
        return STATE_COLOURS[state]
    if state not in _cache:
        _cache[state] = _EXTRA[len(_cache) % len(_EXTRA)]
    return _cache[state]


def fit_transform(points: np.ndarray, size: tuple[int, int], margin: float = 0.06):
    """Scale and offset that fit every found point of the whole try into size (w, h).

    Uses one transform for the whole try, so movement (a jump, sway) stays visible.
    Returns (scale, ox, oy): pixel_out = point * scale + (ox, oy).
    """
    w, h = size
    ok = points[..., 2] > MIN_SCORE
    if not np.any(ok):
        return 1.0, 0.0, 0.0
    xy = points[..., :2][ok]
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)
    bw, bh = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    scale = min(w * (1 - 2 * margin) / bw, h * (1 - 2 * margin) / bh)
    ox = (w - bw * scale) / 2 - x0 * scale
    oy = (h - bh * scale) / 2 - y0 * scale
    return float(scale), float(ox), float(oy)


def draw_figure(img, pts: np.ndarray, scale=1.0, ox=0.0, oy=0.0, labels=True, thick=2):
    """Draw one (25, 3) frame of points onto img (BGR, changed in place)."""
    import cv2

    ok = pts[:, 2] > MIN_SCORE
    xy = np.round(pts[:, :2] * scale + (ox, oy)).astype(int)
    for i, j in b.BONES:
        if ok[i] and ok[j]:
            colour = side_colour(j) if side_colour(j) != MIDDLE else side_colour(i)
            cv2.line(img, tuple(xy[i]), tuple(xy[j]), colour, thick, cv2.LINE_AA)
    for i in np.nonzero(ok)[0]:
        cv2.circle(img, tuple(xy[i]), thick + 2, side_colour(int(i)), -1, cv2.LINE_AA)
        if labels:
            cv2.putText(
                img,
                str(int(i)),
                (int(xy[i, 0]) + 5, int(xy[i, 1]) - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                WHITE,
                1,
                cv2.LINE_AA,
            )
    return img


def state_at(events, t: float) -> str | None:
    for e in events or []:
        if e["start_s"] <= t <= e["end_s"]:
            return e["state"]
    return None


def draw_bar(img, events, t0: float, t1: float, t_now: float, y: int, h: int = BAR_H):
    """Timeline bar from t0 to t1 across the width of img, at rows y .. y+h."""
    import cv2

    w = img.shape[1]
    span = max(t1 - t0, 1e-6)
    cv2.rectangle(img, (0, y), (w - 1, y + h - 1), (30, 30, 30), -1)
    for e in events or []:
        xa = int(round((e["start_s"] - t0) / span * (w - 1)))
        xb = int(round((e["end_s"] - t0) / span * (w - 1)))
        cv2.rectangle(
            img, (max(xa, 0), y + 4), (min(xb, w - 1), y + h - 14), state_colour(e["state"]), -1
        )
    x = int(round((t_now - t0) / span * (w - 1)))
    cv2.line(img, (x, y), (x, y + h - 12), WHITE, 2)
    st = state_at(events, t_now)
    cv2.putText(
        img,
        f"{t_now:5.2f} s  {st or '-'}",
        (6, y + h - 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        state_colour(st) if st else GREY,
        1,
        cv2.LINE_AA,
    )
    return img


def _default_size(tr: Try) -> tuple[int, int]:
    w = int(tr.meta.get("width") or 1280)
    h = int(tr.meta.get("height") or 720)
    s = min(1.0, 960 / w)  # keep files small: at most 960 px wide
    return int(round(w * s)) // 2 * 2, int(round(h * s)) // 2 * 2


def render(
    tr: Try,
    out_path,
    fps: float = 30.0,
    size=None,
    labels: bool = True,
    events=None,
) -> Path:
    """Write tr as a stick-figure mp4 and return its path.

    size: (width, height) of the figure area; default: the source picture scaled to <= 960 px wide.
    The figure is fitted to the area (one scale for the whole try). With events, a timeline bar of
    BAR_H pixels is added under the figure.
    """
    import cv2

    validate(tr)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = size if size is not None else _default_size(tr)
    w, h = int(w) // 2 * 2, int(h) // 2 * 2
    fig_h = h
    total_h = TOP_H + fig_h + (BAR_H if events is not None else 0)
    scale, ox, oy = fit_transform(tr.points, (w, fig_h))
    oy += TOP_H

    writer = cv2.VideoWriter(
        str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, total_h)
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not open a video writer for {out_path}")
    m = tr.meta
    info = " ".join(
        f"{k}={m[k]}" for k in ("source", "child", "task", "leg", "camera") if m.get(k) is not None
    )
    t0, t1 = float(tr.t[0]), float(tr.t[-1])
    try:
        for i in range(tr.n_frames):
            img = np.zeros((total_h, w, 3), dtype=np.uint8)
            draw_figure(img, tr.points[i], scale, ox, oy, labels=labels)
            cv2.putText(
                img,
                f"t {tr.t[i]:6.2f} s  frame {i + 1}/{tr.n_frames}  {info}",
                (6, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                WHITE,
                1,
                cv2.LINE_AA,
            )
            if events is not None:
                draw_bar(img, events, t0, t1, float(tr.t[i]), TOP_H + fig_h)
            writer.write(img)
    finally:
        writer.release()
    return out_path


# ---------------------------------------------------------------- loading files for the CLI


def dataset_json_to_try(d: dict, name: str = "") -> Try:
    """One dataset JSON file (already parsed) -> Try, child picked with common.pick.pick_child."""
    from common.pick import pick_child

    frames = []
    for fr in d["data"]:
        people = [
            np.concatenate(
                [
                    np.asarray(sk["pose"], dtype=np.float32).reshape(25, 2),
                    np.asarray(sk["score"], dtype=np.float32).reshape(25, 1),
                ],
                axis=1,
            )
            for sk in fr.get("skeleton", [])
        ]
        frames.append(np.stack(people) if people else np.zeros((0, 25, 3), np.float32))
    pts = pick_child(frames)
    pts[..., 2] = np.clip(pts[..., 2], 0.0, 1.0)
    meta = {"source": "dataset", "file": Path(name).name, "fps": DATASET_FPS}
    meta.update({"width": 1920, "height": 1080, "label": d.get("label")})
    try:
        from data.load_b import parse_name

        meta.update(parse_name(name))
    except ValueError:
        pass
    t = np.arange(len(frames), dtype=np.float64) / DATASET_FPS
    return validate(Try(points=pts.astype(np.float32), t=t, meta=meta))


def load_any(path: str, zip_path: str | None = None):
    """-> (Try, events or None) from a session JSON, a dataset JSON, a zip member or an .npz."""
    if zip_path:
        with zipfile.ZipFile(zip_path) as z:
            return dataset_json_to_try(json.loads(z.read(path)), path), None
    p = Path(path)
    if p.suffix == ".npz":
        from common.format import load

        return load(p), None
    d = json.loads(p.read_text())
    if "points25" in d:  # saved by app.session
        from app.session import load_record

        rec = load_record(p)
        return rec["try"], rec.get("events")
    if "data" in d:
        return dataset_json_to_try(d, p.name), None
    raise ValueError(f"{path}: not a session JSON, dataset JSON or .npz file")


def demo_events(tr: Try) -> list[dict]:
    """Hand-made events for a fake try (task from meta), to check the bar by eye."""
    task = int(tr.meta.get("task", 1))
    t1 = float(tr.t[-1])
    cuts = {1: [0.4, 1.8, 2.6], 2: [1.0, 1.4, 2.4], 3: [1.0, 1.45]}[task]
    names = {
        1: ["both_feet", "steady", "wobble", "steady"],
        2: ["hop", "pause", "hop", "both_feet"],
        3: ["before", "air", "after"],
    }[task]
    edges = [0.0] + [c for c in cuts if c < t1] + [t1]
    return [
        {"start_s": edges[k], "end_s": edges[k + 1], "state": names[k]}
        for k in range(len(edges) - 1)
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stick-figure replay of one try (mp4)")
    ap.add_argument("path", nargs="?", help="session .json, dataset .json, or .npz")
    ap.add_argument("--zip", help="read PATH as a member of this zip (e.g. data/raw/B_skel.zip)")
    ap.add_argument("--fake", type=int, choices=[1, 2, 3], help="render a fake try of this task")
    ap.add_argument("--leg", choices=["L", "R"], default="L")
    ap.add_argument("--events", help="events JSON file, or 'demo' for hand-made demo events")
    ap.add_argument("--no-labels", action="store_true")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("-o", "--out", help="output .mp4 (default: next to the input, or replay.mp4)")
    args = ap.parse_args(argv)

    if args.fake:
        tr, events = make_fake_try(task=args.fake, leg=args.leg), None
        default_out = Path(f"fake_task{args.fake}_{args.leg}.mp4")
    elif args.path:
        tr, events = load_any(args.path, args.zip)
        default_out = Path(Path(args.path).stem + ".mp4")
        if not args.zip:
            default_out = Path(args.path).with_suffix(".mp4")
    else:
        ap.error("give a file path or --fake TASK")
    if args.events == "demo":
        events = demo_events(tr)
    elif args.events:
        events = json.loads(Path(args.events).read_text())
    out = render(
        tr, args.out or default_out, fps=args.fps, labels=not args.no_labels, events=events
    )
    print(f"wrote {out} ({tr.n_frames} frames, {tr.duration:.2f} s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
