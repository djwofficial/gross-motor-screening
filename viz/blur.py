"""Blur faces in level-3 videos using the RTMPose head points. Owner: Theresia (#52, Nov 22).

Only for children whose parent ticked consent level 3. The original clip stays on the encrypted
drive and is deleted as the consent form says; only the blurred copy may ever be shown.

    python -m viz.blur runs/2026-11-25/T01_1a2b.avi --device cpu     # -> T01_1a2b_blurred.avi
    python -m viz.blur runs/2026-11-25/T01_1a2b_blurred.avi --check  # step through every frame

How it works:
1. RTMPose runs again on the video's own frames (unmirrored, as recorded) and finds EVERY person,
   not just the picked child: anyone in the picture gets blurred. The try JSON is not used for
   the places, because its points start 1 s before the video (pre-roll), only hold the picked
   child, and the recorder may drop frames.
2. Per person, the head points (nose, eyes, ears = BODY_25 0, 15-18) place an ellipse; its size
   comes from the head points' spread and, when found, the shoulder width and torso. When no
   head point is found but neck and shoulders are, the ellipse goes above the neck.
3. Each frame also gets the ellipses of the 2 frames before and after it, so a missed frame or a
   fast jump is still covered.
4. Fail-safe: a frame where nobody was found, or where a person was found but no head could be
   placed, is blurred WHOLE. These frames are listed in the check CSV.
5. Check: <out>.check.csv has one row per frame (status ok / whole_nobody / whole_unsure).
   Then `--check` opens the blurred clip frame by frame: mark any frame where a face can be seen.
   The result is written to the eye_check column. A clip is only fine when every frame was seen
   and none is marked "face".

Ellipse sizes are guesses (2026-11-22), set so the ellipse covers hair and chin with margin;
the eye check is what makes them safe. Pixels here are places in the picture, not measurements.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from common import body25 as b

MIN_SCORE = 0.3
NEIGHBOURS = 2  # frames before and after whose ellipses are added to each frame
BLOCKS = 6  # mosaic blocks across an ellipse before the blur
HEAD = [b.NOSE, b.R_EYE, b.L_EYE, b.R_EAR, b.L_EAR]
CHECK_FIELDS = ["frame", "t_s", "people", "heads", "status", "eye_check"]


# ------------------------------------------------------------------ where the faces are


def head_ellipse(p, min_score: float = MIN_SCORE):
    """One person (25, 3) -> (cx, cy, ax, ay) in pixels, or None if no head can be placed.

    ax, ay = half width and half height. A person with no point found at all also gives None.
    """
    p = np.asarray(p, dtype=np.float32)
    ok = p[:, 2] >= min_score
    sh = [i for i in (b.R_SHOULDER, b.L_SHOULDER) if ok[i]]
    shoulder_w = (
        float(np.linalg.norm(p[b.R_SHOULDER, :2] - p[b.L_SHOULDER, :2])) if len(sh) == 2 else 0.0
    )
    torso = (
        float(np.linalg.norm(p[b.NECK, :2] - p[b.MID_HIP, :2]))
        if ok[b.NECK] and ok[b.MID_HIP]
        else 0.0
    )
    head = [i for i in HEAD if ok[i]]
    if head:
        pts = p[head, :2]
        c = pts.mean(axis=0)
        spread = (
            float(np.max(np.linalg.norm(pts[:, None] - pts[None], axis=2)))
            if len(head) > 1
            else 0.0
        )
        neck_d = float(np.linalg.norm(c - p[b.NECK, :2])) if ok[b.NECK] else 0.0
        # a child's head is about 0.8 shoulder widths wide and 0.5 torso lengths tall (guess)
        r = max(1.1 * spread, 0.6 * shoulder_w, 0.45 * torso, 0.9 * neck_d, 12.0)
        return float(c[0]), float(c[1]), r, 1.25 * r
    if ok[b.NECK] and (shoulder_w > 0 or torso > 0):
        # no head point: put the head above the neck, away from the hip
        if ok[b.MID_HIP] and torso > 0:
            up = (p[b.NECK, :2] - p[b.MID_HIP, :2]) / torso
        else:
            up = np.array([0.0, -1.0], dtype=np.float32)  # picture y points down
        r = max(0.7 * shoulder_w, 0.5 * torso, 12.0)
        c = p[b.NECK, :2] + up * 1.1 * r
        return float(c[0]), float(c[1]), 1.2 * r, 1.4 * r
    return None


def frame_plan(people, min_score: float = MIN_SCORE) -> tuple[list[tuple], str]:
    """All people in one frame (P, 25, 3) -> (ellipses, status).

    status "ok": every person found has an ellipse; "nobody": nobody found;
    "unsure": someone was found but their head could not be placed.
    """
    people = np.asarray(people, dtype=np.float32).reshape(-1, 25, 3)
    found = [q for q in people if np.any(q[:, 2] >= min_score)]
    if not found:
        return [], "nobody"
    ells = [head_ellipse(q, min_score) for q in found]
    status = "ok" if all(e is not None for e in ells) else "unsure"
    return [e for e in ells if e is not None], status


def plan(frames_people, min_score: float = MIN_SCORE, neighbours: int = NEIGHBOURS):
    """Per frame: (ellipses incl. the neighbours' ones, status, people found, heads placed)."""
    own = [frame_plan(pp, min_score) for pp in frames_people]
    out = []
    for i, (ells, status) in enumerate(own):
        near = []
        for j in range(max(0, i - neighbours), min(len(own), i + neighbours + 1)):
            near += own[j][0]
        n_people = int(
            np.sum(
                np.any(
                    np.asarray(frames_people[i]).reshape(-1, 25, 3)[:, :, 2] >= min_score, axis=1
                )
            )
        )
        out.append((near, status, n_people, len(ells)))
    return out


# ------------------------------------------------------------------ blurring


def blur_whole(frame):
    import cv2

    h, w = frame.shape[:2]
    small = cv2.resize(frame, (max(1, w // 40), max(1, h // 40)), interpolation=cv2.INTER_AREA)
    big = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return cv2.GaussianBlur(big, (0, 0), sigmaX=max(w, h) / 60)


def blur_ellipses(frame, ellipses):
    """Mosaic + blur inside each ellipse; the rest of the picture is not changed."""
    import cv2

    out = frame.copy()
    h, w = frame.shape[:2]
    for cx, cy, ax, ay in ellipses:
        x0, x1 = int(max(0, np.floor(cx - ax))), int(min(w, np.ceil(cx + ax) + 1))
        y0, y1 = int(max(0, np.floor(cy - ay))), int(min(h, np.ceil(cy + ay) + 1))
        if x1 <= x0 or y1 <= y0:
            continue
        roi = out[y0:y1, x0:x1]
        rw, rh = x1 - x0, y1 - y0
        small = cv2.resize(
            roi, (BLOCKS, max(1, round(BLOCKS * rh / rw))), interpolation=cv2.INTER_AREA
        )
        mosaic = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)
        mosaic = cv2.GaussianBlur(mosaic, (0, 0), sigmaX=max(rw, rh) / 8)
        mask = np.zeros((rh, rw), np.uint8)
        cv2.ellipse(
            mask, (round(cx - x0), round(cy - y0)), (round(ax), round(ay)), 0, 0, 360, 255, -1
        )
        roi[mask > 0] = mosaic[mask > 0]
    return out


def blur_frame(frame, ellipses, status: str):
    return blur_whole(frame) if status != "ok" else blur_ellipses(frame, ellipses)


# ------------------------------------------------------------------ the video


def _people25(keypoints, scores):
    from live.convert import halpe_to_body25

    kp = np.asarray(keypoints, dtype=np.float32).reshape(-1, 26, 2)
    sc = np.asarray(scores, dtype=np.float32).reshape(-1, 26)
    if len(kp) == 0:
        return np.zeros((0, 25, 3), np.float32)
    return np.asarray(halpe_to_body25(kp, sc), dtype=np.float32).reshape(-1, 25, 3)


def default_out(video_path) -> Path:
    p = Path(video_path)
    return p.with_name(f"{p.stem}_blurred.avi")


def check_path(out_path) -> Path:
    p = Path(out_path)
    return p.with_name(p.stem + ".check.csv")


def blur_video(
    video_path,
    points_json=None,
    out_path=None,
    *,
    device: str = "cpu",
    mode: str = "balanced",
    stream=None,
    min_score: float = MIN_SCORE,
) -> dict:
    """Blur every face in video_path and write out_path (MJPG .avi) and <out>.check.csv.

    points_json: the try JSON, optional; only used to check that it names this video.
    stream: anything with read() -> (t, keypoints (P,26,2), scores (P,26), frame) or None
    (default: live.pose.PoseStream on the video, which reads every frame).
    Returns {"out", "check", "frames", "ok", "whole_nobody", "whole_unsure"}.
    """
    import cv2

    video_path = Path(video_path)
    out_path = Path(out_path) if out_path else default_out(video_path)
    if points_json:
        meta = json.loads(Path(points_json).read_text(encoding="utf-8")).get("meta", {})
        named = meta.get("video_file")
        if named and named != video_path.name:
            raise ValueError(f"{points_json} names video {named}, not {video_path.name}")
    if stream is None:
        from live.pose import PoseStream

        stream = PoseStream(video_path, mode=mode, device=device)
    fps = float(getattr(stream, "source_fps", None) or 30.0)

    # pass 1: find the people (needs the whole clip for the frames after each frame)
    times, people = [], []
    try:
        while (item := stream.read()) is not None:
            t, kp, sc, _ = item
            times.append(float(t))
            people.append(_people25(kp, sc))
    finally:
        stream.close()
    if not times:
        raise ValueError(f"no frames read from {video_path}")
    steps = plan(people, min_score)

    # pass 2: read the frames again and blur them (frames are not kept in memory)
    cap = cv2.VideoCapture(str(video_path))
    writer = None
    n = 0
    try:
        while n < len(steps):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if writer is None:
                h, w = frame.shape[:2]
                out_path.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(
                    str(out_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h)
                )
                if not writer.isOpened():
                    raise OSError("OpenCV cannot open the MJPG video encoder")
            ells, status, _, _ = steps[n]
            writer.write(blur_frame(frame, ells, status))
            n += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if n != len(steps):
        out_path.unlink(missing_ok=True)
        raise ValueError(
            f"read {len(steps)} frames for the pose but {n} for blurring; nothing written"
        )

    names = {"ok": "ok", "nobody": "whole_nobody", "unsure": "whole_unsure"}
    rows = [
        {
            "frame": i,
            "t_s": f"{times[i]:.3f}",
            "people": s[2],
            "heads": s[3],
            "status": names[s[1]],
            "eye_check": "",
        }
        for i, s in enumerate(steps)
    ]
    write_check(check_path(out_path), rows)
    res = {"out": out_path, "check": check_path(out_path), "frames": n}
    for k in ("ok", "whole_nobody", "whole_unsure"):
        res[k] = sum(r["status"] == k for r in rows)
    return res


# ------------------------------------------------------------------ checking every frame


def write_check(path, rows) -> Path:
    path = Path(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CHECK_FIELDS)
        w.writeheader()
        w.writerows(rows)
    return path


def read_check(path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def check_summary(rows) -> dict:
    """Eye check result: every frame seen, and how many show a face."""
    seen = sum(r["eye_check"] in ("ok", "face") for r in rows)
    faces = [int(r["frame"]) for r in rows if r["eye_check"] == "face"]
    return {
        "frames": len(rows),
        "seen": seen,
        "faces": faces,
        "passed": bool(rows) and seen == len(rows) and not faces,
    }


def apply_marks(rows, seen: set[int], faces: set[int]) -> list[dict]:
    """Write the eye check into the rows: 'face' for marked frames, 'ok' for the other seen ones."""
    for r in rows:
        i = int(r["frame"])
        if i in faces:
            r["eye_check"] = "face"
        elif i in seen:
            r["eye_check"] = "ok"
    return rows


KEYS_HELP = (
    "D/right = next   A/left = back   F = face visible here (toggle)   Q/Esc = save and quit"
)


def review(blurred_path, check_csv=None) -> dict:
    """Step through the blurred clip one frame at a time; F marks a frame where a face shows.

    Every frame must be shown once. The marks go into the check CSV's eye_check column.
    """
    import cv2

    check_csv = Path(check_csv) if check_csv else check_path(blurred_path)
    rows = read_check(check_csv)
    cap = cv2.VideoCapture(str(blurred_path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    if len(frames) != len(rows):
        raise ValueError(f"{blurred_path} has {len(frames)} frames, {check_csv} has {len(rows)}")
    seen = {int(r["frame"]) for r in rows if r["eye_check"] in ("ok", "face")}
    faces = {int(r["frame"]) for r in rows if r["eye_check"] == "face"}
    i, win = 0, "blur check"
    print(KEYS_HELP)
    while frames:
        seen.add(i)
        img = frames[i].copy()
        h = img.shape[0]
        s = max(0.5, h / 900)
        label = f"frame {i + 1}/{len(frames)}  seen {len(seen)}  {rows[i]['status']}"
        if i in faces:
            label += "  FACE"
        cv2.putText(img, label, (10, int(30 * s)), cv2.FONT_HERSHEY_SIMPLEX, s, (0, 0, 255), 2)
        cv2.imshow(win, img)  # as recorded (unmirrored), so it matches the CSV
        k = cv2.waitKeyEx(0)
        if k in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if k in (ord("f"), ord("F")):
            faces ^= {i}
        elif k in (ord("a"), 2424832):  # left arrow on Windows
            i = max(0, i - 1)
        elif k in (ord("d"), ord(" "), 2555904):  # right arrow on Windows
            i = min(len(frames) - 1, i + 1)
    cv2.destroyAllWindows()
    write_check(check_csv, apply_marks(rows, seen, faces))
    return check_summary(rows)


def _print_summary(s: dict) -> None:
    print(f"eye check: seen {s['seen']}/{s['frames']} frames, face visible in {len(s['faces'])}")
    if s["faces"]:
        print(f"  frames with a face: {s['faces'][:30]}")
    print("PASSED: no frame shows a face" if s["passed"] else "NOT PASSED yet")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Blur faces in a level-3 try video")
    ap.add_argument(
        "video", help="the try video (.avi from the app) or, with --check, the blurred one"
    )
    ap.add_argument("--json", help="the try JSON (optional, checks that it names this video)")
    ap.add_argument("-o", "--out", help="output .avi (default: <video>_blurred.avi)")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument(
        "--pose-mode", default="balanced", choices=["lightweight", "balanced", "performance"]
    )
    ap.add_argument("--check", action="store_true", help="eye check a blurred clip frame by frame")
    ap.add_argument("--summary", action="store_true", help="print the eye check result and stop")
    args = ap.parse_args(argv)
    if args.summary:
        _print_summary(check_summary(read_check(check_path(args.video))))
        return 0
    if args.check:
        s = review(args.video)
        _print_summary(s)
        return 0 if s["passed"] else 1
    res = blur_video(args.video, args.json, args.out, device=args.device, mode=args.pose_mode)
    print(
        f"wrote {res['out']} ({res['frames']} frames: {res['ok']} face ellipses, "
        f"{res['whole_nobody']} whole-blurred nobody found, {res['whole_unsure']} whole-blurred unsure)"
    )
    print(f"now check every frame: python -m viz.blur {res['out']} --check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
