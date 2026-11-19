"""Measure live pose FPS with video saving off and on, in the real app window. Owner: Revallo.

For #30 (Done when) and #34. Drives app.ui.Runner with PoseStream and TryVideoRecorder exactly
like `python -m app.ui --camera 0 --device cuda --mode manual --view camera`, but presses the keys
itself so both runs have the same length: setup T1 / 45 / consent 3, Enter at readiness,
V (only with --video-on), Space, record --seconds, Space, Enter.

    python -m live.measure_video_fps --video-off --out runs/fps_check
    python -m live.measure_video_fps --video-on  --out runs/fps_check

Stand in view (team members only) before the countdown ends. Prints: pose fps = frames in the
saved try / its duration_s (log.csv), and for V on the recorder counts offered / written / dropped
and whether the .avi opens (frame count read back with OpenCV). Delete --out after copying numbers.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path


def _drive(args) -> dict:
    import cv2

    from app.ui import Runner, draw_screen, key_name
    from live.pose import PoseStream

    stream = PoseStream(source=args.camera, mode=args.pose_mode, device=args.device)
    w = int(stream.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(stream.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    runner = Runner(args.out, mode="manual", frame_size=(w, h), source="webcam")

    counts: dict = {}
    close_video = runner._close_video

    def spy(keep):  # read the counts of the recorder just before it is closed
        if runner.recorder is not None:
            counts["offered"], counts["written"], counts["dropped"] = runner.recorder.counts
        return close_video(keep)

    runner._close_video = spy

    # (seconds after the window opens, key)
    plan = [(0.5, c) for c in "T1"] + [(0.6, "ENTER")] + [(0.7, c) for c in "45"]
    plan += [(0.8, "ENTER"), (0.9, "3"), (1.0, "ENTER"), (args.warmup, "ENTER")]
    if args.video_on:
        plan.append((args.warmup + 0.5, "V"))
    t_space = args.warmup + 1.0
    plan += [(t_space, "SPACE"), (t_space + args.seconds, "SPACE")]
    plan.append((t_space + args.seconds + 1.0, "ENTER"))
    end_at = t_space + args.seconds + 2.0

    name = "Gross motor screening (screening aid, not a diagnosis)"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    t0 = None  # starts at the first frame, after the models have loaded
    loops = 0
    try:
        while t0 is None or time.monotonic() - t0 < end_at:
            item = stream.read()
            if item is None:
                break
            runner.feed(item)
            t0 = time.monotonic() if t0 is None else t0
            cv2.imshow(name, draw_screen(runner, "camera"))
            runner.press(key_name(cv2.waitKeyEx(1)))
            now = time.monotonic() - t0
            while plan and now >= plan[0][0]:
                runner.press(plan.pop(0)[1])
            loops += 1
    finally:
        runner.close()
        stream.close()
        cv2.destroyAllWindows()
    counts["window_loops"] = loops
    counts["saved"] = [str(p) for p in runner.saved]
    counts["message"] = runner.ctrl.message
    counts["camera"] = (w, h)
    return counts


def _report(args, counts: dict) -> None:
    import cv2

    log = Path(args.out) / "log.csv"
    rows = list(csv.DictReader(log.open(encoding="utf-8"))) if log.exists() else []
    if not rows:
        print("no try saved:", counts.get("message"))
        return
    r = rows[-1]
    n, dur = int(r["n_frames"]), float(r["duration_s"])
    print(
        f"V {'on' if args.video_on else 'off'}  camera {counts['camera'][0]}x{counts['camera'][1]}"
    )
    print(f"pose fps = {n} frames / {dur:.2f} s = {n / dur:.1f}")
    print(f"log video column = {r['video']}  app message: {counts.get('message')}")
    if args.video_on:
        print(
            f"recorder offered {counts.get('offered')}  written {counts.get('written')}  "
            f"dropped {counts.get('dropped')}"
        )
        avis = sorted(Path(args.out).glob("T1_*.avi"), key=lambda p: p.stat().st_mtime)
        if not avis:
            print("no .avi found")
            return
        cap = cv2.VideoCapture(str(avis[-1]))
        frames = 0
        while cap.read()[0]:
            frames += 1
        print(
            f"{avis[-1].name}: opens {cap.isOpened()}  frames read back {frames}  "
            f"playback fps {cap.get(cv2.CAP_PROP_FPS):.1f}"
        )
        cap.release()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Live pose FPS with video saving off and on")
    v = ap.add_mutually_exclusive_group(required=True)
    v.add_argument("--video-on", action="store_true")
    v.add_argument("--video-off", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--pose-mode", default="balanced")
    ap.add_argument("--seconds", type=float, default=25.0, help="length of the try")
    ap.add_argument("--warmup", type=float, default=5.0, help="seconds before Space")
    ap.add_argument("--out", type=Path, default=Path("runs") / "fps_check")
    args = ap.parse_args(argv)
    counts = _drive(args)
    _report(args, counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
