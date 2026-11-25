"""The one fullscreen window of the live laptop. Owner: Daryn (cyber HUD look: Marty, 2026-11-26).

Both entry points draw with this module:
    python -m app.ui ...                 testing mode: never a score on screen (show_score=False)
    python -m app.ui ... --show-score    and python -m app.demo: live score + last saved try

Look: a space-cyber game HUD over the edge-to-edge camera (or stick figure): dark space colours,
neon cyan accent with a soft glow, thin cut-corner (chamfered) frames, faint scan lines and a tech
font (Orbitron for big text, Rajdhani for the rest, both SIL OFL 1.1, in app/fonts/; the system
font is used if the files are missing). Every keyboard shortcut is also clickable; F toggles
fullscreen. Setup and confirm states use centred frames for typing or checking a try.

Phases of one try:
    floor plate   a fake-3D plate lying on the floor at the start area (no floor detection: the
                  tilt constants PLATE_* are guesses); it fills with blinking green ("READY") when
                  both ankles stand on it (visual only, it is not a start gate)
    countdown     gesture mode: glitching neon 3-2-1 in a segmented ring, then "GO"
    detection     top right: the task panel (TASK n, task name); show_score: on the left edge an
                  evaluation bar with the live Bad / Good / Perfect probabilities
    result        a big centred box when an attempt ends, which then flies into the right sidebar
    sidebar       show_score: one row per attempt of this recording, newest on top

Scores: Bad red, Good yellow-green, Perfect green. Every animation runs on time (_clock, which is
time.monotonic, or the Runner's stream seconds for windows the Runner defines), never on frame
counts: pose runs at ~8 fps on a CPU.

Size: run_window draws at the screen's own size (set_size(canvas_size(screen_size()))), so the
fullscreen window has no bars and nothing is stretched (Mac: 1512x982 points). The camera picture
fills the height; on a 16:10 screen a little of both sides of a 16:9 camera is cut on screen only
(MAX_SIDE_CROP); pose detection always used the whole picture.

render() only reads the Runner, never changes it (animation state lives here, in _ANIM). With
show_score=False it never reads runner.live or runner.result, so no score can reach the screen in
testing mode. Pose detection always ran on the unmirrored picture; only this picture is mirrored.
Screening aid only, never a diagnosis.
"""

from __future__ import annotations

import math
import os
import sys
import time
import weakref
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.session import AGE_RANGE_MONTHS, age_in_range
from common import body25 as b

W, H = 1280, 720  # changed by set_size() when the window opens
PAD = 24
VX, VY, VW, VH = 0, 0, W, H
BAR_H = 36
BAR_Y = H - BAR_H - 6
RADIUS = 0
CARD_W = 320
MAX_CANVAS_H = 1080  # bigger screens are drawn at this height and scaled up by the window
MIN_CANVAS_H = 720
MAX_SIDE_CROP = 0.08  # on screen only: cut at most this much of each side of the camera picture

# Floor plate: a trapezoid lying on the floor at the start area. GUESSES (a webcam has no depth and
# nothing here detects the floor): tune them by eye on the real room.
PLATE_DEPTH = 1.0  # far edge above the near edge by this fraction of the start-area height
PLATE_TOP_INSET = 0.16  # far edge narrower by this fraction of the plate width, on each side
PLATE_BRACKET = 0.2  # corner bracket arm, as a fraction of the plate side (floor units)
PLATE_GRID = 4  # grid cells per side on the plate
READY_MIN_SCORE = 0.3  # an ankle with a lower score is not counted (as live.start_area.MIN_SCORE)

# Animation lengths in seconds.
TASK_SWAP_S = 0.4  # task panel: glitch out the old task, flicker in the new one
GO_S = 0.5  # "GO" after the countdown
BOX_IN_S = 0.6  # result box opens
BOX_OUT_S = 0.5  # result box closes and flies into the sidebar
BOX_SHOW_S = 2.5  # result box on screen in manual/auto mode and testing mode
ROW_FLASH_S = 0.6  # a sidebar row flashes when its result lands
BAR_TAU_S = 0.25  # evaluation bar easing time constant
SCANLINE_GAIN = 0.86  # every 3rd row of the picture this bright: faint scan lines

_clock = time.monotonic  # replaced by scripts/preview_cyber_ui.py to render chosen moments


def set_size(w: int, h: int) -> None:
    """Draw every picture at w x h from now on (run_window: the screen's own size)."""
    global W, H, VW, VH, BAR_Y
    W, H = int(w), int(h)
    VW, VH = W, H
    BAR_Y = H - BAR_H - 6
    _backdrop.cache_clear()
    _vignette.cache_clear()


def screen_size() -> tuple[int, int] | None:
    """Size of the main display in points (Mac: CoreGraphics; Windows: user32), or None."""
    import ctypes

    try:
        if sys.platform == "darwin":
            import ctypes.util

            cg = ctypes.CDLL(ctypes.util.find_library("CoreGraphics"))
            cg.CGMainDisplayID.restype = ctypes.c_uint32
            for name in ("CGDisplayPixelsWide", "CGDisplayPixelsHigh"):
                getattr(cg, name).restype = ctypes.c_size_t
                getattr(cg, name).argtypes = [ctypes.c_uint32]
            d = cg.CGMainDisplayID()
            w, h = int(cg.CGDisplayPixelsWide(d)), int(cg.CGDisplayPixelsHigh(d))
        elif sys.platform == "win32":
            user32 = ctypes.windll.user32
            w, h = int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
        else:
            return None
    except (OSError, AttributeError, TypeError):
        return None
    return (w, h) if w > 0 and h > 0 else None


def canvas_size(screen: tuple[int, int]) -> tuple[int, int]:
    """The drawing size for a screen: its own shape, MIN..MAX_CANVAS_H tall, even numbers."""
    w, h = screen
    k = min(1.0, MAX_CANVAS_H / h)
    if h * k < MIN_CANVAS_H:
        k = MIN_CANVAS_H / h
    return int(round(w * k / 2)) * 2, int(round(h * k / 2)) * 2


WINDOW_TITLE = "Gross motor screening (screening aid, not a diagnosis)"


def _bgr(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


BG = _bgr("#05070f")  # deep space
SURFACE = _bgr("#0a1124")
RAISED = _bgr("#101b34")
LINE = _bgr("#1f3654")
TEXT = _bgr("#e8f8ff")
DIM = _bgr("#a3bdd1")
FAINT = _bgr("#6d87a0")
ACCENT = _bgr("#00e5ff")  # neon cyan
GREEN = _bgr("#00ff6a")  # Perfect, ready, ok
LIME = _bgr("#c6ff3d")  # Good
RED = _bgr("#ff3355")  # Bad, recording, stop
AMBER = _bgr("#ffb547")  # warnings only (paused, locked, outside the age range)
PURPLE = _bgr("#c58cff")  # check by teacher
LETTERBOX = _bgr("#020308")
NEUTRAL = _bgr("#0a0f1a")
SHADOW = _bgr("#000207")

STATE_COLOURS = {
    "setup": FAINT,
    "readiness": ACCENT,
    "idle": DIM,
    "ready": GREEN,
    "countdown": ACCENT,
    "recording": RED,
    "confirm": AMBER,
}
STATE_NAMES = {
    "setup": "New child",
    "readiness": "Readiness check",
    "idle": "Waiting",
    "ready": "Ready",
    "countdown": "Get ready",
    "recording": "Recording",
    "confirm": "Check the try",
}
SCORE_NAMES = dict(b.LABELS)
SCORE_COLOURS = {0: RED, 1: LIME, 2: GREEN}
SCORE_WORDS = {
    0: "did not complete the task this time",
    1: "completed the task with some difficulty",
    2: "completed the task smoothly",
}
TASK_SHORT = {1: "Stand", 2: "Hop", 3: "Jump"}
MODE_NAMES = {"auto": "Auto", "manual": "Manual", "gesture": "Gesture"}

_RIGHT_PTS = {b.R_SHOULDER, b.R_ELBOW, b.R_WRIST, b.R_HIP, b.R_KNEE, b.R_ANKLE, b.R_EYE, b.R_EAR}
_RIGHT_PTS |= set(b.RIGHT_FOOT)
_LEFT_PTS = {b.L_SHOULDER, b.L_ELBOW, b.L_WRIST, b.L_HIP, b.L_KNEE, b.L_ANKLE, b.L_EYE, b.L_EAR}
_LEFT_PTS |= set(b.LEFT_FOOT)
FIG_RIGHT, FIG_LEFT, FIG_MID = _bgr("#ff9f43"), _bgr("#2fa8ff"), _bgr("#eafcff")
MIN_SCORE = 0.05

# ---------------------------------------------------------------------- fonts and text

FONT_DIR = Path(__file__).resolve().parent / "fonts"
_WEIGHTS = ("regular", "medium", "semibold", "bold")
_RAJDHANI = {"regular": "Regular", "medium": "Medium", "semibold": "SemiBold", "bold": "Bold"}
_ORBITRON = FONT_DIR / "Orbitron-Variable.ttf"


def _system_font_sources() -> dict[str, tuple[str, int, str | None]]:
    """The system font (SF Pro on Mac, Segoe UI on Windows, DejaVu Sans from matplotlib)."""
    if sys.platform == "darwin":
        sf = "/System/Library/Fonts/SFNS.ttf"  # SF Pro, variable
        if os.path.exists(sf):
            return {w: (sf, 0, w.capitalize()) for w in _WEIGHTS}
        hn = "/System/Library/Fonts/HelveticaNeue.ttc"  # 0 Regular, 1 Bold, 10 Medium
        if os.path.exists(hn):
            index = {"regular": 0, "medium": 10, "semibold": 10, "bold": 1}
            return {w: (hn, index[w], None) for w in _WEIGHTS}
    if sys.platform == "win32":
        d = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
        files = {
            "regular": "segoeui.ttf",
            "medium": "seguisb.ttf",
            "semibold": "seguisb.ttf",
            "bold": "segoeuib.ttf",
        }
        if all((d / f).exists() for f in files.values()):
            return {w: (str(d / f), 0, None) for w, f in files.items()}
    try:
        import matplotlib

        d = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        reg, bold = str(d / "DejaVuSans.ttf"), str(d / "DejaVuSans-Bold.ttf")
        if os.path.exists(reg) and os.path.exists(bold):
            return {
                "regular": (reg, 0, None),
                "medium": (reg, 0, None),
                "semibold": (bold, 0, None),
                "bold": (bold, 0, None),
            }
    except ImportError:
        pass
    return {}


@lru_cache(maxsize=1)
def _font_sources() -> dict[str, tuple[str, int, str | None]]:
    """weight -> (font file, index in a .ttc, variation name of a variable font).

    regular..bold: Rajdhani from app/fonts (else the system font); display: Orbitron Bold for big
    text (else the bold weight). Orbitron has no middle dot: never give it a string with one.
    """
    raj = {w: FONT_DIR / f"Rajdhani-{n}.ttf" for w, n in _RAJDHANI.items()}
    if all(p.exists() for p in raj.values()):
        src = {w: (str(p), 0, None) for w, p in raj.items()}
    else:
        src = _system_font_sources()
    if _ORBITRON.exists():
        src["display"] = (str(_ORBITRON), 0, "Bold")
    elif "bold" in src:
        src["display"] = src["bold"]
    return src


@lru_cache(maxsize=96)
def _font(size: int, weight: str = "regular"):
    from PIL import ImageFont

    src = _font_sources().get(weight)
    if src is not None:
        path, index, variation = src
        try:
            font = ImageFont.truetype(path, size, index=index)
            if variation:
                try:
                    font.set_variation_by_name(variation)
                except (OSError, ValueError):
                    pass
            return font
        except OSError:
            pass
    return ImageFont.load_default(size)


@lru_cache(maxsize=96)
def _metrics(size: int, weight: str) -> tuple[int, int, float]:
    """(ascent, descent, middle of the capital letters measured from the top of the line)."""
    f = _font(size, weight)
    asc, desc = f.getmetrics()
    _, y0, _, y1 = f.getbbox("H")
    return asc, desc, (y0 + y1) / 2


@lru_cache(maxsize=4096)
def measure(s: str, size: int = 16, weight: str = "regular") -> float:
    """Width of one line of text in pixels."""
    return float(_font(size, weight).getlength(s))


@lru_cache(maxsize=2048)
def _ink(s: str, size: int, weight: str) -> np.ndarray:
    """Alpha (h, w, 1) of one line of text, drawn 2 px in from the left of the line box."""
    from PIL import Image, ImageDraw

    f = _font(size, weight)
    asc, desc = f.getmetrics()
    im = Image.new("L", (max(1, math.ceil(measure(s, size, weight)) + 4), asc + desc), 0)
    ImageDraw.Draw(im).text((2, 0), s, font=f, fill=255)
    return np.asarray(im, dtype=np.float32)[..., None] / 255.0


@lru_cache(maxsize=512)
def _halo(s: str, size: int, weight: str, spread: float = 2.2) -> tuple[np.ndarray, int]:
    """A soft halo that follows the letters (blurred ink), and its padding in pixels."""
    import cv2

    ink = _ink(s, size, weight)[..., 0]
    pad = max(3, size // 6)
    big = np.pad(ink, pad)
    halo = cv2.GaussianBlur(big, (0, 0), sigmaX=pad / spread)
    return np.clip(halo * 2.2, 0, 1)[..., None], pad


def wrap(s: str, size: int, weight: str, max_w: float, max_lines: int | None = None) -> list[str]:
    """Greedy word wrap; the last kept line ends with an ellipsis if text was cut."""
    lines, cur = [], ""
    for word in str(s).split():
        t = f"{cur} {word}" if cur else word
        if not cur or measure(t, size, weight) <= max_w:
            cur = t
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while last and measure(last + "…", size, weight) > max_w:
            last = last[:-1]
        lines[-1] = last.rstrip() + "…"
    return lines


# ---------------------------------------------------------------------- masks and backdrops


@lru_cache(maxsize=512)
def _rr_mask(w: int, h: int, r: int) -> np.ndarray:
    """Anti-aliased rounded-rectangle alpha (h, w, 1), drawn 4x larger and scaled down."""
    import cv2

    k = 4
    big = np.zeros((h * k, w * k), np.uint8)
    rk = r * k
    if rk <= 0:
        big[:] = 255
    else:
        cv2.rectangle(big, (rk, 0), (w * k - 1 - rk, h * k - 1), 255, -1)
        cv2.rectangle(big, (0, rk), (w * k - 1, h * k - 1 - rk), 255, -1)
        for cx, cy in ((rk, rk), (w * k - 1 - rk, rk), (rk, h * k - 1 - rk)):
            cv2.circle(big, (cx, cy), rk, 255, -1, cv2.LINE_AA)
        cv2.circle(big, (w * k - 1 - rk, h * k - 1 - rk), rk, 255, -1, cv2.LINE_AA)
    small = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)[..., None] / 255.0


@lru_cache(maxsize=64)
def _ring_mask(w: int, h: int, r: int, bw: int) -> np.ndarray:
    outer = _rr_mask(w, h, r)
    inner = np.zeros_like(outer)
    if w > 2 * bw and h > 2 * bw:
        inner[bw : h - bw, bw : w - bw] = _rr_mask(w - 2 * bw, h - 2 * bw, max(r - bw, 0))
    return outer - inner


def _chamfer_points(w: float, h: float, cut: float) -> np.ndarray:
    """Outline of a box with the top-left and bottom-right corners cut at 45 degrees."""
    return np.array(
        [[cut, 0], [w, 0], [w, h - cut], [w - cut, h], [0, h], [0, cut]], dtype=np.float64
    )


@lru_cache(maxsize=256)
def _chamfer_mask(w: int, h: int, cut: int) -> np.ndarray:
    """Anti-aliased alpha (h, w, 1) of a chamfered box (fill)."""
    import cv2

    k = 4
    big = np.zeros((h * k, w * k), np.uint8)
    pts = np.round(_chamfer_points(w * k - 1, h * k - 1, cut * k)).astype(np.int32)
    cv2.fillPoly(big, [pts], 255, cv2.LINE_AA)
    small = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)[..., None] / 255.0


@lru_cache(maxsize=256)
def _chamfer_line(w: int, h: int, cut: int, bw: int = 1) -> np.ndarray:
    """Anti-aliased alpha (h, w, 1) of the chamfered outline only."""
    import cv2

    k = 4
    big = np.zeros((h * k, w * k), np.uint8)
    inset = bw * k / 2
    pts = _chamfer_points(w * k - 1 - 2 * inset, h * k - 1 - 2 * inset, cut * k) + inset
    cv2.polylines(big, [np.round(pts).astype(np.int32)], True, 255, bw * k, cv2.LINE_AA)
    small = cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32)[..., None] / 255.0


@lru_cache(maxsize=128)
def _chamfer_glow(w: int, h: int, cut: int, spread: int = 8) -> tuple[np.ndarray, int]:
    """Soft neon glow (blurred outline) around a chamfered box, and its padding."""
    import cv2

    line = np.pad(_chamfer_line(w, h, cut, 2)[..., 0], spread * 2)
    glow = cv2.GaussianBlur(line, (0, 0), sigmaX=spread / 2)
    return np.clip(glow * 2.0, 0, 1)[..., None], spread * 2


@lru_cache(maxsize=1)
def _backdrop() -> np.ndarray:
    """Deep-space backdrop for the stick figure: a cold glow in the middle and a few stars."""
    import cv2

    yy, xx = np.mgrid[0:VH, 0:VW].astype(np.float32)
    d = np.sqrt(((xx - VW / 2) / (VW / 2)) ** 2 + ((yy - VH * 0.45) / (VH / 2)) ** 2)
    t = np.clip(d / 1.3, 0, 1)[..., None]
    inner, outer = np.array(_bgr("#0d1830"), np.float32), np.array(BG, np.float32)
    noise = np.random.default_rng(0).uniform(-0.5, 0.5, (VH, VW, 1)).astype(np.float32)
    img = np.clip(inner * (1 - t) + outer * t + 0.5 + noise, 0, 255).astype(np.uint8)
    rng = np.random.default_rng(7)
    for _ in range(int(VW * VH / 9000)):
        x, y = int(rng.integers(0, VW)), int(rng.integers(0, VH))
        v = int(rng.integers(70, 200))
        r = 1 if rng.random() < 0.85 else 2
        cv2.circle(img, (x, y), r, (v, v, int(v * 0.85)), -1, cv2.LINE_AA)
    img[::3] = (img[::3] * SCANLINE_GAIN).astype(np.uint8)  # scan lines
    return img


@lru_cache(maxsize=1)
def _vignette() -> np.ndarray:
    """uint8 gain (VH, VW, 3), 255 = unchanged: darkens the corners a little (screen glass).

    Applied with one cv2.multiply (fast at 1080p; a float blend of the whole frame is not)."""
    yy, xx = np.mgrid[0:VH, 0:VW].astype(np.float32)
    d = np.sqrt(((xx - VW / 2) / (VW / 2)) ** 2 + ((yy - VH / 2) / (VH / 2)) ** 2)
    gain = 1.0 - np.clip((d - 0.75) / 0.7, 0, 1) ** 1.6 * 0.55
    gain[::3] *= SCANLINE_GAIN  # the scan lines are baked in (no extra pass per frame)
    g = np.round(gain * 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=2)


def _lighten(c, f: float = 0.14):
    return tuple(int(v + (255 - v) * f) for v in c)


def _mix(c1, c2, f: float):
    return tuple(int(round(a + (b2 - a) * f)) for a, b2 in zip(c1, c2, strict=True))


# ---------------------------------------------------------------------- small animation maths


def _clip01(x: float) -> float:
    return float(min(max(x, 0.0), 1.0))


def _ease(x: float) -> float:
    """Smoothstep 0..1."""
    x = _clip01(x)
    return x * x * (3 - 2 * x)


def _ease_out(x: float) -> float:
    x = _clip01(x)
    return 1 - (1 - x) ** 3


def _seg(p: float, a: float, b2: float) -> float:
    """Progress 0..1 of p inside the window a..b2."""
    return _clip01((p - a) / (b2 - a)) if b2 > a else float(p >= a)


def _blink(t: float, hz: float = 1.5) -> float:
    """0..1 pulse."""
    return 0.5 + 0.5 * math.sin(2 * math.pi * hz * t)


_FLICKER = (0.0, 0.85, 0.05, 1.0, 0.25, 0.9, 0.55, 1.0)


def _flicker(p: float) -> float:
    """Alpha while something 'flickers in' (p 0..1): a few hard on/off steps, then on."""
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    return _FLICKER[min(len(_FLICKER) - 1, int(p * len(_FLICKER)))]


def _approach(old: np.ndarray, new: np.ndarray, dt: float, tau: float = BAR_TAU_S) -> np.ndarray:
    """Exponential easing from old toward new over dt seconds."""
    k = 1.0 - math.exp(-max(dt, 0.0) / tau)
    return old + (new - old) * k


# ---------------------------------------------------------------------- canvas


class Canvas:
    """The window picture plus every string drawn on it (tests read .texts)."""

    def __init__(self, w: int | None = None, h: int | None = None, bg=BG):
        w, h = W if w is None else w, H if h is None else h
        self.img = np.empty((h, w, 3), np.uint8)
        if bg is not None:  # None: the caller paints every pixel (render: _video)
            self.img[:] = bg
        self.texts: list[str] = []

    def paint(self, x: int, y: int, alpha: np.ndarray, colour) -> None:
        """Blend colour into img at (x, y) with alpha (h, w, 1) in 0..1; clipped to the picture."""
        h, w = alpha.shape[:2]
        ih, iw = self.img.shape[:2]
        xa, ya, xb, yb = max(x, 0), max(y, 0), min(x + w, iw), min(y + h, ih)
        if xa >= xb or ya >= yb:
            return
        a = alpha[ya - y : yb - y, xa - x : xb - x]
        reg = self.img[ya:yb, xa:xb]
        c = np.asarray(colour, np.float32)
        reg[:] = (reg + (c - reg) * a + 0.5).astype(np.uint8)

    def text(
        self,
        x,
        y,
        s,
        size=16,
        weight="regular",
        colour=TEXT,
        anchor="lt",
        alpha=1.0,
        shadow: bool = False,
        glow: float = 0.0,
        neon: float = 0.0,
        record: bool = True,
    ):
        """anchor: l/c/r then t (top of the line box), m (middle of capitals) or b (baseline).

        glow (0..1): a soft dark halo around the letters, so text reads on any picture.
        neon (0..1): a soft halo in the text's own colour (the HUD glow).
        alpha 0 draws nothing but still records the string (record=False: never recorded)."""
        s = str(s)
        if not s:
            return 0.0
        if record:
            self.texts.append(s)
        w = measure(s, size, weight)
        if alpha <= 0:
            return w
        asc, _desc, mid = _metrics(size, weight)
        if anchor[0] == "c":
            x -= w / 2
        elif anchor[0] == "r":
            x -= w
        if anchor[1] == "m":
            y -= mid
        elif anchor[1] == "b":
            y -= asc
        ink = _ink(s, size, weight)
        px, py = int(round(x)) - 2, int(round(y))
        if glow > 0:
            halo, pad = _halo(s, size, weight)
            self.paint(px - pad, py - pad + max(1, size // 24), halo * (glow * alpha), SHADOW)
        if shadow:
            self.paint(px + 1, py + 1, ink * (0.75 * alpha), SHADOW)
        if neon > 0:
            halo, pad = _halo(s, size, weight, 1.2)
            self.paint(px - pad, py - pad, halo * (0.55 * neon * alpha), colour)
        self.paint(px, py, ink if alpha >= 1 else ink * alpha, colour)
        return w

    def rrect(self, x, y, w, h, r, fill=None, alpha=1.0, border=None, bw=1, border_alpha=1.0):
        x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
        if w <= 0 or h <= 0:
            return
        r = int(min(r, w // 2, h // 2))
        if fill is not None:
            m = _rr_mask(w, h, r)
            self.paint(x, y, m if alpha >= 1 else m * alpha, fill)
        if border is not None and bw > 0:
            self.ring(x, y, w, h, r, bw, border, border_alpha)

    def ring(self, x, y, w, h, r, bw, colour, alpha=1.0):
        """Border only; paints the four edge bands, not the (large) inside."""
        m = _ring_mask(w, h, r, bw)
        if alpha < 1:
            m = m * alpha
        band = max(r, bw) + 1
        self.paint(x, y, m[:band], colour)
        self.paint(x, y + h - band, m[h - band :], colour)
        self.paint(x, y + band, m[band : h - band, :band], colour)
        self.paint(x + w - band, y + band, m[band : h - band, w - band :], colour)

    def frame(
        self,
        x,
        y,
        w,
        h,
        cut: int = 12,
        colour=ACCENT,
        fill=SURFACE,
        fill_alpha: float = 0.78,
        line_alpha: float = 0.9,
        glow: float = 0.45,
    ):
        """A chamfered HUD frame: dark fill, thin neon outline and a soft glow."""
        x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
        if w < 4 or h < 4:
            return
        cut = int(max(0, min(cut, w // 3, h // 3)))
        if fill is not None and fill_alpha > 0:
            self.paint(x, y, _chamfer_mask(w, h, cut) * fill_alpha, fill)
        if glow > 0 and w * h < 2_000_000:
            g, pad = _chamfer_glow(w, h, cut)
            self.paint(x - pad, y - pad, g * glow, colour)
        if line_alpha > 0:
            self.paint(x, y, _chamfer_line(w, h, cut) * line_alpha, colour)

    def round_corners(self, x, y, w, h, r, bg=BG):
        """Cut the corners of a picture already drawn at (x, y, w, h) back to bg."""
        if r <= 0:
            return
        m = 1.0 - _rr_mask(2 * r, 2 * r, r)
        self.paint(x, y, m[:r, :r], bg)
        self.paint(x + w - r, y, m[:r, r:], bg)
        self.paint(x, y + h - r, m[r:, :r], bg)
        self.paint(x + w - r, y + h - r, m[r:, r:], bg)

    def dim(self, x, y, w, h, factor: float):
        import cv2

        reg = self.img[y : y + h, x : x + w]
        reg[:] = cv2.convertScaleAbs(reg, alpha=factor)

    def fade(self, x, y, w, h, colour, a_top: float, a_bottom: float):
        """Vertical eased fade of colour over the picture."""
        t = np.linspace(0, 1, h, dtype=np.float32)
        t = t * t * (3 - 2 * t)
        a = (a_top + (a_bottom - a_top) * t)[:, None, None]
        self.paint(x, y, np.broadcast_to(a, (h, w, 1)), colour)

    def fade_x(self, x, y, w, h, colour, a_left: float, a_right: float, v_feather: int = 56):
        """Soft 2D feathered horizontal scrim (smooth on top/bottom too, zero hard edges)."""
        tx = np.linspace(0, 1, w, dtype=np.float32)
        tx = tx * tx * (3 - 2 * tx)
        ax = (a_left + (a_right - a_left) * tx)[None, :]
        ay = np.ones((h, 1), dtype=np.float32)
        vf = min(v_feather, h // 2)
        if vf > 1:
            ramp = np.linspace(0, 1, vf, dtype=np.float32)
            ramp = ramp * ramp * (3 - 2 * ramp)
            ay[:vf, 0] = ramp
            ay[-vf:, 0] = ramp[::-1]
        a = (ay * ax)[..., None]
        self.paint(x, y, a, colour)

    def dot(self, cx, cy, r, colour):
        import cv2

        cv2.circle(self.img, (int(cx * 4), int(cy * 4)), int(r * 4), colour, -1, cv2.LINE_AA, 2)

    def poly(self, pts, colour, alpha: float = 1.0, closed: bool = True, thickness: int = 1):
        """Anti-aliased polyline (window pixels) with alpha; thickness -1 fills it."""
        import cv2

        pts = np.asarray(pts, np.float64)
        if alpha <= 0 or len(pts) < 2:
            return
        ih, iw = self.img.shape[:2]
        x0 = int(max(0, math.floor(pts[:, 0].min()) - abs(thickness) - 2))
        y0 = int(max(0, math.floor(pts[:, 1].min()) - abs(thickness) - 2))
        x1 = int(min(iw, math.ceil(pts[:, 0].max()) + abs(thickness) + 3))
        y1 = int(min(ih, math.ceil(pts[:, 1].max()) + abs(thickness) + 3))
        if x0 >= x1 or y0 >= y1:
            return
        q = np.round((pts - (x0, y0)) * 4).astype(np.int32)
        if alpha >= 1:
            reg = self.img[y0:y1, x0:x1]
            if thickness < 0:
                cv2.fillPoly(reg, [q], colour, cv2.LINE_AA, 2)
            else:
                cv2.polylines(reg, [q], closed, colour, thickness, cv2.LINE_AA, 2)
            return
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        if thickness < 0:
            cv2.fillPoly(m, [q], 255, cv2.LINE_AA, 2)
        else:
            cv2.polylines(m, [q], closed, 255, thickness, cv2.LINE_AA, 2)
        self.paint(x0, y0, m.astype(np.float32)[..., None] * (alpha / 255.0), colour)


def _brackets(
    cv: Canvas, x, y, w, h, arm, colour, corners=("tl", "tr", "bl", "br"), th=2, alpha=1.0
):
    """Right-angle corner brackets of a box."""
    spots = {
        "tl": (x, y, 1, 1),
        "tr": (x + w, y, -1, 1),
        "bl": (x, y + h, 1, -1),
        "br": (x + w, y + h, -1, -1),
    }
    for c in corners:
        px, py, sx, sy = spots[c]
        cv.poly(
            [(px + sx * arm, py), (px, py), (px, py + sy * arm)],
            colour,
            alpha,
            closed=False,
            thickness=th,
        )


def _glitch(cv: Canvas, x, y, w, h, amount: float, seed: int) -> None:
    """Slice jitter (bands shifted sideways) and a colour split (red and blue pulled apart)."""
    if amount <= 0:
        return
    ih, iw = cv.img.shape[:2]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(iw, int(x + w)), min(ih, int(y + h))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    reg = cv.img[y0:y1, x0:x1]
    src = reg.copy()
    rng = np.random.default_rng(seed)
    hh = y1 - y0
    for _ in range(int(rng.integers(3, 7))):
        a = int(rng.integers(0, hh))
        bh = int(rng.integers(2, max(3, hh // 5)))
        dx = int(round(rng.choice((-1, 1)) * amount * rng.uniform(6, 26)))
        reg[a : a + bh] = np.roll(src[a : a + bh], dx, axis=1)
    s = max(1, int(round(amount * 7)))
    reg[:, :, 2] = np.roll(reg[:, :, 2], s, axis=1)
    reg[:, :, 0] = np.roll(reg[:, :, 0], -s, axis=1)


# ---------------------------------------------------------------------- per-runner animation state


@dataclass
class _Anim:
    """What the screen remembers between pictures of one Runner (never stored on the Runner)."""

    t_last: float | None = None
    bar: np.ndarray | None = None
    task_key: tuple | None = None
    task_old: tuple | None = None
    task_content: tuple | None = None
    task_since: float = -1e9
    prev_state: str | None = None
    go_since: float = -1e9
    result_key: object = None
    result_since: float = -1e9
    history: list = field(default_factory=list)
    child: str | None = None
    tries_saved: int | None = None
    saved_task: int | None = None
    saved_since: float = -1e9


_ANIM: weakref.WeakKeyDictionary[object, _Anim] = weakref.WeakKeyDictionary()


def _anim(runner) -> _Anim:
    a = _ANIM.get(runner)
    if a is None:
        a = _ANIM[runner] = _Anim()
    return a


# ---------------------------------------------------------------------- stick figure


def _side(i: int):
    if i in _RIGHT_PTS:
        return FIG_RIGHT
    if i in _LEFT_PTS:
        return FIG_LEFT
    return FIG_MID


def draw_skeleton(img, pts: np.ndarray, scale: float, ox: float, oy: float) -> None:
    """One (25, 3) frame of points onto img (BGR, changed in place), with a dark outline."""
    import cv2

    ok = pts[:, 2] > MIN_SCORE
    q = np.round((pts[:, :2] * scale + (ox, oy)) * 4).astype(int)
    p = [(int(u), int(v)) for u, v in q]
    bones = [(i, j) for i, j in b.BONES if ok[i] and ok[j]]
    for i, j in bones:
        cv2.line(img, p[i], p[j], SHADOW, 9, cv2.LINE_AA, 2)
    for i, j in bones:
        colour = _side(j) if _side(j) != FIG_MID else _side(i)
        cv2.line(img, p[i], p[j], colour, 4, cv2.LINE_AA, 2)
    head = set(b.HEAD)
    for i in np.nonzero(ok)[0]:
        r = 3 if i in head else 5
        cv2.circle(img, p[i], (r + 2) * 4, SHADOW, -1, cv2.LINE_AA, 2)
        cv2.circle(img, p[i], r * 4, _side(int(i)), -1, cv2.LINE_AA, 2)


# ---------------------------------------------------------------------- bottom strip (buttons)


@dataclass
class Button:
    """A clickable keyboard shortcut or plain HUD label in the bottom strip."""

    key: str | None
    label: str
    cap: str = ""
    kind: str = "ghost"  # primary, ghost, danger, seg, label, gap
    colour: tuple = ACCENT
    selected: bool = False
    dot: tuple | None = None
    group: str = ""
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    pad: int = 12

    def contains(self, px, py) -> bool:
        return (
            self.key is not None
            and self.x <= px < self.x + self.w
            and self.y <= py < self.y + self.h
        )


_LABEL = (15, "semibold")
_CAP = (12, "bold")
_GROUP_GAP = 20


def _width(btn: Button) -> int:
    if btn.kind == "label":
        return math.ceil(measure(btn.label, 14, "semibold"))
    p = btn.pad if btn.kind != "seg" else max(8, btn.pad - 2)
    w = p + measure(btn.label, *_LABEL) + p
    if btn.dot:
        w += 12
    if btn.cap:
        cap_w = max(18.0, measure(btn.cap, *_CAP) + 10)
        w += cap_w + 7
    return math.ceil(w)


def _place(left: list[Button], right: list[Button], pad: int) -> bool:
    """Lay out the left and right groups inside [VX + PAD, VX + VW - PAD]."""
    x = VX + PAD
    for btn in left:
        if btn.kind == "gap":
            x += 16
            continue
        btn.pad = pad if btn.kind != "seg" else max(8, pad - 2)
        btn.x, btn.y, btn.w, btn.h = x, BAR_Y, _width(btn), BAR_H
        x += btn.w + (10 if btn.kind == "label" else 6)
    x = VX + VW - PAD
    for i, btn in enumerate(reversed(right)):
        btn.pad = pad
        btn.w, btn.y, btn.h = _width(btn), BAR_Y, BAR_H
        btn.x = x - btn.w
        next_idx = len(right) - 2 - i
        extra = 12 if btn.kind == "primary" and next_idx >= 0 else 6
        x = btn.x - extra
    shown = [b for b in left if b.kind != "gap"]
    left_end = max((b.x + b.w for b in shown), default=VX + PAD)
    right_start = min((b.x for b in right), default=VX + VW - PAD)
    gap = _GROUP_GAP if shown and right else 0
    return left_end + gap <= right_start


def buttons(runner) -> list[Button]:
    """Bottom keyboard strip for the current state. Pure: the same Runner gives the same items."""
    c = runner.ctrl
    st = c.state
    left: list[Button] = []
    right: list[Button] = []
    if st == "setup":
        left.append(Button(None, "Type with keyboard · Tab / Enter next", kind="label"))
        nxt = "Continue" if c.field == 2 else "Next"
        right += [
            Button("ESC", "Quit", "Esc", "danger"),
            Button("ENTER", nxt, "Enter", "primary"),
        ]
    elif st == "readiness":
        left.append(Button(None, "Press F to toggle fullscreen", kind="label"))
        right += [
            Button("N", "Next child", "N"),
            Button("Q", "Quit", "Q", "danger"),
            Button("ENTER", "Start", "Enter", "primary", GREEN),
        ]
    elif getattr(runner, "auto_flow", False) and st != "confirm":
        if c.preset_task is not None:
            leg_s = f" · {c.preset_leg}" if c.preset_leg and c.preset_task != 3 else ""
            left.append(
                Button(
                    None,
                    f"LOCKED · Task {c.preset_task} {TASK_SHORT[c.preset_task]}{leg_s}",
                    kind="label",
                    colour=AMBER,
                )
            )
            left.append(Button("0", "Auto", "0", "ghost"))
        elif c.preset_leg is not None:
            left.append(
                Button(
                    None,
                    f"LEG {c.preset_leg} · 1/2/3 lock task",
                    kind="label",
                    colour=AMBER,
                )
            )
            left.append(Button("0", "Auto", "0", "ghost"))
        elif c.mode == "gesture":
            left.append(
                Button(None, "OPEN HAND TO START · ARMS X TO STOP", kind="label", colour=DIM)
            )
        else:
            left.append(
                Button(None, "AUTO DETECT · 1/2/3 lock task · L/R leg", kind="label", colour=DIM)
            )
        right += _idle_or_recording_buttons(c, st, show_pause=True)
    else:
        task = c.choice_task if st == "confirm" else c.preset_task
        leg = c.choice_leg if st == "confirm" else c.preset_leg
        left.append(Button(None, "TASK" if task else "TASK · AUTO", kind="label"))
        for k in (1, 2, 3):
            left.append(
                Button(str(k), TASK_SHORT[k], str(k), "seg", selected=task == k, group="task")
            )
        left.append(Button(None, "", kind="gap"))
        if task == 3:
            left.append(Button(None, "BOTH FEET", kind="label"))
        else:
            left.append(Button(None, "LEG" if leg else "LEG · AUTO", kind="label"))
            for k, name in (("L", "Left"), ("R", "Right")):
                left.append(Button(k, name, k, "seg", selected=leg == k, group="leg"))
        if st == "confirm":
            right += [
                Button("BACKSPACE", "Cancel", "Backspace", "danger"),
                Button("ENTER", "Save", "Enter", "primary"),
            ]
        else:
            right += _idle_or_recording_buttons(c, st, show_pause=False)
    for pad in (12, 10, 8, 6):
        if _place(left, right, pad):
            break
    return [btn for btn in left + right if btn.kind != "gap"]


def _idle_or_recording_buttons(c, st: str, *, show_pause: bool = False) -> list[Button]:
    """Right group of the bottom strip while waiting (idle/ready) or recording."""
    right: list[Button] = []
    if st in ("idle", "ready"):
        if show_pause and c.mode in ("auto", "gesture"):
            paused = bool(getattr(c, "paused", False))
            what = "gesture" if c.mode == "gesture" else "auto"
            right.append(
                Button(
                    "P",
                    f"Resume {what}" if paused else f"Pause {what}",
                    "P",
                    selected=paused,
                    colour=AMBER,
                    dot=AMBER if paused else None,
                )
            )
        if c.consent == 3:
            on = bool(c.video)
            right.append(
                Button(
                    "V",
                    "Video on" if on else "Video off",
                    "V",
                    selected=on,
                    colour=RED,
                    dot=RED if on else FAINT,
                )
            )
        right += [
            Button("N", "Next child", "N"),
            Button("Q", "Quit", "Q", "danger"),
            Button("SPACE", "Start", "Space", "primary", GREEN),
        ]
    elif st == "countdown":
        right += [
            Button("BACKSPACE", "Cancel", "Backspace", "danger"),
            Button("SPACE", "Start now", "Space", "primary", GREEN),
        ]
    elif st == "recording":
        right += [
            Button("BACKSPACE", "Cancel", "Backspace", "danger"),
            Button("SPACE", "Stop", "Space", "primary", RED),
        ]
    return right


def hit(btns: list[Button], x, y) -> str | None:
    """The key of the button under (x, y), or None."""
    for btn in btns:
        if btn.contains(x, y):
            return btn.key
    return None


def _draw_primary_divider(cv: Canvas, btns: list[Button]) -> None:
    primary = next((b for b in btns if b.kind == "primary"), None)
    if primary is not None:
        left_neighbors = [b for b in btns if b.key is not None and b.x + b.w < primary.x]
        if left_neighbors:
            prev_end = max(b.x + b.w for b in left_neighbors)
            if primary.x - prev_end <= 24:
                sx = (prev_end + primary.x) // 2
                cv.rrect(sx, BAR_Y + 9, 1, BAR_H - 18, 0, fill=ACCENT, alpha=0.35)


def _draw_button(cv: Canvas, btn: Button, hovered: bool) -> None:
    cy = btn.y + btn.h / 2
    if btn.kind == "label":
        col = btn.colour if btn.colour != ACCENT else FAINT
        cv.text(btn.x, cy, btn.label, 14, "semibold", col, "lm", shadow=True)
        return

    is_primary = btn.kind == "primary"
    is_seg = btn.kind == "seg"
    is_danger = btn.kind == "danger"
    bx, by, bw, bh = btn.x, btn.y + 3, btn.w, btn.h - 6

    if is_primary:
        cv.frame(
            bx,
            by,
            bw,
            bh,
            8,
            btn.colour,
            fill=btn.colour,
            fill_alpha=0.30 if hovered else 0.16,
            line_alpha=1.0 if hovered else 0.8,
            glow=0.6 if hovered else 0.35,
        )
        fg = TEXT if hovered else _lighten(btn.colour, 0.3)
    elif is_seg:
        if hovered:
            cv.frame(
                bx, by, bw, bh, 6, ACCENT, fill=ACCENT, fill_alpha=0.10, line_alpha=0.4, glow=0
            )
        if btn.selected:
            cv.rrect(bx + 6, btn.y + btn.h - 5, bw - 12, 2, 0, fill=ACCENT, alpha=1.0)
            cv.rrect(bx + 6, btn.y + btn.h - 7, bw - 12, 6, 0, fill=ACCENT, alpha=0.15)
            fg = ACCENT
        else:
            fg = TEXT if hovered else DIM
    elif is_danger:
        line = 0.8 if hovered else 0.0
        cv.frame(
            bx,
            by,
            bw,
            bh,
            6,
            RED,
            fill=RED,
            fill_alpha=0.16 if hovered else 0.0,
            line_alpha=line,
            glow=0,
        )
        fg = RED if hovered else DIM
    else:
        if btn.selected:
            cv.frame(
                bx,
                by,
                bw,
                bh,
                6,
                btn.colour,
                fill=btn.colour,
                fill_alpha=0.22 if hovered else 0.12,
                line_alpha=0.7,
                glow=0,
            )
            fg = _lighten(btn.colour, 0.2)
        else:
            if hovered:
                cv.frame(
                    bx,
                    by,
                    bw,
                    bh,
                    6,
                    ACCENT,
                    fill=ACCENT,
                    fill_alpha=0.10,
                    line_alpha=0.5,
                    glow=0,
                )
            fg = TEXT if hovered else DIM

    x = btn.x + btn.pad
    if btn.cap:
        kw = max(18.0, measure(btn.cap, *_CAP) + 10)
        kh = 19
        cap_col = btn.colour if (is_primary or btn.selected) else ACCENT
        cv.frame(
            x,
            cy - kh / 2,
            kw,
            kh,
            4,
            cap_col,
            fill=BG,
            fill_alpha=0.6,
            line_alpha=0.75 if (is_primary or btn.selected or hovered) else 0.35,
            glow=0,
        )
        cv.text(
            x + kw / 2,
            cy,
            btn.cap,
            *_CAP,
            colour=fg if (is_primary or btn.selected or hovered) else DIM,
            anchor="cm",
        )
        x += kw + 7
    if btn.dot:
        cv.dot(x + 4, cy, 3.5, btn.dot)
        x += 12
    cv.text(x, cy, btn.label, *_LABEL, colour=fg, anchor="lm", shadow=True)


# ---------------------------------------------------------------------- video and floor plate


def _stage_transform(frame_size: tuple[float, float]) -> tuple[float, int, int, int, int]:
    """Return (scale, draw_w, draw_h, offset_x, offset_y) inside the video stage.

    The picture fits inside the window, except that it fills the height when that cuts at most
    MAX_SIDE_CROP of each side (a 16:9 camera on a 16:10 Mac screen); offset_x is then negative.
    """
    fw, fh = frame_size
    sx, sy = VW / fw, VH / fh
    s = min(sx, sy)
    if sx < sy and (1.0 - sx / sy) / 2 <= MAX_SIDE_CROP:
        s = sy
    dw, dh = int(round(fw * s)), int(round(fh * s))
    return s, dw, dh, (VW - dw) // 2, (VH - dh) // 2


def _to_canvas(runner, x: float, y: float) -> tuple[float, float]:
    """Unmirrored frame pixel -> mirrored window pixel."""
    fw, _fh = runner.frame_size
    s, _dw, _dh, ox, oy = _stage_transform(runner.frame_size)
    return VX + ox + (fw - x) * s, VY + oy + y * s


def _start_box(runner) -> tuple[tuple[float, float, float, float], bool]:
    """The start area in unmirrored frame pixels, and whether it was calibrated."""
    from live.start_area import default_start_area

    custom = getattr(runner, "start_area", None) is not None
    return (runner.start_area if custom else default_start_area(runner.frame_size)), custom


def plate_quad(runner) -> np.ndarray:
    """The floor plate in window pixels: (4, 2) far-left, far-right, near-right, near-left.

    Near (bottom) edge: the bottom of the start area (foot level), full width. Far edge:
    PLATE_DEPTH of the start-area height higher and PLATE_TOP_INSET narrower on each side.
    """
    from live.start_area import mirror_box_x

    raw, _custom = _start_box(runner)
    fw, _fh = runner.frame_size
    mx0, my0, mx1, my1 = mirror_box_x(raw, fw)
    s, _dw, _dh, ox, oy = _stage_transform(runner.frame_size)
    x0, x1 = VX + ox + mx0 * s, VX + ox + mx1 * s
    yb = VY + oy + my1 * s
    yt = yb - PLATE_DEPTH * (my1 - my0) * s
    inset = PLATE_TOP_INSET * (x1 - x0)
    return np.array([[x0 + inset, yt], [x1 - inset, yt], [x1, yb], [x0, yb]], np.float64)


def _floor_map(quad: np.ndarray):
    """Floor (u, v) in 0..1 -> window pixels (a perspective map of the unit square)."""
    import cv2

    unit = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    m = cv2.getPerspectiveTransform(unit, quad.astype(np.float32))

    def project(uv) -> np.ndarray:
        p = np.asarray(uv, np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, m).reshape(-1, 2).astype(np.float64)

    return project


def ankles_on_plate(runner, quad: np.ndarray | None = None) -> bool:
    """Both ankles (score >= READY_MIN_SCORE) inside the floor plate (window pixels)."""
    import cv2

    pts = getattr(runner, "last_points", None)
    if pts is None:
        return False
    quad = plate_quad(runner) if quad is None else quad
    contour = quad.astype(np.float32).reshape(-1, 1, 2)
    for i in (b.L_ANKLE, b.R_ANKLE):
        if float(pts[i, 2]) < READY_MIN_SCORE:
            return False
        x, y = _to_canvas(runner, float(pts[i, 0]), float(pts[i, 1]))
        if cv2.pointPolygonTest(contour, (float(x), float(y)), False) < 0:
            return False
    return True


def _floor_plate(cv: Canvas, runner, now: float) -> None:
    """The start area as a fake-3D plate lying on the floor (+ the ~50 % height guide)."""
    from live.start_area import standing_guide_lines

    quad = plate_quad(runner)
    if quad[2, 0] - quad[3, 0] < 8 or quad[2, 1] - quad[0, 1] < 4:
        return
    raw, custom = _start_box(runner)
    project = _floor_map(quad)
    ready = ankles_on_plate(runner, quad)
    col = GREEN if ready else ACCENT

    # One alpha mask for the fill, grid, outline and bracket glow, painted once (fast at 1080p).
    import cv2

    ih, iw = cv.img.shape[:2]
    pad = 8
    x0 = int(max(0, math.floor(quad[:, 0].min()) - pad))
    y0 = int(max(0, math.floor(quad[:, 1].min()) - pad))
    x1 = int(min(iw, math.ceil(quad[:, 0].max()) + pad))
    y1 = int(min(ih, math.ceil(quad[:, 1].max()) + pad))
    if x0 >= x1 or y0 >= y1:
        return
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)

    def draw(pts, a, thickness=1, closed=False):
        q = np.round((np.asarray(pts, np.float64) - (x0, y0)) * 4).astype(np.int32)
        v = int(round(255 * min(a, 1.0)))
        if thickness < 0:
            cv2.fillPoly(m, [q], v, cv2.LINE_AA, 2)
        else:
            cv2.polylines(m, [q], closed, v, thickness, cv2.LINE_AA, 2)

    draw(quad, 0.10 + (0.22 + 0.16 * _blink(now, 1.5) if ready else 0.0), -1)
    for k in range(1, PLATE_GRID):  # grid on the floor (fainter toward the far edge)
        f = k / PLATE_GRID
        draw(project([(0, f), (1, f)]), 0.25 + 0.25 * f)
        draw(project([(f, 0), (f, 1)]), 0.35)
    draw(quad, 0.6, 1, closed=True)
    draw(quad[2:4], 0.95, 2)  # near edge brighter
    arm = PLATE_BRACKET  # four corner brackets: right angles on the floor, projected
    brackets = []
    for u, v in ((0, 0), (1, 0), (1, 1), (0, 1)):
        su, sv = (1 if u == 0 else -1), (1 if v == 0 else -1)
        br = project([(u + su * arm, v), (u, v), (u, v + sv * arm)])
        brackets.append(br)
        draw(br, 0.45, 7)
    cv.paint(x0, y0, m.astype(np.float32)[..., None] * (1 / 255.0), col)
    for br in brackets:
        cv.poly(br, _lighten(col, 0.25), 1.0, closed=False, thickness=3)

    label = "Start area" if custom else "Start area (guide)"
    cv.text(quad[0, 0], quad[0, 1] - 8, label, 14, "semibold", col, "lb", shadow=True)
    if ready:
        cv.text(
            float(quad[1, 0]),
            float(quad[1, 1]) - 8,
            "READY",
            24,
            "display",
            GREEN,
            "rb",
            alpha=0.65 + 0.35 * _blink(now, 1.5),
            glow=0.6,
            neon=0.8,
        )

    fw, fh = runner.frame_size
    y_head, _ = standing_guide_lines((fw, fh), raw)
    _s, _dw, _dh, _ox, oy = _stage_transform(runner.frame_size)
    yh = VY + oy + y_head * _s
    if VY <= yh < VY + VH:
        x0, x1 = float(quad[3, 0]), float(quad[2, 0])
        for sx in np.arange(x0, x1, 18):
            cv.poly([(sx, yh), (min(sx + 10, x1), yh)], ACCENT, 0.6, closed=False)
        cv.text(
            0.5 * (x0 + x1),
            max(VY + 86, yh - 8),
            "Target ~50% height",
            14,
            "semibold",
            ACCENT,
            "cb",
            shadow=True,
        )


def _video(cv: Canvas, runner, view: str, now: float) -> None:
    import cv2

    area = cv.img[VY : VY + VH, VX : VX + VW]
    fw, _fh = runner.frame_size
    s, dw, dh, ox, oy = _stage_transform(runner.frame_size)
    if view == "neutral":
        area[:] = NEUTRAL
        return
    if view == "camera" and runner.last_frame is not None:
        if ox > 0 or oy > 0:  # bars around the picture
            area[:] = LETTERBOX
        pic = cv2.resize(cv2.flip(runner.last_frame, 1), (dw, dh), interpolation=cv2.INTER_LINEAR)
        x0, y0 = max(0, ox), max(0, oy)  # paste, cutting what falls outside the window
        w, h = min(dw - (x0 - ox), VW - x0), min(dh - (y0 - oy), VH - y0)
        area[y0 : y0 + h, x0 : x0 + w] = pic[y0 - oy : y0 - oy + h, x0 - ox : x0 - ox + w]
        area[:] = cv2.multiply(area, _vignette(), scale=1 / 255)
    else:
        area[:] = _backdrop()
    if runner.ctrl.state in ("readiness", "idle", "ready", "countdown"):
        _floor_plate(cv, runner, now)
    if runner.last_points is not None:
        pts = runner.last_points.copy()
        pts[:, 0] = np.where(pts[:, 2] > 0, fw - pts[:, 0], 0)
        draw_skeleton(area, pts, s, ox, oy)


# ---------------------------------------------------------------------- top bar and window frame


def _hud_item(
    cv: Canvas,
    x_right: float,
    cy: float,
    label: str,
    colour=TEXT,
    dot=None,
    size: int = 15,
    weight: str = "semibold",
) -> float:
    w = measure(label, size, weight) + (13 if dot else 0)
    x = x_right - w
    tx = x
    if dot:
        cv.dot(tx + 4, cy, 4, dot)
        tx += 13
    cv.text(tx, cy, label, size, weight, colour, "lm", shadow=True)
    return x


def _fps_colour(fps: float) -> tuple[int, int, int]:
    if fps >= 20.0:
        return GREEN
    if fps >= 15.0:
        return AMBER
    return RED


def _recording_seconds(runner) -> float:
    buf = getattr(runner, "buf", None)
    if not buf or len(buf) < 2:
        return 0.0
    return float(buf[-1][0] - buf[0][0])


def _top_bar(cv: Canvas, runner, now: float) -> None:
    c = runner.ctrl
    cv.fade(VX, VY, VW, 110, BG, 0.9, 0.0)
    x0, cy = VX + PAD, VY + 26
    cv.poly([(x0, cy - 9), (x0 + 9, cy - 9), (x0, cy)], ACCENT, 1.0, thickness=-1)
    title = "GROSS MOTOR SCREENING"
    w_title = cv.text(x0 + 16, cy, title, 15, "display", TEXT, "lm", shadow=True, neon=0.3)
    sub = "Screening aid · not a diagnosis"
    cv.text(x0 + 16, cy + 21, sub, 14, "medium", DIM, "lm", shadow=True)
    limit = x0 + 16 + max(w_title, measure(sub, 14, "medium")) + 24

    paused = c.state == "idle" and bool(getattr(c, "paused", False))
    if paused:
        colour, name = AMBER, "Gesture paused" if c.mode == "gesture" else "Auto paused"
        dot_col = AMBER
    else:
        colour = STATE_COLOURS.get(c.state, DIM)
        name = STATE_NAMES.get(c.state, c.state)
        if c.state == "recording":
            name = f"{name}  {_recording_seconds(runner):4.1f} s"
        pulse = c.state != "recording" or (now % 1.0) < 0.6
        dot_col = colour if pulse else TEXT

    chip_w = measure(name, 16, "bold") + 13 + 24
    chip_x = VX + VW - PAD - chip_w
    cv.frame(chip_x, cy - 14, chip_w, 28, 7, colour, fill=SURFACE, fill_alpha=0.7, glow=0.35)
    _hud_item(cv, VX + VW - PAD - 12, cy, name, colour, dot=dot_col, size=16, weight="bold")
    x = chip_x - 6

    items: list[tuple[str, tuple[int, int, int], tuple[int, int, int] | None, str]] = []
    if c.state != "recording":
        fps = runner.fps() if hasattr(runner, "fps") else 0.0
        mode = MODE_NAMES.get(c.mode, c.mode)
        if fps > 0:
            items.append((f"{fps:.0f} fps", _fps_colour(fps), None, "semibold"))
        items.append((mode, DIM if c.mode == "manual" else ACCENT, None, "medium"))
    if c.state != "setup":
        from live.quality import feet_visible

        feet_ok = feet_visible(getattr(runner, "last_points", None), runner.frame_size)
        if feet_ok:
            items.append(("Both feet visible", GREEN, GREEN, "semibold"))
        else:
            items.append(("Feet not visible", RED, RED, "semibold"))
    if c.child:
        if c.video:
            items.append(("Video on", RED, RED, "semibold"))
        if not age_in_range(c.age):
            lo, hi = AGE_RANGE_MONTHS
            items.append((f"Outside {lo}–{hi} months", AMBER, AMBER, "semibold"))
        if c.state != "recording":
            items.append((f"{c.tries_saved} saved", DIM, None, "medium"))
        items.append((f"{c.child} · {c.age} months · consent {c.consent}", TEXT, None, "medium"))

    for label, fg, dot, weight in items:
        w = measure(label, 15, weight) + (13 if dot else 0)
        if x - 20 - w < limit:
            break
        cv.text(x - 10, cy, "/", 15, "semibold", LINE, "cm", shadow=True)
        x = _hud_item(cv, x - 20, cy, label, fg, dot=dot, size=15, weight=weight)


def _window_frame(cv: Canvas, colour, alpha: float = 1.0) -> None:
    """The state colour as HUD brackets in the window corners plus a faint edge line."""
    cv.ring(VX, VY, VW, VH, 0, 2, colour, alpha * 0.35)
    arm = min(64, VW // 12)
    _brackets(cv, VX + 6, VY + 6, VW - 13, VH - 13, arm, colour, th=3, alpha=alpha)


# ---------------------------------------------------------------------- setup / readiness / confirm


def _panel(cv: Canvas, x, y, w, h, alpha=0.90, colour=ACCENT):
    cv.paint(int(x) + 4, int(y) + 8, _chamfer_mask(int(w), int(h), 18) * 0.5, SHADOW)
    cv.frame(x, y, w, h, 18, colour, fill=SURFACE, fill_alpha=alpha, line_alpha=0.85, glow=0.5)
    _brackets(cv, x - 6, y - 6, w + 12, h + 12, 22, colour, corners=("tr", "bl"), th=2)


def _setup_field_boxes(c) -> tuple[int, int, int, int, list[tuple[int, int, int, int]], list[str]]:
    w = 500
    msg_lines = wrap(c.message, 15, "regular", w - 56, 2)
    h = 28 + 34 + 26 + 18 + 3 * 82 + len(msg_lines) * 20 + 24
    x, y = VX + (VW - w) // 2, VY + (VH - h) // 2
    cx = x + 28
    yy = y + 28 + 38 + 36
    bw = w - 56
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(3):
        yy += 22
        boxes.append((cx, yy, bw, 46))
        yy += 46 + 14
    return x, y, w, h, boxes, msg_lines


def _setup_field_hit(c, mx: int, my: int) -> str | None:
    if c.state != "setup":
        return None
    _, _, _, _, boxes, _ = _setup_field_boxes(c)
    for i, (bx, by, bw, bh) in enumerate(boxes):
        if bx <= mx <= bx + bw and by <= my <= by + bh:
            return f"FIELD_{i}"
    return None


def _setup_card(cv: Canvas, c, now: float) -> None:
    names = ("Child code", "Age in months", "Consent level")
    lo, hi = AGE_RANGE_MONTHS
    hints = ("e.g. K03", f"{lo}–{hi}", "1 points only · 2 stick figure · 3 + video")
    x, y, w, h, boxes, msg_lines = _setup_field_boxes(c)
    _panel(cv, x, y, w, h, 0.94)
    cx = x + 28
    yy = y + 28
    cv.text(cx, yy, "New child", 28, "bold", TEXT, neon=0.2)
    yy += 38
    cv.text(
        cx,
        yy,
        "Enter moves to the next field · Backspace or click a box to go back.",
        15,
        "regular",
        DIM,
    )
    yy += 36
    caret_on = (now % 1.0) < 0.6
    for i, name in enumerate(names):
        focused = i == c.field
        cv.text(cx, yy, name.upper(), 13, "bold", ACCENT if focused else DIM)
        if i == 2:
            cv.text(x + w - 28, yy, "1 points · 2 stick · 3 video", 14, "regular", FAINT, "rt")
        bx, by, bw, bh = boxes[i]
        yy = by
        cv.frame(
            bx,
            by,
            bw,
            bh,
            9,
            ACCENT if focused else LINE,
            fill=RAISED,
            fill_alpha=0.95,
            line_alpha=1.0,
            glow=0.4 if focused else 0.0,
        )
        value = c.form[i]
        tx = bx + 14
        if value:
            tx += cv.text(tx, by + 23, value, 21, "semibold", TEXT, "lm")
        elif not focused:
            cv.text(tx, by + 23, hints[i], 17, "regular", FAINT, "lm")
        if focused:
            if caret_on:
                cv.rrect(tx + 2, by + 12, 2, 22, 0, fill=ACCENT)
            if not value:
                cv.text(tx + 10, by + 23, hints[i], 17, "regular", FAINT, "lm")
        yy = by + bh + 14
    error = c.message and not c.message.startswith(("Type", "Next child"))
    for line in msg_lines:
        cv.text(cx, yy, line, 15, "regular", AMBER if error else DIM)
        yy += 20


def _icon_ok(cv: Canvas, cx, cy):
    import cv2

    cv.dot(cx, cy, 9, GREEN)
    pts = np.array([[cx - 4.5, cy], [cx - 1.2, cy + 3.5], [cx + 4.5, cy - 3.5]]) * 4
    cv2.polylines(cv.img, [pts.astype(np.int32)], False, BG, 2, cv2.LINE_AA, 2)


def _icon_warn(cv: Canvas, cx, cy):
    cv.poly([(cx, cy - 9), (cx + 9, cy + 7), (cx - 9, cy + 7)], AMBER, 1.0, thickness=-1)
    cv.text(cx, cy + 1, "!", 13, "bold", BG, "cm")


def _readiness_overlay(cv: Canvas, runner) -> None:
    c = runner.ctrl
    custom = getattr(runner, "start_area", None) is not None
    w = 460
    rows = [wrap(p, 16, "regular", w - 40, 3) for p in c.problems]
    if c.mode == "gesture":
        foot = "Show an open hand to the camera (or press Start, Enter) to start."
    elif not c.problems:
        foot = "Press Start (Enter) when the child is at the start line."
    else:
        foot = "You can start anyway with Start (Enter)."
    foot_lines = wrap(foot, 15, "medium", w)
    tip_lines = (
        []
        if custom
        else wrap(
            "Tip: run 'python -m live.start_area' to calibrate the taped floor box.",
            13,
            "regular",
            w,
        )
    )
    body_h = 28 if not rows else sum(len(r) * 21 + 10 for r in rows)
    total_h = 34 + 14 + body_h + 12 + len(foot_lines) * 20 + (len(tip_lines) * 18 + 8)
    cv.fade_x(0, 48, w + 160, min(H - 96, total_h + 112), BG, 0.86, 0.0, v_feather=64)

    cx, yy = VX + PAD, VY + 92
    cv.rrect(cx, yy + 2, 3, 22, 0, fill=ACCENT)
    cv.text(cx + 12, yy, "Readiness check", 24, "bold", TEXT, shadow=True, neon=0.2)
    yy += 40
    if not rows:
        _icon_ok(cv, cx + 9, yy + 9)
        cv.text(
            cx + 28,
            yy + 9,
            "Camera and child look ready.",
            16,
            "semibold",
            GREEN,
            "lm",
            shadow=True,
        )
        yy += 30
    for lines in rows:
        _icon_warn(cv, cx + 9, yy + 9)
        for line in lines:
            cv.text(cx + 28, yy + 9, line, 16, "regular", TEXT, "lm", shadow=True)
            yy += 21
        yy += 10
    yy += 6
    for line in foot_lines:
        cv.text(cx, yy, line, 15, "medium", DIM, shadow=True)
        yy += 20
    if tip_lines:
        yy += 6
        for line in tip_lines:
            cv.text(cx, yy, line, 13, "regular", FAINT, shadow=True)
            yy += 18


def _task_leg(task: int, leg) -> str:
    """'Task 1 · stand on one leg · leg L'; task 3 (jump with both feet) has no leg."""
    text = f"Task {task} · {b.TASKS[task]}"
    return text if task == 3 else f"{text} · leg {leg}"


def _confirm_card(cv: Canvas, c) -> None:
    if not c.guess:
        return
    gt, gl, gc = c.guess
    ct, cl = c.choice_task, c.choice_leg
    corrected = ct != gt or (ct != 3 and cl != gl)
    w, h = 520, 268
    x, y = VX + (VW - w) // 2, VY + (VH - h) // 2
    _panel(cv, x, y, w, h, 0.94, AMBER)
    cx, yy = x + 28, y + 28
    cv.text(cx, yy, "Check this try", 26, "bold", TEXT)
    yy += 36
    cv.text(cx, yy, "Correct the task or leg if the guess is wrong, then save.", 15, "regular", DIM)
    yy += 38
    cv.text(cx, yy, "GUESS", 13, "bold", DIM)
    cv.text(x + w - 28, yy, f"{gc:.0%} sure", 14, "semibold", DIM, "rt")
    yy += 22
    cv.text(cx, yy, _task_leg(gt, gl), 18, "medium", TEXT)
    yy += 40
    cv.text(cx, yy, "WILL BE SAVED AS", 13, "bold", ACCENT)
    if corrected:
        cv.text(x + w - 28, yy, "Corrected", 14, "bold", AMBER, "rt")
    yy += 22
    cv.text(cx, yy, _task_leg(ct, cl), 22, "bold", TEXT)
    yy += 44
    keys = "1/2/3 task · " + ("" if ct == 3 else "L/R leg · ") + "Enter save · Backspace cancel"
    cv.text(cx, yy, keys, 14, "regular", FAINT)


def _toast(cv: Canvas, message: str) -> None:
    if not message:
        return
    lines = wrap(message, 16, "semibold", 820, 2)
    yy = BAR_Y - 12 - (len(lines) - 1) * 20
    for line in lines:
        cv.text(VX + VW / 2, yy, line, 16, "semibold", TEXT, "cb", shadow=True, glow=0.6)
        yy += 20


# ---------------------------------------------------------------------- gesture ring and countdown


def _arc(cv: Canvas, cx, cy, r, frac: float, colour, thickness: int = 4, start=-90.0) -> None:
    """Clockwise arc from 12 o'clock covering frac (0..1) of a circle."""
    import cv2

    if frac <= 0:
        return
    c = (int(round(cx * 4)), int(round(cy * 4)))
    ax = (int(round(r * 4)), int(round(r * 4)))
    cv2.ellipse(cv.img, c, ax, start, 0, 360 * min(frac, 1.0), colour, thickness, cv2.LINE_AA, 2)


def _gesture_ring(cv: Canvas, runner) -> None:
    """A ring on the raised hand (palm) or the crossed arms (X) that fills while it is held."""
    import cv2

    g = getattr(runner, "gesture", None)
    if g is None or g.kind is None or g.anchor is None:
        return
    if runner.ctrl.state not in ("readiness", "idle", "ready", "recording"):
        return
    x, y = _to_canvas(runner, *g.anchor)
    col = GREEN if g.kind == "palm" else RED
    c4 = (int(round(x * 4)), int(round(y * 4)))
    cv2.circle(cv.img, c4, 30 * 4, SHADOW, 6, cv2.LINE_AA, 2)
    cv2.circle(cv.img, c4, 30 * 4, ACCENT if not g.waiting_for_open else AMBER, 1, cv2.LINE_AA, 2)
    _arc(cv, x, y, 30, g.progress, col, 5)
    label = g.hint
    cv.text(x, y + 46, label, 15, "bold", AMBER if g.waiting_for_open else TEXT, "ct", shadow=True)


def _segment_ring(cv: Canvas, cx, cy, r, frac: float, colour, n: int = 36) -> None:
    """A HUD ring of n segments; the first frac of them (clockwise from 12) are lit."""
    lit = math.ceil(frac * n - 1e-6)
    step = 360.0 / n
    for i in range(n):
        start = -90.0 + i * step
        on = i < lit
        _arc(cv, cx, cy, r, (step - 3.5) / 360.0, colour if on else LINE, 9 if on else 5, start)


def _countdown_overlay(cv: Canvas, runner, now: float) -> None:
    """Glitching neon 3-2-1 in the middle of the window, in a segmented ring."""
    left = runner.countdown_left() if hasattr(runner, "countdown_left") else None
    if left is None:
        return
    n = max(1, min(3, math.ceil(left - 1e-6))) if left > 0 else 1
    frac = _clip01(left - (n - 1))  # 1 -> 0 during each digit
    p = 1.0 - frac  # time into this digit
    cx, cy, r = VX + VW / 2, VY + VH / 2 - 20, 132
    disc = _rr_mask(2 * r + 40, 2 * r + 40, r + 20)
    cv.paint(int(cx - r - 20), int(cy - r - 20), disc * 0.62, BG)
    _arc(cv, cx, cy, r + 16, 1.0, ACCENT, 1)
    for k in range(12):  # outer ticks
        a = math.radians(k * 30 - 90 + (now * 40) % 30)
        x0, y0 = cx + (r + 22) * math.cos(a), cy + (r + 22) * math.sin(a)
        x1, y1 = cx + (r + 30) * math.cos(a), cy + (r + 30) * math.sin(a)
        cv.poly([(x0, y0), (x1, y1)], ACCENT, 0.8, closed=False, thickness=2)
    _segment_ring(cv, cx, cy, r, frac, ACCENT)
    _arc(cv, cx, cy, r - 16, 1.0, LINE, 1)

    glitch = 1.0 - _seg(p, 0.0, 0.15)  # glitch in at the start of each digit
    alpha = 0.35 + 0.65 * _seg(p, 0.0, 0.12)
    digit = str(n)
    size = 170
    cv.text(cx, cy, digit, size, "display", ACCENT, "cm", alpha=alpha, neon=1.0, glow=0.5)
    cv.text(cx, cy, digit, size, "display", TEXT, "cm", alpha=alpha * 0.85, record=False)
    if glitch > 0:
        gw = measure(digit, size, "display") + 60
        _glitch(cv, cx - gw / 2, cy - size * 0.6, gw, size * 1.2, glitch, int(now * 30) + n)
    cv.text(cx, cy + r + 40, "Get ready", 26, "bold", ACCENT, "ct", shadow=True, neon=0.5)
    cv.text(
        cx,
        cy + r + 74,
        "Recording starts at 0 · Backspace cancels",
        15,
        "medium",
        DIM,
        "ct",
        shadow=True,
    )


def _go_overlay(cv: Canvas, a: _Anim, now: float) -> None:
    p = (now - a.go_since) / GO_S
    if not 0 <= p < 1:
        return
    size = int(120 + 70 * _ease_out(p))
    alpha = 1.0 - _seg(p, 0.55, 1.0)
    cx, cy = VX + VW / 2, VY + VH / 2 - 20
    cv.text(cx, cy, "GO", size, "display", GREEN, "cm", alpha=alpha, neon=1.0, glow=0.5)
    if p < 0.2:
        gw = measure("GO", size, "display") + 60
        _glitch(cv, cx - gw / 2, cy - size * 0.6, gw, size * 1.2, 1 - p / 0.2, int(now * 30))


# ---------------------------------------------------------------------- task panel (detection)


def _task_content(runner, show_score: bool) -> tuple | None:
    """(task or None, sub line, colour) for the task panel, or None when it is not shown.

    Testing mode (show_score False) uses only the Controller: the task set by key, else
    'TASK ?'. It never reads runner.detect or runner.live."""
    c = runner.ctrl
    frames = runner.splitter.current() if getattr(runner, "splitter", None) else None
    in_attempt = c.state == "recording" or bool(frames)
    if c.preset_task is not None and c.state in ("idle", "ready", "recording"):
        leg = f" · leg {c.preset_leg}" if c.preset_leg and c.preset_task != 3 else ""
        return c.preset_task, f"Set by key{leg}", AMBER
    if not in_attempt:
        return None
    if not show_score:
        return None, "AUTO", ACCENT
    detect = getattr(runner, "detect", None)
    if detect:
        return int(detect["task"]), f"Detected · {float(detect['confidence']):.0%}", ACCENT
    live = getattr(runner, "live", None)
    if live and live.get("task") and "score" in live:
        return int(live["task"]), "Detected", ACCENT
    return None, "Detecting…", ACCENT


TASK_PANEL_W, TASK_PANEL_H = 300, 104


def _task_panel_rect() -> tuple[int, int, int, int]:
    return VX + VW - PAD - TASK_PANEL_W, VY + 66, TASK_PANEL_W, TASK_PANEL_H


def _task_panel_text(cv: Canvas, content: tuple, x, y, w, alpha: float, record: bool) -> None:
    task, sub, col = content
    head = f"TASK {task}" if task else "TASK ?"
    name = b.TASKS[task].upper() if task else "WAITING FOR MOVEMENT"
    cv.text(x + 20, y + 16, head, 34, "display", TEXT, "lt", alpha, neon=0.6, record=record)
    cv.text(x + 20, y + 60, name, 17, "bold", ACCENT, "lt", alpha, record=record)
    cv.text(x + w - 16, y + 84, sub, 14, "semibold", col, "rt", alpha, record=record)


def _task_panel(cv: Canvas, runner, show_score: bool, a: _Anim, now: float) -> int | None:
    """Top right: TASK n big and its name; glitch swap when the task changes. Returns its bottom."""
    content = _task_content(runner, show_score)
    key = None if content is None else (content[0], content[1].split(" ·")[0])
    if key != a.task_key:
        a.task_old = a.task_content if a.task_key is not None else None
        a.task_key, a.task_since = key, now
    a.task_content = content
    if content is None:
        return None
    x, y, w, h = _task_panel_rect()
    cv.frame(x, y, w, h, 14, ACCENT, fill=SURFACE, fill_alpha=0.72, glow=0.5)
    _brackets(cv, x - 5, y - 5, w + 10, h + 10, 16, ACCENT, corners=("tr", "bl"), th=2)
    cv.rrect(x + 9, y + 20, 3, 32, 0, fill=ACCENT, alpha=0.9)
    p = (now - a.task_since) / TASK_SWAP_S
    if p < 0.5 and a.task_old is not None:
        _task_panel_text(cv, a.task_old, x, y, w, 1.0 - _seg(p, 0.0, 0.5), record=False)
        _glitch(cv, x + 4, y + 4, w - 8, h - 8, 1.0 - p, int(now * 30))
        _task_panel_text(cv, content, x, y, w, 0.0, record=True)
    else:
        _task_panel_text(cv, content, x, y, w, _flicker(_seg(p, 0.5, 1.0)), record=True)
    return y + h


# ---------------------------------------------------------------------- evaluation bar


def _eval_bar(cv: Canvas, runner, a: _Anim, now: float) -> None:
    """Left edge: stacked Perfect / Good / Bad heights = live probabilities (show_score only)."""
    live = getattr(runner, "live", None) or {}
    proba = live.get("proba")
    target = None if proba is None else np.asarray(proba, np.float64)
    dt = 0.0 if a.t_last is None else now - a.t_last
    if target is not None:
        a.bar = target.copy() if a.bar is None else _approach(a.bar, target, dt)
    bw, bh = 16, int(VH * 0.45)
    x, y = VX + PAD + 4, VY + (VH - bh) // 2
    cv.frame(x - 5, y - 5, bw + 10, bh + 10, 6, ACCENT, fill=BG, fill_alpha=0.75, glow=0.4)
    cv.text(x + bw / 2, y - 14, "EVAL", 12, "bold", ACCENT, "cb", shadow=True)
    if a.bar is None or target is None or float(np.sum(a.bar)) <= 1e-6:
        msg = live.get("error") if live else None
        cv.text(
            x + bw + 14, y + bh / 2, "WAITING FOR A CLEAR VIEW", 13, "bold", DIM, "lm", shadow=True
        )
        if msg:
            cv.text(x + bw + 14, y + bh / 2 + 18, msg, 12, "regular", FAINT, "lm", shadow=True)
        return
    p = np.clip(a.bar, 0, None)
    p = p / max(float(p.sum()), 1e-6)
    hb, hg = bh * p[0], bh * p[1]
    hp = bh - hb - hg
    cv.rrect(x, y, bw, hp, 0, fill=GREEN, alpha=0.95)
    cv.rrect(x, y + hp, bw, hg, 0, fill=LIME, alpha=0.95)
    cv.rrect(x, y + hp + hg, bw, hb, 0, fill=RED, alpha=0.95)
    for yy in (y + hp, y + hp + hg):
        cv.rrect(x, yy - 1, bw, 2, 0, fill=BG, alpha=0.9)
    for k in range(1, 4):  # quarter ticks
        cv.rrect(x + bw + 2, y + bh * k / 4, 5, 1, 0, fill=ACCENT, alpha=0.6)
    lead = int(np.argmax(target))
    col = SCORE_COLOURS[lead]
    centres = (y + hp + hg + hb / 2, y + hp + hg / 2, y + hp / 2)
    ly = float(min(max(centres[lead], y + 14), y + bh - 14))
    cv.poly(
        [(x + bw + 8, ly), (x + bw + 15, ly - 6), (x + bw + 15, ly + 6)], col, 1.0, thickness=-1
    )
    cv.text(x + bw + 20, ly, f"{target[lead]:.0%}", 22, "bold", col, "lm", glow=0.8, neon=0.5)
    cv.text(x + bw + 20, ly + 20, SCORE_NAMES[lead].upper(), 12, "bold", col, "lm", glow=0.8)


# ---------------------------------------------------------------------- result box and sidebar


def _verdict(r: dict) -> tuple[str, tuple[int, int, int]]:
    if r.get("check") or r.get("score") is None:
        return "Check by teacher", PURPLE
    sc = int(r["score"])
    return f"{sc} {SCORE_NAMES[sc]}", SCORE_COLOURS[sc]


def _fade(runner, item: dict, t_in: float = 0.25, t_out: float = 0.45) -> float:
    """0..1: how much of a timed overlay ({"since", "until"} in stream seconds) to show now."""
    now = getattr(runner, "now", None)
    if now is None or "until" not in item:
        return 1.0
    a = (item["until"] - now) / t_out
    if "since" in item:
        a = min(a, (now - item["since"]) / t_in)
    return float(min(max(a, 0.0), 1.0))


def _score_words(r: dict) -> tuple[str, str, tuple[int, int, int]]:
    """(big word, small line, colour) of one attempt: "Perfect" / "Score 2 · 83% sure"."""
    if r.get("check") or r.get("score") is None:
        return "Check by teacher", "The model is unsure", PURPLE
    sc = int(r["score"])
    return (
        SCORE_NAMES[sc],
        f"Score {sc} · {float(r.get('confidence') or 0):.0%} sure",
        SCORE_COLOURS[sc],
    )


def _observations(reasons) -> list[str]:
    """2-3 short lines from the explain() reasons (a preview, not the report)."""
    parts = []
    for chunk in str(reasons or "").replace(";", ",").split(","):
        chunk = chunk.strip().rstrip(".")
        if chunk:
            parts.append(chunk[0].upper() + chunk[1:])
    if not parts:
        parts = ["No observations yet", "The full timeline is in the saved try"]
    return parts[:3]


@dataclass
class _Box:
    """One result for the centred box (and its sidebar row)."""

    task: int
    word: str = ""
    word_col: tuple = ACCENT
    small: str = ""
    headline: str = ""
    tag: tuple | None = None  # (text, colour, dot)
    obs: list = field(default_factory=list)


def _box_from_result(r: dict, show_score: bool) -> _Box:
    task = int(r["task"])
    if not show_score:
        return _Box(task, "SAVED", ACCENT, "", "", None, [])
    word, small, col = _score_words(r)
    tag = None
    if "recorded" in r:
        tag = ("Recorded", RED, True) if r.get("recorded") else ("Not recorded", DIM, False)
    return _Box(task, word, col, small, r.get("headline", ""), tag, _observations(r.get("reasons")))


BOX_W = 560


def _box_size(box: _Box) -> tuple[int, int]:
    h = 28 + 44 + 30 + 60 + 26
    if box.headline or box.tag:
        h += 24
    if box.obs:
        h += 30 + 22 * len(box.obs)
    return BOX_W, h + 18


def _box_content(cv: Canvas, box: _Box, x, y, w, alpha: float) -> None:
    cx = x + w / 2
    yy = y + 28
    cv.text(cx, yy, f"TASK {box.task}", 34, "display", TEXT, "ct", alpha, neon=0.6, glow=0.4)
    yy += 44
    cv.text(cx, yy, b.TASKS[box.task].upper(), 18, "bold", ACCENT, "ct", alpha)
    yy += 30
    big = 50 if len(box.word) < 10 else 34
    cv.text(cx, yy + 28, box.word, big, "display", box.word_col, "cm", alpha, neon=0.9, glow=0.4)
    yy += 60
    if box.small:
        cv.text(cx, yy, box.small, 17, "semibold", TEXT, "ct", alpha)
    yy += 26
    if box.headline or box.tag:
        parts = []
        if box.headline:
            parts.append(box.headline)
        line = "  ·  ".join(parts)
        tw = measure(line, 15, "medium") if line else 0
        tag_w = 0
        if box.tag:
            tag_w = (
                measure(box.tag[0], 14, "bold") + (14 if box.tag[2] else 0) + (16 if line else 0)
            )
        tx = cx - (tw + tag_w) / 2
        if line:
            cv.text(tx, yy, line, 15, "medium", DIM, "lt", alpha)
            tx += tw + 16
        if box.tag:
            text, col, dot = box.tag
            if dot and alpha > 0:
                cv.dot(tx + 4, yy + 10, 4, col)
            cv.text(tx + (14 if dot else 0), yy, text, 14, "bold", col, "lt", alpha)
        yy += 24
    if box.obs:
        yy += 6
        cv.rrect(x + 28, yy + 8, w - 56, 1, 0, fill=LINE, alpha=alpha)
        cv.text(x + 28, yy - 2, "OBSERVATIONS", 13, "bold", ACCENT, "lt", alpha)
        cv.text(x + w - 28, yy - 2, "preview", 13, "semibold", FAINT, "rt", alpha)
        yy += 22
        for line in box.obs:
            for part in wrap(line, 15, "medium", w - 80, 1):
                cv.text(x + 34, yy, "›", 15, "bold", ACCENT, "lt", alpha)
                cv.text(x + 50, yy, part, 15, "medium", DIM, "lt", alpha)
            yy += 22


def _result_box(cv: Canvas, box: _Box, p_in: float, q_out: float, target, now: float) -> None:
    """The big centred box: opens (p_in 0..1), closes and flies to target (q_out 0..1)."""
    w, h = _box_size(box)
    x, y = VX + (VW - w) / 2, VY + (VH - h) / 2 - 20
    cx, cy = x + w / 2, y + h / 2
    if q_out <= 0:
        o = _ease_out(_seg(p_in, 0.0, 0.42))  # scan-line strip opens from the centre
        k = _ease_out(_seg(p_in, 0.17, 0.58))  # brackets slide out to the corners
        blink = _seg(p_in, 0.5, 0.75)
        line_a = 1.0 if blink in (0.0, 1.0) else (1.0 if int(blink * 4) % 2 else 0.2)
        content = _flicker(_seg(p_in, 0.67, 1.0))
        hh = max(4.0, h * o)
        fx, fy, fw, fh = x, cy - hh / 2, w, hh
        fill_a = 0.86
        alpha_all = 1.0
    else:
        blink = _seg(q_out, 0.0, 0.2)
        line_a = 1.0 if blink in (0.0, 1.0) else (1.0 if int(blink * 4) % 2 else 0.2)
        k = 1.0 - _ease(_seg(q_out, 0.2, 0.5))  # brackets slide toward each other and close
        content = 1.0 - _seg(q_out, 0.1, 0.3)
        f = _ease(_seg(q_out, 0.5, 1.0))  # collapse and fly into the sidebar
        tx, ty, tw, th = target
        hh = h + (th - h) * f
        fw = w + (tw - w) * f
        fx = x + (tx - x) * f
        fy = (cy - hh / 2) + (ty - (cy - th / 2)) * f
        fh = hh
        fill_a = 0.86
        alpha_all = 1.0 - 0.4 * f
    cv.frame(
        fx,
        fy,
        fw,
        fh,
        18,
        ACCENT,
        fill=SURFACE,
        fill_alpha=fill_a * alpha_all,
        line_alpha=line_a * alpha_all,
        glow=0.6 * line_a * alpha_all,
    )
    if q_out <= 0 and p_in < 0.5:  # the opening strip shows bright scan lines
        for yy in np.arange(fy + 2, fy + fh - 2, 4):
            cv.rrect(fx + 6, yy, fw - 12, 1, 0, fill=ACCENT, alpha=0.18)
    # two brackets: top-left and bottom-right, sliding between the centre and the corners
    arm = 34 * max(k, 0.15)
    bl = (cx + (x - 10 - cx) * k, cy + (y - 10 - cy) * k)
    br = (cx + (x + w + 10 - cx) * k, cy + (y + h + 10 - cy) * k)
    if q_out < 0.5 and k > 0.02:
        cv.poly([(bl[0] + arm, bl[1]), bl, (bl[0], bl[1] + arm)], ACCENT, line_a, False, 3)
        cv.poly([(br[0] - arm, br[1]), br, (br[0], br[1] - arm)], ACCENT, line_a, False, 3)
    _box_content(cv, box, x, y, w, content if q_out <= 0 or content > 0 else 0.0)


SIDEBAR_ROW_H = 54


def _sidebar_geom(top: int) -> tuple[int, int, int]:
    """(x, y of the first row, width) of the right sidebar."""
    w = CARD_W
    return VX + VW - PAD - w, top + 30, w


def _history_rows(
    cv: Canvas,
    rows: list[dict],
    title: str,
    top: int,
    flash: float,
    extra: list[str],
    reserve: bool = False,
) -> None:
    """Right sidebar: one cyber row per attempt, newest on top; the newest flashes on landing.

    reserve: keep the top slot free for the result box that is flying in."""
    x, y, w = _sidebar_geom(top)
    cv.fade_x(x - 140, top - 20, w + PAD + 140, H - top - 60, BG, 0.0, 0.8, v_feather=64)
    cv.rrect(x, top + 4, 3, 14, 0, fill=ACCENT)
    cv.text(x + 10, top + 11, title, 13, "bold", ACCENT, "lm", shadow=True)
    for line in extra:
        cv.text(x, y + 4, line, 14, "semibold", DIM, "lt", shadow=True)
        y += 22
    if reserve:
        y += SIDEBAR_ROW_H
    for i, r in enumerate(reversed(rows)):
        if y + SIDEBAR_ROW_H > BAR_Y - 40:
            break
        task = int(r["task"])
        word, _small, col = _score_words(r)
        fl = flash if i == 0 else 0.0
        line_col = _mix(ACCENT, TEXT, fl)
        cv.frame(
            x,
            y,
            w,
            SIDEBAR_ROW_H - 8,
            10,
            line_col,
            fill=_mix(SURFACE, ACCENT, 0.35 * fl),
            fill_alpha=0.78,
            line_alpha=0.5 + 0.5 * fl,
            glow=0.2 + 0.8 * fl,
        )
        cv.rrect(x + 1, y + 10, 3, SIDEBAR_ROW_H - 28, 0, fill=col)
        cv.text(x + 14, y + 15, f"T{task} {TASK_SHORT[task].upper()}", 17, "bold", TEXT, "lm")
        cv.text(x + w - 12, y + 15, word.upper(), 17, "bold", col, "rm", neon=0.3)
        when = r.get("headline") or (f"{float(r['duration_s']):.1f} s" if "duration_s" in r else "")
        if not when and r.get("try"):
            when = f"Try #{int(r['try'])}"
        cv.text(x + 14, y + 34, when, 13, "medium", DIM, "lm")
        if r.get("try") and r.get("headline"):
            cv.text(x + w - 12, y + 34, f"Try #{int(r['try'])}", 13, "medium", FAINT, "rm")
        y += SIDEBAR_ROW_H
    cv.text(x, y + 4, "Reference only, not a diagnosis.", 13, "medium", FAINT, shadow=True)


def _summary_moment(cv: Canvas, summary: dict, a: float) -> None:
    """After the X: the saved attempts over the dimmed picture, in a HUD frame."""
    if a <= 0:
        return
    rows = summary.get("rows", [])
    cv.dim(VX, VY, VW, VH, 1.0 - 0.7 * a)
    cx = VX + VW / 2
    n = len(rows)
    total = 92 + max(n, 1) * 64 + 40
    y = VY + (VH - total) / 2
    fw = 640
    cv.frame(
        cx - fw / 2,
        y - 30,
        fw,
        total + 40,
        20,
        ACCENT,
        fill=SURFACE,
        fill_alpha=0.7 * a,
        line_alpha=a,
        glow=0.5 * a,
    )
    if not rows:
        cv.text(cx, y + 20, "No task found in this recording", 30, "bold", TEXT, "cm", alpha=a)
        cv.text(
            cx,
            y + 62,
            "Nothing was saved. Show an open hand to try again.",
            17,
            "regular",
            DIM,
            "cm",
            alpha=a,
        )
        return
    cv.text(cx, y + 20, "Recording saved", 36, "bold", TEXT, "cm", alpha=a, neon=0.4)
    cv.text(
        cx,
        y + 56,
        f"{n} {'try' if n == 1 else 'tries'} · one per task",
        16,
        "medium",
        DIM,
        "cm",
        alpha=a,
        shadow=True,
    )
    y += 104
    for r in rows:
        task = int(r["task"])
        word, small, col = _score_words(r)
        title = f"Task {task} · {b.TASKS[task]}"
        cv.text(cx - 22, y, title, 21, "bold", TEXT, "rm", alpha=a, glow=0.7)
        cv.text(cx - 22, y + 24, r.get("headline", ""), 15, "regular", DIM, "rm", alpha=a, glow=0.7)
        cv.text(cx + 22, y, word, 26, "bold", col, "lm", alpha=a, glow=0.7, neon=0.4)
        cv.text(cx + 22, y + 24, small, 15, "regular", DIM, "lm", alpha=a, glow=0.7)
        y += 64
    cv.text(cx, y + 10, "Reference only, not a diagnosis.", 13, "regular", FAINT, "cm", alpha=a)


# ---------------------------------------------------------------------- the HUD of one picture


def _same_bout(r: dict, card: dict | None) -> bool:
    return card is not None and r.get("bout") is not None and r.get("bout") is card.get("bout")


def _box_phase_stream(runner, card: dict) -> tuple[float, float]:
    now = float(getattr(runner, "now", 0.0))
    p_in = (now - card["since"]) / BOX_IN_S
    q_out = 1.0 - (card["until"] - now) / BOX_OUT_S
    return _clip01(p_in), _clip01(q_out)


def _box_phase_clock(since: float, now: float) -> tuple[float, float] | None:
    t = now - since
    if t < 0 or t >= BOX_SHOW_S:
        return None
    return _clip01(t / BOX_IN_S), _clip01(1.0 - (BOX_SHOW_S - t) / BOX_OUT_S)


def _flash_amount(dt: float) -> float:
    """Row flash 0..1 for dt seconds after the result landed."""
    if not 0 <= dt < ROW_FLASH_S:
        return 0.0
    return (1.0 - dt / ROW_FLASH_S) * (1.0 if int(dt / 0.1) % 2 == 0 else 0.5)


def _gesture_hud(cv: Canvas, runner, a: _Anim, now: float, top: int) -> None:
    """show_score in gesture mode: result box, attempts of the recording, summary."""
    c = runner.ctrl
    summary = runner.summary_card()
    if summary is not None:
        _summary_moment(cv, summary, _fade(runner, summary))
        return
    card = runner.card()
    if c.state == "recording":
        rows, title = list(runner.rec_bouts), "THIS RECORDING"
    else:
        rows, title = list(runner.preview_bouts), "DETECTED · NOT RECORDED"
    shown = [r for r in rows if not _same_bout(r, card)]
    last = getattr(runner, "intermezzo", None)
    flash = 0.0
    if card is None and last is not None and shown and _same_bout(shown[-1], last):
        flash = _flash_amount(float(getattr(runner, "now", 0.0)) - last["until"])
    if shown or (card is not None and rows):
        _history_rows(cv, shown, title, top, flash, [], reserve=card is not None)
    if card is not None:
        x, y, w = _sidebar_geom(top)
        p_in, q_out = _box_phase_stream(runner, card)
        _result_box(
            cv, _box_from_result(card, True), p_in, q_out, (x, y, w, SIDEBAR_ROW_H - 8), now
        )


def _classic_hud(cv: Canvas, runner, a: _Anim, now: float, top: int) -> None:
    """show_score in manual/auto mode: hint, result box after a save, saved tries of the child."""
    c = runner.ctrl
    if a.child != c.child:
        a.child, a.history = c.child, []
    result = getattr(runner, "result", None)
    if result is not None:
        key = result.get("try")
        if key != a.result_key:
            a.result_key, a.result_since = key, now
            a.history = [h for h in a.history if h.get("try") != key] + [dict(result)]
    live = getattr(runner, "live", None)
    hint = live if live and "hint" in live else None
    if c.state not in ("idle", "ready"):
        return
    rows = list(a.history)
    phase = _box_phase_clock(a.result_since, now) if result is not None else None
    extra: list[str] = []
    if hint:
        extra.append(hint["hint"])
        extra += wrap(hint.get("detail", ""), 13, "regular", CARD_W, 3)
    if result is not None:
        if result.get("auto"):
            auto = result["auto"]
            extra.append(
                f"Task detected: {b.TASKS[int(auto['task'])]} · {float(auto['confidence']):.0%}"
            )
        elif result.get("pinned"):
            extra.append("Task prioritized by key")
    flying = phase is not None and phase[1] < 1.0
    shown = rows[:-1] if flying and rows else rows
    flash = 0.0
    if result is not None and not flying:
        flash = _flash_amount(now - (a.result_since + BOX_SHOW_S))
    if rows or extra:
        title = "LAST SAVED TRY" if result is not None else "READY TO RECORD"
        _history_rows(cv, shown, title, top, flash, extra, reserve=flying)
    if flying and result is not None:
        x, y, w = _sidebar_geom(top)
        y += 22 * len(extra)
        _result_box(
            cv,
            _box_from_result(result, True),
            phase[0],
            phase[1],
            (x, y, w, SIDEBAR_ROW_H - 8),
            now,
        )


def _testing_box(cv: Canvas, runner, a: _Anim, now: float) -> None:
    """Testing mode: a result box with the task only (never a score) after a try is saved."""
    c = runner.ctrl
    card = runner.card() if hasattr(runner, "card") else None
    target = (VX + VW - PAD - CARD_W, VY + VH / 2, CARD_W, SIDEBAR_ROW_H - 8)
    if card is not None:
        p_in, q_out = _box_phase_stream(runner, card)
        box = _Box(int(card["task"]), "DETECTED", ACCENT)
        _result_box(cv, box, p_in, q_out, target, now)
        return
    if c.state == "confirm":
        a.saved_task = c.choice_task
    if a.tries_saved is None:
        a.tries_saved = c.tries_saved
    if c.tries_saved > a.tries_saved:
        a.saved_since = now
        a.saved_task = a.saved_task or c.choice_task
    a.tries_saved = c.tries_saved
    phase = _box_phase_clock(a.saved_since, now)
    if phase is None or a.saved_task not in (1, 2, 3) or c.state not in ("idle", "ready"):
        return
    _result_box(cv, _Box(int(a.saved_task), "SAVED", ACCENT), phase[0], phase[1], target, now)


def render(runner, view: str = "stick", show_score: bool = False) -> Canvas:
    """The fullscreen window picture for the Runner's current state.

    view: stick (stick figure on a dark backdrop), camera (mirrored camera picture with the
    stick figure on top), neutral (no person shown).
    """
    c = runner.ctrl
    now = _clock()
    a = _anim(runner)
    if a.prev_state == "countdown" and c.state == "recording":
        a.go_since = now
    cv = Canvas(bg=None)  # _video paints every pixel
    _video(cv, runner, view, now)
    if c.state in ("setup", "confirm"):
        cv.dim(VX, VY, VW, VH, 0.38)
    _top_bar(cv, runner, now)
    cv.fade(VX, VY + VH - 84, VW, 84, BG, 0.0, 0.9)
    paused = c.state == "idle" and bool(getattr(c, "paused", False))
    if c.state in ("readiness", "ready", "countdown", "recording", "confirm"):
        _window_frame(cv, STATE_COLOURS[c.state])
    elif paused:
        _window_frame(cv, AMBER, 0.75)
    if c.state == "setup":
        _setup_card(cv, c, now)
    elif c.state == "readiness":
        _readiness_overlay(cv, runner)
    elif c.state == "confirm":
        _confirm_card(cv, c)
    if getattr(runner, "gesture_flow", False):
        _gesture_ring(cv, runner)

    if c.state not in ("setup", "readiness", "confirm", "countdown"):
        bottom = _task_panel(cv, runner, show_score, a, now)
        top = (bottom + 18) if bottom is not None else VY + 80
        if show_score:
            frames = runner.splitter.current() if getattr(runner, "splitter", None) else None
            if view != "neutral" and (c.state == "recording" or frames):
                _eval_bar(cv, runner, a, now)
            if getattr(runner, "gesture_flow", False):
                _gesture_hud(cv, runner, a, now, top)
            else:
                _classic_hud(cv, runner, a, now, top)
        else:
            _testing_box(cv, runner, a, now)
    else:
        a.task_key = None
        a.task_content = None
        if c.state == "confirm" and not show_score:
            a.saved_task = c.choice_task
    if c.state == "countdown":
        _countdown_overlay(cv, runner, now)
    _go_overlay(cv, a, now)
    if c.state not in ("setup", "confirm", "countdown") and not (
        c.state == "recording" and c.message == "Recording. Space stops."
    ):
        _toast(cv, c.message)
    btns = buttons(runner)
    _draw_primary_divider(cv, btns)
    pointer = getattr(runner, "pointer", None)
    over = hit(btns, *pointer) if pointer else None
    for btn in btns:
        _draw_button(cv, btn, hovered=btn.key is not None and btn.key == over)
    a.prev_state = c.state
    a.t_last = now
    return cv


def run_window(
    runner,
    stream,
    draw,
    paced: bool = False,
    title: str = WINDOW_TITLE,
    fullscreen: bool = True,
):
    """Open the fullscreen window and run until Q (or the window is closed). Returns runner.saved.

    The pictures are drawn at the main screen's size (screen_size, canvas_size), so the fullscreen
    window fits the Mac (or Windows) screen exactly.

    draw(runner) -> picture. Pose detection runs in a PoseWorker thread; this loop feeds every
    new frame to the Runner (in order), draws once, then handles keys and clicks. Pressing F
    (outside setup) toggles fullscreen vs normal window.
    """
    import cv2

    from app.ui import PoseWorker, key_name

    old_size = (W, H)
    if fullscreen and (size := screen_size()) is not None:
        set_size(*canvas_size(size))  # draw at the screen's own shape: no bars, no stretching
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    set_prop = getattr(cv2, "setWindowProperty", None)
    prop_fs = getattr(cv2, "WND_PROP_FULLSCREEN", None)
    win_fs = getattr(cv2, "WINDOW_FULLSCREEN", None)
    win_norm = getattr(cv2, "WINDOW_NORMAL", 0)
    is_fs = bool(fullscreen)
    if set_prop is not None and prop_fs is not None and win_fs is not None:
        set_prop(title, prop_fs, win_fs if is_fs else win_norm)

    clicks: list[str] = []
    set_mouse = getattr(cv2, "setMouseCallback", None)
    if set_mouse is not None:
        set_mouse(title, _mouse_handler(runner, clicks, cv2))
    worker = PoseWorker(stream, paced=paced)
    try:
        while not runner.quit:
            for item in worker.take(wait_s=0.005):
                runner.feed(item)
            if worker.error is not None:
                runner.ctrl.message = (
                    f"Camera stopped ({worker.error}). Earlier tries are saved. Backspace if a "
                    "try is open, then Q; check the camera and restart with the same child code."
                )
            elif worker.ended and worker.items.empty():
                runner.ctrl.message = "The video ended. Press Q to quit."
            cv2.imshow(title, draw(runner))
            k = key_name(cv2.waitKeyEx(1))
            if (
                k == "F"
                and runner.ctrl.state != "setup"
                and set_prop is not None
                and prop_fs is not None
                and win_fs is not None
            ):
                is_fs = not is_fs
                set_prop(title, prop_fs, win_fs if is_fs else win_norm)
            else:
                runner.press(k)
            while clicks and not runner.quit:
                runner.press(clicks.pop(0))
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        stopped = worker.stop()
        runner.close()
        stream.close()
        if not stopped:
            worker.stop(timeout=2.0)
        cv2.destroyAllWindows()
        set_size(*old_size)
    return runner.saved


def _mouse_handler(runner, clicks: list, cv2):
    """A click is a press and release on the same button; it queues that button's key."""
    down = getattr(cv2, "EVENT_LBUTTONDOWN", 1)
    up = getattr(cv2, "EVENT_LBUTTONUP", 4)
    pressed: dict = {"key": None}

    def on_mouse(event, x, y, _flags, _param):
        runner.pointer = (x, y)
        if event == down:
            pressed["key"] = hit(buttons(runner), x, y) or _setup_field_hit(runner.ctrl, x, y)
        elif event == up:
            k = hit(buttons(runner), x, y) or _setup_field_hit(runner.ctrl, x, y)
            if k is not None and k == pressed["key"]:
                clicks.append(k)
            pressed["key"] = None

    return on_mouse
