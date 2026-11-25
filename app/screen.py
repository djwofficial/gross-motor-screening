"""The one fullscreen window of the live laptop. Owner: Daryn.

Both entry points draw with this module:
    python -m app.ui ...                 testing mode: never a score on screen (show_score=False)
    python -m app.ui ... --show-score    and python -m app.demo: live score + last saved try

Look: edge-to-edge fullscreen camera (or stick figure) with a minimal, cardless typographic HUD
over soft edge scrims so every metric (FPS, feet visibility, state, task, live score) stays legible
without heavy boxes. Every keyboard shortcut is also clickable; pressing F toggles fullscreen.
Setup and confirm states use focused centered panels for typing or verifying an uncertain try.
Gesture mode (app.demo, show_score): a big 3-2-1 in the middle, a ring on the wrist or the crossed
arms while a gesture is held, the result of each attempt in large type (no box, fades in and out),
the attempts of the recording on the right and a summary after the X.

Size: run_window draws at the screen's own size (set_size(canvas_size(screen_size()))), so the
fullscreen window has no bars and nothing is stretched (Mac: 1512x982 points). The camera picture
fills the height; on a 16:10 screen a little of both sides of a 16:9 camera is cut on screen only
(MAX_SIDE_CROP); pose detection always used the whole picture.

Text is drawn with Pillow in the system font (SF Pro on Mac, Segoe UI on Windows, DejaVu Sans from
matplotlib otherwise). render() only reads the Runner, never changes it. With show_score=False it
never reads runner.live or runner.result, so no score can reach the screen in testing mode.
Pose detection always ran on the unmirrored picture; only this picture is mirrored.
Screening aid only, never a diagnosis.
"""

from __future__ import annotations

import math
import os
import sys
import time
import weakref
from dataclasses import dataclass
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


def set_size(w: int, h: int) -> None:
    """Draw every picture at w x h from now on (run_window: the screen's own size)."""
    global W, H, VW, VH, BAR_Y
    W, H = int(w), int(h)
    VW, VH = W, H
    BAR_Y = H - BAR_H - 6
    _backdrop.cache_clear()


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


BG = _bgr("#090d12")
SURFACE = _bgr("#131922")
RAISED = _bgr("#1e2632")
LINE = _bgr("#303846")
TEXT = _bgr("#f6f8fa")
DIM = _bgr("#c9d1d9")
FAINT = _bgr("#9ea7b3")
ACCENT = _bgr("#79c0ff")
GREEN = _bgr("#56d364")
RED = _bgr("#ff7b72")
AMBER = _bgr("#f8d46c")
PURPLE = _bgr("#d2a8ff")
LETTERBOX = _bgr("#05070a")
NEUTRAL = _bgr("#181e26")
SHADOW = _bgr("#05070a")

STATE_COLOURS = {
    "setup": _bgr("#9ea7b3"),
    "readiness": ACCENT,
    "idle": _bgr("#c9d1d9"),
    "ready": GREEN,
    "countdown": AMBER,
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
SCORE_COLOURS = {0: RED, 1: AMBER, 2: GREEN}
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
FIG_RIGHT, FIG_LEFT, FIG_MID = _bgr("#ff9f43"), _bgr("#4da3ff"), _bgr("#f0f6fc")
MIN_SCORE = 0.05

_WEIGHTS = ("regular", "medium", "semibold", "bold")


@lru_cache(maxsize=1)
def _font_sources() -> dict[str, tuple[str, int, str | None]]:
    """weight -> (font file, index in a .ttc, variation name of a variable font)."""
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


@lru_cache(maxsize=64)
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


@lru_cache(maxsize=64)
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
def _halo(s: str, size: int, weight: str) -> tuple[np.ndarray, int]:
    """A soft dark halo that follows the letters (blurred ink), and its padding in pixels."""
    import cv2

    ink = _ink(s, size, weight)[..., 0]
    pad = max(3, size // 6)
    big = np.pad(ink, pad)
    halo = cv2.GaussianBlur(big, (0, 0), sigmaX=pad / 2.2)
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


@lru_cache(maxsize=1)
def _backdrop() -> np.ndarray:
    """Dark backdrop for the stick figure: a soft light in the middle."""
    yy, xx = np.mgrid[0:VH, 0:VW].astype(np.float32)
    d = np.sqrt(((xx - VW / 2) / (VW / 2)) ** 2 + ((yy - VH * 0.45) / (VH / 2)) ** 2)
    t = np.clip(d / 1.3, 0, 1)[..., None]
    inner, outer = np.array(_bgr("#18202a"), np.float32), np.array(BG, np.float32)
    noise = np.random.default_rng(0).uniform(-0.5, 0.5, (VH, VW, 1)).astype(np.float32)
    return np.clip(inner * (1 - t) + outer * t + 0.5 + noise, 0, 255).astype(np.uint8)


def _lighten(c, f: float = 0.14):
    return tuple(int(v + (255 - v) * f) for v in c)


class Canvas:
    """The window picture plus every string drawn on it (tests read .texts)."""

    def __init__(self, w: int | None = None, h: int | None = None, bg=BG):
        w, h = W if w is None else w, H if h is None else h
        self.img = np.empty((h, w, 3), np.uint8)
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
    ):
        """anchor: l/c/r then t (top of the line box), m (middle of capitals) or b (baseline).

        glow (0..1): a soft dark halo around the letters, so big text reads on any picture
        without a box behind it."""
        s = str(s)
        if not s:
            return 0.0
        self.texts.append(s)
        w = measure(s, size, weight)
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


_LABEL = (13, "semibold")
_CAP = (11, "semibold")
_GROUP_GAP = 20


def _width(btn: Button) -> int:
    if btn.kind == "label":
        return math.ceil(measure(btn.label, 12, "semibold"))
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
                cv.rrect(sx, BAR_Y + 9, 1, BAR_H - 18, 1, fill=TEXT, alpha=0.20)


def _draw_button(cv: Canvas, btn: Button, hovered: bool) -> None:
    cy = btn.y + btn.h / 2
    if btn.kind == "label":
        col = btn.colour if btn.colour != ACCENT else FAINT
        cv.text(btn.x, cy, btn.label, 12, "semibold", col, "lm", shadow=True)
        return

    is_primary = btn.kind == "primary"
    is_seg = btn.kind == "seg"
    is_danger = btn.kind == "danger"

    if is_primary:
        cv.rrect(
            btn.x,
            btn.y + 3,
            btn.w,
            btn.h - 6,
            8,
            fill=btn.colour,
            alpha=0.28 if hovered else 0.18,
            border=btn.colour,
            border_alpha=0.75 if hovered else 0.45,
        )
        fg = TEXT if hovered else _lighten(btn.colour, 0.25)
    elif is_seg:
        if hovered:
            cv.rrect(btn.x, btn.y + 4, btn.w, btn.h - 8, 7, fill=TEXT, alpha=0.10)
        if btn.selected:
            cv.rrect(btn.x + 8, btn.y + btn.h - 4, btn.w - 16, 2, 1, fill=ACCENT, alpha=0.95)
            fg = TEXT
        else:
            fg = TEXT if hovered else DIM
    elif is_danger:
        if hovered:
            cv.rrect(btn.x, btn.y + 4, btn.w, btn.h - 8, 7, fill=RED, alpha=0.18)
        fg = RED if hovered else DIM
    else:
        if btn.selected:
            cv.rrect(
                btn.x,
                btn.y + 4,
                btn.w,
                btn.h - 8,
                7,
                fill=btn.colour,
                alpha=0.22 if hovered else 0.14,
            )
            fg = _lighten(btn.colour, 0.2)
        else:
            if hovered:
                cv.rrect(btn.x, btn.y + 4, btn.w, btn.h - 8, 7, fill=TEXT, alpha=0.10)
            fg = TEXT if hovered else DIM

    x = btn.x + btn.pad
    if btn.cap:
        kw = max(18.0, measure(btn.cap, *_CAP) + 10)
        kh = 19
        cap_border = btn.colour if (is_primary or btn.selected) else TEXT
        cv.rrect(
            x,
            cy - kh / 2,
            kw,
            kh,
            5,
            fill=BG,
            alpha=0.55,
            border=cap_border,
            border_alpha=0.55 if (is_primary or btn.selected or hovered) else 0.25,
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


def _draw_framing_guide(cv: Canvas, area: np.ndarray, runner, s: float, ox: int, oy: int) -> None:
    import cv2

    from live.start_area import default_start_area, mirror_box_x, standing_guide_lines

    fw, fh = runner.frame_size
    custom = getattr(runner, "start_area", None) is not None
    raw_box = runner.start_area if custom else default_start_area((fw, fh))
    mx0, my0, mx1, my1 = mirror_box_x(raw_box, fw)
    x0, y0 = max(0, int(round(ox + mx0 * s))), max(0, int(round(oy + my0 * s)))
    x1, y1 = min(VW - 1, int(round(ox + mx1 * s))), min(VH - 1, int(round(oy + my1 * s)))
    if x1 <= x0 or y1 <= y0:
        return

    col = GREEN if custom else ACCENT
    arm = max(12, min(26, (x1 - x0) // 5, (y1 - y0) // 3))
    corners = (
        ((x0, y0), (x0 + arm, y0), (x0, y0 + arm)),
        ((x1, y0), (x1 - arm, y0), (x1, y0 + arm)),
        ((x0, y1), (x0 + arm, y1), (x0, y1 - arm)),
        ((x1, y1), (x1 - arm, y1), (x1, y1 - arm)),
    )
    for corner, hx, vy in corners:
        cv2.line(area, corner, hx, col, 2, cv2.LINE_AA)
        cv2.line(area, corner, vy, col, 2, cv2.LINE_AA)
    for sx in range(x0 + arm + 8, x1 - arm - 4, 16):
        cv2.line(area, (sx, y0), (min(sx + 6, x1 - arm), y0), col, 1, cv2.LINE_AA)
        cv2.line(area, (sx, y1), (min(sx + 6, x1 - arm), y1), col, 1, cv2.LINE_AA)

    start_label = "Start area" if custom else "Start area (guide)"
    cv.text(
        VX + x0 + 8,
        max(VY + 84, VY + y0 - 8),
        start_label,
        12,
        "semibold",
        col,
        "lb",
        shadow=True,
    )

    y_head, _ = standing_guide_lines((fw, fh), raw_box)
    yh = int(round(oy + y_head * s))
    if 0 <= yh < VH:
        for sx in range(x0, x1, 18):
            cv2.line(area, (sx, yh), (min(sx + 10, x1), yh), col, 1, cv2.LINE_AA)
        cv.text(
            VX + 0.5 * (x0 + x1),
            max(VY + 86, VY + yh - 8),
            "Target ~50% height",
            12,
            "medium",
            col,
            "cb",
            shadow=True,
        )


def _video(cv: Canvas, runner, view: str) -> None:
    import cv2

    area = cv.img[VY : VY + VH, VX : VX + VW]
    fw, _fh = runner.frame_size
    s, dw, dh, ox, oy = _stage_transform(runner.frame_size)
    if view == "neutral":
        area[:] = NEUTRAL
        return
    if view == "camera" and runner.last_frame is not None:
        area[:] = LETTERBOX
        pic = cv2.resize(cv2.flip(runner.last_frame, 1), (dw, dh), interpolation=cv2.INTER_LINEAR)
        x0, y0 = max(0, ox), max(0, oy)  # paste, cutting what falls outside the window
        w, h = min(dw - (x0 - ox), VW - x0), min(dh - (y0 - oy), VH - y0)
        area[y0 : y0 + h, x0 : x0 + w] = pic[y0 - oy : y0 - oy + h, x0 - ox : x0 - ox + w]
    else:
        area[:] = _backdrop()
    if runner.ctrl.state in ("readiness", "idle", "ready"):
        _draw_framing_guide(cv, area, runner, s, ox, oy)
    if runner.last_points is not None:
        pts = runner.last_points.copy()
        pts[:, 0] = np.where(pts[:, 2] > 0, fw - pts[:, 0], 0)
        draw_skeleton(area, pts, s, ox, oy)


def _hud_item(
    cv: Canvas,
    x_right: float,
    cy: float,
    label: str,
    colour=TEXT,
    dot=None,
    size: int = 13,
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


def _top_bar(cv: Canvas, runner) -> None:
    c = runner.ctrl
    cv.fade(VX, VY, VW, 110, BG, 0.88, 0.0)
    x0, cy = VX + PAD, VY + 26
    w_title = cv.text(x0, cy, "Gross Motor Screening", 17, "semibold", TEXT, "lm", shadow=True)
    cv.text(x0, cy + 21, "Screening aid · not a diagnosis", 12, "regular", DIM, "lm", shadow=True)
    limit = x0 + max(w_title, measure("Screening aid · not a diagnosis", 12)) + 24

    paused = c.state == "idle" and bool(getattr(c, "paused", False))
    if paused:
        colour, name = AMBER, "Gesture paused" if c.mode == "gesture" else "Auto paused"
        dot_col = AMBER
    else:
        colour = STATE_COLOURS.get(c.state, DIM)
        name = STATE_NAMES.get(c.state, c.state)
        if c.state == "recording":
            name = f"{name}  {_recording_seconds(runner):4.1f} s"
        pulse = c.state != "recording" or (time.monotonic() % 1.0) < 0.6
        dot_col = colour if pulse else TEXT

    x = _hud_item(cv, VX + VW - PAD, cy, name, colour, dot=dot_col, size=14, weight="semibold")

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
        w = measure(label, 13, weight) + (13 if dot else 0)
        if x - 20 - w < limit:
            break
        cv.text(x - 10, cy, "·", 13, "semibold", FAINT, "cm", shadow=True)
        x = _hud_item(cv, x - 20, cy, label, fg, dot=dot, size=13, weight=weight)


def _panel(cv: Canvas, x, y, w, h, alpha=0.90):
    cv.rrect(x + 2, y + 6, w, h, 14, fill=SHADOW, alpha=0.45)
    cv.rrect(x, y, w, h, 14, fill=SURFACE, alpha=alpha, border=LINE, border_alpha=0.85)


def _setup_field_boxes(c) -> tuple[int, int, int, int, list[tuple[int, int, int, int]], list[str]]:
    w = 500
    msg_lines = wrap(c.message, 14, "regular", w - 56, 2)
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


def _setup_card(cv: Canvas, c) -> None:
    names = ("Child code", "Age in months", "Consent level")
    lo, hi = AGE_RANGE_MONTHS
    hints = ("e.g. K03", f"{lo}–{hi}", "1 points only · 2 stick figure · 3 + video")
    x, y, w, h, boxes, msg_lines = _setup_field_boxes(c)
    _panel(cv, x, y, w, h, 0.94)
    cx = x + 28
    yy = y + 28
    cv.text(cx, yy, "New child", 26, "semibold", TEXT)
    yy += 38
    cv.text(
        cx,
        yy,
        "Enter moves to the next field · Backspace or click a box to go back.",
        14,
        "regular",
        DIM,
    )
    yy += 36
    caret_on = (time.monotonic() % 1.0) < 0.6
    for i, name in enumerate(names):
        focused = i == c.field
        cv.text(cx, yy, name, 13, "medium", TEXT if focused else DIM)
        if i == 2:
            cv.text(x + w - 28, yy, "1 points · 2 stick · 3 video", 13, "regular", FAINT, "rt")
        bx, by, bw, bh = boxes[i]
        yy = by
        cv.rrect(
            bx,
            by,
            bw,
            bh,
            10,
            fill=RAISED,
            border=ACCENT if focused else LINE,
            bw=2 if focused else 1,
        )
        value = c.form[i]
        tx = bx + 14
        if value:
            tx += cv.text(tx, by + 23, value, 19, "regular", TEXT, "lm")
        elif not focused:
            cv.text(tx, by + 23, hints[i], 16, "regular", FAINT, "lm")
        if focused:
            if caret_on:
                cv.rrect(tx + 2, by + 12, 2, 22, 1, fill=ACCENT)
            if not value:
                cv.text(tx + 10, by + 23, hints[i], 16, "regular", FAINT, "lm")
        yy = by + bh + 14
    error = c.message and not c.message.startswith(("Type", "Next child"))
    for line in msg_lines:
        cv.text(cx, yy, line, 14, "regular", AMBER if error else DIM)
        yy += 20


def _icon_ok(cv: Canvas, cx, cy):
    import cv2

    cv.dot(cx, cy, 9, GREEN)
    pts = np.array([[cx - 4.5, cy], [cx - 1.2, cy + 3.5], [cx + 4.5, cy - 3.5]]) * 4
    cv2.polylines(cv.img, [pts.astype(np.int32)], False, BG, 2, cv2.LINE_AA, 2)


def _icon_warn(cv: Canvas, cx, cy):
    cv.dot(cx, cy, 9, AMBER)
    cv.text(cx, cy, "!", 13, "bold", BG, "cm")


def _readiness_overlay(cv: Canvas, runner) -> None:
    c = runner.ctrl
    custom = getattr(runner, "start_area", None) is not None
    w = 460
    rows = [wrap(p, 15, "regular", w - 40, 3) for p in c.problems]
    if c.mode == "gesture":
        foot = "Show an open hand to the camera (or press Start, Enter) to start."
    elif not c.problems:
        foot = "Press Start (Enter) when the child is at the start line."
    else:
        foot = "You can start anyway with Start (Enter)."
    foot_lines = wrap(foot, 14, "regular", w)
    tip_lines = (
        []
        if custom
        else wrap(
            "Tip: run 'python -m live.start_area' to calibrate the taped floor box.",
            12,
            "regular",
            w,
        )
    )
    body_h = 28 if not rows else sum(len(r) * 21 + 10 for r in rows)
    total_h = 34 + 14 + body_h + 12 + len(foot_lines) * 20 + (len(tip_lines) * 18 + 8)
    cv.fade_x(0, 48, w + 160, min(H - 96, total_h + 112), BG, 0.84, 0.0, v_feather=64)

    cx, yy = VX + PAD, VY + 92
    cv.text(cx, yy, "Readiness check", 21, "semibold", TEXT, shadow=True)
    yy += 38
    if not rows:
        _icon_ok(cv, cx + 9, yy + 9)
        cv.text(
            cx + 28, yy + 9, "Camera and child look ready.", 15, "medium", GREEN, "lm", shadow=True
        )
        yy += 30
    for lines in rows:
        _icon_warn(cv, cx + 9, yy + 9)
        for line in lines:
            cv.text(cx + 28, yy + 9, line, 15, "regular", TEXT, "lm", shadow=True)
            yy += 21
        yy += 10
    yy += 6
    for line in foot_lines:
        cv.text(cx, yy, line, 14, "medium", DIM, shadow=True)
        yy += 20
    if tip_lines:
        yy += 6
        for line in tip_lines:
            cv.text(cx, yy, line, 12, "regular", FAINT, shadow=True)
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
    _panel(cv, x, y, w, h, 0.94)
    cx, yy = x + 28, y + 28
    cv.text(cx, yy, "Check this try", 24, "semibold", TEXT)
    yy += 36
    cv.text(cx, yy, "Correct the task or leg if the guess is wrong, then save.", 14, "regular", DIM)
    yy += 38
    cv.text(cx, yy, "Guess", 13, "medium", DIM)
    cv.text(x + w - 28, yy, f"{gc:.0%} sure", 13, "medium", DIM, "rt")
    yy += 22
    cv.text(cx, yy, _task_leg(gt, gl), 17, "regular", TEXT)
    yy += 40
    cv.text(cx, yy, "Will be saved as", 13, "medium", DIM)
    if corrected:
        cv.text(x + w - 28, yy, "Corrected", 13, "semibold", AMBER, "rt")
    yy += 22
    cv.text(cx, yy, _task_leg(ct, cl), 20, "semibold", TEXT)
    yy += 44
    keys = "1/2/3 task · " + ("" if ct == 3 else "L/R leg · ") + "Enter save · Backspace cancel"
    cv.text(cx, yy, keys, 13, "regular", FAINT)


def _toast(cv: Canvas, message: str) -> None:
    if not message:
        return
    lines = wrap(message, 14, "medium", 820, 2)
    yy = BAR_Y - 12 - (len(lines) - 1) * 20
    for line in lines:
        cv.text(VX + VW / 2, yy, line, 14, "medium", TEXT, "cb", shadow=True)
        yy += 20


_BADGE_POS: weakref.WeakKeyDictionary[object, tuple[float, float]] = weakref.WeakKeyDictionary()


def _shoulder_anchor(runner) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    """Return ((right_x, right_y), (left_x, left_y), shoulder_span_px) in mirrored canvas coords."""
    pts = getattr(runner, "last_points", None)
    if pts is None or len(pts) <= max(b.L_SHOULDER, b.R_SHOULDER):
        return None
    fw, _fh = runner.frame_size
    s, _dw, _dh, ox, oy = _stage_transform(runner.frame_size)

    def to_canvas(idx: int) -> tuple[float, float] | None:
        if float(pts[idx, 2]) < MIN_SCORE:
            return None
        return (VX + ox + (fw - float(pts[idx, 0])) * s, VY + oy + float(pts[idx, 1]) * s)

    found = [p for idx in (b.L_SHOULDER, b.R_SHOULDER) if (p := to_canvas(idx)) is not None]
    if len(found) == 2:
        left_p, right_p = sorted(found, key=lambda p: p[0])
        return right_p, left_p, max(24.0, right_p[0] - left_p[0])
    if len(found) == 1:
        return found[0], found[0], 60.0
    for idx in (b.NECK, b.NOSE):
        if (p := to_canvas(idx)) is not None:
            return p, p, 60.0
    return None


def _draw_floating_score_badge(cv: Canvas, runner, live: dict) -> bool:
    """Lightweight floating shoulder score callout (show_score only)."""
    import cv2

    anchor = _shoulder_anchor(runner)
    if anchor is None:
        return False
    right_sh, left_sh, span = anchor
    sc = int(live["score"])
    task = int(live.get("task") or 1)
    conf = float(live.get("confidence") or 0.0)
    col = SCORE_COLOURS[sc]
    label = SCORE_NAMES[sc]
    is_rec = runner.ctrl.state == "recording"
    mode_tag = "Live" if is_rec else "Preview"
    sub = f"{mode_tag} · {conf:.0%} · {TASK_SHORT.get(task, '')}"

    text_w = max(measure(label, 19, "semibold"), measure(sub, 11, "medium"))
    bw = int(max(156, 14 + 30 + 12 + text_w + 16))
    bh = 50
    stem_dx = int(min(44, max(22, span * 0.26)))
    stem_dy = 20

    right_edge, left_edge = VX + VW - PAD, VX + PAD
    if getattr(runner, "result", None) and runner.ctrl.state in ("idle", "ready"):
        right_edge = VX + VW - PAD - CARD_W - 14
    top_edge, bot_edge = VY + 84, BAR_Y - 18
    go_right = right_sh[0] + stem_dx + 14 + bw <= right_edge
    ax, ay = right_sh if go_right else left_sh
    tx = (ax + stem_dx + 14) if go_right else (ax - stem_dx - 14 - bw)
    tx = max(left_edge, min(tx, right_edge - bw))
    ty = max(top_edge, min(ay - stem_dy - bh / 2, bot_edge - bh))

    prev = _BADGE_POS.get(runner)
    if prev is None or math.hypot(tx - prev[0], ty - prev[1]) > 260.0:
        bx, by = tx, ty
    else:
        bx = prev[0] + 0.28 * (tx - prev[0])
        by = prev[1] + 0.28 * (ty - prev[1])
    _BADGE_POS[runner] = (bx, by)

    elbow_x = int(round(bx - 12)) if go_right else int(round(bx + bw + 12))
    elbow_y = int(round(by + bh / 2))
    attach_x = int(round(bx)) if go_right else int(round(bx + bw))
    cv2.line(cv.img, (int(round(ax)), int(round(ay))), (elbow_x, elbow_y), col, 2, cv2.LINE_AA)
    cv2.line(cv.img, (elbow_x, elbow_y), (attach_x, elbow_y), col, 2, cv2.LINE_AA)
    cv.dot(ax, ay, 4.5, col)
    cv.dot(ax, ay, 2, TEXT)

    cy = by + bh / 2
    cv.dot(bx + 26, cy, 15, col)
    cv.text(bx + 26, cy, str(sc), 18, "bold", BG, "cm")
    cv.text(bx + 50, by + 18, label, 19, "semibold", col, "lm", glow=0.9)
    cv.text(bx + 50, by + 37, sub, 11, "medium", TEXT, "lm", glow=0.9)
    return True


def _detect_label(
    detect: dict | None, preset_task: int | None, preset_leg: str | None = None
) -> str:
    """'Detecting: hop on one leg · 92%' or pinned task label while a try runs."""
    if preset_task:
        base = f"Task set by key: {b.TASKS[preset_task]}"
        if preset_leg and preset_task != 3:
            return f"{base} · leg {preset_leg}"
        return base
    if not detect:
        return "Detecting…"
    return f"Detecting: {b.TASKS[int(detect['task'])]} · {float(detect['confidence']):.0%}"


def _detect_hud(cv: Canvas, detect: dict | None, preset_task: int | None, preset_leg: str | None):
    label = _detect_label(detect, preset_task, preset_leg)
    col = AMBER if preset_task else ACCENT
    cv.dot(VX + PAD + 4, VY + 74, 4, col)
    cv.text(VX + PAD + 16, VY + 74, label, 13, "semibold", col, "lm", shadow=True)


def _score_hud(cv: Canvas, runner, view: str = "camera") -> None:
    c = runner.ctrl
    if c.state in ("setup", "readiness", "confirm"):
        return
    if getattr(runner, "gesture_flow", False):
        _gesture_hud(cv, runner, view)
        return
    if c.state == "recording" and getattr(runner, "auto_flow", False):
        _detect_hud(cv, getattr(runner, "detect", None), c.preset_task, c.preset_leg)
    elif (
        c.state in ("idle", "ready")
        and getattr(runner, "auto_flow", False)
        and c.preset_task is not None
    ):
        _detect_hud(cv, None, c.preset_task, c.preset_leg)

    live = getattr(runner, "live", None)
    anchored = False
    if view != "neutral" and live and "score" in live:
        anchored = _draw_floating_score_badge(cv, runner, live)

    result = getattr(runner, "result", None)
    show_result = bool(result and c.state in ("idle", "ready"))
    show_live_col = not anchored
    if not show_live_col and not show_result:
        return

    w = CARD_W
    x = VX + VW - PAD - w
    cv.fade_x(x - 140, 48, w + PAD + 140, H - 96, BG, 0.0, 0.86, v_feather=64)
    y = VY + 92
    if show_live_col:
        y = _live_text(cv, live, x, y, w)
        if show_result:
            cv.rrect(x, y + 6, w, 1, 0, fill=TEXT, alpha=0.16)
            y += 20
    if show_result:
        _result_text(cv, result, x, y, w)


def _live_text(cv: Canvas, live, x, y, w) -> int:
    if live and "hint" in live:
        task = live.get("task")
        detail_lines = wrap(live.get("detail", ""), 13, "regular", w, 3)
        cv.text(x, y, "READY TO RECORD", 11, "semibold", ACCENT, shadow=True)
        if task:
            cv.text(
                x + w,
                y,
                f"Task {int(task)} · {TASK_SHORT.get(int(task), '')}",
                11,
                "semibold",
                DIM,
                "rt",
                shadow=True,
            )
        y += 22
        cv.text(x, y, live["hint"], 18, "semibold", TEXT, shadow=True)
        y += 26
        for line in detail_lines:
            cv.text(x, y, line, 13, "regular", DIM, shadow=True)
            y += 18
        return y + 8

    if not live or "error" in live:
        detail = (
            live["error"] if live else "The score appears after about 1 s of the child in view."
        )
        lines = wrap(detail, 13, "regular", w, 3)
        cv.text(x, y, "LIVE SCORE", 11, "semibold", DIM, shadow=True)
        y += 22
        cv.text(x, y, "Waiting for a clear view", 17, "semibold", TEXT, shadow=True)
        y += 26
        for line in lines:
            cv.text(x, y, line, 13, "regular", DIM, shadow=True)
            y += 18
        return y + 8

    sc = int(live["score"])
    task = int(live.get("task") or 1)
    conf = float(live.get("confidence") or 0.0)
    col = SCORE_COLOURS[sc]
    reasons = wrap(live.get("reasons", ""), 13, "regular", w, 3)
    cv.text(x, y, "LIVE SCORE", 11, "semibold", DIM, shadow=True)
    cv.text(
        x + w,
        y,
        f"Task {task} · {TASK_SHORT.get(task, '')}",
        11,
        "semibold",
        DIM,
        "rt",
        shadow=True,
    )
    y += 24
    num_w = cv.text(x, y, str(sc), 34, "bold", col, "lt", shadow=True)
    cv.text(x + num_w + 12, y + 3, SCORE_NAMES[sc], 24, "semibold", col, "lt", shadow=True)
    cv.text(x + num_w + 12, y + 31, f"{conf:.0%} sure", 13, "medium", DIM, "lt", shadow=True)
    y += 52
    cv.rrect(x, y, w, 3, 2, fill=TEXT, alpha=0.18)
    if conf > 0:
        cv.rrect(x, y, max(6, w * min(conf, 1.0)), 3, 2, fill=col)
    y += 14
    for line in reasons:
        cv.text(x, y, line, 13, "regular", DIM, shadow=True)
        y += 18
    return y + 8


def _result_text(cv: Canvas, r: dict, x, y, w) -> None:
    check = r.get("check") or r.get("score") is None
    reasons = wrap(r.get("reasons", ""), 13, "regular", w, 3)
    head = _task_leg(r["task"], r["leg"])
    if check:
        words = wrap("Model is unsure: check by teacher.", 14, "regular", w, 2)
        col, verdict = PURPLE, "Check by teacher"
    else:
        s = int(r["score"])
        words = wrap(SCORE_WORDS[s].capitalize() + ".", 14, "regular", w, 2)
        col, verdict = SCORE_COLOURS[s], f"{s} · {SCORE_NAMES[s]}"

    auto = r.get("auto")
    pinned = r.get("pinned")
    cv.text(x, y, "LAST SAVED TRY", 11, "semibold", DIM, shadow=True)
    if r.get("try"):
        cv.text(x + w, y, f"Try #{int(r['try'])}", 11, "semibold", DIM, "rt", shadow=True)
    y += 22
    if auto:
        detected = f"Task detected: {b.TASKS[int(auto['task'])]} · {float(auto['confidence']):.0%}"
        cv.text(x, y, detected, 13, "medium", ACCENT, shadow=True)
        y += 20
    elif pinned:
        cv.text(x, y, "Task prioritized by key", 13, "medium", AMBER, shadow=True)
        y += 20
    cv.text(x, y, head, 15, "medium", TEXT, shadow=True)
    y += 26
    vw = cv.text(x, y, verdict, 22, "semibold", col, "lt", shadow=True)
    if not check:
        cv.text(
            x + vw + 10,
            y + 13,
            f"{float(r.get('confidence') or 0):.0%} sure",
            13,
            "medium",
            DIM,
            "lm",
            shadow=True,
        )
    y += 34
    for line in words:
        cv.text(x, y, line, 14, "regular", TEXT, shadow=True)
        y += 20
    y += 4
    for line in reasons:
        cv.text(x, y, line, 13, "regular", DIM, shadow=True)
        y += 18
    y += 8
    cv.text(x, y, "Reference only, not a diagnosis.", 11, "regular", FAINT, shadow=True)


# ---------------------------------------------------------------------- gesture flow


def _to_canvas(runner, x: float, y: float) -> tuple[float, float]:
    """Unmirrored frame pixel -> mirrored window pixel."""
    fw, _fh = runner.frame_size
    s, _dw, _dh, ox, oy = _stage_transform(runner.frame_size)
    return VX + ox + (fw - x) * s, VY + oy + y * s


def _arc(cv: Canvas, cx, cy, r, frac: float, colour, thickness: int = 4) -> None:
    """Clockwise arc from 12 o'clock covering frac (0..1) of a circle."""
    import cv2

    if frac <= 0:
        return
    c = (int(round(cx * 4)), int(round(cy * 4)))
    ax = (int(round(r * 4)), int(round(r * 4)))
    cv2.ellipse(cv.img, c, ax, -90, 0, 360 * min(frac, 1.0), colour, thickness, cv2.LINE_AA, 2)


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
    cv2.circle(cv.img, c4, 30 * 4, TEXT if not g.waiting_for_open else AMBER, 2, cv2.LINE_AA, 2)
    _arc(cv, x, y, 30, g.progress, col, 5)
    label = g.hint
    cv.text(
        x, y + 46, label, 13, "semibold", AMBER if g.waiting_for_open else TEXT, "ct", shadow=True
    )


def _countdown_overlay(cv: Canvas, runner) -> None:
    """Big 3-2-1 in the middle of the window, with a ring that empties each second."""
    left = runner.countdown_left() if hasattr(runner, "countdown_left") else None
    if left is None:
        return
    n = max(1, min(3, math.ceil(left - 1e-6))) if left > 0 else 1
    frac = left - (n - 1)
    cx, cy, r = VX + VW / 2, VY + VH / 2 - 20, 132
    cv.rrect(cx - r - 10, cy - r - 10, 2 * r + 20, 2 * r + 20, r + 10, fill=BG, alpha=0.62)
    import cv2

    cv2.circle(
        cv.img,
        (int(cx * 4), int(cy * 4)),
        int(r * 4),
        _lighten(BG, 0.25),
        6,
        cv2.LINE_AA,
        2,
    )
    _arc(cv, cx, cy, r, frac, AMBER, 8)
    cv.text(cx, cy, str(n), 190, "bold", TEXT, "cm", shadow=True)
    cv.text(cx, cy + r + 34, "Get ready", 24, "semibold", AMBER, "ct", shadow=True)
    cv.text(
        cx,
        cy + r + 68,
        "Recording starts at 0 · Backspace cancels",
        14,
        "regular",
        DIM,
        "ct",
        shadow=True,
    )


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


def _attempt_moment(cv: Canvas, r: dict, a: float) -> None:
    """The attempt that just ended, in large type at the top middle (no box)."""
    if a <= 0:
        return
    task = int(r["task"])
    word, small, col = _score_words(r)
    cx = VX + VW / 2
    big = 64 if len(word) < 10 else 46
    head = f"TASK {task} · {b.TASKS[task].upper()}"
    detail = f"{r.get('headline', '')}  ·  {small}"
    cv.text(cx, VY + 102, head, 13, "semibold", TEXT, "cm", alpha=a, glow=0.9)
    cv.text(cx, VY + 162, word, big, "bold", col, "cm", alpha=a, glow=0.85)
    cv.text(cx, VY + 214, detail, 18, "medium", TEXT, "cm", alpha=a, glow=0.9)
    tag, tag_col = ("● Recorded", RED) if r.get("recorded") else ("Not recorded", DIM)
    cv.text(cx, VY + 240, tag, 12, "semibold", tag_col, "cm", alpha=a, glow=0.9)


def _attempt_rows(cv: Canvas, rows: list[dict], title: str, x, y, w) -> int:
    cv.text(x, y, title, 11, "semibold", DIM, shadow=True)
    y += 24
    for r in rows:
        task = int(r["task"])
        verdict, col = _verdict(r)
        cv.text(x, y, f"Task {task} · {TASK_SHORT[task]}", 15, "semibold", TEXT, shadow=True)
        cv.text(x + w, y, verdict, 15, "semibold", col, "rt", shadow=True)
        y += 22
        cv.text(x, y, r.get("headline", ""), 13, "regular", DIM, shadow=True)
        y += 26
    return y


def _summary_moment(cv: Canvas, summary: dict, a: float) -> None:
    """After the X: the saved attempts in large type over the dimmed picture (no box)."""
    if a <= 0:
        return
    rows = summary.get("rows", [])
    cv.dim(VX, VY, VW, VH, 1.0 - 0.7 * a)
    cx = VX + VW / 2
    n = len(rows)
    total = 92 + max(n, 1) * 64 + 40
    y = VY + (VH - total) / 2
    if not rows:
        cv.text(cx, y + 20, "No task found in this recording", 30, "semibold", TEXT, "cm", alpha=a)
        cv.text(
            cx,
            y + 62,
            "Nothing was saved. Show an open hand to try again.",
            16,
            "regular",
            DIM,
            "cm",
            alpha=a,
        )
        return
    cv.text(cx, y + 20, "Recording saved", 34, "semibold", TEXT, "cm", alpha=a, glow=0.7)
    cv.text(
        cx,
        y + 56,
        f"{n} {'try' if n == 1 else 'tries'} · one per task",
        15,
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
        cv.text(cx - 22, y, title, 20, "semibold", TEXT, "rm", alpha=a, glow=0.7)
        cv.text(cx - 22, y + 24, r.get("headline", ""), 14, "regular", DIM, "rm", alpha=a, glow=0.7)
        cv.text(cx + 22, y, word, 24, "bold", col, "lm", alpha=a, glow=0.7)
        cv.text(cx + 22, y + 24, small, 14, "regular", DIM, "lm", alpha=a, glow=0.7)
        y += 64
    cv.text(cx, y + 10, "Reference only, not a diagnosis.", 12, "regular", FAINT, "cm", alpha=a)


def _gesture_hud(cv: Canvas, runner, view: str) -> None:
    """show_score in gesture mode: live badge, attempt card, attempt list, summary."""
    c = runner.ctrl
    frames = runner.splitter.current() if getattr(runner, "splitter", None) else None
    if frames:
        _detect_hud(cv, getattr(runner, "detect", None), c.preset_task, c.preset_leg)
    elif c.preset_task is not None:
        _detect_hud(cv, None, c.preset_task, c.preset_leg)

    live = getattr(runner, "live", None)
    if view != "neutral" and frames and live and "score" in live:
        _draw_floating_score_badge(cv, runner, live)

    summary = runner.summary_card()
    if summary is not None:
        _summary_moment(cv, summary, _fade(runner, summary))
        return
    card = runner.card()
    if card is not None:
        _attempt_moment(cv, card, _fade(runner, card))

    if c.state == "recording":
        rows, title = list(runner.rec_bouts), "THIS RECORDING"
    else:
        rows, title = list(runner.preview_bouts), "DETECTED · NOT RECORDED"
    if not rows:
        return
    w = CARD_W
    x = VX + VW - PAD - w
    cv.fade_x(x - 140, 48, w + PAD + 140, H - 96, BG, 0.0, 0.86, v_feather=64)
    y = _attempt_rows(cv, rows, title, x, VY + 92, w)
    cv.text(x, y + 2, "Reference only, not a diagnosis.", 11, "regular", FAINT, shadow=True)


def render(runner, view: str = "stick", show_score: bool = False) -> Canvas:
    """The fullscreen window picture for the Runner's current state.

    view: stick (stick figure on a dark backdrop), camera (mirrored camera picture with the
    stick figure on top), neutral (no person shown).
    """
    c = runner.ctrl
    cv = Canvas()
    _video(cv, runner, view)
    if c.state in ("setup", "confirm"):
        cv.dim(VX, VY, VW, VH, 0.38)
    _top_bar(cv, runner)
    cv.fade(VX, VY + VH - 84, VW, 84, BG, 0.0, 0.88)
    paused = c.state == "idle" and bool(getattr(c, "paused", False))
    if c.state in ("readiness", "ready", "countdown", "recording", "confirm"):
        cv.ring(VX, VY, VW, VH, 0, 3, STATE_COLOURS[c.state])
    elif paused:
        cv.ring(VX, VY, VW, VH, 0, 2, AMBER, alpha=0.75)
    if c.state == "setup":
        _setup_card(cv, c)
    elif c.state == "readiness":
        _readiness_overlay(cv, runner)
    elif c.state == "confirm":
        _confirm_card(cv, c)
    if getattr(runner, "gesture_flow", False):
        _gesture_ring(cv, runner)
    if show_score:
        _score_hud(cv, runner, view)
    if c.state == "countdown":
        _countdown_overlay(cv, runner)
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
