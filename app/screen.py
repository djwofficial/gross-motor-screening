"""The one window of the live laptop. Owner: Daryn.

Both entry points draw with this module:
    python -m app.ui ...                 testing mode: never a score on screen (show_score=False)
    python -m app.ui ... --show-score    and python -m app.demo: live score + last saved try

Look: the camera picture (or the stick figure) in a rounded frame whose edge takes the colour of
the state, a top bar with the state and the child, cards for the setup form, the readiness check
and the confirm step, and a bottom bar of buttons. Every button is also a key (written on the
button), so the keyboard works as before; a click sends the same key to Controller.
With show_score=True a card on the right shows the live score and the last saved try.

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
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from app.session import AGE_RANGE_MONTHS, age_in_range
from common import body25 as b

W, H = 1280, 792  # window picture (pixels)
PAD = 16
VX, VY, VW, VH = PAD, PAD, W - 2 * PAD, 702  # video frame, 1248 x 702 = 16:9
BAR_Y, BAR_H = VY + VH + 12, 46  # bottom bar of buttons
RADIUS = 16
CARD_W = 320  # score cards on the right (show_score only)
WINDOW_TITLE = "Gross motor screening (screening aid, not a diagnosis)"


def _bgr(hex_colour: str) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


BG = _bgr("#0d1117")
SURFACE = _bgr("#161b22")
RAISED = _bgr("#21262d")
LINE = _bgr("#30363d")
TEXT = _bgr("#f0f6fc")
DIM = _bgr("#9198a1")
FAINT = _bgr("#6e7681")
ACCENT = _bgr("#2f81f7")
GREEN = _bgr("#238636")
RED = _bgr("#da3633")
AMBER = _bgr("#d29922")
PURPLE = _bgr("#a371f7")
LETTERBOX = _bgr("#05070a")
NEUTRAL = _bgr("#1c2128")
SHADOW = _bgr("#0b0d10")

STATE_COLOURS = {
    "setup": _bgr("#8b949e"),
    "readiness": _bgr("#58a6ff"),
    "idle": _bgr("#8b949e"),
    "ready": _bgr("#3fb950"),
    "recording": _bgr("#f85149"),
    "confirm": _bgr("#d29922"),
}
STATE_NAMES = {
    "setup": "New child",
    "readiness": "Readiness check",
    "idle": "Waiting",
    "ready": "Ready",
    "recording": "Recording",
    "confirm": "Check the try",
}
SCORE_NAMES = dict(b.LABELS)  # 0 Bad, 1 Good, 2 Perfect
SCORE_COLOURS = {0: _bgr("#f85149"), 1: _bgr("#e3b341"), 2: _bgr("#3fb950")}
SCORE_WORDS = {
    0: "did not complete the task this time",
    1: "completed the task with some difficulty",
    2: "completed the task smoothly",
}
TASK_SHORT = {1: "Stand", 2: "Hop", 3: "Jump"}

# Stick figure: right side orange, left side blue, middle white (as in viz.replay).
_RIGHT_PTS = {b.R_SHOULDER, b.R_ELBOW, b.R_WRIST, b.R_HIP, b.R_KNEE, b.R_ANKLE, b.R_EYE, b.R_EAR}
_RIGHT_PTS |= set(b.RIGHT_FOOT)
_LEFT_PTS = {b.L_SHOULDER, b.L_ELBOW, b.L_WRIST, b.L_HIP, b.L_KNEE, b.L_ANKLE, b.L_EYE, b.L_EAR}
_LEFT_PTS |= set(b.LEFT_FOOT)
FIG_RIGHT, FIG_LEFT, FIG_MID = _bgr("#ff9f43"), _bgr("#4da3ff"), _bgr("#f0f6fc")
MIN_SCORE = 0.05  # same as viz.replay: a point with a lower score is not drawn


# ---------------------------------------------------------------------- fonts and text

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
                    pass  # keep the default weight
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


# ---------------------------------------------------------------------- shapes


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
    inner, outer = np.array(_bgr("#1b222c"), np.float32), np.array(BG, np.float32)
    noise = np.random.default_rng(0).uniform(-0.5, 0.5, (VH, VW, 1)).astype(np.float32)
    return np.clip(inner * (1 - t) + outer * t + 0.5 + noise, 0, 255).astype(np.uint8)  # dither


def _lighten(c, f: float = 0.14):
    return tuple(int(v + (255 - v) * f) for v in c)


class Canvas:
    """The window picture plus every string drawn on it (tests read .texts)."""

    def __init__(self, w: int = W, h: int = H, bg=BG):
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

    def text(self, x, y, s, size=16, weight="regular", colour=TEXT, anchor="lt", alpha=1.0):
        """anchor: l/c/r then t (top of the line box), m (middle of capitals) or b (baseline)."""
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
        self.paint(int(round(x)) - 2, int(round(y)), ink if alpha >= 1 else ink * alpha, colour)
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
        """Vertical fade of colour over the picture (eased), e.g. to make the top bar readable."""
        t = np.linspace(0, 1, h, dtype=np.float32)
        t = t * t * (3 - 2 * t)
        a = (a_top + (a_bottom - a_top) * t)[:, None, None]
        self.paint(x, y, np.broadcast_to(a, (h, w, 1)), colour)

    def dot(self, cx, cy, r, colour):
        import cv2

        cv2.circle(self.img, (int(cx * 4), int(cy * 4)), int(r * 4), colour, -1, cv2.LINE_AA, 2)


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
    q = np.round((pts[:, :2] * scale + (ox, oy)) * 4).astype(int)  # 2 bits of sub-pixel
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


# ---------------------------------------------------------------------- buttons


@dataclass
class Button:
    """A clickable key. key=None is a plain label in the bar."""

    key: str | None
    label: str
    cap: str = ""
    kind: str = "ghost"  # primary, ghost, seg, label
    colour: tuple = ACCENT
    selected: bool = False
    dot: tuple | None = None
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0

    def contains(self, px, py) -> bool:
        return (
            self.key is not None
            and self.x <= px < self.x + self.w
            and self.y <= py < self.y + self.h
        )


_LABEL = (15, "semibold")
_CAP = (12, "medium")


def _width(btn: Button) -> int:
    if btn.kind == "label":
        return math.ceil(measure(btn.label, 14, "medium"))
    w = 16 + measure(btn.label, *_LABEL) + 16
    if btn.dot:
        w += 14
    if btn.cap:  # label, 10 px, key cap, 10 px (see _draw_button)
        w += 10 + measure(btn.cap, *_CAP) + 14 + 10 - 16
    return math.ceil(w)


def buttons(runner) -> list[Button]:
    """The bottom bar for the current state, laid out. Pure: the same Runner gives the same bar."""
    c = runner.ctrl
    st = c.state
    left: list[Button] = []
    right: list[Button] = []
    if st == "setup":
        left.append(Button(None, "Type with the keyboard", kind="label"))
        nxt = "Continue" if c.field == 2 else "Next"
        right += [Button("ESC", "Quit", "Esc"), Button("ENTER", nxt, "Enter", "primary")]
    elif st == "readiness":
        right += [
            Button("N", "Next child", "N"),
            Button("Q", "Quit", "Q"),
            Button("ENTER", "Start", "Enter", "primary", GREEN),
        ]
    else:
        task = c.choice_task if st == "confirm" else c.preset_task
        leg = c.choice_leg if st == "confirm" else c.preset_leg
        left.append(Button(None, "Task" if task else "Task · auto", kind="label"))
        for k in (1, 2, 3):
            left.append(Button(str(k), TASK_SHORT[k], str(k), "seg", selected=task == k))
        left.append(Button(None, "", kind="gap"))
        left.append(Button(None, "Leg" if leg else "Leg · auto", kind="label"))
        for k, name in (("L", "Left"), ("R", "Right")):
            left.append(Button(k, name, k, "seg", selected=leg == k))
        if st in ("idle", "ready"):
            if c.consent == 3:
                on = bool(c.video)
                right.append(
                    Button("V", "Video on" if on else "Video off", "V", dot=RED if on else FAINT)
                )
            right += [
                Button("N", "Next child", "N"),
                Button("Q", "Quit", "Q"),
                Button("SPACE", "Start", "Space", "primary", GREEN),
            ]
        elif st == "recording":
            right += [
                Button("BACKSPACE", "Cancel", "Backspace"),
                Button("SPACE", "Stop", "Space", "primary", RED),
            ]
        elif st == "confirm":
            right += [
                Button("BACKSPACE", "Cancel", "Backspace"),
                Button("ENTER", "Save", "Enter", "primary"),
            ]
    x = VX
    for btn in left:
        if btn.kind == "gap":
            x += 14
            continue
        btn.x, btn.y, btn.w, btn.h = x, BAR_Y, _width(btn), BAR_H
        x += btn.w + (10 if btn.kind == "label" else 6)
    x = VX + VW
    for btn in reversed(right):
        btn.w, btn.y, btn.h = _width(btn), BAR_Y, BAR_H
        btn.x = x - btn.w
        x = btn.x - 10
    return [btn for btn in left + right if btn.kind != "gap"]


def hit(btns: list[Button], x, y) -> str | None:
    """The key of the button under (x, y), or None."""
    for btn in btns:
        if btn.contains(x, y):
            return btn.key
    return None


def _draw_button(cv: Canvas, btn: Button, hovered: bool) -> None:
    cy = btn.y + btn.h / 2
    if btn.kind == "label":
        cv.text(btn.x, cy, btn.label, 14, "medium", DIM, "lm")
        return
    filled = btn.kind == "primary" or btn.selected
    fill = btn.colour if btn.kind == "primary" else (ACCENT if btn.selected else RAISED)
    if hovered:
        fill = _lighten(fill)
    cv.rrect(btn.x, btn.y, btn.w, btn.h, 10, fill=fill, border=None if filled else LINE)
    x = btn.x + 16
    if btn.dot:
        cv.dot(x + 4, cy, 4, btn.dot)
        x += 14
    cv.text(x, cy, btn.label, *_LABEL, colour=TEXT, anchor="lm")
    if btn.cap:
        kw = measure(btn.cap, *_CAP) + 14
        kx = btn.x + btn.w - 10 - kw
        cv.rrect(kx, cy - 11, kw, 22, 6, fill=(0, 0, 0), alpha=0.22, border=TEXT, border_alpha=0.22)
        cv.text(kx + kw / 2, cy, btn.cap, *_CAP, colour=TEXT, anchor="cm", alpha=0.85)


# ---------------------------------------------------------------------- parts of the picture


def _video(cv: Canvas, runner, view: str) -> None:
    import cv2

    area = cv.img[VY : VY + VH, VX : VX + VW]
    fw, fh = runner.frame_size
    s = min(VW / fw, VH / fh)
    dw, dh = int(round(fw * s)), int(round(fh * s))
    ox, oy = (VW - dw) // 2, (VH - dh) // 2
    if view == "neutral":
        area[:] = NEUTRAL
        return
    if view == "camera" and runner.last_frame is not None:
        area[:] = LETTERBOX
        area[oy : oy + dh, ox : ox + dw] = cv2.resize(
            cv2.flip(runner.last_frame, 1), (dw, dh), interpolation=cv2.INTER_LINEAR
        )
    else:
        area[:] = _backdrop()
    if runner.last_points is not None:
        pts = runner.last_points.copy()
        pts[:, 0] = np.where(pts[:, 2] > 0, fw - pts[:, 0], 0)  # mirror for display only
        draw_skeleton(area, pts, s, ox, oy)


def _chip(cv: Canvas, x_right, cy, label, colour=TEXT, fill=SURFACE, fill_alpha=0.78, dot=None):
    """A pill ending at x_right; returns its left edge."""
    w = 14 + (14 if dot else 0) + measure(label, 13, "medium") + 14
    x = x_right - w
    cv.rrect(x, cy - 15, w, 30, 15, fill=fill, alpha=fill_alpha, border=LINE, border_alpha=0.8)
    tx = x + 14
    if dot:
        cv.dot(tx + 4, cy, 4, dot)
        tx += 14
    cv.text(tx, cy, label, 13, "medium", colour, "lm")
    return x


def _recording_seconds(runner) -> float:
    buf = getattr(runner, "buf", None)
    if not buf or len(buf) < 2:
        return 0.0
    return float(buf[-1][0] - buf[0][0])


def _top_bar(cv: Canvas, runner) -> None:
    c = runner.ctrl
    cv.fade(VX, VY, VW, 132, BG, 0.85, 0.0)
    x0, cy = VX + 24, VY + 36
    w_title = cv.text(x0, cy, "Gross Motor Screening", 19, "semibold", TEXT, "lm")
    cv.text(x0, cy + 23, "Screening aid · not a diagnosis", 13, "regular", DIM, "lm")
    limit = x0 + max(w_title, measure("Screening aid · not a diagnosis", 13)) + 24

    colour = STATE_COLOURS.get(c.state, DIM)
    name = STATE_NAMES.get(c.state, c.state)
    if c.state == "recording":
        name = f"{name}  {_recording_seconds(runner):4.1f} s"
    pulse = c.state != "recording" or (time.monotonic() % 1.0) < 0.6
    x = _chip(cv, VX + VW - 20, cy, name, TEXT, colour, 0.28, colour if pulse else FAINT)

    chips = []
    if c.child:
        chips.append((f"{c.child} · {c.age} months · consent {c.consent}", TEXT, None))
        chips.append((f"{c.tries_saved} saved", DIM, None))
        if not age_in_range(c.age):
            lo, hi = AGE_RANGE_MONTHS
            chips.append((f"Outside {lo}–{hi} months", _bgr("#e3b341"), AMBER))
        if c.video:
            chips.append(("Video on", TEXT, RED))
    fps = runner.fps() if hasattr(runner, "fps") else 0.0
    mode = "Auto" if c.mode == "auto" else "Manual"
    chips.append((f"{mode} · {fps:.0f} fps" if fps else mode, DIM, None))
    for label, fg, dot in chips:
        w = 14 + (14 if dot else 0) + measure(label, 13, "medium") + 14
        if x - 8 - w < limit:
            break
        x = _chip(cv, x - 8, cy, label, fg, dot=dot)


def _card(cv: Canvas, x, y, w, h, alpha=0.9):
    cv.rrect(x, y, w, h, 14, fill=SURFACE, alpha=alpha, border=LINE)


def _setup_card(cv: Canvas, c) -> None:
    w = 500
    names = ("Child code", "Age in months", "Consent level")
    lo, hi = AGE_RANGE_MONTHS
    hints = ("e.g. K03", f"{lo}–{hi}", "1, 2 or 3 from the signed form")
    msg_lines = wrap(c.message, 14, "regular", w - 56, 2)
    h = 28 + 34 + 26 + 18 + 3 * 82 + len(msg_lines) * 20 + 24
    x, y = VX + (VW - w) // 2, VY + (VH - h) // 2 + 20
    _card(cv, x, y, w, h, 0.96)
    cx = x + 28
    yy = y + 28
    cv.text(cx, yy, "New child", 26, "semibold", TEXT)
    yy += 38
    cv.text(cx, yy, "Enter moves to the next field.", 14, "regular", DIM)
    yy += 36
    caret_on = (time.monotonic() % 1.0) < 0.6
    for i, name in enumerate(names):
        focused = i == c.field
        cv.text(cx, yy, name, 13, "medium", TEXT if focused else DIM)
        if i == 2:
            cv.text(x + w - 28, yy, "3 allows video", 13, "regular", FAINT, "rt")
        yy += 22
        bw = w - 56
        cv.rrect(
            cx,
            yy,
            bw,
            46,
            10,
            fill=RAISED,
            border=ACCENT if focused else LINE,
            bw=2 if focused else 1,
        )
        value = c.form[i]
        tx = cx + 14
        if value:
            tx += cv.text(tx, yy + 23, value, 19, "regular", TEXT, "lm")
        elif not focused:
            cv.text(tx, yy + 23, hints[i], 16, "regular", FAINT, "lm")
        if focused:
            if caret_on:
                cv.rrect(tx + 2, yy + 12, 2, 22, 1, fill=ACCENT)
            if not value:
                cv.text(tx + 10, yy + 23, hints[i], 16, "regular", FAINT, "lm")
        yy += 46 + 14
    error = c.message and not c.message.startswith(("Type", "Next child"))
    for line in msg_lines:
        cv.text(cx, yy, line, 14, "regular", _bgr("#e3b341") if error else DIM)
        yy += 20


def _icon_ok(cv: Canvas, cx, cy):
    import cv2

    cv.dot(cx, cy, 10, _bgr("#3fb950"))
    pts = np.array([[cx - 5, cy], [cx - 1.5, cy + 4], [cx + 5, cy - 4]]) * 4
    cv2.polylines(cv.img, [pts.astype(np.int32)], False, BG, 2, cv2.LINE_AA, 2)


def _icon_warn(cv: Canvas, cx, cy):
    cv.dot(cx, cy, 10, AMBER)
    cv.text(cx, cy, "!", 14, "bold", BG, "cm")


def _readiness_card(cv: Canvas, c) -> None:
    w = 470
    rows = []
    for p in c.problems:
        rows.append(wrap(p, 15, "regular", w - 90, 3))
    foot = (
        "Press Start (Enter) when the child is at the start line."
        if not c.problems
        else "You can start anyway with Start (Enter)."
    )
    foot_lines = wrap(foot, 14, "regular", w - 56)
    body_h = 30 if not rows else sum(len(r) * 21 + 10 for r in rows)
    h = 24 + 30 + 16 + body_h + 8 + len(foot_lines) * 20 + 22
    x, y = VX + 24, VY + 96
    _card(cv, x, y, w, h)
    cx, yy = x + 24, y + 24
    cv.text(cx, yy, "Readiness check", 20, "semibold", TEXT)
    yy += 46
    if not rows:
        _icon_ok(cv, cx + 10, yy + 10)
        cv.text(cx + 32, yy + 10, "Camera and child look ready.", 15, "regular", TEXT, "lm")
        yy += 30
    for lines in rows:
        _icon_warn(cv, cx + 10, yy + 10)
        for line in lines:
            cv.text(cx + 32, yy + 10, line, 15, "regular", TEXT, "lm")
            yy += 21
        yy += 10
    yy += 8
    for line in foot_lines:
        cv.text(cx, yy, line, 14, "regular", DIM)
        yy += 20


def _confirm_card(cv: Canvas, c) -> None:
    if not c.guess:
        return
    gt, gl, gc = c.guess
    ct, cl = c.choice_task, c.choice_leg
    corrected = ct != gt or (ct != 3 and cl != gl)
    w, h = 520, 268
    x, y = VX + (VW - w) // 2, VY + (VH - h) // 2 + 20
    _card(cv, x, y, w, h, 0.96)
    cx, yy = x + 28, y + 28
    cv.text(cx, yy, "Check this try", 24, "semibold", TEXT)
    yy += 36
    cv.text(cx, yy, "Correct the task or leg if the guess is wrong, then save.", 14, "regular", DIM)
    yy += 38
    cv.text(cx, yy, "Guess", 13, "medium", DIM)
    cv.text(x + w - 28, yy, f"{gc:.0%} sure", 13, "medium", DIM, "rt")
    yy += 22
    cv.text(cx, yy, f"Task {gt} · {b.TASKS[gt]} · leg {gl}", 17, "regular", TEXT)
    yy += 40
    cv.text(cx, yy, "Will be saved as", 13, "medium", DIM)
    if corrected:
        cv.text(x + w - 28, yy, "Corrected", 13, "semibold", _bgr("#e3b341"), "rt")
    yy += 22
    cv.text(cx, yy, f"Task {ct} · {b.TASKS[ct]} · leg {cl}", 20, "semibold", TEXT)
    yy += 44
    cv.text(cx, yy, "1/2/3 task · L/R leg · Enter save · Backspace cancel", 13, "regular", FAINT)


def _toast(cv: Canvas, message: str) -> None:
    if not message:
        return
    lines = wrap(message, 15, "regular", 760, 3)
    w = max(measure(t, 15) for t in lines) + 40
    h = len(lines) * 22 + 22
    x, y = VX + (VW - w) / 2, VY + VH - 20 - h
    cv.rrect(x, y, w, h, 12, fill=SURFACE, alpha=0.9, border=LINE)
    yy = y + 11
    for line in lines:
        cv.text(VX + VW / 2, yy, line, 15, "regular", TEXT, "ct")
        yy += 22


def _score_cards(cv: Canvas, runner) -> None:
    """Live score and last saved try (show_score only). Reference only, never a diagnosis."""
    c = runner.ctrl
    x, y, w = VX + VW - 24 - CARD_W, VY + 96, CARD_W
    live = getattr(runner, "live", None)
    if c.state != "setup":
        y = _live_card(cv, live, x, y, w) + 12
    result = getattr(runner, "result", None)
    if result and c.state in ("idle", "ready"):
        _result_card(cv, result, x, y, w)


def _live_card(cv: Canvas, live, x, y, w) -> int:
    inner = w - 40
    if not live or "error" in live:
        detail = (
            live["error"] if live else "The score appears after about 1 s of the child in view."
        )
        lines = wrap(detail, 13, "regular", inner, 3)
        h = 20 + 18 + 14 + 24 + len(lines) * 18 + 18
        _card(cv, x, y, w, h)
        cx, yy = x + 20, y + 20
        cv.text(cx, yy, "LIVE SCORE", 12, "semibold", DIM)
        yy += 32
        cv.text(cx, yy, "Waiting for a clear view", 17, "semibold", TEXT)
        yy += 30
        for line in lines:
            cv.text(cx, yy, line, 13, "regular", DIM)
            yy += 18
        return y + h
    sc = int(live["score"])
    task = int(live.get("task") or 1)
    conf = float(live.get("confidence") or 0.0)
    col = SCORE_COLOURS[sc]
    reasons = wrap(live.get("reasons", ""), 13, "regular", inner, 3)
    h = 20 + 18 + 16 + 64 + 14 + 6 + 16 + len(reasons) * 18 + 18
    _card(cv, x, y, w, h)
    cx, yy = x + 20, y + 20
    cv.text(cx, yy, "LIVE SCORE", 12, "semibold", DIM)
    cv.text(x + w - 20, yy, f"Task {task} · {TASK_SHORT.get(task, '')}", 12, "semibold", DIM, "rt")
    yy += 34
    cv.dot(cx + 30, yy + 30, 30, col)
    cv.text(cx + 30, yy + 30, str(sc), 32, "bold", BG, "cm")
    cv.text(cx + 76, yy + 20, SCORE_NAMES[sc], 28, "semibold", col, "lm")
    cv.text(cx + 76, yy + 48, f"{conf:.0%} sure", 14, "regular", DIM, "lm")
    yy += 64 + 14
    cv.rrect(cx, yy, inner, 6, 3, fill=RAISED)
    if conf > 0:
        cv.rrect(cx, yy, max(6, inner * min(conf, 1.0)), 6, 3, fill=col)
    yy += 6 + 16
    for line in reasons:
        cv.text(cx, yy, line, 13, "regular", DIM)
        yy += 18
    return y + h


def _result_card(cv: Canvas, r: dict, x, y, w) -> None:
    inner = w - 40
    check = r.get("check") or r.get("score") is None
    reasons = wrap(r.get("reasons", ""), 13, "regular", inner, 3)
    head = f"Task {r['task']} · {b.TASKS[r['task']]} · leg {r['leg']}"
    if check:
        words = wrap("Model is unsure: check by teacher.", 15, "regular", inner, 2)
        col, chip = PURPLE, "Check by teacher"
    else:
        s = int(r["score"])
        words = wrap(SCORE_WORDS[s].capitalize() + ".", 15, "regular", inner, 2)
        col, chip = SCORE_COLOURS[s], f"{s} · {SCORE_NAMES[s]}"
    h = 20 + 18 + 12 + 22 + 12 + 30 + 12 + len(words) * 21 + 6 + len(reasons) * 18 + 10 + 18 + 18
    _card(cv, x, y, w, h)
    cx, yy = x + 20, y + 20
    cv.text(cx, yy, "LAST SAVED TRY", 12, "semibold", DIM)
    yy += 30
    cv.text(cx, yy, head, 15, "medium", TEXT)
    yy += 34
    cw = measure(chip, 14, "semibold") + 28
    cv.rrect(cx, yy, cw, 30, 15, fill=col, alpha=0.22, border=col, border_alpha=0.7)
    cv.text(cx + 14, yy + 15, chip, 14, "semibold", col, "lm")
    if not check:
        cv.text(
            cx + cw + 12,
            yy + 15,
            f"{float(r.get('confidence') or 0):.0%} sure",
            13,
            "regular",
            DIM,
            "lm",
        )
    yy += 42
    for line in words:
        cv.text(cx, yy, line, 15, "regular", TEXT)
        yy += 21
    yy += 6
    for line in reasons:
        cv.text(cx, yy, line, 13, "regular", DIM)
        yy += 18
    yy += 10
    cv.text(cx, yy, "Reference only, not a diagnosis.", 12, "regular", FAINT)


# ---------------------------------------------------------------------- the whole picture


def render(runner, view: str = "stick", show_score: bool = False) -> Canvas:
    """The window picture for the Runner's current state.

    view: stick (stick figure on a dark backdrop), camera (mirrored camera picture with the
    stick figure on top), neutral (no person shown).
    """
    c = runner.ctrl
    cv = Canvas()
    _video(cv, runner, view)
    if c.state in ("setup", "confirm"):
        cv.dim(VX, VY, VW, VH, 0.4)
    cv.round_corners(VX, VY, VW, VH, RADIUS)
    _top_bar(cv, runner)
    strong = c.state in ("readiness", "ready", "recording", "confirm")
    cv.ring(VX, VY, VW, VH, RADIUS, 3 if strong else 1, STATE_COLOURS[c.state] if strong else LINE)
    if c.state == "setup":
        _setup_card(cv, c)
    elif c.state == "readiness":
        _readiness_card(cv, c)
    elif c.state == "confirm":
        _confirm_card(cv, c)
    if show_score:
        _score_cards(cv, runner)
    if c.state not in ("setup", "confirm"):
        _toast(cv, c.message)
    btns = buttons(runner)
    pointer = getattr(runner, "pointer", None)
    over = hit(btns, *pointer) if pointer else None
    for btn in btns:
        _draw_button(cv, btn, hovered=btn.key is not None and btn.key == over)
    return cv


# ---------------------------------------------------------------------- the window loop


def run_window(runner, stream, draw, paced: bool = False, title: str = WINDOW_TITLE):
    """Open the window and run until Q (or the window is closed). Returns runner.saved.

    draw(runner) -> picture. Pose detection runs in a PoseWorker thread; this loop feeds every
    new frame to the Runner (in order), draws once, then handles the keys and mouse clicks.
    """
    import cv2

    from app.ui import PoseWorker, key_name

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    clicks: list[str] = []
    set_mouse = getattr(cv2, "setMouseCallback", None)
    if set_mouse is not None:
        set_mouse(title, _mouse_handler(runner, clicks, cv2))
    worker = PoseWorker(stream, paced=paced)
    try:
        while not runner.quit:
            for item in worker.take(wait_s=0.005):  # short wait: no busy loop when idle
                runner.feed(item)
            if worker.error is not None:  # e.g. no camera frame for 5 s (PoseStream closes)
                runner.ctrl.message = (
                    f"Camera stopped ({worker.error}). Earlier tries are saved. Backspace if a "
                    "try is open, then Q; check the camera and restart with the same child code."
                )
            elif worker.ended and worker.items.empty():
                runner.ctrl.message = "The video ended. Press Q to quit."
            cv2.imshow(title, draw(runner))
            runner.press(key_name(cv2.waitKeyEx(1)))
            while clicks and not runner.quit:
                runner.press(clicks.pop(0))
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        stopped = worker.stop()
        runner.close()  # an unconfirmed video clip is discarded
        stream.close()  # also wakes a camera read that is still waiting
        if not stopped:
            worker.stop(timeout=2.0)
        cv2.destroyAllWindows()
    return runner.saved


def _mouse_handler(runner, clicks: list, cv2):
    """A click is a press and release on the same button; it queues that button's key."""
    down = getattr(cv2, "EVENT_LBUTTONDOWN", 1)
    up = getattr(cv2, "EVENT_LBUTTONUP", 4)
    pressed: dict = {"key": None}

    def on_mouse(event, x, y, _flags, _param):
        runner.pointer = (x, y)  # render() lights up the button under the pointer
        if event == down:
            pressed["key"] = hit(buttons(runner), x, y)
        elif event == up:
            k = hit(buttons(runner), x, y)
            if k is not None and k == pressed["key"]:
                clicks.append(k)
            pressed["key"] = None

    return on_mouse
