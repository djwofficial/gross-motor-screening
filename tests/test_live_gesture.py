"""live.gesture (Daryn, 2026-11-25): open palm and crossed arms from the body points, open fingers
from a (stub) hand model, and the fallback when the hand model is missing."""

import time

import numpy as np
import pytest

from common import body25 as b
from common.format import make_fake_try
from live import gesture as g

FPS = 30.0
STAND = make_fake_try(1, "R", seconds=0.1, idle_before=1.0).points[0].astype(np.float32)


def raised(pose, side="R"):
    p = pose.copy()
    sh, el, wr = (b.R_SHOULDER, b.R_ELBOW, b.R_WRIST) if side == "R" else g._ARM["L"]
    dx = -1 if side == "R" else 1
    p[el, :2] = p[sh, :2] + (25 * dx, -45)
    p[wr, :2] = p[sh, :2] + (30 * dx, -100)
    return p


def crossed(pose):
    p = pose.copy()
    hip = p[b.MID_HIP, :2].copy()
    p[b.L_ELBOW, :2] = hip + (45, -60)
    p[b.L_WRIST, :2] = hip + (-25, -85)
    p[b.R_ELBOW, :2] = hip + (-45, -60)
    p[b.R_WRIST, :2] = hip + (25, -85)
    return p


def one_foot_up(pose):
    p = pose.copy()
    for i in b.LEFT_FOOT:
        p[i, 1] -= 40
    return p


def run(det, pose, seconds, want, t0=0.0, frame=None, after=None):
    """Feed the same pose for seconds; return [(t, event)] of the events."""
    out = []
    for i in range(int(round(seconds * FPS))):
        t = t0 + i / FPS
        ev = det.update(pose, t, want, frame)
        if after is not None:
            after(t)
        if ev:
            out.append((t, ev))
    return out


def hand_points(open_=True):
    kp = np.zeros((21, 2))
    sc = np.full(21, 0.9)
    kp[1:5] = [(-20, -10), (-30, -20), (-38, -28), (-44, -34)]  # thumb
    for f, x in enumerate((-12, -4, 4, 12)):
        mcp, pip, dip, tip = 5 + 4 * f, 6 + 4 * f, 7 + 4 * f, 8 + 4 * f
        kp[mcp], kp[pip], kp[dip] = (x, -40), (x, -60), (x, -72)
        kp[tip] = (x, -82) if open_ else (x, -35)
    return kp, sc


def test_raised_hand_only_with_feet_down():
    assert g.raised_sides(STAND) == []
    assert g.raised_sides(raised(STAND, "R")) == ["R"]
    assert g.raised_sides(raised(STAND, "L")) == ["L"]
    assert g.raised_sides(one_foot_up(raised(STAND, "R"))) == []
    assert g.raised_sides(None) == []


def test_palm_fires_once_after_hold_and_again_after_lowering():
    det = g.GestureDetector()  # no hand model: a raised hand is enough
    ev = run(det, raised(STAND), 1.5, "palm")
    assert len(ev) == 1 and ev[0][1] == "palm"
    assert g.PALM_HOLD_S - 0.05 <= ev[0][0] <= g.PALM_HOLD_S + 0.1
    assert run(det, STAND, 0.5, "palm", t0=1.5) == []  # hand down: armed again
    ev = run(det, raised(STAND, "L"), 1.0, "palm", t0=2.0)
    assert [e for _, e in ev] == ["palm"]


def test_no_palm_on_one_leg_or_when_not_watched():
    assert run(g.GestureDetector(), one_foot_up(raised(STAND)), 2.0, "palm") == []
    assert run(g.GestureDetector(), raised(STAND), 2.0, "cross") == []
    assert run(g.GestureDetector(), raised(STAND), 2.0, None) == []


def test_cross_fires_after_hold_and_remembers_when_it_began():
    assert g.cross_point(STAND) is None
    assert g.cross_point(crossed(STAND)) is not None
    assert g.cross_point(one_foot_up(crossed(STAND))) is None
    det = g.GestureDetector()
    run(det, STAND, 1.0, "cross")
    ev = run(det, crossed(STAND), 1.5, "cross", t0=1.0)
    assert [e for _, e in ev] == ["cross"]
    assert ev[0][0] - 1.0 == pytest.approx(g.CROSS_HOLD_S, abs=0.1)
    assert det.cross_since == pytest.approx(1.0, abs=1e-6)


def test_wrists_on_the_wrong_side_also_count_as_an_x():
    p = crossed(STAND)
    hip = p[b.MID_HIP, :2]
    p[b.L_ELBOW, :2] = hip + (45, -60)  # forearms short of each other, wrists swapped sides
    p[b.L_WRIST, :2] = hip + (-5, -85)
    p[b.R_ELBOW, :2] = hip + (-45, -60)
    p[b.R_WRIST, :2] = hip + (5, -60)
    assert (
        g._intersection(p[b.L_ELBOW, :2], p[b.L_WRIST, :2], p[b.R_ELBOW, :2], p[b.R_WRIST, :2])
        is None
    )
    assert g.cross_point(p) is not None


def test_hand_is_open_and_hand_box():
    assert g.hand_is_open(*hand_points(True))
    assert not g.hand_is_open(*hand_points(False))
    kp, sc = hand_points(True)
    assert not g.hand_is_open(kp, sc * 0.2)  # model unsure
    box = g.hand_box(raised(STAND), "R", (1280, 720))
    x0, y0, x1, y1 = box
    wx, wy = raised(STAND)[b.R_WRIST, :2]
    assert x0 < wx < x1 and y1 > wy - 1  # the box sits on and beyond the wrist
    assert g.hand_box(STAND, "R", (1280, 720)) is not None


class StubHand:
    def __init__(self, open_):
        self.open_, self.calls = open_, 0

    def __call__(self, image, bboxes):
        self.calls += 1
        kp, sc = hand_points(self.open_)
        return kp[None], sc[None]


def wait_for(cond, timeout=2.0):
    end = time.monotonic() + timeout
    while not cond() and time.monotonic() < end:
        time.sleep(0.002)
    return cond()


@pytest.mark.parametrize("open_, fires", [(True, True), (False, False)])
def test_hand_model_decides_open_palm(open_, fires):
    checker = g.HandChecker(model=StubHand(open_))
    det = g.GestureDetector(hand_checker=checker)
    frame = np.zeros((720, 1280, 3), np.uint8)

    def settle(t):  # let the hand thread answer the request of this frame
        if det._last_submit == t:
            wait_for(lambda: (checker.latest() or {}).get("t") == t)

    try:
        ev = run(det, raised(STAND), 2.0, "palm", frame=frame, after=settle)
    finally:
        checker.close()
    assert checker.model.calls > 0
    assert bool(ev) is fires
    if not fires:
        assert det.waiting_for_open


def test_without_hand_model_a_raised_hand_is_enough(monkeypatch):
    def broken(self):
        raise RuntimeError("offline")

    monkeypatch.setattr(g.HandChecker, "load", broken)
    checker = g.HandChecker()
    assert wait_for(lambda: checker.available is False)
    det = g.GestureDetector(hand_checker=checker)
    assert det.status == "no hand model" and "offline" in checker.error
    ev = run(det, raised(STAND), 1.5, "palm", frame=np.zeros((720, 1280, 3), np.uint8))
    assert [e for _, e in ev] == ["palm"]
    checker.close()


def test_palm_at_face_or_chest_height_counts():
    p = STAND.copy()
    p[b.R_ELBOW, :2] = p[b.R_SHOULDER, :2] + (-35, 40)
    p[b.R_WRIST, :2] = p[b.R_SHOULDER, :2] + (-30, 5)  # hand up beside the chest, below the chin
    assert g.raised_sides(p) == ["R"]
    p[b.R_WRIST, :2] = p[b.R_SHOULDER, :2] + (-30, 60)  # wrist below the elbow: arm hanging
    assert g.raised_sides(p) == []


def test_feet_or_hips_out_of_the_picture_do_not_block():
    p = raised(STAND)
    p[[*b.LEFT_FOOT, *b.RIGHT_FOOT], 2] = 0.0  # standing close: no feet
    assert g.raised_sides(p) == ["R"]
    p[[b.MID_HIP, b.L_HIP, b.R_HIP, b.L_KNEE, b.R_KNEE], 2] = 0.0  # no hips: shoulder ruler
    assert g._torso(p) == pytest.approx(g.SHOULDER_TO_TORSO * 60, rel=0.05)
    assert g.raised_sides(p) == ["R"]
    assert g.cross_point(crossed(p)) is not None


def test_unsure_fingers_do_not_count_and_hint_says_what_is_missing():
    kp, sc = hand_points(True)
    assert g.open_fingers(kp, sc) == 4
    assert g.open_fingers(kp, np.full(21, 0.2)) == 0
    checker = g.HandChecker(model=StubHand(False))
    det = g.GestureDetector(hand_checker=checker)
    frame = np.zeros((720, 1280, 3), np.uint8)
    try:
        det.update(raised(STAND), 0.0, "palm", frame)
        assert wait_for(lambda: checker.latest() is not None)
        det.update(raised(STAND), 0.05, "palm", frame)
    finally:
        checker.close()
    assert det.hint == "Open your hand · 0/4 fingers"
    assert g.GestureDetector().hint == ""
