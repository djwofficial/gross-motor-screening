"""Task 3 (standing long jump) uses both feet: the window offers no Left/Right leg."""

from app.screen import _task_leg
from app.ui import Controller, big_lines, status_lines


def _idle_controller() -> tuple[Controller, list[float]]:
    c = Controller("manual", debounce_s=0.0)
    now = [0.0]

    def press(k):
        now[0] += 1.0
        c.key(k, now[0])

    for k in ("T", "1", "ENTER", "4", "5", "ENTER", "1", "ENTER", "ENTER"):
        press(k)
    assert c.state == "idle"
    return c, press


def test_preset_leg_ignored_for_task3():
    c, press = _idle_controller()
    press("3")
    press("L")
    assert c.preset_task == 3 and c.preset_leg is None
    assert "both feet" in c.message
    assert "both feet" in status_lines(c)[2] and "leg" not in status_lines(c)[2]


def test_preset_leg_still_works_for_task1():
    c, press = _idle_controller()
    press("1")
    press("R")
    assert c.preset_leg == "R"


def test_confirm_task3_shows_no_leg_and_ignores_lr():
    c, press = _idle_controller()
    press("SPACE")
    press("SPACE")
    c.set_guess(3, "L", 0.9)
    press("R")
    assert c.choice_leg == "L"  # unchanged: L/R does nothing for task 3
    assert not any("leg" in line for line in big_lines(c))


def test_task_leg_text():
    assert _task_leg(3, "L") == "Task 3 · standing long jump"
    assert _task_leg(1, "R").endswith("· leg R")
