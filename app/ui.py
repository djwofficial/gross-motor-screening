"""The GUI window. Owner: Freya. TODO (Nov 11 skeleton, Nov 19 testing mode, Nov 21 automatic mode).

One window on the live laptop (OpenCV window is enough; no installer, web or phone app):
- Large area: camera picture, stick figure or neutral picture (still open: ask KEE).
  Never a score in testing mode.
- Status bar: state (ready / recording / confirm), child code, age, task, leg, try number,
  video on/off, readiness problems from live.quality.check in plain words.
- Keys: 1/2/3 task, L/R leg, Space manual start/stop, Enter confirm the guess,
  Backspace cancel the try, N next child, V video on/off, Q quit.
app/main.py calls this; keep the drawing here and the logic in main.py / session.py.
"""


def run_gui(*args, **kwargs):
    raise NotImplementedError("Freya, Nov 11: app/ui.py")
