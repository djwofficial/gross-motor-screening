"""Try -> stick-figure mp4 with point numbers. Owner: Freya (Nov 10).

    render(tr, out_path, fps=30, size=None, labels=True, events=None) -> Path
Draw common.body25.BONES on a black picture, one frame per row of tr.points; write the point
number next to each point when labels=True (used to check left/right by eye).
Nov 13: when events (features.clues.timeline) are given, draw a coloured bar under the figure,
one colour per state, so the moments can be checked by eye.
"""


def render(tr, out_path, fps: float = 30.0, size=None, labels: bool = True, events=None):
    raise NotImplementedError("Freya, Nov 10: viz/replay.py")
