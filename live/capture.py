"""Capture thread for the webcam. Owner: Revallo. TODO (Nov 13; video saving Nov 15).

Goal: grab frames in their own thread at 1280x720, 30 fps, with a time stamp per frame
(time.perf_counter), so pose estimation never blocks the camera. Turn off auto-framing
(Mac Center Stage, Windows Studio Effects "Automatic framing") before testing.

Also add a `--video FILE` mode so recorded team clips go through exactly the same code.
"""


def resample_to_fps(points, t, fps: float = 30.0):
    """TODO Revallo: resample a try (points (T,25,3), t (T,)) to a fixed fps using the time stamps,
    because the dataset is 30 fps and live fps can drop to about 15 in a dim room."""
    raise NotImplementedError("Revallo, Nov 13: resample_to_fps")
