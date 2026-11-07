"""Blur faces in level-3 videos using the RTMPose head points. Owner: Theresia (Nov 22).

Only for children whose parent ticked consent level 3. Use nose, eyes and ears (BODY_25 0, 15-18)
to place an ellipse per frame, blur it, then check every frame by eye: no face may be visible.
"""


def blur_video(video_path, points_json, out_path):
    raise NotImplementedError("Theresia, Nov 22: viz/blur.py")
