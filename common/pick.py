"""Pick the child when 2+ people are in the picture. Owner: Marty (Nov 15). Revallo reuses it live.

Shared interface (final, change only through Daryn):

    pick_child(frames) -> np.ndarray float32 (T, 25, 3)
        frames: list of length T; frames[i] is a float32 array (P_i, 25, 3) with every person
                found in frame i (BODY_25, pixels). P_i may be 0.
        returns the child's points per frame; (0, 0, 0) for frames where the child wasn't found.

Plan (Marty): lock on the person at the start (dataset: the child near the centre / start line),
then follow the nearest neck frame by frame. 28% of group B files have 2+ people in some frame.

STUB: takes the first person in each frame. Replace with the real version.
"""

from __future__ import annotations

import numpy as np

from common import body25 as b


def pick_child(frames: list[np.ndarray]) -> np.ndarray:
    out = np.zeros((len(frames), b.N_POINTS, 3), dtype=np.float32)
    for i, people in enumerate(frames):
        if len(people):
            out[i] = people[0]
    return out
