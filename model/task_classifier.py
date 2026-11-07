"""Guess which task the child did and on which leg. Owner: Marty (Nov 22).

Shared interface (final, change only through Daryn):

    predict_task(points, t) -> (task, leg, confidence)
        points: float32 (T, 25, 3) of the child, pixels; t: float64 (T,) seconds
        task:   1, 2 or 3
        leg:    "L" or "R" = the support leg (the one the child stands or hops on);
                for task 3 (both feet) it is still returned but the app ignores it
        confidence: float 0..1 for the task guess

The operator always confirms (Enter) or corrects (1/2/3, L/R), so a wrong guess costs one key.
If accuracy on group B (GroupKFold by child) is low, the app uses keys only.

STUB: leg = the foot that stays lower on average (real, simple); task = 1 with confidence 0.0.
"""

from __future__ import annotations

import numpy as np

from common import body25 as b
from common.geometry import normalize


def predict_task(points: np.ndarray, t: np.ndarray) -> tuple[int, str, float]:
    n = normalize(points)
    ly = np.nanmean(n[:, b.L_ANKLE, 1])
    ry = np.nanmean(n[:, b.R_ANKLE, 1])
    leg = "L" if (np.isnan(ry) or ly <= ry) else "R"  # y points up: the lower foot supports
    return 1, leg, 0.0
