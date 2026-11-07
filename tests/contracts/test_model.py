"""Contract: model/predict.py and model/task_classifier.py (owner Marty)."""

import pytest

from common.format import make_fake_try
from features.clues import compute
from model.predict import predict
from model.task_classifier import predict_task


@pytest.mark.parametrize("task", [1, 2, 3])
def test_predict_returns_score_and_confidence(task):
    tr = make_fake_try(task=task)
    score, conf = predict(task, compute(task, tr.points, tr.t))
    assert isinstance(score, int) and score in (0, 1, 2)
    assert isinstance(conf, float) and 0.0 <= conf <= 1.0


@pytest.mark.parametrize("leg", ["L", "R"])
@pytest.mark.parametrize("task", [1, 2])
def test_predict_task_types_and_support_leg(task, leg):
    tr = make_fake_try(task=task, leg=leg)
    guessed_task, guessed_leg, conf = predict_task(tr.points, tr.t)
    assert guessed_task in (1, 2, 3) and 0.0 <= conf <= 1.0
    assert guessed_leg == leg
