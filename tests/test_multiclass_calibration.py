#!/usr/bin/env python
"""Multiclass probability calibration (temperature / one-vs-rest isotonic).

Run:  python -m pytest tests/test_multiclass_calibration.py -q
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Agents.TrainModel.agent_train_model import TrainModelAgent
from model_scoring import apply_multiclass_calibration


def _overconfident(n=3000, k=4, seed=0):
    """True class probabilities q, and a model output that is far too sharp (q**4)."""
    rs = np.random.RandomState(seed)
    logits = rs.normal(size=(n, k)) * 1.2
    q = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    y = np.array([rs.choice(k, p=row) for row in q])
    sharp = q ** 4
    return sharp / sharp.sum(1, keepdims=True), y


@pytest.mark.parametrize("method", ["temperature", "ovr_isotonic"])
def test_calibration_reduces_logloss_and_rows_sum_to_one(method):
    P, y = _overconfident()
    cal = TrainModelAgent._fit_multiclass_calibrator(P[:2000], y[:2000], method)
    assert cal["kind"] == "multiclass" and cal["method"] == method
    Q = apply_multiclass_calibration(P[2000:], cal)        # held-out rows
    assert np.allclose(Q.sum(1), 1.0)
    labels = list(range(P.shape[1]))
    assert log_loss(y[2000:], Q, labels=labels) < log_loss(y[2000:], P[2000:], labels=labels)


def test_temperature_keeps_the_predicted_class_and_softens():
    P, y = _overconfident()
    cal = TrainModelAgent._fit_multiclass_calibrator(P, y, "temperature")
    assert cal["T"] > 1.0                                  # over-confident → T > 1 softens
    Q = apply_multiclass_calibration(P, cal)
    assert (Q.argmax(1) == P.argmax(1)).all()
    assert Q.max(1).mean() < P.max(1).mean()


def test_calibrator_is_a_plain_picklable_dict():
    import pickle
    P, y = _overconfident(n=500)
    for method in ("temperature", "ovr_isotonic"):
        cal = TrainModelAgent._fit_multiclass_calibrator(P, y, method)
        back = pickle.loads(pickle.dumps(cal))
        assert np.allclose(apply_multiclass_calibration(P, back), apply_multiclass_calibration(P, cal))
