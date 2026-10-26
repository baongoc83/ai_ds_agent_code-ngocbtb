"""Turn a fitted final_model.pkl artifact + an encoded feature matrix into
predictions. Shared by the UI (ui/scoring.py) and replay_driver.py so every
consumer handles regression / binary / multiclass and label decoding the same way.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd

import target_encoding


def _aligned_proba(model, X, n_classes: int) -> np.ndarray:
    """predict_proba with columns aligned to class codes 0..n_classes-1 (a seed
    model that never saw a rare class returns fewer columns)."""
    p = np.asarray(model.predict_proba(X), dtype=float)
    classes = getattr(model, "classes_", np.arange(p.shape[1]))
    out = np.zeros((p.shape[0], n_classes))
    for j, c in enumerate(classes):
        if 0 <= int(c) < n_classes:
            out[:, int(c)] = p[:, j]
    return out


def apply_multiclass_calibration(P: np.ndarray, cal: Dict[str, Any]) -> np.ndarray:
    """Calibrate an (n, k) probability matrix with a fitted multiclass calibrator dict
    (see TrainModelAgent._fit_multiclass_calibrator). Rows always sum to 1."""
    P = np.clip(np.asarray(P, dtype=float), 1e-15, 1.0)
    if cal.get("method") == "temperature":
        z = np.log(P) / float(cal["T"])
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)
    if cal.get("method") == "ovr_isotonic":
        Q = np.column_stack([m.predict(P[:, c]) for c, m in enumerate(cal["models"])])
        s = Q.sum(axis=1, keepdims=True)
        # a row the per-class maps sent to all-zero keeps its raw probabilities
        return np.where(s > 1e-12, Q / np.where(s > 1e-12, s, 1.0), P / P.sum(axis=1, keepdims=True))
    return P / P.sum(axis=1, keepdims=True)


def task_of(artifact: Dict[str, Any]) -> str:
    return target_encoding.task_of(artifact.get("task") or "classification")


def class_labels(artifact: Dict[str, Any]) -> List[Any]:
    tm = artifact.get("target_mapping") or {}
    if tm.get("classes"):
        return list(tm["classes"])
    n = int(artifact.get("n_classes") or 2)
    return list(range(n))


def predict_frame(artifact: Dict[str, Any], X: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Columns: regression → prediction | binary → probability, prediction, label |
    multiclass → proba_<label>..., prediction (class code), label."""
    models = artifact.get("ensemble_models") or [artifact["model"]]
    task = task_of(artifact)
    tm = artifact.get("target_mapping")
    out = pd.DataFrame(index=range(len(X)))
    if task == "regression":
        out["prediction"] = np.mean([m.predict(X) for m in models], axis=0)
        return out
    if task == "multiclass":
        labels = class_labels(artifact)
        P = np.mean([_aligned_proba(m, X, len(labels)) for m in models], axis=0)
        if isinstance(artifact.get("calibrator"), dict):
            P = apply_multiclass_calibration(P, artifact["calibrator"])
        for j, lab in enumerate(labels):
            out[f"proba_{lab}"] = P[:, j]
        out["prediction"] = P.argmax(axis=1)
        out["label"] = target_encoding.decode(out["prediction"], tm)
        return out
    p = np.mean([_aligned_proba(m, X, 2)[:, 1] for m in models], axis=0)
    cal = artifact.get("calibrator")
    if cal is not None:
        p = np.clip(cal.predict(p), artifact.get("pd_floor", 1e-15), artifact.get("pd_cap", 1 - 1e-15))
    out["probability"] = p
    out["prediction"] = (p >= threshold).astype(int)
    out["label"] = target_encoding.decode(out["prediction"], tm)
    return out
