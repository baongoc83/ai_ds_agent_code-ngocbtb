"""Target (label) encoding shared by the pipeline, Agent 3 and scoring.

Every downstream step reads the target with ``pd.to_numeric`` — a text label
such as "yes"/"no" (bank-marketing) or "A"/"B"/"C" would silently become NaN
and the run would train on nothing. The pipeline therefore encodes the target
ONCE, before the split, and freezes the mapping:

    regression              : must already be numeric (non-numeric -> error)
    binary_classification   : two labels -> 0 / 1 (positive = 1)
    multiclass              : k labels  -> 0 .. k-1 (sorted)

The frozen plan is written to ``{RUN_DIR}/target_mapping.json`` and stored in
the model artifact, so predictions can be decoded back to the original labels.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

MAX_CLASSES = 50  # beyond this a "multiclass" target is almost surely an ID / free text

# Lower-cased labels read as the POSITIVE class of a binary target.
_POSITIVE_TOKENS = {"1", "1.0", "yes", "y", "true", "t", "positive", "pos", "bad", "default",
                    "defaulted", "churn", "churned", "fraud", "fraudulent", "event", "success"}


def task_of(model_type: str) -> str:
    """'regression' | 'multiclass' | 'classification' (binary) from a model_type string."""
    mt = str(model_type or "").lower()
    if "regress" in mt:
        return "regression"
    if "multi" in mt:
        return "multiclass"
    return "classification"


def _jsonable(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        f = float(v)
        return int(f) if f.is_integer() else f
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v if isinstance(v, (int, float, str, bool)) else str(v)


def _sort_key(v: Any):
    try:
        return (0, float(v), "")
    except (TypeError, ValueError):
        return (1, 0.0, str(v))


def plan(y: pd.Series, model_type: str) -> Dict[str, Any]:
    """Decide the encoding from the TRAIN target. Raises ValueError on an
    unusable target (non-numeric regression, wrong class count)."""
    task = task_of(model_type)
    s = y.dropna()
    if task == "regression":
        num = pd.to_numeric(s, errors="coerce")
        bad = int(num.isna().sum())
        if bad > max(0, 0.01 * len(s)):
            raise ValueError(
                f"Regression target '{y.name}' is not numeric ({bad} non-numeric values, "
                f"e.g. {s[num.isna()].astype(str).unique()[:5].tolist()}). Pick a numeric target "
                "or switch model_type to binary_classification / multiclass.")
        return {"task": task, "encoded": False, "classes": None}

    values = sorted(s.unique().tolist(), key=_sort_key)
    k = len(values)
    if k < 2:
        raise ValueError(f"Target '{y.name}' has {k} distinct value(s) — need at least 2 classes.")
    if task == "classification" and k > 2:
        raise ValueError(
            f"binary_classification needs exactly 2 target values, '{y.name}' has {k} "
            f"(e.g. {[str(v) for v in values[:8]]}). Use model_type='multiclass' or recode the target.")
    if task == "multiclass" and k > MAX_CLASSES:
        raise ValueError(
            f"Target '{y.name}' has {k} distinct values (> {MAX_CLASSES}) — looks like an ID or a "
            "continuous value. Use regression or group the labels first.")

    if task == "classification":
        neg, pos = values
        lowered = [str(v).strip().lower() for v in values]
        hits = [v for v, lv in zip(values, lowered) if lv in _POSITIVE_TOKENS]
        if all(_sort_key(v)[0] == 0 for v in values):
            how = "larger numeric value"            # 0/1, 1/2, -1/1 ... (sorted ascending)
        elif len(hits) == 1:
            pos = hits[0]
            neg = values[0] if values[1] == pos else values[1]
            how = "recognised positive token"
        else:
            counts = s.value_counts()
            pos = counts.idxmin()
            neg = counts.idxmax()
            how = "minority class"
        classes = [neg, pos]
        positive_rule = how
    else:
        classes = values
        positive_rule = None

    identity = all(_sort_key(c)[0] == 0 and float(c) == i for i, c in enumerate(classes))
    return {"task": task, "encoded": not identity,
            "classes": [_jsonable(c) for c in classes], "positive_rule": positive_rule}


def apply(y: pd.Series, p: Dict[str, Any]) -> pd.Series:
    """Encode a target column with a frozen plan. Unknown labels -> NaN."""
    if not p or p.get("task") == "regression":
        return pd.to_numeric(y, errors="coerce")
    classes = p["classes"]
    lookup = {str(c): i for i, c in enumerate(classes)}
    # numeric labels may arrive as 1 vs 1.0 vs "1" across partitions — compare as text,
    # with integral floats normalised ("1.0" -> "1")
    def key(v):
        if pd.isna(v):
            return None
        try:
            f = float(v)
            return str(int(f)) if f.is_integer() else str(f)
        except (TypeError, ValueError):
            return str(v)
    lookup.update({key(c): i for i, c in enumerate(classes)})
    return y.map(lambda v: lookup.get(key(v), np.nan)).astype("float64")


def decode(codes, p: Optional[Dict[str, Any]]) -> List[Any]:
    """Map integer codes back to the original labels (identity when no plan)."""
    if not p or not p.get("classes"):
        return list(codes)
    classes = p["classes"]
    return [classes[int(c)] if c is not None and not pd.isna(c) and 0 <= int(c) < len(classes) else None
            for c in codes]


def save(p: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(p, f, indent=2, ensure_ascii=False)


def load(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None
