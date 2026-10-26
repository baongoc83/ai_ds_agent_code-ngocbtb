"""Score raw input rows with a finished run's artifacts (no LLM, no retraining).

Replays the exact chain the run used: CleaningSpec -> NullProcessor ->
FeatureSpec -> categorical encoding -> ensemble predict (+ calibration for
classification). Artifacts are loaded only after their SHA-256 matches the
run's replay_manifest.json, since joblib.load executes code.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


class ScoringError(Exception):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _checked_load(path: Path, expected: Dict[str, str]):
    import joblib
    if not path.exists():
        raise ScoringError(f"Thiếu file {path.name} trong run.")
    want = expected.get(path.name)
    if want and _sha256(path) != want:
        raise ScoringError(f"{path.name} không khớp SHA-256 trong replay_manifest.json — từ chối load.")
    return joblib.load(path)


def load_bundle(run_dir: str) -> Dict[str, Any]:
    rd = Path(run_dir)
    mpath = rd / "replay_manifest.json"
    if not mpath.exists():
        raise ScoringError("Run này không có replay_manifest.json nên không thể chấm điểm.")
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    hashes = {name: (v.get("sha256") if isinstance(v, dict) else v)
              for name, v in (manifest.get("artifacts") or {}).items()}

    bundle = {
        "manifest": manifest,
        "clean_spec": _checked_load(rd / manifest["cleaning"]["spec_file"], hashes),
        "feat_spec": _checked_load(rd / manifest["feature_engineering"]["spec_file"], hashes),
        "artifact": _checked_load(rd / manifest["training"]["model_file"], hashes),
        "null_proc": None,
    }
    info = manifest.get("null_processing")
    if info:
        from preprocessing import NullProcessor
        bundle["null_proc"] = NullProcessor.load(rd / info["file"], expected_sha256=info.get("sha256"))
    return bundle


def input_columns(run_dir: str) -> List[str]:
    """Raw columns the user is expected to provide (from the cleaner report)."""
    rd = Path(run_dir)
    dc = rd / "data_cleaner_report.json"
    cols: List[str] = []
    if dc.exists():
        rep = json.loads(dc.read_text(encoding="utf-8"))
        cols = list(rep.get("columns_remaining") or rep.get("original_columns") or [])
    m = rd / "replay_manifest.json"
    if m.exists():
        target = json.loads(m.read_text(encoding="utf-8")).get("target_column")
        cols = [c for c in cols if c != target]
    return cols


def task_of(bundle: Dict[str, Any]) -> str:
    import target_encoding
    return target_encoding.task_of(bundle["artifact"].get("task")
                                   or bundle["manifest"].get("training", {}).get("task")
                                   or "classification")


def predict(bundle: Dict[str, Any], raw: pd.DataFrame) -> pd.DataFrame:
    """Return the input rows with a `prediction` column (value or probability)."""
    manifest, art = bundle["manifest"], bundle["artifact"]
    df = raw.copy()
    target = manifest.get("target_column")
    if target in df.columns:
        df = df.drop(columns=[target])

    df = bundle["clean_spec"].apply(df)
    if bundle["null_proc"] is not None:
        df = bundle["null_proc"].transform(df)
    df = bundle["feat_spec"].apply(df, strict=False)

    features = art["feature_cols"]
    absent = [c for c in features if c not in df.columns]
    if absent:
        raise ScoringError(f"Thiếu {len(absent)} feature sau khi xử lý, ví dụ {absent[:8]}. "
                           "Kiểm tra input có đủ các cột gốc.")
    X = df[features].copy()
    for col in X.columns:
        le = art["cat_encoders"].get(col)
        if le is not None:
            vals = X[col].fillna("__NA__").astype(str)
            X[col] = le.transform(vals.where(vals.isin(set(le.classes_)), "__NA__"))
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(-999)

    import model_scoring
    art = {**art, "target_mapping": art.get("target_mapping") or manifest.get("target_mapping")}
    pred = model_scoring.predict_frame(art, X.reset_index(drop=True))
    out = raw.reset_index(drop=True).copy()
    for col in pred.columns:
        out[col] = pred[col].values
    return out


def class_labels(bundle: Dict[str, Any]) -> List[Any]:
    import model_scoring
    art = bundle["artifact"]
    return model_scoring.class_labels(
        {**art, "target_mapping": art.get("target_mapping") or bundle["manifest"].get("target_mapping")})
