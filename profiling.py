#!/usr/bin/env python
"""Lightweight dataset profiling → auto-config heuristics.

The pipeline reads its tuning knobs (feature-count caps, row-sampling sizes,
LLM metadata caps) from env-var-backed Config class attributes. This module
lets those be derived automatically from the shape of the raw input dataset
(rows, columns, numeric/categorical mix) so a user running ~1000 features /
~1M rows gets sensible values without hand-setting ~10 env vars.

Two functions:
  profile_raw(path)  — read ONLY the shape of a raw dataset (never the full
                       frame), returning {n_rows, n_cols, n_numeric, n_categorical}.
  derive_config(profile) — pure arithmetic heuristic over a profile dict.

Both are deliberately dependency-light and never raise on a probing failure:
if a dataset can't be profiled cheaply, auto-config is simply skipped.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


def _is_numeric_dtype(dtype) -> bool:
    import numpy as np
    try:
        return bool(np.issubdtype(dtype, np.number))
    except TypeError:
        return False


def _split_compression(path: str):
    """Mirror BaseAgent's compression/suffix split for the common local cases."""
    suffix = Path(path).suffix.lower()
    compression = None
    comp_map = {".gz": "gzip", ".bz2": "bz2", ".b2": "bz2", ".xz": "xz",
                ".zst": "zstd", ".zip": "zip"}
    if suffix in comp_map:
        compression = comp_map[suffix]
        suffix = Path(path[:-len(suffix)]).suffix.lower()
    return suffix, compression


def _count_csv_lines(path: str, compression: Optional[str]) -> Optional[int]:
    """Count data rows of a local CSV cheaply (one pass over bytes)."""
    try:
        import gzip
        if compression == "gzip":
            with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
                return sum(1 for _ in fh) - 1
        if compression is None:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                return sum(1 for _ in fh) - 1
        return None  # bz2/xz/zst: not worth a streaming decode here
    except OSError:
        return None




def _shape(path: str, suffix: str, compression: Optional[str]) -> Optional[Dict]:
    """{n_rows, n_cols, n_numeric, n_categorical} from metadata / a small head."""
    import pandas as pd
    from Agents.BaseAgent.base_agent import BaseAgent

    if compression is None and suffix == ".parquet":
        # schema + row-group metadata only — never materialise the table
        import pyarrow.parquet as pq
        import pyarrow.types as pat
        pf = pq.ParquetFile(path)
        fields = list(pf.schema_arrow)
        n_numeric = sum(pat.is_integer(f.type) or pat.is_floating(f.type) or pat.is_decimal(f.type)
                        for f in fields)
        return {"n_rows": int(pf.metadata.num_rows), "n_cols": len(fields),
                "n_numeric": int(n_numeric), "n_categorical": len(fields) - int(n_numeric)}
    if compression is None and suffix in (".feather", ".ftr", ".orc"):
        head = pd.read_feather(path) if suffix != ".orc" else pd.read_orc(path)
        n_numeric = int(sum(_is_numeric_dtype(d) for d in head.dtypes))
        return {"n_rows": int(len(head)), "n_cols": head.shape[1],
                "n_numeric": n_numeric, "n_categorical": head.shape[1] - n_numeric}
    if suffix in (".csv", ".tsv", ".txt", "") or (compression is not None and suffix not in
                                                    (".json", ".jsonl", ".ndjson")):
        # Separator / encoding / decimal sniffed; a 500-row sample gives
        # meaningful numeric-vs-categorical dtypes; rows counted by a byte scan.
        head = BaseAgent.read_csv_smart(path, compression=compression, nrows=500)
        n_numeric = int(sum(_is_numeric_dtype(d) for d in head.dtypes))
        n_rows = _count_csv_lines(path, compression)
        return {"n_rows": int(n_rows if n_rows is not None else len(head)), "n_cols": head.shape[1],
                "n_numeric": n_numeric, "n_categorical": max(0, head.shape[1] - n_numeric)}
    # JSON / Excel: no cheap header — load (these files are small in practice)
    df = BaseAgent.load_dataframe(path)
    n_numeric = int(sum(_is_numeric_dtype(d) for d in df.dtypes))
    return {"n_rows": int(len(df)), "n_cols": df.shape[1], "n_numeric": n_numeric,
            "n_categorical": max(0, df.shape[1] - n_numeric)}


def _target_stats(path: str, target: str, suffix: str, compression: Optional[str]) -> Dict:
    """Class count / smallest class size of the raw target (one column read)."""
    import pandas as pd
    from Agents.BaseAgent.base_agent import BaseAgent
    if compression is None and suffix == ".parquet":
        y = pd.read_parquet(path, columns=[target])[target]
    elif suffix in (".csv", ".tsv", ".txt", "") or compression is not None:
        y = BaseAgent.read_csv_smart(path, compression=compression, usecols=[target])[target]
    else:
        y = BaseAgent.load_dataframe(path)[target]
    y = y.dropna()
    counts = y.value_counts()
    return {"target_n_unique": int(len(counts)),
            "target_minority": int(counts.min()) if len(counts) else 0,
            "target_is_numeric": bool(pd.api.types.is_numeric_dtype(y))}


def profile_raw(path: str, target_column: Optional[str] = None) -> Optional[Dict]:
    """Return {n_rows, n_cols, n_numeric, n_categorical[, target_*]} for a raw dataset.

    Reads only metadata / a small head (plus the target column when given),
    never the full frame for columnar / CSV inputs. Returns None if the dataset
    cannot be profiled (unsupported format, remote path, read error) so callers
    can safely skip auto-config.
    """
    if not path or path.startswith(("s3://", "s3a://", "s3n://", "gs://", "az://")):
        return None
    from Agents.BaseAgent.base_agent import BaseAgent
    if not os.path.exists(BaseAgent.file_part(path)):     # "book.xlsx::Sheet" → the workbook
        return None
    suffix, compression = _split_compression(BaseAgent.file_part(path))
    try:
        prof = _shape(path, suffix, compression)
    except Exception:
        return None
    if prof and target_column:
        try:
            prof.update(_target_stats(path, target_column, suffix, compression))
        except Exception:
            pass        # target stats are a refinement — shape alone still sizes the run
    return prof


# ── sizing ────────────────────────────────────────────────────────────────

ROW_TIERS = [(2_000, "tiny"), (20_000, "small"), (200_000, "medium"), (1_000_000, "large")]
COL_TIERS = [(50, "narrow"), (300, "medium"), (1_000, "wide")]


def size_tier(profile: Dict) -> Dict[str, str]:
    """Human-readable size class used by derive_config (also shown in the UI/log)."""
    n_rows, n_cols = profile.get("n_rows") or 0, profile.get("n_cols") or 0
    rows = next((name for lim, name in ROW_TIERS if n_rows < lim), "xlarge")
    cols = next((name for lim, name in COL_TIERS if n_cols < lim), "very_wide")
    return {"rows": rows, "cols": cols, "cells": f"{n_rows * n_cols:,}"}


def _clamp(value: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(value))))


def derive_config(profile: Dict, model_type: Optional[str] = None) -> Dict:
    """Map a profile dict → auto-config overrides for the WHOLE pipeline.

    Keys match Config class-attribute names (see Config.AUTO_CONFIG_ENV). Every
    knob depends on dataset size, not just the final feature count:

      feature caps     ← n_cols (and n_rows / smallest class: ≥ ~10 events per feature)
      search budget    ← rows × cols (FLAML / Optuna time, Optuna trials, benchmark)
      validation       ← n_rows / smallest class (CV folds, seeds, early stopping)
      model capacity   ← n_rows (max leaves / depth / trees — small data overfits)
      sampling         ← n_rows × n_cols (FLAML rows, SHAP rows, drift rows)
      binning / charts ← n_rows (IV bins, min bin count, chart deciles)

    Returns an empty dict when the profile lacks the fields the formulas need.
    """
    n_rows = int(profile.get("n_rows") or 0)
    n_cols = int(profile.get("n_cols") or 0)
    n_numeric = profile.get("n_numeric")
    n_categorical = profile.get("n_categorical")
    if not n_rows and not n_cols:
        return {}
    task = "regression" if "regress" in str(model_type or "").lower() else (
        "multiclass" if "multi" in str(model_type or "").lower() else "classification")
    minority = profile.get("target_minority")
    # "events": rows that carry the signal — the smallest class for classification
    events = int(minority) if (minority and task != "regression") else n_rows
    cells = n_rows * n_cols
    overrides: Dict = {}

    # ── feature caps (n_cols, bounded by events per feature) ─────────────
    if n_cols:
        overrides["TOP_K_FEATURES_CAP"] = n_cols
        overrides["RFE_TARGET_FEATURES"] = min(n_cols, _clamp(n_cols * 0.5, 20, 800))
        final = _clamp(n_cols * 0.25, 10, 300)
        if events:
            final = min(final, max(5, events // 10))   # ≥ ~10 events per final feature
        overrides["MAX_FINAL_FEATURES"] = min(n_cols, final)
        overrides["TARGET_NEW_FEATURE_COUNT"] = _clamp(n_cols * 0.04, 5, 60)
        overrides["SHAP_PSI_MIN_FEATURES_FLOOR"] = min(5, n_cols)
        # wide data: drop bigger batches per prune round, fewer rounds needed
        overrides["PRUNE_BATCH_FRACTION"] = 0.2 if n_cols >= 1000 else 0.15 if n_cols >= 300 else 0.1
    if n_numeric:
        overrides["FEATURE_META_MAX_NUMERIC_COLS"] = _clamp(n_numeric, 50, 1000)
    if n_categorical:
        overrides["FEATURE_META_MAX_CATEGORICAL_COLS"] = _clamp(n_categorical, 20, 400)

    if n_rows:
        # ── search budget (seconds) scales with rows × cols ───────────────
        if cells < 200_000:
            flaml, optuna, trials, bench = 90, 180, 40, 120
        elif cells < 5_000_000:
            flaml, optuna, trials, bench = 300, 600, 80, 300
        elif cells < 100_000_000:
            flaml, optuna, trials, bench = 900, 1800, 150, 600
        elif cells < 1_000_000_000:
            flaml, optuna, trials, bench = 1800, 3600, 200, 1200
        else:
            flaml, optuna, trials, bench = 3600, 7200, 300, 1800
        overrides["FLAML_TIME_BUDGET"] = flaml
        overrides["OPTUNA_TIMEOUT"] = optuna
        overrides["OPTUNA_N_TRIALS"] = trials
        overrides["BENCHMARK_TIME_BUDGET"] = bench

        # ── validation: folds / seeds / early stopping ───────────────────
        folds = 5 if n_rows < 200_000 else 4 if n_rows < 1_000_000 else 3
        if task != "regression" and events:
            folds = max(2, min(folds, events // 20))   # ≥ ~20 minority rows per fold
        overrides["CV_N_SPLITS"] = folds
        overrides["FLAML_N_SPLITS"] = folds
        overrides["MULTI_SEED_N"] = (5 if cells < 5_000_000 else 3 if cells < 100_000_000
                                     else 2 if cells < 1_000_000_000 else 1)
        overrides["EARLY_STOPPING_ROUNDS"] = 50 if n_rows < 20_000 else 100 if n_rows < 200_000 else 200

        # ── model capacity: small data overfits deep / leafy trees ───────
        if n_rows < 2_000:
            leaves, depth, trees = 31, 6, 600
        elif n_rows < 20_000:
            leaves, depth, trees = 63, 8, 1000
        elif n_rows < 200_000:
            leaves, depth, trees = 255, 10, 2000
        else:
            leaves, depth, trees = 512, 12, 3000
        overrides["NUM_LEAVES_MAX"] = leaves
        overrides["MAX_DEPTH_MAX"] = depth
        overrides["N_ESTIMATORS_MAX"] = trees
        overrides["TREE_ENSEMBLE_MAX_ESTIMATORS"] = 300 if n_rows >= 1_000_000 else 500
        overrides["FEATURE_RANK_N_ESTIMATORS"] = 200 if n_rows < 20_000 else 500

        # ── sampling (memory / time grow with rows × cols) ───────────────
        overrides["FLAML_MAX_ROWS"] = max(min(n_rows, 20_000),
                                          min(n_rows, 500_000, int(200_000_000 / max(n_cols, 1))))
        overrides["SHAP_SAMPLE_SIZE"] = min(n_rows, 20_000 if n_cols < 300 else 10_000 if n_cols < 1000 else 5_000)
        overrides["DRIFT_SAMPLE_N"] = max(min(n_rows, 10_000), min(n_rows, 100_000))

        # ── binning / charts: enough rows per bin ────────────────────────
        overrides["IV_BINS_DEFAULT"] = 5 if events < 500 else 10 if events < 50_000 else 20
        overrides["FE_IV_STABILITY_MIN_BIN_COUNT"] = 10 if n_rows < 5_000 else 30
        eval_rows = n_rows * 0.2
        overrides["CHART_N_DECILES"] = 5 if eval_rows < 500 else 10 if eval_rows < 200_000 else 20
    return overrides
