"""Helpers for the data-import panel: col-description preview + descriptive stats."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


# ── display ───────────────────────────────────────────────────────────────

def arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Copy of `df` that st.dataframe can always serialise.

    Streamlit converts frames to Arrow, which rejects an object column holding
    several Python types (e.g. a hyper-parameter table where `value` mixes 300,
    0.05 and 'log2', or raw data with numbers and text in one column). Such
    columns are shown as text; homogeneous columns are left untouched."""
    out = df.copy()
    for c in out.columns:
        s = out[c]
        if s.dtype != object:
            continue
        kinds = {type(v) for v in s.dropna().head(5000)}
        if len(kinds) > 1 or kinds & {dict, list, tuple, set, bytes}:
            out[c] = s.map(lambda v: "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
    return out


# ── col-description file ──────────────────────────────────────────────────

def read_desc_file(path: str):
    """Return ("mapping", {col: desc}) when the file is already name->description,
    else ("table", DataFrame) when the user must say which fields to use.
    Nested {group: {col: desc}} JSON is returned as ("nested", dict)."""
    suffix = Path(path).suffix.lower()
    if suffix in (".yaml", ".yml", ".json"):
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) if suffix != ".json" else json.load(f)
        if isinstance(data, dict):
            if data and isinstance(next(iter(data.values())), dict):
                return "nested", data
            return "mapping", {str(k): "" if v is None else str(v) for k, v in data.items()}
        if isinstance(data, list):
            return "table", pd.DataFrame(data)
        raise ValueError(f"Định dạng {suffix} không hợp lệ: cần dict hoặc list.")
    if suffix in (".csv", ".tsv", ".txt"):
        from Agents.BaseAgent.base_agent import BaseAgent
        return "table", BaseAgent.read_csv_smart(path)
    if suffix == ".parquet":
        return "table", pd.read_parquet(path)
    return "table", pd.read_excel(path)


def table_to_mapping(df: pd.DataFrame, name_f: str, desc_f: str,
                     group_f: Optional[str]) -> dict:
    """Same rule as FeatureEngineer._parse_tabular_descriptions."""
    out = {}
    has_group = bool(group_f and group_f in df.columns)
    for _, row in df.iterrows():
        name = str(row[name_f]).strip()
        if not name or name == "nan":
            continue
        desc = str(row[desc_f]).strip() if pd.notna(row[desc_f]) else ""
        if has_group and pd.notna(row.get(group_f)):
            g = str(row[group_f]).strip()
            desc = f"[{g}] {desc}" if desc else f"[{g}]"
        out[name] = desc
    return out


def to_yaml(obj: dict) -> str:
    return yaml.safe_dump(obj, allow_unicode=True, sort_keys=False, width=100)


# ── descriptive statistics ────────────────────────────────────────────────

def _skew_label(s: float) -> str:
    if pd.isna(s):
        return ""
    if s > 0.5:
        return "right (lệch phải)"
    if s < -0.5:
        return "left (lệch trái)"
    return "symmetric (cân xứng)"


def describe_dataset(df: pd.DataFrame):
    """Return (numeric_stats, categorical_stats) DataFrames, one row per column."""
    n = len(df)
    num_rows, cat_rows = [], []
    for c in df.columns:
        s = df[c]
        null_pct = round(float(s.isna().mean() * 100), 2) if n else 0.0
        is_num = pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)
        if is_num:
            x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if len(x):
                q1, med, q3 = x.quantile([0.25, 0.5, 0.75])
                skew = float(x.skew()) if len(x) > 2 else float("nan")
            else:
                q1 = med = q3 = skew = float("nan")
            num_rows.append({
                "column": c, "dtype": str(s.dtype), "null_%": null_pct,
                "distinct": int(s.nunique()),
                "min": x.min() if len(x) else np.nan, "mean": x.mean() if len(x) else np.nan,
                "median": med, "IQR": q3 - q1, "max": x.max() if len(x) else np.nan,
                "skew": round(skew, 3) if not pd.isna(skew) else np.nan,
                "skew_dir": _skew_label(skew),
            })
        else:
            vc = s.value_counts(dropna=True)
            cat_rows.append({
                "column": c, "dtype": str(s.dtype), "null_%": null_pct,
                "distinct": int(s.nunique()),
                "top": str(vc.index[0]) if len(vc) else "",
                "top_freq_%": round(float(vc.iloc[0] / n * 100), 2) if len(vc) and n else np.nan,
            })
    return pd.DataFrame(num_rows), pd.DataFrame(cat_rows)
