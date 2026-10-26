#!/usr/bin/env python
"""Tests for profiling (profile_raw / derive_config) and Config.apply_auto_config.

Run:  py -m pytest tests/test_profiling.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import Config
from profiling import derive_config, profile_raw


# ── derive_config (pure arithmetic) ────────────────────────────────────────

def test_derive_config_wide_dataset():
    # ~1000 cols / ~1M rows → large / very wide tier.
    ov = derive_config({"n_rows": 1_000_000, "n_cols": 1000,
                        "n_numeric": 800, "n_categorical": 200})
    assert ov["TOP_K_FEATURES_CAP"] == 1000
    assert ov["RFE_TARGET_FEATURES"] == 500
    assert ov["MAX_FINAL_FEATURES"] == 250
    assert ov["TARGET_NEW_FEATURE_COUNT"] == 40
    assert ov["FEATURE_META_MAX_NUMERIC_COLS"] == 800
    assert ov["FEATURE_META_MAX_CATEGORICAL_COLS"] == 200
    assert ov["FLAML_MAX_ROWS"] == 200_000            # 200M cells / 1000 cols
    assert ov["SHAP_SAMPLE_SIZE"] == 5_000            # very wide → fewer SHAP rows
    assert ov["DRIFT_SAMPLE_N"] == 100_000
    assert ov["CV_N_SPLITS"] == 3 and ov["MULTI_SEED_N"] == 1
    assert ov["FLAML_TIME_BUDGET"] == 3600 and ov["OPTUNA_TIMEOUT"] == 7200
    assert ov["PRUNE_BATCH_FRACTION"] == 0.2


def test_derive_config_small_dataset():
    # Few cols / few rows → caps never exceed the data, small search, shallow trees.
    ov = derive_config({"n_rows": 1000, "n_cols": 10,
                        "n_numeric": 5, "n_categorical": 5})
    assert ov["RFE_TARGET_FEATURES"] == 10            # ≤ n_cols
    assert ov["MAX_FINAL_FEATURES"] == 10             # ≤ n_cols
    assert ov["TARGET_NEW_FEATURE_COUNT"] == 5        # floor
    assert ov["FLAML_MAX_ROWS"] == 1000               # never more than the data
    assert ov["SHAP_SAMPLE_SIZE"] == 1000
    assert ov["DRIFT_SAMPLE_N"] == 1000
    assert ov["FLAML_TIME_BUDGET"] == 90 and ov["OPTUNA_N_TRIALS"] == 40
    assert ov["NUM_LEAVES_MAX"] == 31 and ov["MAX_DEPTH_MAX"] == 6
    assert ov["MULTI_SEED_N"] == 5 and ov["EARLY_STOPPING_ROUNDS"] == 50
    assert ov["CHART_N_DECILES"] == 5 and ov["IV_BINS_DEFAULT"] == 10   # 1000 rows ≥ 500 events
    ov = derive_config({"n_rows": 1000, "n_cols": 10, "target_minority": 120}, "binary")
    assert ov["IV_BINS_DEFAULT"] == 5                                   # few events → coarse bins


def test_derive_config_caps_upper_bounds():
    # Very wide / very tall → capped at the ceilings.
    ov = derive_config({"n_rows": 10_000_000, "n_cols": 5000,
                        "n_numeric": 3000, "n_categorical": 2000})
    assert ov["RFE_TARGET_FEATURES"] == 800          # ceiling
    assert ov["MAX_FINAL_FEATURES"] == 300           # ceiling
    assert ov["TARGET_NEW_FEATURE_COUNT"] == 60      # ceiling
    assert ov["FEATURE_META_MAX_NUMERIC_COLS"] == 1000  # ceiling
    assert ov["FEATURE_META_MAX_CATEGORICAL_COLS"] == 400  # ceiling
    assert ov["FLAML_MAX_ROWS"] == 40_000            # 200M cells / 5000 cols
    assert ov["SHAP_SAMPLE_SIZE"] == 5_000
    assert ov["DRIFT_SAMPLE_N"] == 100_000


def test_derive_config_minority_class_limits_features_and_folds():
    # 50k rows but only 300 positives: ≤ ~10 events per feature, ≥ ~20 per fold.
    ov = derive_config({"n_rows": 50_000, "n_cols": 400, "n_numeric": 400, "n_categorical": 0,
                        "target_minority": 300}, "binary_classification")
    assert ov["MAX_FINAL_FEATURES"] == 30
    assert ov["CV_N_SPLITS"] == 5
    ov = derive_config({"n_rows": 50_000, "n_cols": 400, "n_numeric": 400, "n_categorical": 0,
                        "target_minority": 60}, "multiclass")
    assert ov["CV_N_SPLITS"] == 3
    # regression ignores class counts
    ov = derive_config({"n_rows": 50_000, "n_cols": 400, "n_numeric": 400, "n_categorical": 0,
                        "target_minority": 60}, "regression")
    assert ov["MAX_FINAL_FEATURES"] == 100


def test_derive_config_missing_fields():
    assert derive_config({}) == {}
    ov = derive_config({"n_cols": 50})
    assert ov["TOP_K_FEATURES_CAP"] == 50
    assert ov["MAX_FINAL_FEATURES"] == 12
    assert "FLAML_TIME_BUDGET" not in ov             # rows unknown → no row-based knobs


# ── profile_raw ────────────────────────────────────────────────────────────

def test_profile_raw_csv(tmp_path):
    p = tmp_path / "data.csv"
    df = pd.DataFrame({
        "a": np.arange(100, dtype=float),
        "b": np.random.randint(0, 10, 100),
        "c": np.random.choice(["x", "y", "z"], 100),
    })
    df.to_csv(p, index=False)
    prof = profile_raw(str(p))
    assert prof is not None
    assert prof["n_rows"] == 100
    assert prof["n_cols"] == 3
    assert prof["n_numeric"] == 2
    assert prof["n_categorical"] == 1


def test_profile_raw_parquet(tmp_path):
    p = tmp_path / "data.parquet"
    df = pd.DataFrame({
        "a": np.arange(50, dtype=float),
        "b": np.random.choice(["m", "f"], 50),
    })
    df.to_parquet(p)
    prof = profile_raw(str(p))
    assert prof is not None
    assert prof["n_rows"] == 50
    assert prof["n_cols"] == 2
    assert prof["n_numeric"] == 1
    assert prof["n_categorical"] == 1


def test_profile_raw_missing_path():
    assert profile_raw("/no/such/file.csv") is None


def test_profile_raw_remote_path():
    assert profile_raw("s3://bucket/key.csv") is None


# ── Config.apply_auto_config ───────────────────────────────────────────────

def test_apply_auto_config_respects_env(monkeypatch):
    # Pin MAX_FINAL_FEATURES via env; auto-config must NOT override it.
    monkeypatch.setenv("MAX_FINAL_FEATURES", "150")
    before = Config.MAX_FINAL_FEATURES  # import-time snapshot, whatever it is
    overrides = {"MAX_FINAL_FEATURES": 250, "RFE_TARGET_FEATURES": 400}
    applied = Config.apply_auto_config(overrides)
    assert "MAX_FINAL_FEATURES" not in applied
    assert applied["RFE_TARGET_FEATURES"] == 400
    assert Config.MAX_FINAL_FEATURES == before  # untouched
    assert Config.RFE_TARGET_FEATURES == 400


def test_apply_auto_config_unpinned(monkeypatch):
    monkeypatch.delenv("RFE_TARGET_FEATURES", raising=False)
    applied = Config.apply_auto_config({"RFE_TARGET_FEATURES": 400})
    assert applied == {"RFE_TARGET_FEATURES": 400}
    assert Config.RFE_TARGET_FEATURES == 400


def test_apply_auto_config_unknown_key():
    # Unknown knob → ignored silently, no crash.
    applied = Config.apply_auto_config({"NO_SUCH_KNOB": 1})
    assert applied == {}

def test_reset_auto_config_restores_defaults(monkeypatch):
    monkeypatch.delenv("NUM_LEAVES_MAX", raising=False)
    default = Config._auto_defaults()["NUM_LEAVES_MAX"]
    Config.apply_auto_config({"NUM_LEAVES_MAX": 31, "FLAML_TIME_BUDGET": 90})
    assert Config.NUM_LEAVES_MAX == 31 and Config.TIME_BUDGET_AUTO_SIZED
    Config.reset_auto_config()
    assert Config.NUM_LEAVES_MAX == default and not Config.TIME_BUDGET_AUTO_SIZED


def test_time_budget_used_as_is_when_auto_sized(monkeypatch):
    from Agents.TrainModel.agent_train_model import TrainModelAgent
    monkeypatch.delenv("FLAML_TIME_BUDGET", raising=False)
    monkeypatch.delenv("OPTUNA_TIMEOUT", raising=False)
    Config.apply_auto_config({"FLAML_TIME_BUDGET": 90, "OPTUNA_TIMEOUT": 180})
    assert TrainModelAgent._compute_time_budget(1000, 10) == (90, 180)
    Config.reset_auto_config()


def test_yaml_editor_pins_are_session_scoped(monkeypatch):
    # Pins are returned for the session / job env — the shared Config and
    # os.environ of the UI server must stay untouched (multi-user safety).
    import os
    from ui import config_editor
    monkeypatch.delenv("MULTI_SEED_N", raising=False)
    before = Config.MULTI_SEED_N
    ok, errs, pins = config_editor.parse_yaml("MULTI_SEED_N: 2\nPRUNE_BATCH_FRACTION: 0.2")
    assert ok, errs
    assert pins == {"MULTI_SEED_N": 2, "PRUNE_BATCH_FRACTION": 0.2}
    assert Config.MULTI_SEED_N == before and "MULTI_SEED_N" not in os.environ
    assert config_editor.pins_to_env(pins) == {"MULTI_SEED_N": "2", "PRUNE_BATCH_FRACTION": "0.2"}
    ok, errs, _ = config_editor.parse_yaml("MULTI_SEED_N: two")
    assert not ok and errs


def test_s3_settings_are_thread_local():
    # Two UI sessions = two threads: each sees only its own credentials.
    import threading
    from Agents.BaseAgent.base_agent import BaseAgent
    seen = {}

    def session(name, key):
        BaseAgent.set_thread_s3_settings({"S3_ACCESS_KEY": key, "S3_SECRET_KEY": key + "-s"})
        seen[name] = BaseAgent.s3_settings()[:2]
    threads = [threading.Thread(target=session, args=(n, k)) for n, k in (("a", "AK1"), ("b", "AK2"))]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert seen == {"a": ("AK1", "AK1-s"), "b": ("AK2", "AK2-s")}
    assert BaseAgent.s3_settings()[0] == Config.S3_ACCESS_KEY      # main thread untouched
