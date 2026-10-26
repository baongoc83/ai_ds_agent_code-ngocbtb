"""YAML config editor: show the auto-config table as editable YAML and turn
the user's edits into per-SESSION pins.

The tunable knobs are those in Config.AUTO_CONFIG_ENV, which profiling.py sizes
from the input (rows × cols, smallest target class). Pins are NOT written to
os.environ or Config (both are shared by every browser session of the server):
the UI keeps them in st.session_state and hands them to the job process as
environment variables (ui/runner.submit_job), where they win over auto-config.
Knobs pinned in the server's own .env are shown as pinned for everyone.
"""
from __future__ import annotations

import os
from typing import Dict, List, Tuple

import yaml

from config import Config


GROUPS = [
    ("Feature caps", ["TOP_K_FEATURES_CAP", "RFE_TARGET_FEATURES", "MAX_FINAL_FEATURES",
                      "TARGET_NEW_FEATURE_COUNT", "SHAP_PSI_MIN_FEATURES_FLOOR", "PRUNE_BATCH_FRACTION"]),
    ("LLM prompt (Agent 2)", ["FEATURE_META_MAX_NUMERIC_COLS", "FEATURE_META_MAX_CATEGORICAL_COLS"]),
    ("Thời gian tìm kiếm", ["FLAML_TIME_BUDGET", "OPTUNA_TIMEOUT", "OPTUNA_N_TRIALS", "BENCHMARK_TIME_BUDGET"]),
    ("Validation", ["CV_N_SPLITS", "FLAML_N_SPLITS", "MULTI_SEED_N", "EARLY_STOPPING_ROUNDS"]),
    ("Độ phức tạp model", ["NUM_LEAVES_MAX", "MAX_DEPTH_MAX", "N_ESTIMATORS_MAX",
                           "TREE_ENSEMBLE_MAX_ESTIMATORS", "FEATURE_RANK_N_ESTIMATORS"]),
    ("Lấy mẫu", ["FLAML_MAX_ROWS", "SHAP_SAMPLE_SIZE", "DRIFT_SAMPLE_N"]),
    ("Binning / chart", ["IV_BINS_DEFAULT", "FE_IV_STABILITY_MIN_BIN_COUNT", "CHART_N_DECILES"]),
]


def knob_descriptions() -> Dict[str, str]:
    return {
        "TOP_K_FEATURES_CAP": "Tổng số cột được phép đưa vào vòng chọn lọc",
        "RFE_TARGET_FEATURES": "Số feature ứng viên sau ranking/RFE",
        "MAX_FINAL_FEATURES": "Số feature tối đa ở model cuối (≤ ~1/10 số mẫu lớp nhỏ nhất)",
        "TARGET_NEW_FEATURE_COUNT": "Số feature tương tác mới LLM sinh mỗi run",
        "SHAP_PSI_MIN_FEATURES_FLOOR": "Số feature tối thiểu khi prune",
        "PRUNE_BATCH_FRACTION": "Tỷ lệ feature bỏ mỗi vòng prune (dữ liệu rộng → lớn hơn)",
        "FEATURE_META_MAX_NUMERIC_COLS": "Số cột numeric đưa vào LLM prompt",
        "FEATURE_META_MAX_CATEGORICAL_COLS": "Số cột categorical đưa vào LLM prompt",
        "FLAML_TIME_BUDGET": "Giây cho FLAML chọn loại model",
        "OPTUNA_TIMEOUT": "Giây cho Optuna tune hyperparameter",
        "OPTUNA_N_TRIALS": "Số trial Optuna tối đa",
        "BENCHMARK_TIME_BUDGET": "Giây cho baseline & benchmark",
        "CV_N_SPLITS": "Số fold CV của model cuối (≥ ~20 mẫu lớp nhỏ/fold)",
        "FLAML_N_SPLITS": "Số fold CV của FLAML",
        "MULTI_SEED_N": "Số seed bagging model cuối (dữ liệu lớn → ít seed)",
        "EARLY_STOPPING_ROUNDS": "Số vòng early stopping",
        "NUM_LEAVES_MAX": "num_leaves tối đa trong search (dữ liệu nhỏ → nhỏ, tránh overfit)",
        "MAX_DEPTH_MAX": "max_depth tối đa trong search",
        "N_ESTIMATORS_MAX": "Số cây tối đa trong search",
        "TREE_ENSEMBLE_MAX_ESTIMATORS": "Số cây tối đa cho RF/ExtraTrees",
        "FEATURE_RANK_N_ESTIMATORS": "Số cây của model xếp hạng feature",
        "FLAML_MAX_ROWS": "Số dòng tối đa đưa vào FLAML (tránh OOM)",
        "SHAP_SAMPLE_SIZE": "Số dòng lấy mẫu để tính SHAP",
        "DRIFT_SAMPLE_N": "Số dòng mỗi partition dùng cho kiểm tra PSI drift",
        "IV_BINS_DEFAULT": "Số bin khi tính IV/WoE",
        "FE_IV_STABILITY_MIN_BIN_COUNT": "Số mẫu tối thiểu mỗi bin khi kiểm tra ổn định IV",
        "CHART_N_DECILES": "Số nhóm cho chart decile / calibration",
    }


def _fmt(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def env_pinned(key: str) -> bool:
    """Pinned for everyone via the server's environment / .env."""
    return Config.AUTO_CONFIG_ENV.get(key) in os.environ


def build_yaml(auto_enabled: bool, auto_values: Dict = None, pins: Dict = None) -> str:
    """Return an editable YAML body, grouped by pipeline stage.

    - auto_enabled True: every knob commented out with its auto value — the user
      uncomments only the ones to force.
    - auto_enabled False: every knob active with its default value.
    Knobs pinned by this session (`pins`) or by the server env are shown active.
    """
    desc = knob_descriptions()
    auto_values, pins = auto_values or {}, pins or {}
    defaults = Config._auto_defaults()
    lines = (["# Auto-config ĐANG BẬT — giá trị dưới đây được tính từ kích thước dữ liệu.",
              "# Bỏ comment (xóa '# ') dòng nào để ÉP giá trị khác; xóa dòng để bỏ ép.",
              "# Giá trị ép chỉ áp dụng cho phiên của bạn."]
             if auto_enabled else
             ["# Auto-config ĐANG TẮT — bạn quản lý toàn bộ các giá trị bên dưới."])
    for group, keys in GROUPS:
        lines.append(f"\n# ── {group}")
        for key in keys:
            if key not in Config.AUTO_CONFIG_ENV:
                continue
            if key in pins:
                val, note, active = pins[key], " (đang ép)", True
            elif env_pinned(key):
                val, note, active = os.environ[Config.AUTO_CONFIG_ENV[key]], " (ép trong .env server)", True
            else:
                val = auto_values.get(key) if auto_enabled else defaults.get(key, getattr(Config, key, ""))
                note, active = "", not auto_enabled
            lines.append(f"{'' if active else '# '}{key}: {_fmt(val) if not isinstance(val, str) else val}"
                         f"    # {desc.get(key, '')}{note}")
    return "\n".join(lines)


def parse_yaml(text: str) -> Tuple[bool, List[str], Dict]:
    """Parse + validate the YAML into session pins {knob: value}.

    Returns (ok, errors, pins). Errors are human-readable messages in Vietnamese;
    nothing is applied anywhere — the caller stores the pins in its session.
    Knobs already pinned by the server env are left out (env wins anyway).
    """
    errors: List[str] = []
    try:
        data = yaml.safe_load(text or "") or {}
    except yaml.YAMLError as e:
        return False, [f"Lỗi YAML: {e}"], {}
    if not isinstance(data, dict):
        return False, ["YAML phải là một mapping (các dòng key: value)."], {}

    valid_keys = set(Config.AUTO_CONFIG_ENV)
    defaults = Config._auto_defaults()
    pins: Dict = {}
    for key, value in data.items():
        if key not in valid_keys:
            errors.append(f"Key không hợp lệ: '{key}' (hợp lệ: {', '.join(sorted(valid_keys))})")
            continue
        is_float = isinstance(defaults.get(key), float)
        ok = (isinstance(value, (int, float)) if is_float else isinstance(value, int)) and not isinstance(value, bool)
        if not ok:
            errors.append(f"'{key}' phải là {'số' if is_float else 'số nguyên'}, nhận được: {value!r}")
        elif value <= 0:
            errors.append(f"'{key}' phải > 0, nhận được: {value!r}")
        elif not env_pinned(key):
            pins[key] = float(value) if is_float else int(value)
    if errors:
        return False, errors, {}
    return True, [], pins


def pins_to_env(pins: Dict) -> Dict[str, str]:
    """Session pins → environment for the job process."""
    return {Config.AUTO_CONFIG_ENV[k]: str(v) for k, v in (pins or {}).items()}
