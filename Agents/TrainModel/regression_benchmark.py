"""Regression baseline + benchmark for Agent 3.

Fits a fixed ladder of models on the SAME train split and scores them with the
SAME metrics (MAE / RMSE / R² / MAPE) on every validation / OOT / test split:

    Mean baseline -> Median baseline -> Linear Regression -> Ridge
    -> Random Forest -> LightGBM -> XGBoost -> Pipeline-tuned model

then draws the 7 diagnostic charts on the validation set and on the OOT/test
hold-out, and turns the numbers into a rule-based "decision checklist".

Honesty note: every model here is fitted on `train` only, so `valid` is a real
hold-out for them. The DEPLOYED model is refit on train+valid (its valid score is
in-sample), which is why the charts use the train-only "Pipeline tuned" twin.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SENTINEL = -999  # TrainModelAgent._fit_prepare_X fills NaN with this

BASELINES = ("Mean baseline", "Median baseline")
VALID_CANDIDATES = ("valid_temporal", "valid", "valid_random")
HOLDOUT_CANDIDATES = ("oot", "test")
ALL_SPLITS = ("train", "valid_temporal", "valid_random", "valid", "oot", "test")

# (key, label, files{set: filename}, how-to-read guide) — single source for
# Config.chart_files("regression"), the UI and the markdown report.
CHART_SPECS: List[Dict[str, Any]] = [
    {"key": "model_comparison", "label": "1. Model comparison (MAE / RMSE / R²)",
     "files": {"valid": "reg_model_comparison_valid.png", "holdout": "reg_model_comparison_holdout.png"},
     "guide": "MAE, RMSE càng thấp càng tốt; R² càng cao càng tốt (R² có thể âm nếu kém hơn baseline "
              "trung bình). MAE là sai số tuyệt đối trung bình; RMSE nhạy hơn với các lỗi rất lớn. "
              "Chênh lệch nhỏ giữa các model chỉ đáng tin khi được xác nhận trên cả Validation và "
              "OOT/Test, không chỉ một lần chia dữ liệu."},
    {"key": "train_vs_valid_mae", "label": "2. Train vs Validation MAE",
     "files": {"valid": "reg_train_vs_valid_mae.png"},
     "guide": "Cả hai MAE thấp và gần nhau: tổng quát hóa tốt. Train rất thấp nhưng Validation cao: có thể "
              "overfit, leakage hoặc lệch phân phối dữ liệu. Cả hai cùng cao: model chưa đủ năng lực, "
              "feature chưa tốt hoặc target nhiễu. Không có ngưỡng chênh lệch chung - hãy so với baseline "
              "và theo dõi qua các kỳ."},
    {"key": "actual_vs_predicted", "label": "3. Actual vs Predicted",
     "files": {"valid": "reg_actual_vs_predicted_valid.png", "holdout": "reg_actual_vs_predicted_holdout.png"},
     "guide": "Đường chéo y = x là dự báo hoàn hảo. Điểm nằm trên đường chéo: dự báo cao hơn thực tế; "
              "nằm dưới: dự báo thấp hơn. Nếu target càng lớn điểm càng xa đường chéo: model dự báo kém ở "
              "nhóm giá trị cao - nên kiểm tra riêng nhóm này thay vì chỉ nhìn chart tổng thể."},
    {"key": "residual_vs_predicted", "label": "4. Residual vs Predicted",
     "files": {"valid": "reg_residual_vs_predicted_valid.png", "holdout": "reg_residual_vs_predicted_holdout.png"},
     "guide": "Residual = Actual - Predicted. Điểm phân bố ngẫu nhiên quanh 0: tốt. Residual dương ở vùng "
              "dự báo cao: model thường dự báo thấp ở đó; âm: thường dự báo cao. Độ phân tán mở rộng theo "
              "Predicted: sai số tăng theo quy mô target. Đường cong/mẫu hình rõ: còn quan hệ phi tuyến "
              "chưa được học."},
    {"key": "residual_distribution", "label": "5. Residual distribution",
     "files": {"valid": "reg_residual_distribution_valid.png", "holdout": "reg_residual_distribution_holdout.png"},
     "guide": "Tập trung quanh 0: ít bias tổng thể. Trung tâm lệch xa 0: model có xu hướng dự báo cao/thấp. "
              "Đuôi dài: một số quan sát sai rất lớn. Nhiều đỉnh: dữ liệu có thể gồm các nhóm hành vi khác "
              "nhau. Residual không cần phân phối chuẩn để dự báo tốt."},
    {"key": "mae_by_target_bin", "label": "6. MAE theo nhóm target",
     "files": {"valid": "reg_mae_by_target_bin_valid.png", "holdout": "reg_mae_by_target_bin_holdout.png"},
     "guide": "Target được chia tối đa 5 nhóm theo phân vị (thấp -> cao). MAE tuyệt đối thường cao hơn ở nhóm "
              "target lớn ngay cả khi sai số tương đối tương đương, vì vậy panel bên phải cho thấy sai số "
              "tương đối (MAE / mean|y|). Nếu MAE tăng mạnh ở nhóm cao: thêm feature về quy mô/biến động, "
              "đổi loss (MAE, Huber, Poisson, Tweedie) hoặc thử dự báo log1p(target) khi target dương, lệch phải."},
    {"key": "permutation_importance", "label": "7. Permutation feature importance (Validation)",
     "files": {"valid": "reg_permutation_importance.png"},
     "guide": "Đảo ngẫu nhiên một feature trên Validation rồi đo MAE tăng bao nhiêu. Dương lớn: model phụ thuộc "
              "nhiều. Gần 0: ít ảnh hưởng. Âm: đảo feature lại làm điểm tốt hơn - biến nhiễu hoặc do biến động "
              "mẫu. Đây là mức độ quan trọng đối với dự báo, không phải quan hệ nhân quả; hai feature tương "
              "quan cao sẽ chia sẻ importance."},
]


def chart_file_list() -> List[tuple]:
    """Flat [(filename, label)] for every regression benchmark chart."""
    out = []
    for spec in CHART_SPECS:
        for which, fname in spec["files"].items():
            suffix = {"valid": " - Validation", "holdout": " - OOT/Test"}[which] if len(spec["files"]) > 1 else ""
            out.append((fname, spec["label"] + suffix))
    return out


# ── metrics ───────────────────────────────────────────────────────────────

def reg_metrics(y_true, y_pred) -> Dict[str, Any]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "mape": None,
    }
    # MAPE only when the target is not (near) zero - otherwise it explodes.
    eps = max(1e-8, 0.01 * float(np.median(np.abs(y_true))))
    nz = np.abs(y_true) > eps
    if len(y_true) and nz.mean() >= 0.99:
        m["mape"] = float(np.mean(np.abs(y_true[nz] - y_pred[nz]) / np.abs(y_true[nz])) * 100)
    return m


# ── model ladder ──────────────────────────────────────────────────────────

def _to_nan(X: pd.DataFrame) -> pd.DataFrame:
    """Undo the -999 NaN sentinel so linear models / RF can impute properly."""
    return X.where(X != SENTINEL)


class _NanAware:
    """Wrap an sklearn pipeline so it first converts the -999 sentinel back to NaN."""

    def __init__(self, est):
        self.est = est

    def fit(self, X, y):
        self.est.fit(_to_nan(X), y)
        return self

    def predict(self, X):
        return self.est.predict(_to_nan(X))

    def predict_proba(self, X):
        return self.est.predict_proba(_to_nan(X))

    def get_params(self, deep=True):
        return {"est": self.est}

    def set_params(self, **p):
        self.est = p.get("est", self.est)
        return self


def linear_preprocessor(categorical_cols: Optional[List[str]] = None):
    """Preprocessing for the LINEAR baselines (Linear / Ridge / Logistic).

    Categorical columns arrive label-encoded (integer codes); feeding codes to a
    linear model imposes a fake order, so they are one-hot encoded (rare levels
    grouped, unseen levels ignored). Numeric columns: median-impute + scale.
    Tree models do not need this and keep the raw codes."""
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    cats = list(categorical_cols or [])
    numeric = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
    if not cats:
        return numeric
    cat_set = set(cats)
    onehot = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=0.005, max_categories=50))
    return ColumnTransformer(
        [("cat", onehot, lambda X: [c for c in X.columns if c in cat_set]),
         ("num", numeric, lambda X: [c for c in X.columns if c not in cat_set])],
        remainder="drop")


def _build_ladder(seed: int, n_train: int,
                  categorical_cols: Optional[List[str]] = None) -> Dict[str, Callable[[], Any]]:
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LinearRegression, RidgeCV
    from sklearn.pipeline import make_pipeline

    def linear():
        return _NanAware(make_pipeline(linear_preprocessor(categorical_cols), LinearRegression()))

    def ridge():
        return _NanAware(make_pipeline(linear_preprocessor(categorical_cols),
                                       RidgeCV(alphas=np.logspace(-3, 3, 13))))

    def rf():
        return _NanAware(make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestRegressor(n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=seed,
                                  max_samples=min(1.0, 200_000 / max(n_train, 1)))))

    ladder: Dict[str, Callable[[], Any]] = {
        "Mean baseline": lambda: DummyRegressor(strategy="mean"),
        "Median baseline": lambda: DummyRegressor(strategy="median"),
        "Linear Regression": linear,
        "Ridge": ridge,
        "Random Forest": rf,
    }
    try:
        import lightgbm as lgb
        ladder["LightGBM"] = lambda: lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1)
    except ImportError:
        pass
    try:
        import xgboost as xgb
        ladder["XGBoost"] = lambda: xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", random_state=seed, n_jobs=-1, verbosity=0)
    except ImportError:
        pass
    return ladder


# ── main entry ────────────────────────────────────────────────────────────

def run_regression_benchmark(
    X: Dict[str, pd.DataFrame],
    y: Dict[str, np.ndarray],
    chart_dir: str,
    report_dir: str,
    tuned_factory: Optional[Callable[[], Any]] = None,
    tuned_label: Optional[str] = None,
    deployed_metrics: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    time_budget_s: int = 300,
    log: Callable[[str, str], None] = lambda stage, msg: None,
    categorical_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run baseline + benchmark. X/y are keyed by split name (must include "train").
    `categorical_cols` (label-encoded columns) are one-hot encoded for the linear models."""
    t0 = time.time()
    splits = [s for s in ALL_SPLITS if s in X and len(X[s]) >= 2]
    if "train" not in splits:
        raise ValueError("benchmark needs a non-empty train split")
    valid_set = next((s for s in VALID_CANDIDATES if s in splits), None)
    holdout_set = next((s for s in HOLDOUT_CANDIDATES if s in splits), None)
    if valid_set is None and holdout_set is None:
        raise ValueError("benchmark needs at least one validation / OOT / test split")

    ladder = _build_ladder(seed, len(X["train"]), categorical_cols)
    if tuned_factory is not None:
        ladder[tuned_label or "Pipeline tuned"] = tuned_factory
    tuned_name = (tuned_label or "Pipeline tuned") if tuned_factory is not None else None

    rows: List[Dict[str, Any]] = []
    preds: Dict[str, Dict[str, np.ndarray]] = {}
    fitted: Dict[str, Any] = {}
    failed: Dict[str, str] = {}
    fit_seconds: Dict[str, float] = {}

    for name, make in ladder.items():
        if time.time() - t0 > time_budget_s:
            failed[name] = f"skipped: benchmark time budget {time_budget_s}s exceeded"
            continue
        try:
            t1 = time.time()
            model = make()
            model.fit(X["train"], y["train"])
            fit_seconds[name] = round(time.time() - t1, 2)
            fitted[name] = model
            preds[name] = {}
            for s in splits:
                p = np.asarray(model.predict(X[s]), dtype=float)
                preds[name][s] = p
                rows.append({"model": name, "split": s, "n": int(len(p)), **reg_metrics(y[s], p)})
            log("Benchmark", f"{name}: fit {fit_seconds[name]}s | " + " | ".join(
                f"{r['split']} MAE={r['mae']:.4g}" for r in rows if r["model"] == name and r["split"] in (valid_set, holdout_set)))
        except Exception as e:  # noqa: BLE001 - one bad model must not kill the benchmark
            failed[name] = f"{type(e).__name__}: {e}"
            log("Benchmark WARN", f"{name} failed: {failed[name]}")

    if "Median baseline" not in fitted:
        raise RuntimeError("baseline models failed - cannot benchmark")

    def mae_of(model: str, split: Optional[str]) -> Optional[float]:
        for r in rows:
            if r["model"] == model and r["split"] == split:
                return r["mae"]
        return None

    ref_split = valid_set or holdout_set
    real_models = [m for m in fitted if m not in BASELINES]
    best_valid_model = min(real_models, key=lambda m: mae_of(m, ref_split) or np.inf) if real_models else None
    chart_model = tuned_name if tuned_name in fitted else best_valid_model
    if chart_model is None:
        raise RuntimeError("no non-baseline model could be fitted")

    result: Dict[str, Any] = {
        "task": "regression",
        "eval_sets": {"valid": valid_set, "holdout": holdout_set},
        "n_rows": {s: int(len(X[s])) for s in splits},
        "models": list(fitted), "failed": failed, "fit_seconds": fit_seconds,
        "rows": rows, "chart_model": chart_model, "best_valid_model": best_valid_model,
        "tuned_model": tuned_name, "deployed_metrics": deployed_metrics or {},
        "charts": {},
    }

    # ── charts + chart-derived diagnostics ───────────────────────────────
    diag: Dict[str, Any] = {}
    try:
        diag = _make_charts(result, X, y, preds, fitted, chart_dir, seed, log)
    except Exception as e:  # noqa: BLE001
        log("Charts ERROR", f"benchmark charts failed: {e}")
    result["diagnostics"] = diag.get("diagnostics", {})
    result["importance"] = diag.get("importance", [])
    result["charts"] = diag.get("charts", {})

    result["findings"] = build_findings(result)
    result["recommended_model"] = _recommend(result)

    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "regression_benchmark_report.md"
    md_path.write_text(render_markdown(result), encoding="utf-8")
    result["report_md"] = str(md_path)
    json_path = out_dir / "regression_benchmark.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=_json_default)
    result["report_json"] = str(json_path)
    result["seconds"] = round(time.time() - t0, 1)
    return result


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return str(o)


# ── charts ────────────────────────────────────────────────────────────────

def _make_charts(result, X, y, preds, fitted, chart_dir, seed, log) -> Dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cdir = Path(chart_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    rows = result["rows"]
    models = result["models"]
    cm = result["chart_model"]
    sets = {k: v for k, v in result["eval_sets"].items() if v}
    saved: Dict[str, str] = {}
    diagnostics: Dict[str, Any] = {}
    BLUE, RED, GREY, GREEN = "#2196F3", "#F44336", "#9E9E9E", "#4CAF50"
    STYLE = {"dpi": 120, "bbox_inches": "tight"}
    spec_files = {s["key"]: s["files"] for s in CHART_SPECS}
    set_title = {"valid": "Validation", "holdout": None}

    def title_of(which):
        return "Validation" if which == "valid" else str(sets["holdout"]).upper()

    def get(model, split, metric):
        for r in rows:
            if r["model"] == model and r["split"] == split:
                return r.get(metric)
        return None

    def save(key, which, fig):
        path = str(cdir / spec_files[key][which])
        fig.savefig(path, **STYLE)
        plt.close(fig)
        saved[f"{key}:{which}"] = path

    def guard(name, fn):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            log("Chart WARN", f"{name} failed: {e}")
            plt.close("all")

    # 1. model comparison
    def chart1(which):
        split = sets[which]
        order = sorted(models, key=lambda m: get(m, split, "mae") if get(m, split, "mae") is not None else np.inf)
        fig, axes = plt.subplots(1, 3, figsize=(15, max(3.5, 0.5 * len(order) + 1.5)))
        for ax, metric, lower_better in zip(axes, ("mae", "rmse", "r2"), (True, True, False)):
            vals = [get(m, split, metric) for m in order]
            colors = [GREY if m in BASELINES else (GREEN if m == cm else BLUE) for m in order]
            ax.barh(order, vals, color=colors)
            ax.invert_yaxis()
            ax.set_title(f"{metric.upper() if metric != 'r2' else 'R²'} ({'lower' if lower_better else 'higher'} is better)")
            ax.grid(axis="x", linestyle=":", alpha=0.4)
            for i, v in enumerate(vals):
                if v is not None and np.isfinite(v):
                    ax.text(v, i, f" {v:.4g}", va="center", fontsize=8)
            if metric != "mae":
                ax.set_yticklabels([])
        fig.suptitle(f"Model comparison - {title_of(which)} (grey=baseline, green=pipeline model)")
        plt.tight_layout()
        save("model_comparison", which, fig)

    for which in sets:
        guard(f"model_comparison:{which}", lambda w=which: chart1(w))

    # 2. train vs validation MAE
    def chart2():
        order = [m for m in models]
        fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(order)), 5))
        series = [("train", "Train", GREEN)]
        if "valid" in sets:
            series.append((sets["valid"], "Validation", BLUE))
        if "holdout" in sets:
            series.append((sets["holdout"], str(sets["holdout"]).upper(), RED))
        w = 0.8 / len(series)
        for i, (split, lab, col) in enumerate(series):
            vals = [get(m, split, "mae") or 0 for m in order]
            ax.bar(np.arange(len(order)) + i * w, vals, w, label=lab, color=col)
        ax.set_xticks(np.arange(len(order)) + 0.4 - w / 2)
        ax.set_xticklabels(order, rotation=30, ha="right")
        ax.set_ylabel("MAE")
        ax.set_title("Train vs Validation MAE (large gap = overfit / leakage / drift)")
        ax.legend()
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        plt.tight_layout()
        save("train_vs_valid_mae", "valid", fig)

    guard("train_vs_valid_mae", chart2)

    # 3-6 per-set diagnostics for the chart model
    for which, split in sets.items():
        if len(y[split]) < 10:
            log("Chart WARN", f"{split} has <10 rows - skipping diagnostic charts for {which}")
            continue
        yt = np.asarray(y[split], dtype=float)
        yp = preds[cm][split]
        resid = yt - yp
        m = reg_metrics(yt, yp)
        tt = f"{cm} - {title_of(which)} (n={len(yt)})"

        def chart3(yt=yt, yp=yp, m=m, tt=tt, which=which):
            fig, ax = plt.subplots(figsize=(7, 6))
            ax.scatter(yt, yp, s=10, alpha=0.5, color=BLUE)
            lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
            ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="Perfect (y = x)")
            ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
            ax.set_title(f"Actual vs Predicted - {tt}\nMAE={m['mae']:.4g} | RMSE={m['rmse']:.4g} | R²={m['r2']:.4f}")
            ax.legend(); ax.grid(linestyle=":", alpha=0.4)
            plt.tight_layout(); save("actual_vs_predicted", which, fig)

        def chart4(yp=yp, resid=resid, tt=tt, which=which):
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(yp, resid, s=10, alpha=0.5, color=BLUE)
            ax.axhline(0, color=RED, linestyle="--", lw=1.5)
            try:  # rolling mean trend makes curvature / bias visible
                order = np.argsort(yp)
                k = max(5, len(yp) // 20)
                trend = pd.Series(resid[order]).rolling(k, center=True, min_periods=1).mean()
                ax.plot(yp[order], trend, color="black", lw=1.5, label="rolling mean residual")
                ax.legend()
            except Exception:  # noqa: BLE001
                pass
            ax.set_xlabel("Predicted"); ax.set_ylabel("Residual (actual - predicted)")
            ax.set_title(f"Residual vs Predicted - {tt}")
            ax.grid(linestyle=":", alpha=0.4)
            plt.tight_layout(); save("residual_vs_predicted", which, fig)

        def chart5(resid=resid, tt=tt, which=which):
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(resid, bins=min(50, max(10, len(resid) // 5)), color=BLUE, alpha=0.8)
            ax.axvline(0, color=RED, linestyle="--", lw=1.5, label="0")
            ax.axvline(float(np.mean(resid)), color="black", lw=1.2, label=f"mean={np.mean(resid):.4g}")
            ax.set_xlabel("Residual"); ax.set_ylabel("Count")
            ax.set_title(f"Residual distribution - {tt}")
            ax.legend(); ax.grid(linestyle=":", alpha=0.4)
            plt.tight_layout(); save("residual_distribution", which, fig)

        def chart6(yt=yt, yp=yp, split=split, tt=tt, which=which):
            q = min(5, max(2, len(yt) // 10))
            bins = pd.qcut(pd.Series(yt), q=q, duplicates="drop")
            df = pd.DataFrame({"bin": bins, "ae": np.abs(yt - yp),
                               "ae_base": np.abs(yt - preds["Median baseline"][split]), "ay": np.abs(yt)})
            g = df.groupby("bin", observed=True)
            tab = pd.DataFrame({"mae": g["ae"].mean(), "mae_base": g["ae_base"].mean(),
                                "rel": g["ae"].mean() / g["ay"].mean().replace(0, np.nan) * 100,
                                "n": g.size()})
            if len(tab) < 2:
                return
            labels = [f"Q{i+1}" + (" (low)" if i == 0 else " (high)" if i == len(tab) - 1 else "")
                      for i in range(len(tab))]
            diagnostics.setdefault(which, {})["bins"] = [
                {"bin": lab, "range": str(idx), "n": int(r.n), "mae": float(r.mae),
                 "mae_median_baseline": float(r.mae_base),
                 "rel_error_pct": None if pd.isna(r.rel) else float(r.rel)}
                for lab, (idx, r) in zip(labels, tab.iterrows())]
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            x = np.arange(len(tab))
            axes[0].bar(x - 0.2, tab["mae"], 0.4, color=BLUE, label=cm)
            axes[0].bar(x + 0.2, tab["mae_base"], 0.4, color=GREY, label="Median baseline")
            axes[0].set_xticks(x); axes[0].set_xticklabels(labels); axes[0].set_ylabel("MAE")
            axes[0].set_title("MAE by target quantile"); axes[0].legend()
            axes[1].bar(x, tab["rel"], color=RED, alpha=0.8)
            axes[1].set_xticks(x); axes[1].set_xticklabels(labels); axes[1].set_ylabel("MAE / mean|y| (%)")
            axes[1].set_title("Relative error by target quantile")
            for a in axes:
                a.grid(axis="y", linestyle=":", alpha=0.4)
            fig.suptitle(f"MAE by target group - {tt}")
            plt.tight_layout(); save("mae_by_target_bin", which, fig)

        for nm, fn in (("actual_vs_predicted", chart3), ("residual_vs_predicted", chart4),
                       ("residual_distribution", chart5), ("mae_by_target_bin", chart6)):
            guard(f"{nm}:{which}", fn)

        diagnostics.setdefault(which, {}).update({
            "split": split, "n": int(len(yt)), "mean_residual": float(np.mean(resid)),
            "mae": m["mae"], "rmse": m["rmse"], "r2": m["r2"],
            "n_extreme": int((np.abs(resid) > 3 * max(m["mae"], 1e-12)).sum()),
        })

    # 7. permutation importance (validation, MAE)
    importance: List[Dict[str, Any]] = []
    vs = sets.get("valid") or sets.get("holdout")
    if vs and cm in fitted and len(y[vs]) >= 10:
        def chart7():
            from sklearn.inspection import permutation_importance
            Xv, yv = X[vs], np.asarray(y[vs], dtype=float)
            if len(Xv) > 3000:
                idx = np.random.RandomState(seed).choice(len(Xv), 3000, replace=False)
                Xv, yv = Xv.iloc[idx], yv[idx]
            pi = permutation_importance(fitted[cm], Xv, yv, scoring="neg_mean_absolute_error",
                                        n_repeats=5 if Xv.shape[1] <= 80 else 3,
                                        random_state=seed, n_jobs=1)
            df = pd.DataFrame({"feature": Xv.columns, "mean": pi.importances_mean, "std": pi.importances_std})
            df = df.sort_values("mean", ascending=False).reset_index(drop=True)
            importance.extend(df.head(30).to_dict(orient="records"))
            top = df.head(20)[::-1]
            fig, ax = plt.subplots(figsize=(8, max(3.5, 0.32 * len(top) + 1.5)))
            ax.barh(top["feature"], top["mean"], xerr=top["std"],
                    color=[RED if v < 0 else BLUE for v in top["mean"]])
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("MAE increase when feature is shuffled")
            ax.set_title(f"Permutation importance - {cm} on {vs} (n={len(Xv)})")
            ax.grid(axis="x", linestyle=":", alpha=0.4)
            plt.tight_layout(); save("permutation_importance", "valid", fig)
            diagnostics["importance_total_pos"] = float(df.loc[df["mean"] > 0, "mean"].sum())
            diagnostics["importance_n_negative"] = int((df.head(20)["mean"] < 0).sum())
        guard("permutation_importance", chart7)

    log("Charts Complete", f"{len(saved)} regression benchmark charts saved")
    return {"charts": saved, "diagnostics": diagnostics, "importance": importance}


# ── interpretation ────────────────────────────────────────────────────────

def _row(result, model, split, metric):
    for r in result["rows"]:
        if r["model"] == model and r["split"] == split:
            return r.get(metric)
    return None


def build_findings(result: Dict[str, Any]) -> List[Dict[str, str]]:
    """Rule-based decision checklist. Thresholds are heuristics, not standards."""
    f: List[Dict[str, str]] = []
    cm = result["chart_model"]
    vs = result["eval_sets"].get("valid")
    hs = result["eval_sets"].get("holdout")
    ref = vs or hs
    diag = result.get("diagnostics", {})

    def add(sev, title, finding, action):
        f.append({"severity": sev, "title": title, "finding": finding, "action": action})

    # 1. versus median baseline
    mae_m, mae_b = _row(result, cm, ref, "mae"), _row(result, "Median baseline", ref, "mae")
    if mae_m is not None and mae_b:
        gain = 1 - mae_m / mae_b
        if gain <= 0:
            add("bad", "MAE không tốt hơn Median baseline",
                f"{cm}: MAE={mae_m:.4g} vs Median baseline {mae_b:.4g} trên {ref}.",
                "Xem lại feature, định nghĩa target và cách chia dữ liệu; mô hình hiện chưa học được tín hiệu hữu ích.")
        elif gain < 0.05:
            add("warn", "Cải thiện so với Median baseline rất nhỏ",
                f"MAE giảm chỉ {gain:.1%} so với Median baseline ({mae_m:.4g} vs {mae_b:.4g}).",
                "Kiểm tra chất lượng feature và độ nhiễu của target trước khi tin vào mô hình.")
        else:
            add("ok", "Tốt hơn Median baseline",
                f"MAE giảm {gain:.1%} so với Median baseline ({mae_m:.4g} vs {mae_b:.4g}) trên {ref}.",
                "Xác nhận thêm trên OOT/Test và qua nhiều kỳ.")
    negs = [r["model"] for r in result["rows"] if r["split"] == ref and r["model"] not in BASELINES
            and r["r2"] is not None and r["r2"] < 0]
    if negs:
        add("info", "Có model R² < 0",
            f"Các model sau kém hơn dự báo bằng trung bình trên {ref}: {', '.join(negs)}.",
            "Loại các model này khỏi lựa chọn; R² âm nghĩa là sai số lớn hơn việc luôn dự báo mean.")

    # 2. train vs validation
    tr, va = _row(result, cm, "train", "mae"), _row(result, cm, vs, "mae") if vs else None
    if tr is not None and va is not None:
        base_tr = _row(result, "Median baseline", "train", "mae")
        close_to_base = base_tr and tr > 0.95 * base_tr
        if close_to_base:
            add("warn", "Train và Validation đều cao (có thể underfit)",
                f"Train MAE={tr:.4g} gần Median baseline ({base_tr:.4g}); Validation MAE={va:.4g}.",
                "Thêm/đổi feature, tăng năng lực model hoặc kiểm tra target có quá nhiễu.")
        elif va > 3 * max(tr, 1e-12):
            add("warn", "Train tốt, Validation kém (nghi overfit / leakage / lệch phân phối)",
                f"Train MAE={tr:.4g} nhưng Validation MAE={va:.4g} (x{va / max(tr, 1e-12):.1f}).",
                "Kiểm tra overfitting (tăng regularization, giảm độ phức tạp) và leakage; so phân phối train/valid.")
        elif va > 1.5 * max(tr, 1e-12):
            add("info", "Chênh lệch Train/Validation đáng chú ý",
                f"Train MAE={tr:.4g}, Validation MAE={va:.4g} (x{va / max(tr, 1e-12):.1f}). "
                "Mức này thường gặp với model cây; quan trọng hơn là Validation vẫn tốt hơn baseline.",
                "Theo dõi qua các kỳ; nếu chênh lệch tăng dần, tăng regularization hoặc giảm độ phức tạp.")
        else:
            add("ok", "Train và Validation gần nhau",
                f"Train MAE={tr:.4g}, Validation MAE={va:.4g}.", "Khả năng tổng quát hóa ổn; tiếp tục theo dõi qua các kỳ.")

    # 3. error growth with target level
    for which in ("valid", "holdout"):
        bins = diag.get(which, {}).get("bins")
        if bins and len(bins) >= 2 and bins[0]["mae"] > 0:
            ratio = bins[-1]["mae"] / bins[0]["mae"]
            rel = [b["rel_error_pct"] for b in bins if b["rel_error_pct"] is not None]
            if ratio > 2:
                scale_only = rel and max(rel) / max(min(rel), 1e-12) < 1.5
                add("warn", f"Sai số tăng theo target ({'Validation' if which == 'valid' else str(hs).upper()})",
                    f"MAE nhóm cao nhất gấp {ratio:.1f} lần nhóm thấp nhất."
                    + (" Sai số tương đối khá đều, nên phần lớn là hiệu ứng quy mô." if scale_only else
                       " Sai số tương đối cũng chênh lệch rõ."),
                    "Phân tích riêng nhóm giá trị lớn; thử loss phù hợp (MAE/Huber/Poisson/Tweedie) hoặc dự báo log1p(target) "
                    "nếu target dương và lệch phải; bổ sung feature về quy mô/biến động.")
                break

    # 4. MAE good but RMSE high
    for which in ("holdout", "valid"):
        d = diag.get(which)
        if d and d["mae"] > 0 and d["rmse"] / d["mae"] > 1.5:
            add("warn", "RMSE cao so với MAE (có lỗi cực lớn)",
                f"{'Validation' if which == 'valid' else str(hs).upper()}: RMSE/MAE={d['rmse'] / d['mae']:.2f}; "
                f"{d['n_extreme']} quan sát có |residual| > 3×MAE.",
                "Kiểm tra từng quan sát sai cực lớn (outlier, lỗi dữ liệu, nhóm khách hàng đặc biệt) trước khi chốt model.")
            break

    # 5. validation vs hold-out
    if vs and hs:
        mv, mh = _row(result, cm, vs, "mae"), _row(result, cm, hs, "mae")
        if mv and mh is not None and mh > 1.2 * mv:
            add("warn", f"Validation tốt nhưng {str(hs).upper()} kém",
                f"MAE Validation={mv:.4g}, {str(hs).upper()}={mh:.4g} (x{mh / mv:.2f}).",
                "Kiểm tra drift, thay đổi hành vi và tính ổn định theo thời gian (PSI, stability); "
                "xem xét train lại theo kỳ gần nhất." if hs == "oot" else
                "Kiểm tra biến động do kích thước mẫu nhỏ và khác biệt phân phối giữa valid/test.")
        elif mv and mh is not None:
            add("ok", f"Validation và {str(hs).upper()} nhất quán",
                f"MAE Validation={mv:.4g}, {str(hs).upper()}={mh:.4g}.", "Kết quả ổn định giữa hai tập.")

    # 6. bias
    for which in ("holdout", "valid"):
        d = diag.get(which)
        if d and d["mae"] > 0 and abs(d["mean_residual"]) > 0.1 * d["mae"]:
            direction = "thấp hơn" if d["mean_residual"] > 0 else "cao hơn"
            add("warn", "Residual lệch khỏi 0 (bias tổng thể)",
                f"{'Validation' if which == 'valid' else str(hs).upper()}: mean residual={d['mean_residual']:.4g} "
                f"(~{abs(d['mean_residual']) / d['mae']:.0%} MAE) - model dự báo {direction} thực tế trung bình.",
                "Kiểm tra lệch phân phối target giữa các tập; cân nhắc hiệu chỉnh bias hoặc đổi loss.")
            break

    # 7. feature importance sanity
    imp = result.get("importance") or []
    total = diag.get("importance_total_pos") or 0
    if imp and total > 0 and imp[0]["mean"] / total > 0.5:
        add("warn", "Một feature chiếm phần lớn importance",
            f"'{imp[0]['feature']}' chiếm {imp[0]['mean'] / total:.0%} tổng permutation importance dương.",
            "Kiểm tra leakage (feature suy ra từ target hoặc chỉ biết sau thời điểm dự báo) và cách định nghĩa feature.")
    elif imp and diag.get("importance_n_negative", 0) >= 5:
        add("info", "Nhiều feature có importance âm",
            f"{diag['importance_n_negative']}/20 feature hàng đầu có importance âm (đảo lại lại cải thiện MAE).",
            "Các biến này có thể là nhiễu - cân nhắc loại bỏ và kiểm tra lại đầu vào.")
    elif imp:
        add("ok", "Importance tập trung hợp lý",
            f"Top feature: {', '.join(i['feature'] for i in imp[:3])}.",
            "Đối chiếu với kiến thức nghiệp vụ xem thứ hạng có hợp lý; importance không phải quan hệ nhân quả.")
    return f


def _recommend(result: Dict[str, Any]) -> Dict[str, Any]:
    """Pick the model to ship: best MAE, but prefer a simpler one within 3%."""
    ref = result["eval_sets"].get("valid") or result["eval_sets"].get("holdout")
    cand = [m for m in result["models"] if m not in BASELINES]
    maes = {m: _row(result, m, ref, "mae") for m in cand}
    maes = {m: v for m, v in maes.items() if v is not None}
    if not maes:
        return {}
    best = min(maes, key=maes.get)
    simple_order = ["Linear Regression", "Ridge"]  # explainable alternatives only
    near = [] if best in simple_order else [
        m for m in simple_order if m in maes and maes[m] <= maes[best] * 1.03]
    pick = near[0] if near else best
    note = (f"{best} có MAE thấp nhất trên {ref} ({maes[best]:.4g}).")
    if pick != best:
        note += f" {pick} chỉ kém ≤3% ({maes[pick]:.4g}) nhưng là model tuyến tính, dễ giải thích hơn - cân nhắc ưu tiên."
    return {"best_by_mae": best, "suggested": pick, "reference_split": ref, "note": note}


# ── markdown ──────────────────────────────────────────────────────────────

def _fmt(v, nd=4):
    return "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}g}"


def results_table(result: Dict[str, Any], split: str) -> pd.DataFrame:
    recs = [{"Model": r["model"], "MAE": r["mae"], "RMSE": r["rmse"], "R²": r["r2"],
             "MAPE %": r["mape"], "n": r["n"]} for r in result["rows"] if r["split"] == split]
    df = pd.DataFrame(recs)
    return df.sort_values("MAE").reset_index(drop=True) if len(df) else df


def render_markdown(result: Dict[str, Any]) -> str:
    sets = result["eval_sets"]
    L: List[str] = ["# Regression baseline & benchmark\n"]
    L.append(f"Tất cả model được fit trên **train** ({result['n_rows'].get('train')} dòng) và chấm cùng metric "
             f"trên Validation (`{sets['valid']}`) và OOT/Test (`{sets['holdout']}`). "
             f"Model dùng cho chart: **{result['chart_model']}**.\n")
    rec = result.get("recommended_model") or {}
    if rec:
        L.append(f"**Khuyến nghị:** {rec['note']}\n")
    for which, split in (("Validation", sets["valid"]), ("OOT/Test", sets["holdout"])):
        if not split:
            continue
        t = results_table(result, split)
        if t.empty:
            continue
        L.append(f"## Bảng so sánh - {which} (`{split}`)\n")
        L.append("| Model | MAE | RMSE | R² | MAPE % |\n|---|---|---|---|---|")
        for _, r in t.iterrows():
            L.append(f"| {r['Model']} | {_fmt(r['MAE'])} | {_fmt(r['RMSE'])} | {_fmt(r['R²'])} | {_fmt(r['MAPE %'], 3)} |")
        L.append("")
    if result.get("failed"):
        L.append("Model không chạy được: " + "; ".join(f"{k} ({v})" for k, v in result["failed"].items()) + "\n")
    L.append("## Chart và cách đọc\n")
    for spec in CHART_SPECS:
        shown = [(w, p) for w, p in ((w, result["charts"].get(f"{spec['key']}:{w}")) for w in spec["files"]) if p]
        if not shown:
            continue
        L.append(f"### {spec['label']}\n\n{spec['guide']}\n")
        for w, p in shown:
            L.append(f"![{spec['label']} {w}](charts/{Path(p).name})\n")
    L.append("## Đánh giá sau huấn luyện và khuyến nghị\n")
    icon = {"ok": "OK", "warn": "CẢNH BÁO", "bad": "NGHIÊM TRỌNG", "info": "LƯU Ý"}
    L.append("| Mức | Phát hiện | Chi tiết | Hành động đề xuất |\n|---|---|---|---|")
    for x in result["findings"]:
        L.append(f"| {icon.get(x['severity'], x['severity'])} | {x['title']} | {x['finding']} | {x['action']} |")
    L.append("\n## Lưu ý phương pháp\n")
    L.append("- Các model benchmark dùng tham số mặc định hợp lý (không tune) và chỉ fit trên train; "
             "model `Pipeline tuned` dùng hyperparameter do pipeline chọn, cũng fit trên train để so sánh công bằng.\n"
             "- Model triển khai thật được refit trên train+valid nên điểm valid của nó là in-sample; "
             "chart dùng bản train-only để Validation là hold-out thật.\n"
             "- Linear/Ridge: biến category được one-hot (gom nhóm hiếm), biến số median-impute + chuẩn hóa.\n"
             "- Ngưỡng trong checklist (ví dụ RMSE/MAE > 1.5, Validation > 1.5× Train) là heuristic, "
             "không phải chuẩn - luôn so với baseline và theo dõi qua các kỳ.\n"
             "- MAPE chỉ tính khi target không bằng hoặc gần 0.\n")
    return "\n".join(L)


# ── UI hints (shared shape with classification_benchmark) ─────────────────

TABLE_FORMAT = {"MAE": "{:.4g}", "RMSE": "{:.4g}", "R²": "{:.4f}", "MAPE %": "{:.2f}"}
HIGHER_BETTER = ["R²"]
LOWER_BETTER = ["MAE", "RMSE"]

CHECKLIST = [
    ("MAE không tốt hơn Median baseline", "Xem lại feature, target và cách chia dữ liệu"),
    ("Train tốt, Validation kém", "Kiểm tra overfitting và leakage"),
    ("Sai số tăng theo target", "Phân tích riêng nhóm giá trị lớn; thử loss hoặc biến đổi target phù hợp"),
    ("MAE tốt nhưng RMSE cao", "Kiểm tra các trường hợp sai số cực lớn"),
    ("Validation tốt nhưng OOT kém", "Kiểm tra drift, thay đổi hành vi và tính ổn định theo thời gian"),
    ("Feature importance không hợp lý", "Kiểm tra dữ liệu đầu vào, leakage và cách định nghĩa feature"),
]
