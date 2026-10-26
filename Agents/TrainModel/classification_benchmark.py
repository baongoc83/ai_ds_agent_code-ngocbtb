"""Binary-classification baseline + benchmark for Agent 3.

Mirror of regression_benchmark.py: fits a fixed ladder of models on the SAME
train split and scores them with the SAME metrics (AUC / Gini / KS / PR-AUC /
LogLoss / Brier) on every validation / OOT / test split:

    Prior baseline -> Logistic Regression -> Random Forest -> LightGBM
    -> XGBoost -> Pipeline-tuned model

then draws 7 diagnostic charts on the validation set and on the OOT/test
hold-out and turns the numbers into a rule-based decision checklist.

Every model here is fitted on `train` only, so `valid` is a real hold-out. The
DEPLOYED model (refit on train+valid, calibrated) keeps its own 9 charts.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss,
                             precision_recall_curve, roc_auc_score, roc_curve)

from Agents.TrainModel.regression_benchmark import (
    ALL_SPLITS, HOLDOUT_CANDIDATES, VALID_CANDIDATES, _NanAware, _fmt, _json_default,
    linear_preprocessor)

BASELINES = ("Prior baseline",)

CHART_SPECS: List[Dict[str, Any]] = [
    {"key": "model_comparison", "label": "1. Model comparison (AUC / KS / Brier)",
     "files": {"valid": "clf_model_comparison_valid.png", "holdout": "clf_model_comparison_holdout.png"},
     "guide": "AUC, KS càng cao càng tốt (khả năng phân biệt Good/Bad); Brier càng thấp càng tốt (độ chính "
              "xác của xác suất). Gini = 2×AUC − 1. Prior baseline luôn có AUC = 0.5 - model nào gần 0.5 "
              "là gần như không phân biệt được. Chênh lệch nhỏ giữa các model chỉ đáng tin khi lặp lại trên "
              "cả Validation và OOT/Test."},
    {"key": "train_vs_valid_auc", "label": "2. Train vs Validation AUC",
     "files": {"valid": "clf_train_vs_valid_auc.png"},
     "guide": "Cả hai AUC cao và gần nhau: tổng quát hóa tốt. Train rất cao nhưng Validation thấp hơn nhiều: "
              "có thể overfit, leakage hoặc lệch phân phối. Validation tốt nhưng OOT giảm: dấu hiệu drift "
              "theo thời gian. Không có ngưỡng chung - so với baseline và theo dõi qua các kỳ."},
    {"key": "roc_curves", "label": "3. ROC curves",
     "files": {"valid": "clf_roc_curves_valid.png", "holdout": "clf_roc_curves_holdout.png"},
     "guide": "Đường càng gần góc trên-trái càng tốt; đường chéo là phân loại ngẫu nhiên (AUC = 0.5). Nếu các "
              "đường cắt nhau, model tốt hơn phụ thuộc vào vùng ngưỡng bạn quan tâm (ví dụ vùng FPR thấp khi "
              "duyệt tự động)."},
    {"key": "pr_curves", "label": "4. Precision-Recall curves",
     "files": {"valid": "clf_pr_curves_valid.png", "holdout": "clf_pr_curves_holdout.png"},
     "guide": "Quan trọng khi tỷ lệ Bad thấp (mất cân bằng): đường ngang là tỷ lệ Bad (mức của dự đoán ngẫu "
              "nhiên). Đường càng cao càng tốt; PR-AUC phản ánh chất lượng bắt Bad tốt hơn AUC khi dữ liệu "
              "rất lệch."},
    {"key": "calibration", "label": "5. Calibration (reliability)",
     "files": {"valid": "clf_calibration_valid.png", "holdout": "clf_calibration_holdout.png"},
     "guide": "Mỗi điểm là một nhóm điểm số: trục ngang là xác suất dự báo trung bình, trục dọc là tỷ lệ Bad "
              "thực tế. Nằm trên đường chéo: xác suất đáng tin. Nằm dưới: model dự báo cao hơn thực tế "
              "(over-predict); nằm trên: dự báo thấp hơn thực tế. Model triển khai đã được hiệu chỉnh "
              "(calibrator) riêng - chart này là bản train-only chưa hiệu chỉnh."},
    {"key": "bad_rate_decile", "label": "6. Bad rate theo decile điểm số",
     "files": {"valid": "clf_bad_rate_decile_valid.png", "holdout": "clf_bad_rate_decile_holdout.png"},
     "guide": "Chia khách hàng thành các nhóm theo điểm số (D1 = rủi ro thấp nhất -> D10 = cao nhất). Tỷ lệ Bad "
              "nên tăng đều (rank-ordering đơn điệu). Các chỗ đảo chiều cho thấy model xếp hạng chưa ổn định "
              "ở vùng đó - cần xem lại khi dùng để đặt ngưỡng hoặc phân hạng."},
    {"key": "permutation_importance", "label": "7. Permutation feature importance (Validation)",
     "files": {"valid": "clf_permutation_importance.png"},
     "guide": "Đảo ngẫu nhiên một feature trên Validation rồi đo AUC giảm bao nhiêu. Dương lớn: model phụ thuộc "
              "nhiều. Gần 0: ít ảnh hưởng. Âm: biến nhiễu hoặc do biến động mẫu. Đây là mức độ quan trọng "
              "với dự báo, không phải quan hệ nhân quả; hai feature tương quan cao sẽ chia sẻ importance."},
]


def chart_file_list() -> List[tuple]:
    out = []
    for spec in CHART_SPECS:
        for which, fname in spec["files"].items():
            suffix = {"valid": " - Validation", "holdout": " - OOT/Test"}[which] if len(spec["files"]) > 1 else ""
            out.append((fname, spec["label"] + suffix))
    return out


# ── metrics ───────────────────────────────────────────────────────────────

def clf_metrics(y_true, p) -> Dict[str, Any]:
    y = np.asarray(y_true).astype(int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    m: Dict[str, Any] = {"n_pos": int(y.sum()), "bad_rate": float(y.mean()) if len(y) else None,
                         "auc": None, "gini": None, "ks": None, "pr_auc": None,
                         "logloss": None, "brier": float(brier_score_loss(y, p)) if len(y) else None}
    if len(np.unique(y)) == 2:
        m["auc"] = float(roc_auc_score(y, p))
        m["gini"] = 2 * m["auc"] - 1
        fpr, tpr, _ = roc_curve(y, p)
        m["ks"] = float(np.max(tpr - fpr))
        m["pr_auc"] = float(average_precision_score(y, p))
        m["logloss"] = float(log_loss(y, p, labels=[0, 1]))
    return m


def _proba(model, X) -> np.ndarray:
    return np.asarray(model.predict_proba(X)[:, 1], dtype=float)


# ── model ladder ──────────────────────────────────────────────────────────

def _build_ladder(seed: int, n_train: int,
                  categorical_cols: Optional[List[str]] = None) -> Dict[str, Callable[[], Any]]:
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    ladder: Dict[str, Callable[[], Any]] = {
        "Prior baseline": lambda: DummyClassifier(strategy="prior"),
        "Logistic Regression": lambda: _NanAware(make_pipeline(
            linear_preprocessor(categorical_cols), LogisticRegression(max_iter=2000, C=1.0))),
        "Random Forest": lambda: _NanAware(make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestClassifier(n_estimators=150, min_samples_leaf=5, n_jobs=-1, random_state=seed,
                                   max_samples=min(1.0, 200_000 / max(n_train, 1))))),
    }
    try:
        import lightgbm as lgb
        ladder["LightGBM"] = lambda: lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbose=-1)
    except ImportError:
        pass
    try:
        import xgboost as xgb
        ladder["XGBoost"] = lambda: xgb.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0)
    except ImportError:
        pass
    return ladder


# ── main entry ────────────────────────────────────────────────────────────

def run_classification_benchmark(
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
    """Run baseline + benchmark for a binary target. X/y keyed by split (must include "train")."""
    t0 = time.time()
    splits = [s for s in ALL_SPLITS if s in X and len(X[s]) >= 2]
    if "train" not in splits:
        raise ValueError("benchmark needs a non-empty train split")
    if len(np.unique(y["train"])) != 2:
        raise ValueError("classification benchmark supports binary targets only")
    valid_set = next((s for s in VALID_CANDIDATES if s in splits), None)
    holdout_set = next((s for s in HOLDOUT_CANDIDATES if s in splits), None)
    if valid_set is None and holdout_set is None:
        raise ValueError("benchmark needs at least one validation / OOT / test split")

    ladder = _build_ladder(seed, len(X["train"]), categorical_cols)
    tuned_name = None
    if tuned_factory is not None:
        tuned_name = tuned_label or "Pipeline tuned"
        ladder[tuned_name] = tuned_factory

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
                p = _proba(model, X[s])
                preds[name][s] = p
                rows.append({"model": name, "split": s, "n": int(len(p)), **clf_metrics(y[s], p)})
            log("Benchmark", f"{name}: fit {fit_seconds[name]}s | " + " | ".join(
                f"{r['split']} AUC={_fmt(r['auc'])}" for r in rows
                if r["model"] == name and r["split"] in (valid_set, holdout_set)))
        except Exception as e:  # noqa: BLE001 - one bad model must not kill the benchmark
            failed[name] = f"{type(e).__name__}: {e}"
            log("Benchmark WARN", f"{name} failed: {failed[name]}")

    if "Prior baseline" not in fitted:
        raise RuntimeError("baseline model failed - cannot benchmark")

    ref = valid_set or holdout_set
    real = [m for m in fitted if m not in BASELINES]
    best_valid_model = max(real, key=lambda m: _row_of(rows, m, ref, "auc") or -1) if real else None
    chart_model = tuned_name if tuned_name in fitted else best_valid_model
    if chart_model is None:
        raise RuntimeError("no non-baseline model could be fitted")

    result: Dict[str, Any] = {
        "task": "classification",
        "eval_sets": {"valid": valid_set, "holdout": holdout_set},
        "n_rows": {s: int(len(X[s])) for s in splits},
        "models": list(fitted), "failed": failed, "fit_seconds": fit_seconds,
        "rows": rows, "chart_model": chart_model, "best_valid_model": best_valid_model,
        "tuned_model": tuned_name, "deployed_metrics": deployed_metrics or {}, "charts": {},
    }

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
    md_path = out_dir / "classification_benchmark_report.md"
    md_path.write_text(render_markdown(result), encoding="utf-8")
    result["report_md"] = str(md_path)
    json_path = out_dir / "classification_benchmark.json"
    result["report_json"] = str(json_path)
    result["seconds"] = round(time.time() - t0, 1)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=_json_default)
    return result


def _row_of(rows, model, split, metric):
    for r in rows:
        if r["model"] == model and r["split"] == split:
            return r.get(metric)
    return None


def _decile_table(y, p, n_max: int = 10) -> pd.DataFrame:
    q = int(min(n_max, max(2, len(y) // 20)))
    rank = pd.Series(p).rank(method="first")
    grp = pd.qcut(rank, q=q, labels=False, duplicates="drop")
    df = pd.DataFrame({"g": grp, "y": y, "p": p})
    t = df.groupby("g").agg(n=("y", "size"), bad_rate=("y", "mean"), avg_score=("p", "mean")).reset_index()
    t["decile"] = [f"D{i + 1}" for i in range(len(t))]
    return t


# ── charts ────────────────────────────────────────────────────────────────

def _make_charts(result, X, y, preds, fitted, chart_dir, seed, log) -> Dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cdir = Path(chart_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    rows, models, cm = result["rows"], result["models"], result["chart_model"]
    sets = {k: v for k, v in result["eval_sets"].items() if v}
    saved: Dict[str, str] = {}
    diagnostics: Dict[str, Any] = {}
    BLUE, RED, GREY, GREEN = "#2196F3", "#F44336", "#9E9E9E", "#4CAF50"
    STYLE = {"dpi": 120, "bbox_inches": "tight"}
    spec_files = {s["key"]: s["files"] for s in CHART_SPECS}
    real = [m for m in models if m not in BASELINES]

    def title_of(which):
        return "Validation" if which == "valid" else str(sets["holdout"]).upper()

    def get(model, split, metric):
        return _row_of(rows, model, split, metric)

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

    def two_classes(split):
        return len(np.unique(y[split])) == 2

    # 1. model comparison
    def chart1(which):
        split = sets[which]
        order = sorted(models, key=lambda m: -(get(m, split, "auc") or 0))
        fig, axes = plt.subplots(1, 3, figsize=(15, max(3.5, 0.5 * len(order) + 1.5)))
        for ax, metric, title in zip(axes, ("auc", "ks", "brier"),
                                     ("AUC (higher is better)", "KS (higher is better)", "Brier (lower is better)")):
            vals = [get(m, split, metric) for m in order]
            colors = [GREY if m in BASELINES else (GREEN if m == cm else BLUE) for m in order]
            ax.barh(order, [v if v is not None else 0 for v in vals], color=colors)
            ax.invert_yaxis()
            ax.set_title(title)
            ax.grid(axis="x", linestyle=":", alpha=0.4)
            for i, v in enumerate(vals):
                if v is not None:
                    ax.text(v, i, f" {v:.4f}", va="center", fontsize=8)
            if metric == "auc":
                ax.axvline(0.5, color="black", lw=0.8, linestyle="--")
            else:
                ax.set_yticklabels([])
        fig.suptitle(f"Model comparison - {title_of(which)} (grey=baseline, green=pipeline model)")
        plt.tight_layout()
        save("model_comparison", which, fig)

    for which, split in sets.items():
        if two_classes(split):
            guard(f"model_comparison:{which}", lambda w=which: chart1(w))

    # 2. train vs validation AUC
    def chart2():
        fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(real)), 5))
        series = [("train", "Train", GREEN)]
        if "valid" in sets:
            series.append((sets["valid"], "Validation", BLUE))
        if "holdout" in sets:
            series.append((sets["holdout"], str(sets["holdout"]).upper(), RED))
        w = 0.8 / len(series)
        for i, (split, lab, col) in enumerate(series):
            vals = [get(m, split, "auc") or 0 for m in real]
            ax.bar(np.arange(len(real)) + i * w, vals, w, label=lab, color=col)
        ax.set_xticks(np.arange(len(real)) + 0.4 - w / 2)
        ax.set_xticklabels(real, rotation=30, ha="right")
        ax.set_ylim(0.4, 1.0)
        ax.axhline(0.5, color="black", lw=0.8, linestyle="--")
        ax.set_ylabel("AUC")
        ax.set_title("Train vs Validation AUC (large gap = overfit / leakage / drift)")
        ax.legend()
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        plt.tight_layout()
        save("train_vs_valid_auc", "valid", fig)

    guard("train_vs_valid_auc", chart2)

    for which, split in sets.items():
        if not two_classes(split) or len(y[split]) < 10:
            log("Chart WARN", f"{split}: <10 rows or one class only - skipping diagnostic charts for {which}")
            continue
        yt = np.asarray(y[split]).astype(int)
        tt = f"{title_of(which)} (n={len(yt)}, bad={int(yt.sum())})"

        def chart3(yt=yt, split=split, tt=tt, which=which):
            fig, ax = plt.subplots(figsize=(7, 6))
            for m in real:
                fpr, tpr, _ = roc_curve(yt, preds[m][split])
                ax.plot(fpr, tpr, lw=2.2 if m == cm else 1.2,
                        label=f"{m} (AUC={get(m, split, 'auc'):.3f})")
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (0.5)")
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
            ax.set_title(f"ROC curves - {tt}")
            ax.legend(fontsize=8, loc="lower right"); ax.grid(linestyle=":", alpha=0.4)
            plt.tight_layout(); save("roc_curves", which, fig)

        def chart4(yt=yt, split=split, tt=tt, which=which):
            fig, ax = plt.subplots(figsize=(7, 6))
            for m in real:
                pr, rc, _ = precision_recall_curve(yt, preds[m][split])
                ax.plot(rc, pr, lw=2.2 if m == cm else 1.2,
                        label=f"{m} (PR-AUC={get(m, split, 'pr_auc'):.3f})")
            ax.axhline(yt.mean(), color="black", linestyle="--", lw=1, label=f"Bad rate={yt.mean():.3f}")
            ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
            ax.set_title(f"Precision-Recall curves - {tt}")
            ax.legend(fontsize=8); ax.grid(linestyle=":", alpha=0.4)
            plt.tight_layout(); save("pr_curves", which, fig)

        def chart5(yt=yt, split=split, tt=tt, which=which):
            p = preds[cm][split]
            q = int(min(10, max(2, len(yt) // 20)))
            grp = pd.qcut(pd.Series(p).rank(method="first"), q=q, labels=False, duplicates="drop")
            t = pd.DataFrame({"g": grp, "y": yt, "p": p}).groupby("g").mean()
            fig, ax = plt.subplots(figsize=(7, 6))
            hi = float(max(t["p"].max(), t["y"].max())) * 1.05
            ax.plot([0, hi], [0, hi], "k--", lw=1, label="Perfect calibration")
            ax.plot(t["p"], t["y"], "o-", color=BLUE, label=cm)
            ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed bad rate")
            ax.set_title(f"Calibration - {cm} - {tt}")
            ax.legend(); ax.grid(linestyle=":", alpha=0.4)
            plt.tight_layout(); save("calibration", which, fig)

        def chart6(yt=yt, split=split, tt=tt, which=which):
            t = _decile_table(yt, preds[cm][split])
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.bar(t["decile"], t["bad_rate"], color=BLUE, label="Observed bad rate")
            ax.plot(t["decile"], t["avg_score"], "o-", color=RED, label="Avg predicted")
            ax.set_xlabel("Score decile (D1 = lowest risk)"); ax.set_ylabel("Rate")
            ax.set_title(f"Bad rate by score decile - {cm} - {tt}")
            ax.legend(); ax.grid(axis="y", linestyle=":", alpha=0.4)
            plt.tight_layout(); save("bad_rate_decile", which, fig)

        for nm, fn in (("roc_curves", chart3), ("pr_curves", chart4),
                       ("calibration", chart5), ("bad_rate_decile", chart6)):
            guard(f"{nm}:{which}", fn)

        p = preds[cm][split]
        dec = _decile_table(yt, p)
        br = dec["bad_rate"].to_numpy()
        diagnostics[which] = {
            "split": split, "n": int(len(yt)), "n_pos": int(yt.sum()), "bad_rate": float(yt.mean()),
            "mean_pred": float(p.mean()), "auc": get(cm, split, "auc"), "ks": get(cm, split, "ks"),
            "brier": get(cm, split, "brier"), "brier_prior": get("Prior baseline", split, "brier"),
            "n_deciles": int(len(br)), "decile_reversals": int((np.diff(br) < 0).sum()),
            "deciles": dec[["decile", "n", "bad_rate", "avg_score"]].to_dict(orient="records"),
        }

    # 7. permutation importance (AUC drop) on validation
    importance: List[Dict[str, Any]] = []
    vs = sets.get("valid") or sets.get("holdout")
    if vs and cm in fitted and len(y[vs]) >= 10 and two_classes(vs):
        def chart7():
            Xv, yv = X[vs], np.asarray(y[vs]).astype(int)
            rs = np.random.RandomState(seed)
            if len(Xv) > 3000:
                idx = rs.choice(len(Xv), 3000, replace=False)
                Xv, yv = Xv.iloc[idx], yv[idx]
            if len(np.unique(yv)) < 2:
                return
            model = fitted[cm]
            base = roc_auc_score(yv, _proba(model, Xv))
            n_rep = 5 if Xv.shape[1] <= 80 else 3
            recs = []
            for col in Xv.columns:
                drops = []
                for _ in range(n_rep):
                    Xp = Xv.copy()
                    Xp[col] = rs.permutation(Xp[col].to_numpy())
                    drops.append(base - roc_auc_score(yv, _proba(model, Xp)))
                recs.append({"feature": col, "mean": float(np.mean(drops)), "std": float(np.std(drops))})
            df = pd.DataFrame(recs).sort_values("mean", ascending=False).reset_index(drop=True)
            importance.extend(df.head(30).to_dict(orient="records"))
            top = df.head(20)[::-1]
            fig, ax = plt.subplots(figsize=(8, max(3.5, 0.32 * len(top) + 1.5)))
            ax.barh(top["feature"], top["mean"], xerr=top["std"],
                    color=[RED if v < 0 else BLUE for v in top["mean"]])
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("AUC decrease when feature is shuffled")
            ax.set_title(f"Permutation importance - {cm} on {vs} (n={len(Xv)})")
            ax.grid(axis="x", linestyle=":", alpha=0.4)
            plt.tight_layout(); save("permutation_importance", "valid", fig)
            diagnostics["importance_total_pos"] = float(df.loc[df["mean"] > 0, "mean"].sum())
            diagnostics["importance_n_negative"] = int((df.head(20)["mean"] < 0).sum())
        guard("permutation_importance", chart7)

    log("Charts Complete", f"{len(saved)} classification benchmark charts saved")
    return {"charts": saved, "diagnostics": diagnostics, "importance": importance}


# ── interpretation ────────────────────────────────────────────────────────

def build_findings(result: Dict[str, Any]) -> List[Dict[str, str]]:
    """Rule-based decision checklist. Thresholds are heuristics, not standards."""
    f: List[Dict[str, str]] = []
    rows, cm = result["rows"], result["chart_model"]
    vs, hs = result["eval_sets"].get("valid"), result["eval_sets"].get("holdout")
    ref = vs or hs
    diag = result.get("diagnostics", {})
    H = str(hs).upper() if hs else ""

    def add(sev, title, finding, action):
        f.append({"severity": sev, "title": title, "finding": finding, "action": action})

    def name_of(which):
        return "Validation" if which == "valid" else H

    # 0. sample size - AUC on few positives is unstable
    for which in ("valid", "holdout"):
        d = diag.get(which)
        if d and d["n_pos"] < 30:
            add("warn", f"Ít quan sát Bad trên {name_of(which)}",
                f"Chỉ có {d['n_pos']} Bad / {d['n']} dòng - AUC/KS dao động mạnh giữa các lần chia.",
                "Dùng thêm dữ liệu, gộp nhiều kỳ hoặc xem khoảng tin cậy (bootstrap) trước khi kết luận.")

    # 1. discrimination vs prior baseline
    auc = _row_of(rows, cm, ref, "auc")
    if auc is not None:
        gini = 2 * auc - 1
        if auc <= 0.55:
            add("bad", "Gần như không phân biệt được Good/Bad",
                f"{cm}: AUC={auc:.4f} (Gini={gini:.3f}) trên {ref}, sát mức ngẫu nhiên 0.5.",
                "Xem lại feature, định nghĩa target (cửa sổ quan sát/performance) và cách chia dữ liệu.")
        elif auc < 0.65:
            add("warn", "Khả năng phân biệt yếu",
                f"{cm}: AUC={auc:.4f} (Gini={gini:.3f}) trên {ref}.",
                "Bổ sung feature hành vi/giao dịch, kiểm tra chất lượng dữ liệu; cân nhắc model hiện chỉ dùng tham khảo.")
        else:
            add("ok", "Tốt hơn rõ rệt so với baseline",
                f"{cm}: AUC={auc:.4f} (Gini={gini:.3f}) trên {ref} so với 0.5 của Prior baseline.",
                "Xác nhận thêm trên OOT/Test và qua nhiều kỳ.")

    # 2. train vs validation
    tr = _row_of(rows, cm, "train", "auc")
    va = _row_of(rows, cm, vs, "auc") if vs else None
    if tr is not None and va is not None:
        gap = tr - va
        if gap > 0.10:
            add("warn", "Train tốt, Validation kém (nghi overfit / leakage / lệch phân phối)",
                f"AUC Train={tr:.4f}, Validation={va:.4f} (chênh {gap:.3f}).",
                "Tăng regularization / giảm độ phức tạp, kiểm tra leakage; so phân phối train/valid.")
        elif gap > 0.05:
            add("info", "Chênh lệch Train/Validation đáng chú ý",
                f"AUC Train={tr:.4f}, Validation={va:.4f} (chênh {gap:.3f}). Thường gặp với model cây.",
                "Theo dõi qua các kỳ; nếu chênh lệch tăng dần, tăng regularization.")
        else:
            add("ok", "Train và Validation gần nhau",
                f"AUC Train={tr:.4f}, Validation={va:.4f}.", "Khả năng tổng quát hóa ổn.")

    # 3. validation vs hold-out (drift)
    if vs and hs:
        av, ah = _row_of(rows, cm, vs, "auc"), _row_of(rows, cm, hs, "auc")
        if av is not None and ah is not None:
            if av - ah > 0.03:
                add("warn", f"Validation tốt nhưng {H} giảm",
                    f"AUC Validation={av:.4f}, {H}={ah:.4f} (giảm {av - ah:.3f}).",
                    "Kiểm tra drift (PSI), thay đổi chính sách/hành vi và tính ổn định theo thời gian; "
                    "cân nhắc train lại theo kỳ gần nhất." if hs == "oot" else
                    "Kiểm tra biến động do mẫu nhỏ và khác biệt phân phối giữa valid/test.")
            else:
                add("ok", f"Validation và {H} nhất quán",
                    f"AUC Validation={av:.4f}, {H}={ah:.4f}.", "Kết quả ổn định giữa hai tập.")

    # 4. calibration level
    for which in ("holdout", "valid"):
        d = diag.get(which)
        if d and d["bad_rate"] > 0:
            ratio = d["mean_pred"] / d["bad_rate"]
            if abs(ratio - 1) > 0.2:
                add("warn", f"Xác suất dự báo lệch mức Bad thực tế ({name_of(which)})",
                    f"Xác suất trung bình {d['mean_pred']:.3f} vs tỷ lệ Bad {d['bad_rate']:.3f} "
                    f"(x{ratio:.2f}) - model {'dự báo cao hơn' if ratio > 1 else 'dự báo thấp hơn'} thực tế.",
                    "Dùng calibrator (model triển khai đã có) và kiểm tra lệch tỷ lệ Bad giữa các kỳ "
                    "trước khi dùng xác suất cho PD/định giá.")
            if d["brier_prior"] and d["brier"] is not None and d["brier"] >= d["brier_prior"]:
                add("bad", f"Brier không tốt hơn baseline ({name_of(which)})",
                    f"Brier={d['brier']:.4f} vs Prior baseline {d['brier_prior']:.4f}.",
                    "Xác suất chưa đáng tin: kiểm tra calibration và overfitting.")
            break

    # 5. rank ordering
    for which in ("holdout", "valid"):
        d = diag.get(which)
        if d and d["n_deciles"] >= 4:
            if d["decile_reversals"] > max(1, d["n_deciles"] // 4):
                add("warn", f"Rank-ordering không đơn điệu ({name_of(which)})",
                    f"Tỷ lệ Bad đảo chiều {d['decile_reversals']}/{d['n_deciles'] - 1} lần giữa các decile.",
                    "Cẩn trọng khi đặt ngưỡng/phân hạng; xem lại các feature gây nhiễu hoặc gộp hạng.")
            else:
                add("ok", f"Rank-ordering ổn ({name_of(which)})",
                    f"Tỷ lệ Bad tăng gần đơn điệu theo decile ({d['decile_reversals']} lần đảo chiều).",
                    "Có thể dùng điểm số để phân hạng/đặt ngưỡng.")
            break

    # 6. imbalance
    d = diag.get("valid") or diag.get("holdout")
    if d and d["bad_rate"] < 0.05:
        add("info", "Dữ liệu mất cân bằng mạnh",
            f"Tỷ lệ Bad chỉ {d['bad_rate']:.2%}.",
            "Ưu tiên AUC, PR-AUC, KS thay vì accuracy; chọn ngưỡng theo chi phí nghiệp vụ.")

    # 7. feature importance sanity
    imp = result.get("importance") or []
    total = diag.get("importance_total_pos") or 0
    if imp and total > 0 and imp[0]["mean"] / total > 0.5:
        add("warn", "Một feature chiếm phần lớn importance",
            f"'{imp[0]['feature']}' chiếm {imp[0]['mean'] / total:.0%} tổng permutation importance dương.",
            "Kiểm tra leakage (feature suy ra từ target hoặc chỉ biết sau thời điểm quyết định).")
    elif imp and diag.get("importance_n_negative", 0) >= 5:
        add("info", "Nhiều feature có importance âm",
            f"{diag['importance_n_negative']}/20 feature hàng đầu có importance âm.",
            "Các biến này có thể là nhiễu - cân nhắc loại bỏ.")
    elif imp:
        add("ok", "Importance tập trung hợp lý",
            f"Top feature: {', '.join(i['feature'] for i in imp[:3])}.",
            "Đối chiếu với kiến thức nghiệp vụ; importance không phải quan hệ nhân quả.")
    return f


def _recommend(result: Dict[str, Any]) -> Dict[str, Any]:
    """Best AUC, but suggest Logistic Regression when it is within 0.01 AUC."""
    ref = result["eval_sets"].get("valid") or result["eval_sets"].get("holdout")
    aucs = {m: _row_of(result["rows"], m, ref, "auc") for m in result["models"] if m not in BASELINES}
    aucs = {m: v for m, v in aucs.items() if v is not None}
    if not aucs:
        return {}
    best = max(aucs, key=aucs.get)
    note = f"{best} có AUC cao nhất trên {ref} ({aucs[best]:.4f})."
    pick = best
    lr = aucs.get("Logistic Regression")
    if best != "Logistic Regression" and lr is not None and aucs[best] - lr <= 0.01:
        pick = "Logistic Regression"
        note += (f" Logistic Regression chỉ kém {aucs[best] - lr:.3f} AUC ({lr:.4f}) nhưng dễ giải thích "
                 "(scorecard) - cân nhắc ưu tiên.")
    return {"best_by_auc": best, "suggested": pick, "reference_split": ref, "note": note}


# ── tables / markdown ─────────────────────────────────────────────────────

def results_table(result: Dict[str, Any], split: str) -> pd.DataFrame:
    recs = [{"Model": r["model"], "AUC": r["auc"], "Gini": r["gini"], "KS": r["ks"], "PR-AUC": r["pr_auc"],
             "LogLoss": r["logloss"], "Brier": r["brier"], "n": r["n"], "Bad": r["n_pos"]}
            for r in result["rows"] if r["split"] == split]
    df = pd.DataFrame(recs)
    return df.sort_values("AUC", ascending=False, na_position="last").reset_index(drop=True) if len(df) else df


TABLE_FORMAT = {"AUC": "{:.4f}", "Gini": "{:.4f}", "KS": "{:.4f}", "PR-AUC": "{:.4f}",
                "LogLoss": "{:.4f}", "Brier": "{:.4f}"}
HIGHER_BETTER = ["AUC", "Gini", "KS", "PR-AUC"]
LOWER_BETTER = ["LogLoss", "Brier"]

CHECKLIST = [
    ("AUC sát 0.5 / không tốt hơn baseline", "Xem lại feature, định nghĩa target và cách chia dữ liệu"),
    ("Train tốt, Validation kém", "Kiểm tra overfitting và leakage"),
    ("Validation tốt nhưng OOT kém", "Kiểm tra drift, thay đổi hành vi và tính ổn định theo thời gian"),
    ("Xác suất lệch tỷ lệ Bad thực tế", "Hiệu chỉnh (calibration) trước khi dùng cho PD/định giá"),
    ("Rank-ordering không đơn điệu", "Cẩn trọng khi đặt ngưỡng; xem lại feature nhiễu, gộp hạng"),
    ("Ít quan sát Bad", "Thêm dữ liệu / gộp kỳ; xem khoảng tin cậy của AUC"),
    ("Feature importance không hợp lý", "Kiểm tra dữ liệu đầu vào, leakage và cách định nghĩa feature"),
]


def render_markdown(result: Dict[str, Any]) -> str:
    sets = result["eval_sets"]
    L: List[str] = ["# Classification baseline & benchmark\n"]
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
        L.append("| Model | AUC | Gini | KS | PR-AUC | LogLoss | Brier |\n|---|---|---|---|---|---|---|")
        for _, r in t.iterrows():
            L.append(f"| {r['Model']} | {_fmt(r['AUC'])} | {_fmt(r['Gini'])} | {_fmt(r['KS'])} | "
                     f"{_fmt(r['PR-AUC'])} | {_fmt(r['LogLoss'])} | {_fmt(r['Brier'])} |")
        L.append("")
    if result.get("failed"):
        L.append("Model không chạy được: " + "; ".join(f"{k} ({v})" for k, v in result["failed"].items()) + "\n")
    L.append("## Chart và cách đọc\n")
    for spec in CHART_SPECS:
        shown = [p for p in (result["charts"].get(f"{spec['key']}:{w}") for w in spec["files"]) if p]
        if not shown:
            continue
        L.append(f"### {spec['label']}\n\n{spec['guide']}\n")
        for p in shown:
            L.append(f"![{spec['label']}](charts/{Path(p).name})\n")
    L.append("## Đánh giá sau huấn luyện và khuyến nghị\n")
    icon = {"ok": "OK", "warn": "CẢNH BÁO", "bad": "NGHIÊM TRỌNG", "info": "LƯU Ý"}
    L.append("| Mức | Phát hiện | Chi tiết | Hành động đề xuất |\n|---|---|---|---|")
    for x in result["findings"]:
        L.append(f"| {icon.get(x['severity'], x['severity'])} | {x['title']} | {x['finding']} | {x['action']} |")
    L.append("\n## Lưu ý phương pháp\n")
    L.append("- Các model benchmark dùng tham số mặc định hợp lý (không tune) và chỉ fit trên train; "
             "`Pipeline tuned` dùng hyperparameter do pipeline chọn, cũng fit trên train để so sánh công bằng.\n"
             "- Model triển khai thật được refit trên train+valid và hiệu chỉnh xác suất; các chart riêng của nó "
             "(ROC, KS, Lift, Gain, Calibration...) vẫn được giữ.\n"
             "- Logistic Regression: biến category được one-hot (gom nhóm hiếm), biến số median-impute + "
             "chuẩn hóa; scorecard thực tế nên dùng WoE binning.\n"
             "- Ngưỡng trong checklist (ví dụ chênh AUC > 0.10, OOT giảm > 0.03) là heuristic, không phải chuẩn.\n")
    return "\n".join(L)
