"""Multiclass baseline + benchmark for Agent 3.

Same contract as regression_benchmark.py / classification_benchmark.py: a fixed
ladder fitted on the SAME train split, scored with the SAME metrics (macro
one-vs-rest AUC, accuracy, balanced accuracy, macro / weighted F1, log-loss) on
every validation / OOT / test split:

    Prior baseline -> Logistic Regression (multinomial) -> Random Forest
    -> LightGBM -> XGBoost -> Pipeline-tuned model

then 7 diagnostic charts on Validation and OOT/Test plus a decision checklist.
Labels are the integer codes 0..k-1 produced by target_encoding; `class_labels`
carries the original names for display.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score,
                             log_loss, precision_recall_fscore_support, roc_auc_score, roc_curve)

from Agents.TrainModel.regression_benchmark import (
    ALL_SPLITS, HOLDOUT_CANDIDATES, VALID_CANDIDATES, _NanAware, _fmt, _json_default,
    linear_preprocessor)

BASELINES = ("Prior baseline",)

CHART_SPECS: List[Dict[str, Any]] = [
    {"key": "model_comparison", "label": "1. Model comparison (macro AUC / Accuracy / Macro F1)",
     "files": {"valid": "mc_model_comparison_valid.png", "holdout": "mc_model_comparison_holdout.png"},
     "guide": "Macro AUC là trung bình AUC one-vs-rest của từng lớp (0.5 = ngẫu nhiên). Accuracy dễ hiểu nhưng "
              "bị lớp đa số chi phối; Macro F1 coi mọi lớp ngang nhau nên phản ánh lớp hiếm tốt hơn. Prior "
              "baseline luôn đoán lớp phổ biến nhất - accuracy của nó chính là tỷ lệ lớp đa số."},
    {"key": "train_vs_valid_auc", "label": "2. Train vs Validation macro AUC",
     "files": {"valid": "mc_train_vs_valid_auc.png"},
     "guide": "Hai cột gần nhau và cao: tổng quát hóa tốt. Train cao hơn nhiều: overfit/leakage. Validation tốt "
              "nhưng OOT giảm: drift theo thời gian."},
    {"key": "confusion_matrix", "label": "3. Confusion matrix (chuẩn hóa theo lớp thực tế)",
     "files": {"valid": "mc_confusion_matrix_valid.png", "holdout": "mc_confusion_matrix_holdout.png"},
     "guide": "Mỗi hàng là lớp thực tế, mỗi cột là lớp dự đoán; ô đường chéo là recall của lớp đó. Ô ngoài đường "
              "chéo đậm cho biết hai lớp hay bị nhầm với nhau - thường do định nghĩa lớp chồng lấn hoặc thiếu "
              "feature phân biệt chúng."},
    {"key": "per_class", "label": "4. Precision / Recall / F1 theo từng lớp",
     "files": {"valid": "mc_per_class_valid.png", "holdout": "mc_per_class_holdout.png"},
     "guide": "Lớp có F1/recall thấp là lớp model chưa học được - hay gặp ở lớp hiếm (số mẫu ghi trong nhãn). "
              "Cân nhắc gom lớp, thêm dữ liệu, class weight hoặc ngưỡng riêng cho lớp quan trọng."},
    {"key": "roc_ovr", "label": "5. ROC one-vs-rest theo từng lớp",
     "files": {"valid": "mc_roc_ovr_valid.png", "holdout": "mc_roc_ovr_holdout.png"},
     "guide": "Mỗi đường là khả năng tách một lớp khỏi các lớp còn lại. Lớp có đường sát đường chéo là lớp model "
              "gần như không nhận diện được."},
    {"key": "confidence", "label": "6. Độ tin cậy dự đoán (top-label calibration)",
     "files": {"valid": "mc_confidence_valid.png", "holdout": "mc_confidence_holdout.png"},
     "guide": "Nhóm các dự đoán theo xác suất cao nhất; so với tỷ lệ đoán đúng thực tế. Nằm dưới đường chéo: model "
              "quá tự tin (over-confident); nằm trên: thiếu tự tin. Quan trọng khi dùng xác suất để ra quyết định "
              "tự động hoặc đặt ngưỡng từ chối."},
    {"key": "permutation_importance", "label": "7. Permutation feature importance (Validation)",
     "files": {"valid": "mc_permutation_importance.png"},
     "guide": "Đảo ngẫu nhiên một feature trên Validation rồi đo macro AUC giảm bao nhiêu. Dương lớn: model phụ "
              "thuộc nhiều; âm: biến nhiễu. Không phải quan hệ nhân quả."},
]


def chart_file_list() -> List[tuple]:
    out = []
    for spec in CHART_SPECS:
        for which, fname in spec["files"].items():
            suffix = {"valid": " - Validation", "holdout": " - OOT/Test"}[which] if len(spec["files"]) > 1 else ""
            out.append((fname, spec["label"] + suffix))
    return out


# ── metrics ───────────────────────────────────────────────────────────────

def macro_auc(y, P) -> Optional[float]:
    y = np.asarray(y).astype(int)
    aucs = [roc_auc_score(y == c, P[:, c]) for c in range(P.shape[1]) if 0 < int((y == c).sum()) < len(y)]
    return float(np.mean(aucs)) if aucs else None


def mc_metrics(y, P) -> Dict[str, Any]:
    y = np.asarray(y).astype(int)
    k = P.shape[1]
    pred = P.argmax(axis=1)
    Pn = np.clip(P, 1e-15, 1.0)
    Pn = Pn / Pn.sum(axis=1, keepdims=True)
    labels = list(range(k))
    return {
        "auc": macro_auc(y, P),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else None,
        "f1_macro": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "f1_weighted": float(f1_score(y, pred, average="weighted", labels=labels, zero_division=0)),
        "logloss": float(log_loss(y, Pn, labels=labels)),
    }


def _proba(model, X, k: int) -> np.ndarray:
    p = np.asarray(model.predict_proba(X), dtype=float)
    classes = getattr(model, "classes_", None)
    if classes is None and hasattr(model, "est"):
        classes = getattr(model.est, "classes_", None)
    classes = np.arange(p.shape[1]) if classes is None else classes
    out = np.zeros((p.shape[0], k))
    for j, c in enumerate(classes):
        if 0 <= int(c) < k:
            out[:, int(c)] = p[:, j]
    return out


# ── model ladder ──────────────────────────────────────────────────────────

def _build_ladder(seed: int, n_train: int, k: int,
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
            tree_method="hist", eval_metric="mlogloss", random_state=seed, n_jobs=-1, verbosity=0)
    except ImportError:
        pass
    return ladder


def _row_of(rows, model, split, metric):
    for r in rows:
        if r["model"] == model and r["split"] == split:
            return r.get(metric)
    return None


# ── main entry ────────────────────────────────────────────────────────────

def run_multiclass_benchmark(
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
    class_labels: Optional[List[Any]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Run baseline + benchmark for a multiclass target coded 0..k-1."""
    t0 = time.time()
    splits = [s for s in ALL_SPLITS if s in X and len(X[s]) >= 2]
    if "train" not in splits:
        raise ValueError("benchmark needs a non-empty train split")
    k = len(class_labels) if class_labels else int(max(int(np.max(y[s])) for s in splits)) + 1
    labels = [str(c) for c in (class_labels or range(k))]
    valid_set = next((s for s in VALID_CANDIDATES if s in splits), None)
    holdout_set = next((s for s in HOLDOUT_CANDIDATES if s in splits), None)
    if valid_set is None and holdout_set is None:
        raise ValueError("benchmark needs at least one validation / OOT / test split")

    ladder = _build_ladder(seed, len(X["train"]), k, categorical_cols)
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
                P = _proba(model, X[s], k)
                preds[name][s] = P
                rows.append({"model": name, "split": s, "n": int(len(P)), **mc_metrics(y[s], P)})
            log("Benchmark", f"{name}: fit {fit_seconds[name]}s | " + " | ".join(
                f"{r['split']} macroAUC={_fmt(r['auc'])}" for r in rows
                if r["model"] == name and r["split"] in (valid_set, holdout_set)))
        except Exception as e:  # noqa: BLE001
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
        "task": "multiclass", "class_labels": labels, "n_classes": k,
        "eval_sets": {"valid": valid_set, "holdout": holdout_set},
        "n_rows": {s: int(len(X[s])) for s in splits},
        "class_counts": {s: np.bincount(np.asarray(y[s]).astype(int), minlength=k).tolist() for s in splits},
        "models": list(fitted), "failed": failed, "fit_seconds": fit_seconds,
        "rows": rows, "chart_model": chart_model, "best_valid_model": best_valid_model,
        "tuned_model": tuned_name, "deployed_metrics": deployed_metrics or {}, "charts": {},
    }

    diag: Dict[str, Any] = {}
    try:
        diag = _make_charts(result, X, y, preds, fitted, chart_dir, seed, log, k, labels)
    except Exception as e:  # noqa: BLE001
        log("Charts ERROR", f"benchmark charts failed: {e}")
    result["diagnostics"] = diag.get("diagnostics", {})
    result["importance"] = diag.get("importance", [])
    result["charts"] = diag.get("charts", {})
    result["findings"] = build_findings(result)
    result["recommended_model"] = _recommend(result)

    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "multiclass_benchmark_report.md"
    md_path.write_text(render_markdown(result), encoding="utf-8")
    result["report_md"] = str(md_path)
    json_path = out_dir / "multiclass_benchmark.json"
    result["report_json"] = str(json_path)
    result["seconds"] = round(time.time() - t0, 1)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False, default=_json_default)
    return result


# ── charts ────────────────────────────────────────────────────────────────

def _make_charts(result, X, y, preds, fitted, chart_dir, seed, log, k, labels) -> Dict[str, Any]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cdir = Path(chart_dir)
    cdir.mkdir(parents=True, exist_ok=True)
    rows, models, cm = result["rows"], result["models"], result["chart_model"]
    sets = {w: s for w, s in result["eval_sets"].items() if s}
    saved: Dict[str, str] = {}
    diagnostics: Dict[str, Any] = {}
    BLUE, RED, GREY, GREEN = "#2196F3", "#F44336", "#9E9E9E", "#4CAF50"
    STYLE = {"dpi": 120, "bbox_inches": "tight"}
    spec_files = {s["key"]: s["files"] for s in CHART_SPECS}
    real = [m for m in models if m not in BASELINES]

    def title_of(which):
        return "Validation" if which == "valid" else str(sets["holdout"]).upper()

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

    def chart1(which):
        split = sets[which]
        order = sorted(models, key=lambda m: -(_row_of(rows, m, split, "auc") or 0))
        fig, axes = plt.subplots(1, 3, figsize=(15, max(3.5, 0.5 * len(order) + 1.5)))
        for ax, metric, title in zip(axes, ("auc", "accuracy", "f1_macro"),
                                     ("Macro AUC (higher)", "Accuracy (higher)", "Macro F1 (higher)")):
            vals = [_row_of(rows, m, split, metric) for m in order]
            ax.barh(order, [v or 0 for v in vals],
                    color=[GREY if m in BASELINES else (GREEN if m == cm else BLUE) for m in order])
            ax.invert_yaxis()
            ax.set_title(title)
            ax.grid(axis="x", linestyle=":", alpha=0.4)
            for i, v in enumerate(vals):
                if v is not None:
                    ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
            if metric != "auc":
                ax.set_yticklabels([])
        fig.suptitle(f"Model comparison - {title_of(which)} (grey=baseline, green=pipeline model)")
        plt.tight_layout()
        save("model_comparison", which, fig)

    for which in sets:
        guard(f"model_comparison:{which}", lambda w=which: chart1(w))

    def chart2():
        fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(real)), 5))
        series = [("train", "Train", GREEN)]
        if "valid" in sets:
            series.append((sets["valid"], "Validation", BLUE))
        if "holdout" in sets:
            series.append((sets["holdout"], str(sets["holdout"]).upper(), RED))
        w = 0.8 / len(series)
        for i, (split, lab, col) in enumerate(series):
            ax.bar(np.arange(len(real)) + i * w, [_row_of(rows, m, split, "auc") or 0 for m in real],
                   w, label=lab, color=col)
        ax.set_xticks(np.arange(len(real)) + 0.4 - w / 2)
        ax.set_xticklabels(real, rotation=30, ha="right")
        ax.set_ylim(0.4, 1.0)
        ax.axhline(0.5, color="black", lw=0.8, linestyle="--")
        ax.set_ylabel("Macro AUC")
        ax.set_title("Train vs Validation macro AUC (large gap = overfit / leakage / drift)")
        ax.legend()
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        plt.tight_layout()
        save("train_vs_valid_auc", "valid", fig)

    guard("train_vs_valid_auc", chart2)

    for which, split in sets.items():
        yt = np.asarray(y[split]).astype(int)
        if len(yt) < 10 or len(np.unique(yt)) < 2:
            log("Chart WARN", f"{split}: <10 rows or one class - skipping diagnostic charts for {which}")
            continue
        P = preds[cm][split]
        pred = P.argmax(axis=1)
        tt = f"{title_of(which)} (n={len(yt)})"
        support = np.bincount(yt, minlength=k)

        def chart3(yt=yt, pred=pred, tt=tt, which=which):
            cmx = confusion_matrix(yt, pred, labels=list(range(k)))
            norm = cmx / np.maximum(cmx.sum(axis=1, keepdims=True), 1)
            fig, ax = plt.subplots(figsize=(max(5, 0.7 * k + 3), max(4.5, 0.6 * k + 2.5)))
            im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(k)); ax.set_yticks(range(k))
            ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            if k <= 15:
                for i in range(k):
                    for j in range(k):
                        ax.text(j, i, f"{norm[i, j]:.2f}\n({cmx[i, j]})", ha="center", va="center", fontsize=7,
                                color="white" if norm[i, j] > 0.5 else "black")
            fig.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(f"Confusion matrix (row-normalised) - {cm} - {tt}")
            plt.tight_layout(); save("confusion_matrix", which, fig)

        def chart4(yt=yt, pred=pred, tt=tt, which=which, support=support):
            pr, rc, f1, _ = precision_recall_fscore_support(yt, pred, labels=list(range(k)), zero_division=0)
            fig, ax = plt.subplots(figsize=(max(8, 1.0 * k + 3), 5))
            x = np.arange(k)
            ax.bar(x - 0.27, pr, 0.27, label="Precision", color=BLUE)
            ax.bar(x, rc, 0.27, label="Recall", color=GREEN)
            ax.bar(x + 0.27, f1, 0.27, label="F1", color=RED)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{lab}\n(n={n})" for lab, n in zip(labels, support)], rotation=0 if k <= 8 else 45)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"Per-class precision / recall / F1 - {cm} - {tt}")
            ax.legend(); ax.grid(axis="y", linestyle=":", alpha=0.4)
            plt.tight_layout(); save("per_class", which, fig)

        def chart5(yt=yt, P=P, tt=tt, which=which):
            fig, ax = plt.subplots(figsize=(7, 6))
            for c in range(k):
                if 0 < (yt == c).sum() < len(yt):
                    fpr, tpr, _ = roc_curve(yt == c, P[:, c])
                    ax.plot(fpr, tpr, lw=1.4, label=f"{labels[c]} (AUC={roc_auc_score(yt == c, P[:, c]):.3f})")
            ax.plot([0, 1], [0, 1], "k--", lw=1)
            ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
            ax.set_title(f"One-vs-rest ROC - {cm} - {tt}")
            ax.legend(fontsize=7, loc="lower right"); ax.grid(linestyle=":", alpha=0.4)
            plt.tight_layout(); save("roc_ovr", which, fig)

        def chart6(yt=yt, P=P, pred=pred, tt=tt, which=which):
            conf = P.max(axis=1)
            q = int(min(10, max(2, len(yt) // 20)))
            grp = pd.qcut(pd.Series(conf).rank(method="first"), q=q, labels=False, duplicates="drop")
            t = pd.DataFrame({"g": grp, "conf": conf, "ok": (pred == yt).astype(float)}).groupby("g").mean()
            fig, ax = plt.subplots(figsize=(7, 6))
            lo = float(min(t["conf"].min(), t["ok"].min())) * 0.95
            ax.plot([lo, 1], [lo, 1], "k--", lw=1, label="Perfect")
            ax.plot(t["conf"], t["ok"], "o-", color=BLUE, label=cm)
            ax.set_xlabel("Mean top-class probability"); ax.set_ylabel("Observed accuracy")
            ax.set_title(f"Confidence vs accuracy - {tt}")
            ax.legend(); ax.grid(linestyle=":", alpha=0.4)
            plt.tight_layout(); save("confidence", which, fig)

        for nm, fn in (("confusion_matrix", chart3), ("per_class", chart4),
                       ("roc_ovr", chart5), ("confidence", chart6)):
            guard(f"{nm}:{which}", fn)

        pr, rc, f1, _ = precision_recall_fscore_support(yt, pred, labels=list(range(k)), zero_division=0)
        cmx = confusion_matrix(yt, pred, labels=list(range(k)))
        off = cmx.astype(float) / np.maximum(cmx.sum(axis=1, keepdims=True), 1)
        np.fill_diagonal(off, 0)
        i, j = np.unravel_index(off.argmax(), off.shape)
        diagnostics[which] = {
            "split": split, "n": int(len(yt)), "support": support.tolist(),
            "precision": pr.tolist(), "recall": rc.tolist(), "f1": f1.tolist(),
            "top_confusion": {"actual": labels[i], "predicted": labels[j], "share": float(off[i, j])},
            "mean_confidence": float(P.max(axis=1).mean()), "accuracy": float((pred == yt).mean()),
            "auc": _row_of(rows, cm, split, "auc"),
        }

    importance: List[Dict[str, Any]] = []
    vs = sets.get("valid") or sets.get("holdout")
    if vs and cm in fitted and len(y[vs]) >= 10:
        def chart7():
            Xv, yv = X[vs], np.asarray(y[vs]).astype(int)
            rs = np.random.RandomState(seed)
            if len(Xv) > 3000:
                idx = rs.choice(len(Xv), 3000, replace=False)
                Xv, yv = Xv.iloc[idx], yv[idx]
            model = fitted[cm]
            base = macro_auc(yv, _proba(model, Xv, k))
            if base is None:
                return
            n_rep = 5 if Xv.shape[1] <= 80 else 3
            recs = []
            for col in Xv.columns:
                drops = []
                for _ in range(n_rep):
                    Xp = Xv.copy()
                    Xp[col] = rs.permutation(Xp[col].to_numpy())
                    drops.append(base - (macro_auc(yv, _proba(model, Xp, k)) or base))
                recs.append({"feature": col, "mean": float(np.mean(drops)), "std": float(np.std(drops))})
            df = pd.DataFrame(recs).sort_values("mean", ascending=False).reset_index(drop=True)
            importance.extend(df.head(30).to_dict(orient="records"))
            top = df.head(20)[::-1]
            fig, ax = plt.subplots(figsize=(8, max(3.5, 0.32 * len(top) + 1.5)))
            ax.barh(top["feature"], top["mean"], xerr=top["std"],
                    color=[RED if v < 0 else BLUE for v in top["mean"]])
            ax.axvline(0, color="black", lw=0.8)
            ax.set_xlabel("Macro AUC decrease when feature is shuffled")
            ax.set_title(f"Permutation importance - {cm} on {vs} (n={len(Xv)})")
            ax.grid(axis="x", linestyle=":", alpha=0.4)
            plt.tight_layout(); save("permutation_importance", "valid", fig)
            diagnostics["importance_total_pos"] = float(df.loc[df["mean"] > 0, "mean"].sum())
            diagnostics["importance_n_negative"] = int((df.head(20)["mean"] < 0).sum())
        guard("permutation_importance", chart7)

    log("Charts Complete", f"{len(saved)} multiclass benchmark charts saved")
    return {"charts": saved, "diagnostics": diagnostics, "importance": importance}


# ── interpretation ────────────────────────────────────────────────────────

def build_findings(result: Dict[str, Any]) -> List[Dict[str, str]]:
    """Rule-based decision checklist. Thresholds are heuristics, not standards."""
    f: List[Dict[str, str]] = []
    rows, cm = result["rows"], result["chart_model"]
    vs, hs = result["eval_sets"].get("valid"), result["eval_sets"].get("holdout")
    ref = vs or hs
    diag = result.get("diagnostics", {})
    labels = result.get("class_labels") or []
    H = str(hs).upper() if hs else ""

    def add(sev, title, finding, action):
        f.append({"severity": sev, "title": title, "finding": finding, "action": action})

    def name_of(which):
        return "Validation" if which == "valid" else H

    for which in ("valid", "holdout"):
        d = diag.get(which)
        if d:
            small = [f"{labels[c]} ({n})" for c, n in enumerate(d["support"]) if 0 < n < 30]
            missing = [labels[c] for c, n in enumerate(d["support"]) if n == 0]
            if small or missing:
                add("warn", f"Lớp quá ít mẫu trên {name_of(which)}",
                    (f"Ít hơn 30 mẫu: {', '.join(small)}. " if small else "")
                    + (f"Không có mẫu: {', '.join(missing)}." if missing else ""),
                    "Metric theo lớp của các lớp này không ổn định; cân nhắc gom lớp hoặc thêm dữ liệu.")

    auc = _row_of(rows, cm, ref, "auc")
    if auc is not None:
        if auc <= 0.55:
            add("bad", "Gần như không phân biệt được các lớp",
                f"{cm}: macro AUC={auc:.4f} trên {ref}.",
                "Xem lại feature, định nghĩa các lớp và cách chia dữ liệu.")
        elif auc < 0.65:
            add("warn", "Khả năng phân biệt yếu", f"{cm}: macro AUC={auc:.4f} trên {ref}.",
                "Bổ sung feature phân biệt các lớp; kiểm tra các lớp có bị chồng lấn định nghĩa.")
        else:
            add("ok", "Tốt hơn rõ rệt so với baseline", f"{cm}: macro AUC={auc:.4f} trên {ref} (baseline 0.5).",
                "Xác nhận thêm trên OOT/Test và theo từng lớp.")
    acc, acc_b = _row_of(rows, cm, ref, "accuracy"), _row_of(rows, "Prior baseline", ref, "accuracy")
    if acc is not None and acc_b is not None and acc <= acc_b + 0.01:
        add("warn", "Accuracy không hơn việc luôn đoán lớp đa số",
            f"Accuracy={acc:.3f} vs Prior baseline {acc_b:.3f} (tỷ lệ lớp đa số).",
            "Model có thể đang thiên về lớp đa số - xem Macro F1, recall theo lớp; cân nhắc class weight.")

    tr = _row_of(rows, cm, "train", "auc")
    va = _row_of(rows, cm, vs, "auc") if vs else None
    if tr is not None and va is not None:
        gap = tr - va
        if gap > 0.10:
            add("warn", "Train tốt, Validation kém (nghi overfit / leakage)",
                f"Macro AUC Train={tr:.4f}, Validation={va:.4f} (chênh {gap:.3f}).",
                "Tăng regularization / giảm độ phức tạp; kiểm tra leakage.")
        elif gap > 0.05:
            add("info", "Chênh lệch Train/Validation đáng chú ý",
                f"Macro AUC Train={tr:.4f}, Validation={va:.4f}.", "Theo dõi qua các kỳ.")
        else:
            add("ok", "Train và Validation gần nhau", f"Macro AUC Train={tr:.4f}, Validation={va:.4f}.",
                "Khả năng tổng quát hóa ổn.")

    if vs and hs:
        av, ah = _row_of(rows, cm, vs, "auc"), _row_of(rows, cm, hs, "auc")
        if av is not None and ah is not None:
            if av - ah > 0.03:
                add("warn", f"Validation tốt nhưng {H} giảm", f"Macro AUC Validation={av:.4f}, {H}={ah:.4f}.",
                    "Kiểm tra drift và thay đổi tỷ lệ các lớp theo thời gian." if hs == "oot" else
                    "Kiểm tra biến động do mẫu nhỏ.")
            else:
                add("ok", f"Validation và {H} nhất quán", f"Macro AUC Validation={av:.4f}, {H}={ah:.4f}.",
                    "Kết quả ổn định giữa hai tập.")

    for which in ("holdout", "valid"):
        d = diag.get(which)
        if not d:
            continue
        weak = [f"{labels[c]} (recall {r:.2f}, F1 {f1:.2f})" for c, (r, f1, n) in
                enumerate(zip(d["recall"], d["f1"], d["support"])) if n > 0 and (r < 0.2 or f1 < 0.3)]
        if weak:
            add("warn", f"Lớp model gần như không nhận diện được ({name_of(which)})", "; ".join(weak) + ".",
                "Thêm dữ liệu / feature cho các lớp này, class weight, hoặc gom với lớp gần nghĩa.")
        tc = d["top_confusion"]
        if tc["share"] > 0.3:
            add("info", "Cặp lớp hay bị nhầm",
                f"{tc['share']:.0%} mẫu lớp '{tc['actual']}' bị dự đoán thành '{tc['predicted']}' ({name_of(which)}).",
                "Xem lại định nghĩa hai lớp và feature giúp tách chúng.")
        gap = d["mean_confidence"] - d["accuracy"]
        if abs(gap) > 0.1:
            add("warn", f"Xác suất {'quá tự tin' if gap > 0 else 'thiếu tự tin'} ({name_of(which)})",
                f"Xác suất top-class trung bình {d['mean_confidence']:.3f} vs accuracy {d['accuracy']:.3f}.",
                "Hiệu chỉnh xác suất (temperature/isotonic theo lớp) trước khi dùng xác suất để ra quyết định.")
        break

    counts = (result.get("class_counts") or {}).get("train")
    if counts and sum(counts):
        share = min(c for c in counts if c > 0) / sum(counts)
        if share < 0.05:
            add("info", "Các lớp mất cân bằng mạnh",
                f"Lớp nhỏ nhất chỉ chiếm {share:.1%} tập train.",
                "Ưu tiên Macro F1 / balanced accuracy / macro AUC thay vì accuracy.")

    imp = result.get("importance") or []
    total = diag.get("importance_total_pos") or 0
    if imp and total > 0 and imp[0]["mean"] / total > 0.5:
        add("warn", "Một feature chiếm phần lớn importance",
            f"'{imp[0]['feature']}' chiếm {imp[0]['mean'] / total:.0%} tổng permutation importance dương.",
            "Kiểm tra leakage (feature suy ra từ nhãn hoặc chỉ biết sau thời điểm dự đoán).")
    elif imp:
        add("ok", "Importance tập trung hợp lý", f"Top feature: {', '.join(i['feature'] for i in imp[:3])}.",
            "Đối chiếu với kiến thức nghiệp vụ.")
    return f


def _recommend(result: Dict[str, Any]) -> Dict[str, Any]:
    ref = result["eval_sets"].get("valid") or result["eval_sets"].get("holdout")
    aucs = {m: _row_of(result["rows"], m, ref, "auc") for m in result["models"] if m not in BASELINES}
    aucs = {m: v for m, v in aucs.items() if v is not None}
    if not aucs:
        return {}
    best = max(aucs, key=aucs.get)
    note, pick = f"{best} có macro AUC cao nhất trên {ref} ({aucs[best]:.4f}).", best
    lr = aucs.get("Logistic Regression")
    if best != "Logistic Regression" and lr is not None and aucs[best] - lr <= 0.01:
        pick = "Logistic Regression"
        note += f" Logistic Regression chỉ kém {aucs[best] - lr:.3f} nhưng dễ giải thích hơn - cân nhắc ưu tiên."
    return {"best_by_auc": best, "suggested": pick, "reference_split": ref, "note": note}


# ── tables / markdown ─────────────────────────────────────────────────────

def results_table(result: Dict[str, Any], split: str) -> pd.DataFrame:
    recs = [{"Model": r["model"], "Macro AUC": r["auc"], "Accuracy": r["accuracy"],
             "Balanced acc": r["balanced_accuracy"], "Macro F1": r["f1_macro"],
             "Weighted F1": r["f1_weighted"], "LogLoss": r["logloss"], "n": r["n"]}
            for r in result["rows"] if r["split"] == split]
    df = pd.DataFrame(recs)
    return df.sort_values("Macro AUC", ascending=False, na_position="last").reset_index(drop=True) if len(df) else df


TABLE_FORMAT = {"Macro AUC": "{:.4f}", "Accuracy": "{:.4f}", "Balanced acc": "{:.4f}",
                "Macro F1": "{:.4f}", "Weighted F1": "{:.4f}", "LogLoss": "{:.4f}"}
HIGHER_BETTER = ["Macro AUC", "Accuracy", "Balanced acc", "Macro F1", "Weighted F1"]
LOWER_BETTER = ["LogLoss"]

CHECKLIST = [
    ("Macro AUC sát 0.5 / accuracy ≈ tỷ lệ lớp đa số", "Xem lại feature, định nghĩa lớp; class weight"),
    ("Train tốt, Validation kém", "Kiểm tra overfitting và leakage"),
    ("Validation tốt nhưng OOT kém", "Kiểm tra drift và thay đổi tỷ lệ lớp theo thời gian"),
    ("Có lớp recall/F1 rất thấp", "Thêm dữ liệu/feature cho lớp đó, gom lớp hoặc class weight"),
    ("Cặp lớp hay bị nhầm", "Xem lại định nghĩa lớp và feature phân biệt"),
    ("Xác suất quá/thiếu tự tin", "Hiệu chỉnh xác suất trước khi dùng cho quyết định tự động"),
    ("Feature importance không hợp lý", "Kiểm tra dữ liệu đầu vào, leakage và cách định nghĩa feature"),
]


def render_markdown(result: Dict[str, Any]) -> str:
    sets = result["eval_sets"]
    L: List[str] = ["# Multiclass baseline & benchmark\n"]
    L.append(f"{result['n_classes']} lớp: {', '.join(map(str, result['class_labels']))}. Tất cả model fit trên "
             f"**train** ({result['n_rows'].get('train')} dòng), chấm cùng metric trên Validation "
             f"(`{sets['valid']}`) và OOT/Test (`{sets['holdout']}`). Model dùng cho chart: "
             f"**{result['chart_model']}**.\n")
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
        L.append("| Model | Macro AUC | Accuracy | Balanced acc | Macro F1 | LogLoss |\n|---|---|---|---|---|---|")
        for _, r in t.iterrows():
            L.append(f"| {r['Model']} | {_fmt(r['Macro AUC'])} | {_fmt(r['Accuracy'])} | {_fmt(r['Balanced acc'])} | "
                     f"{_fmt(r['Macro F1'])} | {_fmt(r['LogLoss'])} |")
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
    L.append("- Model benchmark dùng tham số mặc định và chỉ fit trên train; `Pipeline tuned` dùng hyperparameter "
             "pipeline chọn, cũng fit trên train để so sánh công bằng.\n"
             "- Model triển khai được refit trên train+valid; xác suất multiclass KHÔNG được hiệu chỉnh "
             "(calibrator isotonic/sigmoid chỉ áp dụng cho nhị phân).\n"
             "- Pipeline chọn model và feature theo macro AUC one-vs-rest.\n"
             "- Ngưỡng trong checklist là heuristic, không phải chuẩn.\n")
    return "\n".join(L)
