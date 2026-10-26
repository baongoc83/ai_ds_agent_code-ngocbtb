"""Render pipeline report artifacts as Streamlit elements.

Each function is defensive: a missing file or missing key renders an st.info
hint rather than crashing, so a partial/old run directory still renders the
tabs that have data.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

from ui.data_view import arrow_safe


_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")


def markdown_with_images(text: str, base_dir: Path) -> None:
    """st.markdown can't load local files, so split out ![alt](path) and use st.image."""
    pos = 0
    for m in _IMG_RE.finditer(text):
        chunk = text[pos:m.start()].strip()
        if chunk:
            st.markdown(chunk)
        alt, src = m.group(1), m.group(2)
        path = Path(src) if Path(src).is_absolute() else base_dir / src
        if src.startswith(("http://", "https://")):
            st.image(src, caption=alt or None)
        elif path.exists():
            st.image(str(path), caption=alt or None)
        else:
            st.caption(f"(Không tìm thấy ảnh: {src})")
        pos = m.end()
    tail = text[pos:].strip()
    if tail:
        st.markdown(tail)


def _load_json(path: str) -> Optional[dict]:
    if not path or not Path(path).exists():
        return None
    try:
        return json.load(open(path, "r", encoding="utf-8"))
    except Exception:
        return None


def _as_df(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            return pd.DataFrame(value)
        return pd.DataFrame({"value": list(value)})
    if isinstance(value, dict):
        return pd.DataFrame(list(value.items()), columns=["key", "value"])
    return value


def _shape_str(shape) -> str:
    if isinstance(shape, (list, tuple)) and len(shape) == 2:
        return f"{shape[0]:,} × {shape[1]}"
    return str(shape)


def render_preprocessing(dc_path: str, dq_path: str) -> None:
    dc = _load_json(dc_path)
    if dc is None:
        st.info("Chưa có báo cáo Data Cleaner cho run này.")
        return
    st.subheader("Preprocessing (Data Cleaner)")
    c1, c2 = st.columns(2)
    c1.metric("Shape ban đầu", _shape_str(dc.get("original_shape")))
    c2.metric("Shape sau làm sạch", _shape_str(dc.get("final_shape")))

    remaining = dc.get("columns_remaining")
    if isinstance(remaining, list):
        st.metric("Số cột giữ lại", len(remaining))

    actions = dc.get("actions_taken") or []
    if actions:
        with st.expander(f"Actions ({len(actions)})", expanded=False):
            for a in actions:
                st.write(f"- {a}")
    summary = dc.get("summary")
    if summary:
        st.markdown(summary)

    dq = _load_json(dq_path)
    if dq:
        st.markdown("**Data quality issues**")
        st.json(dq)


def render_feature_selection(fe_path: str, tm_path: str) -> None:
    fe = _load_json(fe_path)
    tm = _load_json(tm_path)
    if fe is None and tm is None:
        st.info("Chưa có báo cáo Feature Engineer cho run này.")
        return

    st.subheader("Feature selection")
    fp = (tm or {}).get("feature_pipeline") or {}
    funnel = [
        ("Ban đầu (n_init)", fp.get("n_init")),
        ("Sau RFE", fp.get("n_after_rfe")),
        ("Sau PSI", fp.get("n_after_psi")),
        ("Sau Stability", fp.get("n_after_stability")),
        ("Sau SHAP+PSI", fp.get("n_after_shap_psi")),
        ("Cuối (n_final)", fp.get("n_final")),
    ]
    cols = st.columns(len(funnel))
    for col, (label, val) in zip(cols, funnel):
        col.metric(label, val if val is not None else "—")

    if fe:
        feats = fe.get("final_features")
        if isinstance(feats, list):
            with st.expander(f"Final features ({len(feats)})", expanded=False):
                st.dataframe(pd.DataFrame({"feature": feats}), width="stretch")
        actions = fe.get("actions_taken") or []
        if actions:
            with st.expander(f"Feature engineering actions ({len(actions)})", expanded=False):
                for a in actions:
                    st.write(f"- {a}")
        if fe.get("summary"):
            st.markdown(fe["summary"])


def render_model(tm_path: str) -> None:
    tm = _load_json(tm_path)
    if tm is None:
        st.info("Chưa có báo cáo Model Trainer cho run này.")
        return
    st.subheader("Model results")
    metrics = tm.get("metrics") or {}

    task = _task_of(tm)
    if task == "regression":
        auc_keys = [
            ("cv_r2_mean", "CV R² (mean)"),
            ("cv_rmse", "CV RMSE"),
            ("test_r2", "Test R²"),
            ("test_rmse", "Test RMSE"),
            ("test_mae", "Test MAE"),
            ("oot_r2", "OOT R²"),
            ("oot_rmse", "OOT RMSE"),
        ]
    elif task == "multiclass":
        auc_keys = [
            ("cv_auc_mean", "CV macro AUC"),
            ("cv_f1_macro", "CV macro F1"),
            ("test_auc", "Test macro AUC"),
            ("test_accuracy", "Test accuracy"),
            ("test_f1_macro", "Test macro F1"),
            ("oot_auc", "OOT macro AUC"),
            ("oot_f1_macro", "OOT macro F1"),
        ]
        if metrics.get("classes"):
            st.caption(f"{len(metrics['classes'])} lớp: " + ", ".join(f"`{c}`" for c in metrics["classes"]))
    else:
        auc_keys = [
            ("cv_auc_mean", "CV AUC (mean)"),
            ("cv_auc_std", "CV AUC (std)"),
            ("valid_auc", "Valid AUC"),
            ("valid_temporal_auc", "Valid temporal AUC"),
            ("oot_auc", "OOT AUC"),
            ("test_auc", "Test AUC"),
        ]
    cols = st.columns(len(auc_keys))
    for col, (key, label) in zip(cols, auc_keys):
        val = metrics.get(key)
        col.metric(label, f"{val:.4f}" if isinstance(val, (int, float)) else "—")

    st.markdown(f"**Best estimator:** `{tm.get('best_estimator', '—')}`")

    params = metrics.get("best_params") or tm.get("best_params")
    if isinstance(params, dict):
        with st.expander("Best hyperparameters", expanded=False):
            # values mix ints, floats and strings ('log2', 'gbdt') → shown as text
            st.dataframe(
                arrow_safe(pd.DataFrame(list(params.items()), columns=["param", "value"])),
                width="stretch",
            )

    splits = tm.get("splits")
    if isinstance(splits, dict):
        with st.expander("Split sizes", expanded=False):
            st.dataframe(
                arrow_safe(pd.DataFrame(list(splits.items()), columns=["partition", "rows"])),
                width="stretch",
            )

    render_benchmark_summary(str(Path(tm_path).parent), _task_of(tm))

    if tm.get("summary"):
        st.markdown(tm["summary"])


def render_explanations(run_dir: str) -> None:
    st.subheader("Feature explanations")

    shap_csv = Path(run_dir) / "shap_feature_explanations.csv"
    shap_md = Path(run_dir) / "final_model_shap_report.md"
    shap_png = Path(run_dir) / "shap_summary.png"
    beeswarm = Path(run_dir) / "shap_beeswarm.png"

    if shap_csv.exists():
        df = pd.read_csv(shap_csv)
        with st.expander("Top features theo SHAP", expanded=True):
            st.dataframe(arrow_safe(df), width="stretch")
    else:
        st.info("Chưa có SHAP feature CSV cho run này.")

    if shap_md.exists():
        # The report already embeds the SHAP images inline.
        markdown_with_images(shap_md.read_text(encoding="utf-8", errors="replace"), Path(run_dir))
    else:
        st.info("Chưa có báo cáo giải thích SHAP (LLM) cho run này.")
        for path, label in ((shap_png, "SHAP summary"), (beeswarm, "SHAP beeswarm")):
            if path.exists():
                st.image(str(path), caption=label, width="stretch")


_SEV = {"ok": ("✅", "success"), "warn": ("⚠️", "warning"), "bad": ("❌", "error"), "info": ("ℹ️", "info")}

_BENCH_INTRO = {
    "regression": "MAE, RMSE càng thấp càng tốt; R² càng cao càng tốt (có thể âm nếu kém hơn baseline trung bình).",
    "classification": "AUC, Gini, KS, PR-AUC càng cao càng tốt; LogLoss, Brier càng thấp càng tốt. "
                      "Prior baseline luôn có AUC = 0.5.",
    "multiclass": "Macro AUC, accuracy, balanced accuracy, Macro F1 càng cao càng tốt; LogLoss càng thấp càng "
                  "tốt. Accuracy của Prior baseline = tỷ lệ lớp đa số.",
}


def _task_of(tm: dict) -> str:
    import target_encoding
    return target_encoding.task_of((tm.get("metrics") or {}).get("task") or tm.get("task") or "classification")


def _bench_module(task: str):
    if task == "regression":
        from Agents.TrainModel import regression_benchmark as mod
    elif task == "multiclass":
        from Agents.TrainModel import multiclass_benchmark as mod
    else:
        from Agents.TrainModel import classification_benchmark as mod
    return mod


def _benchmark(run_dir: str, task: str) -> Optional[dict]:
    return _load_json(str(Path(run_dir) / f"{task}_benchmark.json"))


def render_benchmark_summary(run_dir: str, task: str) -> None:
    """Benchmark tables + post-training evaluation & recommendations (both tasks)."""
    mod = _bench_module(task)
    bm = _benchmark(run_dir, task)
    if not bm:
        return
    st.markdown("### Baseline & benchmark")
    st.caption("Mọi model fit trên **train**, chấm cùng metric trên Validation và OOT/Test. " + _BENCH_INTRO[task])
    sets = bm.get("eval_sets") or {}
    tabs = [(lab, sets.get(k)) for k, lab in (("valid", "Validation"), ("holdout", "OOT/Test")) if sets.get(k)]
    for (lab, split), tab in zip(tabs, st.tabs([f"{lab} ({split})" for lab, split in tabs])):
        with tab:
            df = mod.results_table(bm, split)
            if df.empty:
                st.info("Không có kết quả cho tập này.")
                continue
            st.dataframe(df.style.highlight_min(subset=mod.LOWER_BETTER, color="#c8e6c9")
                         .highlight_max(subset=mod.HIGHER_BETTER, color="#c8e6c9")
                         .format(mod.TABLE_FORMAT, na_rep="—"),
                         width="stretch", hide_index=True)
    if bm.get("failed"):
        st.caption("Không chạy được: " + "; ".join(f"{k} ({v})" for k, v in bm["failed"].items()))
    rec = bm.get("recommended_model") or {}
    if rec:
        st.info(f"**Khuyến nghị model:** {rec.get('note', '')}")

    st.markdown("### Đánh giá sau huấn luyện & khuyến nghị")
    for f in bm.get("findings", []):
        icon, kind = _SEV.get(f.get("severity"), ("•", "info"))
        getattr(st, kind)(f"{icon} **{f['title']}** — {f['finding']}\n\n👉 {f['action']}")
    with st.expander("Checklist ra quyết định", expanded=False):
        st.markdown("| Điều kiện | Hành động |\n|---|---|\n"
                    + "\n".join(f"| {c} | {a} |" for c, a in mod.CHECKLIST))


def _render_benchmark_charts(run_dir: str, bm: dict, task: str) -> int:
    mod = _bench_module(task)
    chart_dir = Path(run_dir) / "charts"
    shown = 0
    st.caption(f"Model dùng cho chart 3–7: **{bm.get('chart_model')}** (bản fit trên train, "
               "nên Validation là hold-out thật).")
    for spec in mod.CHART_SPECS:
        files = [(w, chart_dir / fn) for w, fn in spec["files"].items() if (chart_dir / fn).exists()]
        if not files:
            continue
        st.markdown(f"#### {spec['label']}")
        with st.expander("Cách đọc chart", expanded=False):
            st.markdown(spec["guide"])
        cols = st.columns(len(files))
        for col, (w, p) in zip(cols, files):
            col.image(str(p), caption="Validation" if w == "valid" else "OOT/Test", width="stretch")
            shown += 1
    return shown


def render_charts(run_dir: str) -> None:
    from config import Config
    tm = _load_json(str(Path(run_dir) / "model_trainer_report.json")) or {}
    task = _task_of(tm)
    bm = _benchmark(run_dir, task)
    shown = _render_benchmark_charts(run_dir, bm, task) if bm else 0
    if shown and task in ("regression", "multiclass"):
        return
    # Classification keeps the deployed (calibrated) model's own charts.
    deployed = [(Path(run_dir) / "charts" / Path(path).name, label)
                for path, label in Config.chart_files(task)
                if not Path(path).name.startswith(("clf_", "reg_", "mc_"))]
    deployed = [(p, label) for p, label in deployed if p.exists()]
    if deployed:
        box = st.expander("Chart của model triển khai (refit train+valid, đã hiệu chỉnh)", expanded=not shown) \
            if shown else st.container()
        with box:
            for p, label in deployed:
                st.image(str(p), caption=label.split("—")[0].strip())
    if not shown and not deployed:
        st.info("Run này không có chart.")


def render_predict(run_dir: str) -> None:
    from ui import scoring

    st.subheader("Dự đoán với model của run này")
    try:
        bundle = _cached_bundle(run_dir)
    except Exception as e:  # noqa: BLE001
        st.error(f"Không load được model: {e}")
        return
    task = scoring.task_of(bundle)
    target = bundle["manifest"].get("target_column")
    st.caption(f"Loại bài toán: **{task}** · Target: `{target}` · "
               f"Model: `{bundle['artifact'].get('estimator_name', '—')}`")

    mode = st.radio("Cách nhập", ["Nhập tay 1 bản ghi", "Upload file nhiều bản ghi"],
                    horizontal=True, key=f"pred_mode_{run_dir}")

    if mode == "Nhập tay 1 bản ghi":
        cols = scoring.input_columns(run_dir)
        if not cols:
            st.info("Không xác định được danh sách cột input của run này.")
            return
        defaults = _input_defaults(run_dir, cols)
        with st.form(key=f"pred_form_{run_dir}"):
            grid = st.columns(3)
            values = {}
            for i, c in enumerate(cols):
                d = defaults.get(c)
                with grid[i % 3]:
                    if isinstance(d, list):
                        values[c] = st.selectbox(c, d, key=f"in_{run_dir}_{c}")
                    elif isinstance(d, (int, float)):
                        values[c] = st.number_input(c, value=float(d), key=f"in_{run_dir}_{c}")
                    else:
                        values[c] = st.text_input(c, value="" if d is None else str(d),
                                                  key=f"in_{run_dir}_{c}")
            submitted = st.form_submit_button("Dự đoán", type="primary")
        if submitted:
            _show_prediction(bundle, pd.DataFrame([values]), task, single=True)
    else:
        up = st.file_uploader("File input (csv / tsv / txt / parquet / xlsx / json ...)",
                              type=["csv", "tsv", "txt", "parquet", "feather", "orc", "xlsx", "xls",
                                    "xlsm", "json", "jsonl", "gz", "zip"],
                              key=f"pred_up_{run_dir}")
        if up is not None:
            import tempfile
            from Agents.BaseAgent.base_agent import BaseAgent
            # same loader as training: sniffs the CSV separator / encoding
            suffix = "".join(Path(up.name).suffixes[-2:]) or Path(up.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(up.getbuffer())
            try:
                df = BaseAgent.load_dataframe(tmp.name)
            finally:
                Path(tmp.name).unlink(missing_ok=True)
            st.write(f"Đã đọc {len(df):,} dòng × {df.shape[1]} cột.")
            if st.button("Dự đoán", type="primary", key=f"pred_btn_{run_dir}"):
                _show_prediction(bundle, df, task, single=False)


@st.cache_resource(show_spinner="Đang load model…")
def _cached_bundle(run_dir: str):
    from ui import scoring
    return scoring.load_bundle(run_dir)


def _input_defaults(run_dir: str, cols: list) -> dict:
    """Median for numeric / category list for text, taken from the run's source data."""
    m = _load_json(str(Path(run_dir) / "replay_manifest.json")) or {}
    src = m.get("source_input")
    df = None
    if src and Path(src).exists():
        try:
            from Agents.BaseAgent.base_agent import BaseAgent
            # only the first rows are needed — never read a multi-GB source fully
            suffix, _ = BaseAgent._split_compression(src)
            if suffix in (".csv", ".tsv", ".txt", ""):
                df = BaseAgent.read_csv_smart(src, nrows=5000)
            elif suffix == ".parquet":
                import pyarrow.parquet as pq
                df = next(pq.ParquetFile(src).iter_batches(batch_size=5000)).to_pandas()
            else:
                df = BaseAgent.load_dataframe(src).head(5000)
        except Exception:  # noqa: BLE001
            df = None
    out = {}
    for c in cols:
        if df is None or c not in df.columns:
            out[c] = None
            continue
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            med = s.median()
            out[c] = 0.0 if pd.isna(med) else float(med)
        else:
            vals = s.dropna().astype(str).value_counts().index.tolist()
            out[c] = vals[:200] if 0 < len(vals) <= 200 else (vals[0] if vals else None)
    return out


def _show_prediction(bundle, df: pd.DataFrame, task: str, single: bool) -> None:
    from ui import scoring
    try:
        with st.spinner("Đang dự đoán…"):
            out = scoring.predict(bundle, df)
    except Exception as e:  # noqa: BLE001
        st.error(f"Dự đoán lỗi: {e}")
        return
    if single:
        if task == "regression":
            st.success(f"Giá trị dự đoán: **{out['prediction'].iloc[0]:,.4f}**")
        elif task == "multiclass":
            proba = {c[len("proba_"):]: float(out[c].iloc[0]) for c in out.columns if c.startswith("proba_")}
            st.success(f"Lớp dự đoán: **{out['label'].iloc[0]}** "
                       f"(xác suất {max(proba.values()):.4f})")
            st.bar_chart(pd.Series(proba, name="xác suất"))
        else:
            st.success(f"Xác suất lớp dương (`{scoring.class_labels(bundle)[-1]}`): "
                       f"**{out['probability'].iloc[0]:.4f}** · "
                       f"Nhãn (ngưỡng 0.5): **{out['label'].iloc[0]}**")
        return
    st.success(f"Đã dự đoán {len(out):,} dòng.")
    lead = [c for c in ("label", "prediction", "probability") if c in out.columns] + \
           [c for c in out.columns if c.startswith("proba_")]
    st.dataframe(arrow_safe(out[lead + [c for c in out.columns if c not in lead]].head(1000)), width="stretch")
    st.download_button("Tải kết quả (CSV)", out.to_csv(index=False).encode("utf-8"),
                       file_name="predictions.csv", mime="text/csv")


def render_final_report(run_dir: str) -> None:
    path = Path(run_dir) / "final_report.md"
    if path.exists():
        markdown_with_images(path.read_text(encoding="utf-8", errors="replace"), Path(run_dir))
    else:
        st.info("Chưa có final report cho run này.")