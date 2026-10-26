"""Streamlit web UI for the Multi-Agent AutoML pipeline.

Run:  streamlit run ui/app.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import workspace
from Agents.BaseAgent.base_agent import BaseAgent
from config import Config
from ui import config_editor, data_view, render, runner

st.set_page_config(page_title="AutoML Pipeline UI", layout="wide")

# ── data loading helpers ──────────────────────────────────────────────────
# Every view of the input (columns, preview, stats, target preview) is derived
# from ONE in-memory frame per (path, version): a multi-GB file is read once,
# not once per widget. cache_resource shares that frame across reruns and
# sessions without copying it (consumers must not mutate it), bounded to a
# couple of datasets so memory stays predictable.

REMOTE_SCHEMES = BaseAgent._S3_SCHEMES + ("gs://", "az://", "abfs://")
STATS_SAMPLE_ROWS = int(os.getenv("UI_STATS_SAMPLE_ROWS", 500_000))


def _is_remote(path: Optional[str]) -> bool:
    return bool(path) and path.strip().lower().startswith(REMOTE_SCHEMES)


def _path_ok(path: Optional[str]) -> bool:
    """Local file that exists, or a remote URI (existence is checked when read)."""
    return bool(path) and (_is_remote(path) or os.path.exists(BaseAgent.file_part(path)))


def _mtime(path: str):
    """Cache key part: file mtime for local files. For remote URIs it is a hash
    of this session's effective S3 settings instead, so data read with one
    user's credentials is never served from the cache to a session using
    different (or no) keys."""
    if _is_remote(path):
        sig = "|".join(map(str, BaseAgent.s3_settings()[:5]))
        return "remote:" + hashlib.sha256(sig.encode()).hexdigest()[:16]
    try:
        return Path(BaseAgent.file_part(path)).stat().st_mtime
    except OSError:
        return 0.0


@st.cache_resource(show_spinner="Đang đọc dữ liệu…", max_entries=int(os.getenv("UI_DATA_CACHE_ENTRIES", 2)))
def _load_df(path: str, mtime) -> pd.DataFrame:
    return BaseAgent.load_dataframe(path)


@st.cache_data(show_spinner=False)
def _excel_sheets(path: str, mtime) -> list:
    return BaseAgent.excel_sheets(path)


@st.cache_data(show_spinner=False)
def _load_columns(path: str, mtime) -> list:
    return list(_load_df(path, mtime).columns)


@st.cache_data(show_spinner=False)
def _load_head(path: str, mtime, n: int = 100) -> pd.DataFrame:
    return _load_df(path, mtime).head(n).copy()


@st.cache_data(show_spinner=False)
def _load_shape(path: str, mtime) -> tuple:
    return _load_df(path, mtime).shape


@st.cache_data(show_spinner="Đang tính thống kê…")
def _load_stats(path: str, mtime):
    """Descriptive stats — on a random sample for very large frames (exact
    enough for null % / quantiles / skew, and seconds instead of minutes)."""
    df = _load_df(path, mtime)
    if len(df) > STATS_SAMPLE_ROWS:
        df = df.sample(STATS_SAMPLE_ROWS, random_state=0)
    return data_view.describe_dataset(df)


@st.cache_data(show_spinner=False)
def _profile(path: str, mtime, target: Optional[str]):
    from profiling import profile_raw
    return profile_raw(path, target)


@st.cache_data(show_spinner=False)
def _target_preview(path: str, mtime, target: str, model_type: str):
    """(plan, error) — the exact target encoding the pipeline will freeze."""
    import target_encoding
    try:
        df = _load_df(path, mtime)
        if target not in df.columns:
            return None, None
        return target_encoding.plan(df[target], model_type), None
    except ValueError as e:
        return None, str(e)
    except Exception:  # noqa: BLE001 — preview only; the run reports real errors
        return None, None


# Everything BaseAgent.load_dataframe reads. Compressed uploads keep the inner
# extension (data.csv.gz) so the loader still knows the format.
DATA_FILE_TYPES = ["csv", "tsv", "txt", "parquet", "feather", "ftr", "orc", "xlsx", "xls", "xlsm",
                   "json", "jsonl", "ndjson", "gz", "bz2", "xz", "zip", "zst"]


def _save_upload(uploaded) -> str:
    """Write an upload into this workspace's _uploads/ once per (name, size) and
    reuse it on reruns (path/mtime-keyed caches stay valid). Runs copy their
    input into the run dir, so uploads are only a staging area — the workspace
    clean-up proposes deleting them after RETENTION_DAYS."""
    key = f"_upload::{WS_NAME}::{uploaded.name}::{uploaded.size}"
    path = st.session_state.get(key)
    if path and os.path.exists(path):
        return path
    suffix = "".join(Path(uploaded.name).suffixes[-2:]) or Path(uploaded.name).suffix
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(uploaded.name).name[: -len(suffix) or None])[:60]
    dest = workspace.uploads_dir(Config.OUTPUT_DIR, WS_NAME) / f"{time.strftime('%Y%m%d_%H%M%S')}_{stem}{suffix}"
    dest.write_bytes(uploaded.getbuffer())
    st.session_state[key] = str(dest)
    return str(dest)


# ── S3 credentials (per browser session) ──────────────────────────────────
# Keys typed in the UI live only in this session (st.session_state). They are
# applied to THIS session's thread for previews (BaseAgent.set_thread_s3_settings)
# and passed to the job process as environment variables — never written to the
# shared Config, os.environ or disk, so concurrent users cannot see or use each
# other's keys. Empty fields fall back to .env (S3_*) and the AWS default chain.

_S3_FIELDS = [  # (Config attr, label, is_secret, help)
    ("S3_ACCESS_KEY", "Access key ID", False, "AWS access key ID hoặc access key của MinIO/Wasabi..."),
    ("S3_SECRET_KEY", "Secret access key", True, "Secret đi cùng access key."),
    ("S3_SESSION_TOKEN", "Session token (tùy chọn)", True, "Chỉ cần với credential tạm thời (STS/SSO)."),
    ("S3_REGION", "Region (tùy chọn)", False, "vd: ap-southeast-1. Để trống = tự nhận."),
    ("S3_ENDPOINT_URL", "Endpoint URL (tùy chọn)", False,
     "Cho S3-compatible (MinIO...), vd: http://minio.local:9000. Để trống nếu dùng AWS S3."),
]


def _session_s3() -> dict:
    """This session's UI-entered S3 values (only the non-empty ones)."""
    vals = {attr: (st.session_state.get(f"s3_{attr}") or "").strip() for attr, *_ in _S3_FIELDS}
    return {k: v for k, v in vals.items() if v}


_S3_PANEL_SHOWN = False  # module globals reset on every Streamlit rerun


def _s3_credentials_panel(path: str) -> None:
    global _S3_PANEL_SHOWN
    if _S3_PANEL_SHOWN:  # typed path and a viewed run can both be S3 — one panel only
        BaseAgent.set_thread_s3_settings(_session_s3())
        return
    _S3_PANEL_SHOWN = True
    with st.expander("🔑 Thông tin truy cập S3", expanded=True):
        st.caption("Key chỉ lưu trong phiên trình duyệt này (không dùng chung với người khác, "
                   "không ghi ra file). Để trống để dùng cấu hình trong .env hoặc AWS credential "
                   "chain (biến môi trường, ~/.aws/credentials, IAM role).")
        for attr, label, secret, help_ in _S3_FIELDS:
            st.text_input(label, key=f"s3_{attr}", type="password" if secret else "default",
                          help=help_, placeholder="(dùng .env)" if os.getenv(attr) else "")
        BaseAgent.set_thread_s3_settings(_session_s3())
        access, secret = BaseAgent.s3_settings()[:2]
        if bool(access) != bool(secret):
            st.warning("Cần nhập cả Access key ID và Secret access key.")
        if st.button("Kiểm tra kết nối", key="s3_check"):
            try:
                st.success(BaseAgent.check_s3_access(path))
            except Exception as e:  # noqa: BLE001
                st.error(f"Không truy cập được: {type(e).__name__}: {e}")


# ── source file selection ─────────────────────────────────────────────────

st.title("AutoML Pipeline")
st.caption("Chọn dữ liệu, cấu hình, bấm Train — xem báo cáo đầy đủ.")

# ── workspace (per user) ──────────────────────────────────────────────────
# Each user works in a workspace: runs, uploads and clean-up are scoped to it.
# The choice is kept in the URL (?ws=name) so a reload / bookmark keeps it.
# "default" = OUTPUT_DIR itself, so every run made before workspaces existed stays visible.

_NEW_WS = "➕ Tạo workspace mới…"


@st.cache_data(ttl=60, show_spinner=False)
def _scan_workspace(ws_path: str, active: tuple, days: int):
    return workspace.scan(Path(ws_path), days, active)


with st.sidebar:
    _ws_list = workspace.list_workspaces(Config.OUTPUT_DIR)
    _ws_q = st.query_params.get("ws", workspace.DEFAULT)
    _ws_idx = _ws_list.index(_ws_q) if _ws_q in _ws_list else 0
    _ws_pick = st.selectbox("🗂 Workspace", _ws_list + [_NEW_WS], index=_ws_idx, key="ws_pick",
                            help="Không gian làm việc riêng: run, file upload và việc dọn dẹp "
                                 "chỉ áp dụng trong workspace đã chọn.")
    if _ws_pick == _NEW_WS:
        _new = st.text_input("Tên workspace mới", placeholder="vd: ngoc_credit", key="ws_new_name")
        if st.button("Tạo workspace", key="ws_create"):
            if workspace.valid_name(_new):
                workspace.create(Config.OUTPUT_DIR, _new)
                st.query_params["ws"] = _new
                st.session_state.pop("ws_pick", None)
                st.rerun()
            else:
                st.error("Tên chỉ gồm chữ, số, '_', '.', '-' (tối đa 64 ký tự).")
        WS_NAME = _ws_q if _ws_q in _ws_list else workspace.DEFAULT
    else:
        WS_NAME = _ws_pick
        if st.query_params.get("ws") != WS_NAME:
            st.query_params["ws"] = WS_NAME
    WS_DIR = workspace.ws_dir(Config.OUTPUT_DIR, WS_NAME)
    WS_DIR.mkdir(parents=True, exist_ok=True)

    # Retention: never deletes on its own — shows a red warning and asks.
    _scan = _scan_workspace(str(WS_DIR), tuple(runner.active_run_dirs()), workspace.RETENTION_DAYS)
    _old = _scan["old_runs"] + _scan["old_uploads"]
    _dismiss_key = f"ws_dismiss::{WS_NAME}"
    if _old and not st.session_state.get(_dismiss_key):
        _mb = sum(i["size_mb"] for i in _old)
        st.error(f"⚠️ Workspace **{WS_NAME}** có {len(_scan['old_runs'])} run và "
                 f"{len(_scan['old_uploads'])} file upload đã quá {workspace.RETENTION_DAYS} ngày "
                 f"({_mb:,.0f} MB). Bạn có muốn xóa không?")
        with st.expander("Xem danh sách", expanded=False):
            st.dataframe(pd.DataFrame(_old)[["name", "age_days", "size_mb"]], hide_index=True, width="stretch")
        _c1, _c2, _c3 = st.columns(3)
        if _c1.button("🗑 Xóa", key="ws_del_old", help="Xóa vĩnh viễn các run/file trong danh sách"):
            _r = workspace.delete(WS_DIR, [i["path"] for i in _old])
            _scan_workspace.clear()
            st.success(f"Đã xóa {_r['deleted']} mục, giải phóng {_r['freed_mb']:,.0f} MB.")
            st.rerun()
        if _c2.button("📌 Giữ lại", key="ws_keep_old", help="Đánh dấu giữ các run này — không nhắc lại"):
            workspace.keep([i["path"] for i in _scan["old_runs"]])
            _scan_workspace.clear()
            st.session_state[_dismiss_key] = True
            st.rerun()
        if _c3.button("Để sau", key="ws_later", help="Ẩn cảnh báo trong phiên này"):
            st.session_state[_dismiss_key] = True
            st.rerun()
    if _scan["failed_runs"]:
        with st.expander(f"🧹 Run lỗi / dở dang ({len(_scan['failed_runs'])})", expanded=False):
            st.caption("Run không tạo ra báo cáo model (lỗi hoặc bị dừng) và không còn chạy.")
            st.dataframe(pd.DataFrame(_scan["failed_runs"])[["name", "age_days", "size_mb"]],
                         hide_index=True, width="stretch")
            if st.button("Xóa tất cả run lỗi", key="ws_del_failed"):
                _r = workspace.delete(WS_DIR, [i["path"] for i in _scan["failed_runs"]])
                _scan_workspace.clear()
                st.success(f"Đã xóa {_r['deleted']} run lỗi ({_r['freed_mb']:,.0f} MB).")
                st.rerun()
    st.divider()

with st.sidebar:
    st.header("① Dữ liệu & cấu hình")

    src_mode = st.radio(
        "Nguồn dữ liệu",
        ["Đường dẫn local", "Upload file"],
        help="Chọn file dataset: csv/tsv/txt (tự nhận dấu phân cách , ; tab |), parquet, "
             "feather, orc, xlsx/xls/xlsm, json/jsonl; có thể nén .gz/.bz2/.xz/.zip/.zst. "
             "Với dataset lớn (nhiều GB) nên dùng đường dẫn local cho nhanh.",
    )

    input_path = None
    if src_mode == "Đường dẫn local":
        input_path = st.text_input(
            "Đường dẫn file input",
            placeholder="vd: data/train.csv hoặc s3://bucket/path/train.parquet",
            help="Đường dẫn tuyệt đối/tương đối tới file trên máy, hoặc URI "
                 "s3:// · s3a:// · s3n:// (AWS S3 hoặc S3-compatible như MinIO). "
                 "Với S3 sẽ hiện thêm ô nhập key truy cập.",
        )
        input_path = (input_path or "").strip() or None
        if input_path and input_path.lower().startswith(BaseAgent._S3_SCHEMES):
            _s3_credentials_panel(input_path)
    else:
        uploaded = st.file_uploader(
            "Chọn file dataset",
            type=DATA_FILE_TYPES,
            help="Upload file dataset. File được lưu tạm để pipeline đọc.",
        )
        if uploaded is not None:
            input_path = _save_upload(uploaded)
    if not (input_path and input_path.lower().startswith(BaseAgent._S3_SCHEMES)):
        BaseAgent.set_thread_s3_settings(None)  # source is not S3 → no session keys

    # Excel workbook with several sheets → let the user pick (path becomes "book.xlsx::Sheet")
    _sheets = (_excel_sheets(BaseAgent.file_part(input_path), _mtime(input_path))
               if input_path and BaseAgent.file_part(input_path).lower().endswith(BaseAgent._EXCEL_SUFFIXES)
               else [])
    if len(_sheets) > 1:
        _AUTO, _ALL = "(tự động: gộp nếu cùng cột, nếu không lấy sheet lớn nhất)", "(gộp tất cả các sheet)"
        if st.session_state.get("excel_sheet") not in [_AUTO, _ALL] + _sheets:
            st.session_state.pop("excel_sheet", None)
        _choice = st.selectbox("Sheet Excel", [_AUTO, _ALL] + _sheets, key="excel_sheet",
                               help=f"File có {len(_sheets)} sheet: {', '.join(_sheets)}.")
        if _choice == _ALL:
            input_path = f"{BaseAgent.file_part(input_path)}::*"
        elif _choice != _AUTO:
            input_path = f"{BaseAgent.file_part(input_path)}::{_choice}"

    col_desc_path = None
    col_desc_kind, col_desc_obj = None, None
    col_desc_uploaded = st.file_uploader(
        "Col-description (tùy chọn)",
        type=["csv", "tsv", "txt", "json", "yaml", "yml", "parquet", "xlsx", "xls", "xlsm"],
        help="File mô tả cột giúp Agent 2 (Feature Engineer) hiểu ngữ cảnh từng "
             "cột. Nếu file đã ở dạng `tên_cột: mô_tả` (json/yaml) thì không cần "
             "khai báo field.",
    )
    if col_desc_uploaded is not None:
        col_desc_path = _save_upload(col_desc_uploaded)
        try:
            col_desc_kind, col_desc_obj = data_view.read_desc_file(col_desc_path)
        except Exception as e:  # noqa: BLE001
            st.error(f"Không đọc được file col-desc: {e}")
            col_desc_path = None

    col_name_field, col_desc_field, col_group_field = "column_name", "description", ""
    if col_desc_kind in ("mapping", "nested") and Path(col_desc_path).suffix.lower() != ".json":
        # pipeline only reads .json: hand it a normalised copy of the yaml (same
        # staging area as uploads — written once per source file)
        _json_copy = Path(col_desc_path).with_suffix(".normalised.json")
        if not _json_copy.exists():
            _json_copy.write_text(json.dumps(col_desc_obj, ensure_ascii=False), encoding="utf-8")
        col_desc_path = str(_json_copy)
    if col_desc_kind == "mapping":
        st.caption("File đã ở dạng `tên cột: mô tả` — không cần chọn field.")
    elif col_desc_kind in ("table", "nested"):
        if col_desc_kind == "table":
            col_name_field = st.text_input("col_name_field", value="column_name",
                                           help="Tên cột chứa tên cột trong file col-desc.")
            col_desc_field = st.text_input("col_desc_field", value="description",
                                           help="Tên cột chứa mô tả trong file col-desc.")
        col_group_field = st.text_input("col_group_field", value="",
                                        help="(Tùy chọn) Tên cột nhóm/table trong file col-desc. Để trống nếu không có.")

    auto_config = st.toggle(
        "Auto-config từ dữ liệu",
        value=True,
        help="Tự động suy ra các config (số feature giữ lại, kích thước sample, cap "
             "metadata) từ shape dữ liệu. Các giá trị bạn set thủ công (qua YAML bên "
             "dưới hoặc env) vẫn luôn được ưu tiên.",
    )

    # Placeholders keep the sidebar order: ② config is filled later (after the
    # data is loaded), but the run viewer must stay usable before any file is chosen.
    cfg_box = st.container()
    out_box = st.container()

with out_box:
    st.divider()
    st.header("📂 Xem output đã chạy")
    st.caption("Chọn một run đã hoàn tất trong outputs/ để xem lại kết quả "
               "(Preprocessing / Feature selection / Model / Explanations).")
    runs = runner.list_successful_runs(str(WS_DIR))   # this workspace only
    if not runs:
        st.caption("Chưa có run output nào.")
    else:
        run_options = [f"{d}/{r}" for d, r, _ in runs]
        sel = st.selectbox(
            "Chọn run",
            options=run_options,
            index=0,
            key="run_selector",
            help="Run mới nhất ở trên. Chọn run rồi bấm 'Xem run này'.",
        )
        if st.button("Xem run này", key="load_run", width="stretch"):
            sel_dir = dict(((f"{d}/{r}"), p) for d, r, p in runs).get(sel)
            if sel_dir:
                st.session_state["run_dir"] = sel_dir
                st.session_state["run_id"] = None
                st.session_state["view_src_run"] = sel_dir
                st.rerun()

# ── dataset of a past run ─────────────────────────────────────────────────
# Viewing a past run switches the data panel to the dataset that run used
# (replay_manifest.json -> source_input). Choosing another file in the sidebar
# switches back.

_sel_sig = (src_mode, input_path if src_mode == "Đường dẫn local"
            else (uploaded.name, uploaded.size) if uploaded is not None else None)
if st.session_state.get("_input_sig", _sel_sig) != _sel_sig:
    # New dataset → start clean: drop the previously viewed run's results.
    # A job still training keeps its run_id so its progress stays visible.
    st.session_state.pop("view_src_run", None)
    st.session_state.pop("run_dir", None)
    st.session_state.pop("done_msg", None)
    if not runner.is_running([st.session_state.get("run_id")]):
        st.session_state.pop("run_id", None)
st.session_state["_input_sig"] = _sel_sig

run_src_note = None
_view_run = st.session_state.get("view_src_run")
if _view_run:
    try:
        with open(Path(_view_run) / "replay_manifest.json", encoding="utf-8") as _f:
            _run_src = json.load(_f).get("source_input")
    except (OSError, ValueError):
        _run_src = None
    if _run_src and _path_ok(_run_src):
        input_path = _run_src
        if _run_src.lower().startswith(BaseAgent._S3_SCHEMES):
            with st.sidebar:
                _s3_credentials_panel(_run_src)
        run_src_note = (f"Đang hiển thị dataset của run `{Path(_view_run).parent.name}/"
                        f"{Path(_view_run).name}`: `{_run_src}`")
    else:
        st.sidebar.warning(
            f"Không tìm thấy dataset gốc của run này ({_run_src or 'manifest không ghi đường dẫn'}). "
            "File upload tạm có thể đã bị xóa — hãy chọn lại file dữ liệu.")

# ── columns for the selected file ─────────────────────────────────────────

columns: list = []
target_ok = False
if input_path:
    if not _path_ok(input_path):
        st.sidebar.error("Không tìm thấy file. Kiểm tra lại đường dẫn.")
    else:
        try:
            columns = _load_columns(input_path, _mtime(input_path))
        except Exception as e:  # noqa: BLE001
            st.sidebar.error(f"Không đọc được file: {e}")

# ── main area ─────────────────────────────────────────────────────────────

# Render a previously-completed run first so it works even before an input
# file is chosen — the run directory comes from the sidebar run selector or a
# freshly-finished training job (st.session_state["run_dir"]).
done_msg = st.session_state.pop("done_msg", None)
if done_msg:
    st.success(done_msg)
    st.toast(done_msg, icon="✅")
    st.balloons()

run_dir = st.session_state.get("run_dir")
if run_dir and Path(run_dir).exists():
    rd = Path(run_dir)
    st.subheader(f"Kết quả run: `{rd.parent.name}/{rd.name}`")
    tab1, tab2, tab3, tab_ch, tab4, tab_pred = st.tabs(
        ["Preprocessing", "Feature selection", "Model results", "Charts",
         "Feature explanations", "🔮 Dự đoán"]
    )
    with tab_ch:
        render.render_charts(str(rd))
    with tab_pred:
        render.render_predict(str(rd))
    with tab1:
        render.render_preprocessing(
            str(rd / "data_cleaner_report.json"),
            str(rd / "data_quality_report.json"),
        )
    with tab2:
        render.render_feature_selection(
            str(rd / "feature_engineer_report.json"),
            str(rd / "model_trainer_report.json"),
        )
    with tab3:
        render.render_model(str(rd / "model_trainer_report.json"))
    with tab4:
        render.render_explanations(str(rd))

if not input_path or not columns:
    st.info("Chọn một file dữ liệu ở sidebar để bắt đầu.")
    st.stop()

# Head preview
st.subheader("Xem nhanh dữ liệu")
if run_src_note:
    st.info(run_src_note + " — chọn file khác ở sidebar để quay lại.")
head = _load_head(input_path, _mtime(input_path))
st.dataframe(data_view.arrow_safe(head), width="stretch")
_n_rows, _n_cols = _load_shape(input_path, _mtime(input_path))
st.write(f"**Shape:** {_n_rows:,} dòng (xem {min(100, _n_rows)}) × {_n_cols:,} cột")

st.subheader("Thống kê mô tả")
try:
    _num, _cat = _load_stats(input_path, _mtime(input_path))
    if len(_num):
        st.caption("Cột số — null %, distinct, min / mean / median / IQR / max, skew")
        st.dataframe(_num, width="stretch", hide_index=True)
    if len(_cat):
        st.caption("Cột category/text — null %, count distinct, giá trị phổ biến nhất")
        st.dataframe(_cat, width="stretch", hide_index=True)
except Exception as e:  # noqa: BLE001
    st.warning(f"Không tính được thống kê: {e}")

if col_desc_kind in ("mapping", "nested", "table"):
    st.subheader("Mô tả cột (YAML)")
    if col_desc_kind == "table":
        if col_name_field in col_desc_obj.columns and col_desc_field in col_desc_obj.columns:
            st.code(data_view.to_yaml(data_view.table_to_mapping(
                col_desc_obj, col_name_field, col_desc_field, col_group_field or None)),
                language="yaml")
        else:
            st.warning(f"Không thấy field trong file. Các cột có sẵn: {list(col_desc_obj.columns)}")
    else:
        st.code(data_view.to_yaml(col_desc_obj), language="yaml")

# ── config panel ──────────────────────────────────────────────────────────

with st.expander("⚙️ Config / Auto-config", expanded=False):
    # Target / model type come from the sidebar widgets (their state from the
    # previous rerun) so the sizing can use the smallest target class.
    _tgt = st.session_state.get("target_column")
    _mt = st.session_state.get("model_type", "binary_classification")
    profile = _profile(input_path, _mtime(input_path), _tgt if _tgt in columns else None)
    auto_overrides: dict = {}
    if profile:
        from profiling import derive_config, size_tier
        auto_overrides = derive_config(profile, _mt)
        tier = size_tier(profile)
        st.caption(
            f"Kích thước: **{profile.get('n_rows'):,} dòng** ({tier['rows']}) × **{profile.get('n_cols')} cột** "
            f"({tier['cols']}) = {tier['cells']} ô"
            + (f" · target `{_tgt}`: {profile.get('target_n_unique')} giá trị, lớp nhỏ nhất "
               f"{profile.get('target_minority'):,} dòng" if profile.get("target_n_unique") else "")
            + f" · model type `{_mt}`.")
    # Pins from the YAML editor belong to THIS session only (passed to the job
    # process as env); knobs pinned in the server's .env apply to everyone.
    cfg_pins: dict = st.session_state.get("cfg_pins", {})
    if profile:
        desc = config_editor.knob_descriptions()
        defaults = Config._auto_defaults()
        rows = []
        for group, keys in config_editor.GROUPS:
            for k in keys:
                if k in cfg_pins:
                    used, source = cfg_pins[k], "ép (phiên này)"
                elif config_editor.env_pinned(k):
                    used, source = os.environ[Config.AUTO_CONFIG_ENV[k]], "ép (.env server)"
                else:
                    used = auto_overrides.get(k) if auto_config else defaults.get(k)
                    source = "auto" if auto_config else "mặc định"
                show = lambda v: "—" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))  # noqa: E731
                rows.append({"nhóm": group, "knob": k, "mặc định": show(defaults.get(k)),
                             "auto": show(auto_overrides.get(k)), "sẽ dùng": show(used),
                             "nguồn": source, "ý nghĩa": desc.get(k, "")})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    else:
        st.write("Không thể tự suy config (file không profile được, ví dụ file trên S3).")

    yaml_text = st.text_area(
        "Config (YAML)",
        value=config_editor.build_yaml(auto_config, auto_overrides, cfg_pins),
        height=320,
        help="Bỏ comment key nào để ép giá trị đó (thắng auto-config); xóa dòng để bỏ ép. "
             "Chỉ áp dụng cho phiên của bạn.",
    )
    if st.button("Áp dụng config", key="apply_cfg"):
        ok, errors, pins = config_editor.parse_yaml(yaml_text)
        if ok:
            st.session_state["cfg_pins"] = pins
            st.success(f"Đã áp dụng {len(pins)} giá trị ép cho phiên này — dùng ở lần Train tới.")
            st.rerun()
        else:
            st.error("Không áp dụng được:\n" + "\n".join(f"- {e}" for e in errors))

# ── run config (model / split knobs) ──────────────────────────────────────

with cfg_box:
    st.header("② Cấu hình model")

    if st.session_state.get("target_column") not in columns:
        st.session_state.pop("target_column", None)       # dataset changed → stale choice
    target_column = st.selectbox(
        "Target/Label column",
        options=columns,
        key="target_column",
        help="Cột nhãn cần dự đoán (nhị phân / đa lớp / regression).",
    )
    entity_id_col = st.selectbox(
        "PK / entity_id (tùy chọn)",
        options=[""] + columns,
        help="Cột định danh chính của 1 dòng, vd customer_id, SK_ID_CURR. Dùng để "
             "kiểm tra khóa chính và tách OOT.",
    )
    entity_id_col = entity_id_col or None
    composite_key_cols = st.multiselect(
        "Composite keys (tùy chọn)",
        options=columns,
        help="Các cột tạo khóa ghép. Agent 3 dùng để trích cột ngày và tách OOT "
             "theo thời gian.",
    ) or None

    domain = st.selectbox(
        "Domain",
        ["generic", "credit_risk", "propensity", "fraud"],
        help="credit_risk: chấm điểm tín dụng; propensity: xu hướng mua/churn; "
             "fraud: phát hiện gian lận; generic: mặc định.",
    )
    product_type = st.selectbox(
        "Product type",
        ["generic", "consumer_unsecured", "credit_card", "mortgage", "auto",
         "overdraft", "bnpl", "sme"],
        help="Loại sản phẩm cho hướng dẫn feature engineering theo lĩnh vực.",
    )
    model_type = st.selectbox(
        "Model type",
        ["binary_classification", "regression", "multiclass"],
        key="model_type",
        help="binary_classification: target 2 giá trị (0/1, yes/no...). "
             "multiclass: 3–50 lớp (chữ hoặc số). regression: target số liên tục.",
    )
    # Probability calibration (fit on CV out-of-fold predictions, no leakage)
    calibration, calib_env = None, {}
    if model_type != "regression":
        calibration = st.toggle(
            "Hiệu chỉnh xác suất (calibration)", value=Config.CALIBRATION_ENABLED, key="calibration",
            help="Fit trên dự đoán out-of-fold của CV (không leak). Nhị phân: isotonic/sigmoid "
                 "(xác suất ≈ tỷ lệ thực tế, dùng cho PD). Multiclass: temperature (giữ lớp dự đoán, "
                 "chỉnh độ tự tin) hoặc isotonic theo từng lớp.")
        if calibration:
            if model_type == "multiclass":
                _opts = ["temperature", "ovr_isotonic"]
                _m = st.selectbox("Phương pháp hiệu chỉnh", _opts, key="calib_method_mc",
                                  index=_opts.index(Config.MULTICLASS_CALIBRATION_METHOD)
                                  if Config.MULTICLASS_CALIBRATION_METHOD in _opts else 0,
                                  help="temperature: 1 tham số, ổn định với dữ liệu nhỏ. "
                                       "ovr_isotonic: linh hoạt theo từng lớp, cần nhiều mẫu mỗi lớp.")
                calib_env = {"MULTICLASS_CALIBRATION_METHOD": _m}
            else:
                _opts = ["isotonic", "sigmoid"]
                _m = st.selectbox("Phương pháp hiệu chỉnh", _opts, key="calib_method_bin",
                                  index=_opts.index(Config.CALIBRATION_METHOD)
                                  if Config.CALIBRATION_METHOD in _opts else 0,
                                  help="isotonic: linh hoạt (chuẩn ngân hàng, cần ≥ ~1000 dòng). "
                                       "sigmoid (Platt): ổn định với dữ liệu nhỏ.")
                calib_env = {"CALIBRATION_METHOD": _m}

    target_plan, target_err = _target_preview(input_path, _mtime(input_path), target_column, model_type)
    if target_err:
        st.error(f"Target không hợp với '{model_type}': {target_err}")
    elif target_plan and target_plan.get("classes"):
        cls = target_plan["classes"]
        if target_plan["task"] == "classification":
            st.caption(f"Target nhị phân: `{cls[0]}` → 0, `{cls[1]}` → 1 (lớp dương; "
                       f"{target_plan.get('positive_rule')}).")
        else:
            st.caption(f"Target {len(cls)} lớp → mã 0..{len(cls) - 1}: "
                       + ", ".join(f"`{c}`" for c in cls[:12]) + (" …" if len(cls) > 12 else ""))

    # Each job runs in its own process; this session can run one at a time and
    # the server caps concurrent jobs (UI_MAX_CONCURRENT_JOBS).
    _my_job_running = runner.is_running([st.session_state.get("run_id")])
    _server_full = not runner.can_submit()
    train_clicked = st.button(
        "🚀 Train",
        type="primary",
        width="stretch",
        disabled=_my_job_running or _server_full,
    )
    if _server_full and not _my_job_running:
        st.caption(f"Máy chủ đang chạy {runner.running_count()}/{runner.MAX_CONCURRENT_JOBS} job — "
                   "đợi một job xong để Train.")

# ── validation ────────────────────────────────────────────────────────────

def _validate() -> list:
    errs = []
    if not _path_ok(input_path):
        errs.append("Chưa chọn file input hợp lệ.")
    elif input_path.lower().startswith(BaseAgent._S3_SCHEMES) and \
            bool(BaseAgent.s3_settings()[0]) != bool(BaseAgent.s3_settings()[1]):
        errs.append("S3: cần nhập cả Access key ID và Secret access key.")
    if not target_column:
        errs.append("Chưa chọn target column.")
    elif target_column not in columns:
        errs.append(f"Target '{target_column}' không có trong dữ liệu.")
    if target_err:
        errs.append(f"Target không hợp với model type: {target_err}")
    if entity_id_col and entity_id_col not in columns:
        errs.append(f"PK '{entity_id_col}' không có trong dữ liệu.")
    if composite_key_cols:
        bad = [c for c in composite_key_cols if c not in columns]
        if bad:
            errs.append(f"Composite keys không có trong dữ liệu: {bad}")
    if col_desc_path and col_desc_kind == "table":
        if not col_name_field:
            errs.append("Có file col-desc nhưng thiếu col_name_field.")
        if not col_desc_field:
            errs.append("Có file col-desc nhưng thiếu col_desc_field.")
    return errs


def _build_run_params() -> dict:
    return {
        "input_path": input_path,
        "target_column": target_column,
        "col_descriptions_path": col_desc_path,
        "col_descriptions_kwargs": (
            {
                "col_name_field": col_name_field,
                "col_desc_field": col_desc_field,
                "col_group_field": col_group_field or None,
            }
            if col_desc_path and col_desc_kind != "mapping" else None
        ),
        "entity_id_col": entity_id_col,
        "composite_key_cols": composite_key_cols,
        "valid_path": None,
        "oot_path": None,
        "train_sample_ratio": None,
        "prefilter": True,
        "check_distribution": True,
        "domain": domain,
        "model_type": model_type,
        "product_type": product_type,
        "auto_config": auto_config,
        "calibration": calibration,
    }


# ── submit / poll job ─────────────────────────────────────────────────────

if train_clicked:
    errs = _validate()
    if errs:
        st.error("Chưa đủ thông tin để chạy:\n" + "\n".join(f"- {e}" for e in errs))
    else:
        # Parse the edited YAML once more (in case user didn't click Áp dụng).
        ok, yerr, pins = config_editor.parse_yaml(yaml_text)
        if not ok:
            st.error("Config YAML không hợp lệ:\n" + "\n".join(f"- {e}" for e in yerr))
        else:
            st.session_state["cfg_pins"] = pins
            # Session-scoped settings travel to the job process via its env only.
            # the job writes its run into this session's workspace
            job_env = {**config_editor.pins_to_env(pins), **calib_env, "OUTPUT_DIR": str(WS_DIR)}
            if input_path.lower().startswith(BaseAgent._S3_SCHEMES):
                job_env.update(_session_s3())
            run_id = runner.submit_job(_build_run_params(), env_overrides=job_env)
            st.session_state["run_id"] = run_id
            st.session_state["train_errs"] = None

# ── show job progress / results ───────────────────────────────────────────

run_id = st.session_state.get("run_id")
if run_id:
    job = runner.get_status(run_id)
    if job:
        if job["status"] == "running":
            with st.status("Pipeline đang chạy…", expanded=True):
                st.write("Đang train trong tiến trình riêng (có thể mất vài phút đến vài giờ). "
                         "Bạn có thể xem run cũ hoặc đổi dataset trong lúc chờ.")
                if job.get("run_dir"):
                    st.caption(f"Run dir: `{job['run_dir']}`")
                if job.get("log"):
                    st.code(job["log"], language="text")
                if st.button("⏹ Dừng job", key=f"cancel_{run_id}"):
                    runner.cancel_job(run_id)
            time.sleep(2)
            st.rerun()

        elif job["status"] == "done":
            res = job.get("result") or {}
            if res.get("task") == "regression":
                score = f"Test R² = {res['test_r2']:.4f}" if isinstance(res.get("test_r2"), float) else ""
            else:
                k = "oot_auc" if isinstance(res.get("oot_auc"), float) else "test_auc"
                score = f"{k} = {res[k]:.4f}" if isinstance(res.get(k), float) else ""
            rdn = Path(job.get("run_dir") or "")
            st.session_state["done_msg"] = (
                f"Train thành công! Run `{rdn.parent.name}/{rdn.name}` · "
                f"model {res.get('best_model', '—')}" + (f" · {score}" if score else "")
                + ". Xem tab 🔮 Dự đoán để chạy thử model.")
            st.session_state["run_dir"] = job.get("run_dir")
            st.session_state["run_id"] = None  # done — stop polling
            st.rerun()

        else:  # error
            st.error("Pipeline gặp lỗi.")
            st.code(job.get("error", ""), language="text")
            if job.get("traceback"):
                with st.expander("Traceback"):
                    st.code(job["traceback"], language="text")
            st.session_state["run_id"] = None