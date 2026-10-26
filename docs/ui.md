# Giao diện Web (Streamlit)

```powershell
.\.venv\Scripts\python.exe -m streamlit run ui/app.py      # Linux: .venv/bin/python -m streamlit run ui/app.py
```

Luôn chạy bằng interpreter của `.venv` (xem [environment.md](environment.md)). Gọi lệnh
`streamlit` trần có thể trỏ sang một Python khác thiếu thư viện.

> **Sau khi cập nhật code ngoài `ui/app.py`** (Agents, pipeline, config, …) phải **khởi
> động lại server**: Streamlit chỉ nạp lại `ui/app.py`, các module đã import vẫn là bản cũ
> trong bộ nhớ (lỗi kiểu `has no attribute …`). Job train không bị ảnh hưởng — mỗi job là
> một process mới, luôn nạp code mới nhất.

Code: [`ui/app.py`](../ui/app.py) (trang chính), [`ui/render.py`](../ui/render.py) (các tab
kết quả), [`ui/runner.py`](../ui/runner.py) + [`ui/job_worker.py`](../ui/job_worker.py)
(chạy job), [`ui/config_editor.py`](../ui/config_editor.py), [`ui/data_view.py`](../ui/data_view.py),
[`ui/scoring.py`](../ui/scoring.py), [`workspace.py`](../workspace.py).

---

## 1. Workspace (mỗi người dùng một không gian)

Ô **🗂 Workspace** ở đầu sidebar. Chọn workspace có sẵn hoặc **➕ Tạo workspace mới…**
(tên gồm chữ, số, `_ . -`). Lựa chọn được giữ trên URL (`?ws=<tên>`) nên reload / bookmark
vẫn giữ đúng workspace.

| Workspace | Thư mục |
|---|---|
| `default` | `OUTPUT_DIR/` (mọi run trước khi có workspace vẫn nằm ở đây) |
| `<tên>` | `OUTPUT_DIR/workspaces/<tên>/` |

Trong mỗi workspace: `YYYY-MM-DD/run_NN/` (các run) và `_uploads/` (file upload).
Danh sách *Xem output đã chạy*, file upload, job train và việc dọn dẹp đều **chỉ trong
workspace đang chọn**.

### Dọn dẹp (không bao giờ tự xoá)

- Có run hoặc file upload **quá 7 ngày** → hiện **cảnh báo đỏ** kèm dung lượng, danh sách
  và 3 nút: **🗑 Xóa** (xoá vĩnh viễn) · **📌 Giữ lại** (đánh dấu `.keep`, không nhắc lại) ·
  **Để sau** (ẩn trong phiên).
- **🧹 Run lỗi / dở dang**: run không tạo được báo cáo model và không ghi gì trong 6 giờ
  gần nhất → nút *Xóa tất cả run lỗi*.
- Chỉ xoá được thư mục `run_NN` và file trong `_uploads` **bên trong workspace**; không
  bao giờ đụng tới job đang chạy.
- Dòng lệnh: `python tools/workspace_cleanup.py [--workspace ws] [--delete failed|old|all] [--keep old] [--days 7]`
  (luôn hỏi xác nhận, `--yes` cho cron).

---

## 2. Nguồn dữ liệu

- **Đường dẫn local** — file trên máy chạy server, hoặc URI `s3://` · `s3a://` · `s3n://`.
- **Upload file** — lưu vào `workspace/_uploads/` (một lần cho mỗi file).

Định dạng: csv / tsv / txt (tự nhận dấu phân cách `, ; tab |`, encoding và dấu thập phân
`1,5`), parquet, feather, orc, xlsx / xls / xlsm, json / jsonl; nén `.gz .bz2 .xz .zip .zst`.

**Excel nhiều sheet** → hiện ô **Sheet Excel**: *tự động* (gộp nếu cùng cột, nếu không
lấy sheet lớn nhất), *gộp tất cả*, hoặc chọn một sheet (đường dẫn thành `book.xlsx::Sheet2`).

**S3** → hiện mục **🔑 Thông tin truy cập S3**: Access key, Secret (ẩn), Session token
(tuỳ chọn, STS/SSO), Region, Endpoint URL (MinIO…), nút **Kiểm tra kết nối**. Key chỉ
nằm trong phiên trình duyệt: không ghi ra file, không dùng chung với người khác, được
truyền riêng cho job của bạn. Để trống → dùng `S3_*` trong `.env` / AWS credential chain.

**File mô tả cột (tuỳ chọn)**: csv / tsv / txt / json / yaml / parquet / xlsx. File dạng
`tên_cột: mô_tả` không cần khai báo field; file dạng bảng thì nhập
`col_name_field` / `col_desc_field` / `col_group_field`. Nội dung hiển thị dạng YAML dưới
phần thống kê.

---

## 3. Xem dữ liệu

- **Xem nhanh dữ liệu** — 100 dòng đầu + shape thật của dataset.
- **Thống kê mô tả** — cột số: null %, distinct, min / mean / median / IQR / max, skew và
  hướng lệch; cột category: null %, count distinct, giá trị phổ biến nhất. Dataset lớn hơn
  `UI_STATS_SAMPLE_ROWS` (500k) được tính trên mẫu ngẫu nhiên.
- Mỗi file chỉ được đọc **một lần** cho mọi khung xem (cache dùng chung, tối đa
  `UI_DATA_CACHE_ENTRIES` dataset).

---

## 4. Cấu hình

### ⚙️ Config / Auto-config

Bảng mọi tham số auto-config (27 knob): nhóm, giá trị mặc định, giá trị auto (tính từ số
dòng × số cột × lớp nhỏ nhất của target — xem [config_params.md](config_params.md#auto-config--co-giãn-theo-kích-thước-dữ-liệu)),
giá trị **sẽ dùng** và **nguồn** (`auto` / `mặc định` / `ép (phiên này)` / `ép (.env server)`).

YAML bên dưới: bỏ comment dòng nào để ép giá trị đó, xoá dòng để bỏ ép → **Áp dụng
config**. Giá trị ép chỉ áp dụng cho phiên của bạn và cho job bạn bấm Train.

### ② Cấu hình model

- **Target / PK / Composite keys / Domain / Product type / Model type.**
- **Xem trước mã hoá target** ngay dưới Model type, vd *"Target nhị phân: `no` → 0, `yes` → 1"*
  hoặc *"Target 4 lớp → mã 0..3"*. Target không hợp với model type → lỗi đỏ, nút Train bị khoá.
- **Hiệu chỉnh xác suất (calibration)** (binary / multiclass) + phương pháp:
  binary `isotonic` / `sigmoid`, multiclass `temperature` / `ovr_isotonic`
  (chi tiết: [model-evaluation.md](model-evaluation.md#3-hiệu-chỉnh-xác-suất-calibration)).

---

## 5. Train

- Mỗi job chạy trong **một process riêng** với cấu hình của phiên (run dir, key S3, giá trị
  ép, calibration, workspace) → nhiều người train cùng lúc không ghi đè nhau.
- Mỗi phiên chạy 1 job một lúc; server giới hạn `UI_MAX_CONCURRENT_JOBS` (mặc định 2) job
  đồng thời — đủ thì nút Train bị khoá và hiện số job đang chạy.
- Trong lúc chạy: log trực tiếp, run dir, nút **⏹ Dừng job**. Có thể xem run cũ hoặc đổi
  dataset trong lúc chờ.
- Mỗi run lưu bản sao dữ liệu vào `run_dir/input/` (xem `INPUT_COPY_MODE`).

---

## 6. Xem kết quả

Chọn run trong **📂 Xem output đã chạy** → **Xem run này**. Phần xem nhanh / thống kê
chuyển sang **đúng dataset run đó đã dùng** (bản sao trong `run_dir/input/`). Chọn dataset
khác ở sidebar → quay về màn hình ban đầu.

| Tab | Nội dung |
|---|---|
| Preprocessing | báo cáo Agent 1 + data quality |
| Feature selection | báo cáo Agent 2 + funnel feature Agent 3 |
| Model results | metric theo task, bảng baseline & benchmark, khuyến nghị model, đánh giá & khuyến nghị, checklist |
| Charts | 7 chart benchmark (Validation / OOT-Test) + cách đọc; binary thêm chart model triển khai |
| Feature explanations | SHAP + giải thích của LLM |
| 🔮 Dự đoán | nhập tay 1 bản ghi hoặc upload file (mọi định dạng ở mục 2) → regression: giá trị; binary: xác suất + nhãn gốc; multiclass: lớp dự đoán + xác suất từng lớp; tải CSV |

---

## 7. Biến môi trường của UI

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `UI_MAX_CONCURRENT_JOBS` | `2` | số job train đồng thời trên server |
| `UI_DATA_CACHE_ENTRIES` | `2` | số dataset giữ trong bộ nhớ để xem trước |
| `UI_STATS_SAMPLE_ROWS` | `500000` | trên ngưỡng này, thống kê mô tả tính trên mẫu |
| `UI_JOB_DIR` | `%TEMP%/automl_ui_jobs` | nơi lưu `params.json` / `status.json` / `output.log` của job |
