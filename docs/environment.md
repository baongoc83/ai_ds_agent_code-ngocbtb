# Môi trường: cài một lần, chạy đúng mọi nơi

Pickle (`final_model.pkl`, `cleaning_spec.pkl`, `feature_spec.pkl`) và booster nhạy với
phiên bản thư viện; `replay_pipeline.py` và `final_model_code.py` từ chối chạy khi lệch.
Vì vậy môi trường được khoá chính xác.

## Các file

| File | Vai trò |
|---|---|
| `.python-version` | Python **3.12** |
| `requirements.txt` | thư viện code import trực tiếp, pin chính xác (kể cả `streamlit==1.49.1`) |
| `requirements-lock.txt` | **mọi** package (trực tiếp + gián tiếp, 84 package) pin theo môi trường tham chiếu — *sinh tự động* |
| `requirements-dev.txt` | công cụ test (`pytest`), cài thêm trên lock |
| `requirements-proxy.txt` | chỉ khi host LiteLLM proxy trên máy — cài vào venv **riêng** |
| `setup_env.ps1` / `setup_env.sh` | tạo `.venv`, cài lock (+ dev), kiểm tra |
| `tools/check_env.py` | so interpreter đang chạy với lock |
| `tools/freeze_requirements.py` | sinh lại lock từ môi trường tham chiếu |

## Cài đặt

```powershell
# Windows
.\setup_env.ps1          # runtime
.\setup_env.ps1 -Dev     # + pytest
```

```bash
# Linux / macOS
./setup_env.sh
./setup_env.sh --dev
```

Script: tạo `.venv` bằng Python trong `.python-version` → `pip install -r requirements-lock.txt`
(+ dev) → chạy `tools/check_env.py`. Sau đó **luôn** dùng interpreter của venv:

```powershell
.\.venv\Scripts\python.exe main.py data/train.csv TARGET
.\.venv\Scripts\python.exe -m streamlit run ui/app.py
.\.venv\Scripts\python.exe -m pytest tests -q
```

## Kiểm tra môi trường

```powershell
python tools/check_env.py          # Python + 84 package của lock
python tools/check_env.py --dev    # + pytest
```

In `Environment OK …` hoặc danh sách lệch (`need X, installed Y`) kèm cách sửa.

## Nâng cấp một thư viện

1. Đổi version trong `requirements.txt`, cài vào môi trường tham chiếu, chạy test.
2. `py -3.12 tools/freeze_requirements.py` → sinh lại `requirements-lock.txt`.
3. `py -3.12 tools/freeze_requirements.py --check` (CI) báo lock đã cũ hay chưa.
4. Commit cả hai file. Run cũ train bằng phiên bản trước vẫn replay được trong môi trường
   cũ, hoặc với `--allow-version-mismatch` (kết quả có thể lệch).

## Ghi chú

- Lock được sinh trên Windows; mục riêng Windows duy nhất (`colorama`) là thuần Python,
  cài trên Linux không sao.
- `zstandard` không nằm trong lock — chỉ cần khi đọc file `.zst`.
- LiteLLM proxy kéo ~80 package và pin cứng `pydantic` → cài vào `.venv-proxy` riêng
  (`requirements-proxy.txt`), pipeline không import `litellm`.
