# Loại bài toán, baseline & benchmark, đánh giá sau huấn luyện

Tài liệu này mô tả phần **đánh giá model** của Agent 3 cho cả 3 loại bài toán:
mã hoá target, chỉ số chọn model, hiệu chỉnh xác suất (calibration), bộ
baseline & benchmark, 7 chart chẩn đoán và checklist ra quyết định.

Code liên quan:

| File | Vai trò |
|---|---|
| [`target_encoding.py`](../target_encoding.py) | Mã hoá target (một lần, trước khi chia dữ liệu) + giải mã nhãn |
| [`Agents/TrainModel/agent_train_model.py`](../Agents/TrainModel/agent_train_model.py) | Chọn model / feature theo task, calibration, `_tool_model_benchmark` |
| [`Agents/TrainModel/regression_benchmark.py`](../Agents/TrainModel/regression_benchmark.py) | Benchmark + chart + checklist cho regression |
| [`Agents/TrainModel/classification_benchmark.py`](../Agents/TrainModel/classification_benchmark.py) | … cho phân loại nhị phân |
| [`Agents/TrainModel/multiclass_benchmark.py`](../Agents/TrainModel/multiclass_benchmark.py) | … cho multiclass |
| [`model_scoring.py`](../model_scoring.py) | Dự đoán dùng chung cho UI + replay (theo task, có calibration, giải mã nhãn) |

---

## 1. Ba loại bài toán

Chọn bằng `model_type` (UI: ô *Model type*; CLI: `--model-type`).

| `model_type` | Target hợp lệ | Chỉ số chọn model / feature | Metric báo cáo |
|---|---|---|---|
| `binary_classification` | đúng 2 giá trị (0/1, yes/no, Y/N, Good/Bad, …) | AUC | AUC, Gini, KS, PR-AUC, LogLoss, Brier |
| `multiclass` | 3–50 nhãn (chữ hoặc số) | **macro one-vs-rest AUC** | macro AUC, accuracy, balanced accuracy, macro / weighted F1, LogLoss |
| `regression` | số liên tục | R² | MAE, RMSE, R², MAPE |

Mọi bước tìm kiếm (FLAML, Optuna, xếp hạng, prune, overfit check) đều tối ưu chỉ số
ở cột 3. Với multiclass, macro AUC vẫn được lưu dưới các key `*_auc` nên toàn bộ luồng
không phải đổi tên.

### Mã hoá target (`target_encoding.py`)

Mọi bước downstream đọc target bằng `pd.to_numeric`; một target dạng chữ ("yes"/"no")
trước đây sẽ thành NaN. Pipeline vì vậy mã hoá target **một lần, trước khi chia dữ liệu**,
rồi đóng băng mapping:

| Task | Quy tắc |
|---|---|
| binary | 2 nhãn → 0/1. Lớp dương (=1) chọn theo thứ tự: target số → giá trị **lớn hơn** (0/1, 1/2); nhận diện từ khoá (`yes, y, true, bad, default, churn, fraud, 1` …); còn lại → **lớp thiểu số** |
| multiclass | k nhãn → 0..k-1 (sắp theo số nếu là số, theo chữ nếu là chữ). Tối đa 50 lớp |
| regression | phải là số (≤ 1% giá trị không parse được), nếu không báo lỗi rõ ràng |

- Mapping lưu ở `run_dir/target_mapping.json`, trong `final_model.pkl` (`target_mapping`)
  và `replay_manifest.json` → dự đoán được giải mã về **nhãn gốc**.
- Nhãn chỉ xuất hiện ở valid/oot mà không có trong train → target = NaN + cảnh báo
  `target_unseen_label` trong `data_quality_report.json`.
- Dòng **thiếu target** bị Agent 3 loại trước khi train (log số dòng theo từng split) —
  XGBoost/LightGBM không nhận nhãn NaN.
- Chọn sai loại bài toán (vd binary cho target 5 lớp, regression cho target chữ) → pipeline
  dừng ngay với thông báo; UI hiện lỗi đỏ và khoá nút Train trước khi chạy.

---

## 2. Multiclass trong Agent 3

| Bước | Hành vi |
|---|---|
| FLAML | `metric="roc_auc_ovr"` |
| Optuna / early stopping | LightGBM `multi_logloss`, XGBoost `mlogloss`; trial chấm bằng macro AUC |
| Trọng số lớp | lệch ≥ `AUTO_CLASS_WEIGHT_THRESHOLD` → `class_weight="balanced"` (LightGBM / RF / ExtraTrees) |
| RFE (nếu bật) | `scoring="roc_auc_ovr"` |
| Stability | Gini one-vs-rest trung bình theo kỳ |
| SHAP | trung bình \|SHAP\| qua các lớp (beeswarm chỉ thể hiện độ lớn) |
| Model cuối | OOF là ma trận xác suất; metric CV: macro AUC, accuracy, macro F1, logloss |
| Chart model triển khai | ROC/KS/Lift/decile là khái niệm nhị phân → dùng 7 chart benchmark multiclass |
| Inference | `final_model_code.py`: `predict_proba` (n × k), `predict` trả **nhãn gốc** |

---

## 3. Hiệu chỉnh xác suất (calibration)

Fit trên **dự đoán out-of-fold của CV** trên train+valid (không leak — mỗi dòng được chấm
bởi model chưa từng thấy nó). Bật/tắt chung bằng `CALIBRATION_ENABLED` (mặc định bật),
ghi đè từng run bằng UI (ô *Hiệu chỉnh xác suất*) hoặc CLI `--calibration on|off`.

| Task | Phương pháp | Chọn bằng |
|---|---|---|
| binary | `isotonic` (mặc định, cần ≥ ~1000 dòng) · `sigmoid` (Platt, dữ liệu nhỏ) | `CALIBRATION_METHOD` / UI / `--calibration-method` |
| multiclass | `temperature` (mặc định: 1 tham số T trên log-xác suất; **giữ nguyên lớp dự đoán**, chỉ chỉnh độ tự tin) · `ovr_isotonic` (isotonic từng lớp rồi chuẩn hoá tổng = 1; linh hoạt hơn, cần nhiều mẫu mỗi lớp) | `MULTICLASS_CALIBRATION_METHOD` / UI / `--calibration-method` |
| regression | không áp dụng | — |

- Multiclass chỉ áp dụng khi log-loss OOF giảm ≥ `MULTICLASS_CALIBRATION_MIN_GAIN`
  (mặc định 0.5%) — nếu model đã gần chuẩn, một ánh xạ gần đồng nhất chỉ thêm nhiễu.
  Log ghi rõ "NOT applied … gain x% < 0.5%" kèm T đã fit.
- Calibrator multiclass lưu dạng dict thuần (+ `IsotonicRegression` của sklearn) → artifact
  load được không cần repo, `final_model_code.py` tự áp dụng.
- Metric từng split báo cáo sau calibration; log-loss trước calibration ở `*_logloss_raw`.
- Binary: xác suất sau calibration bị chặn trong `[PD_FLOOR, PD_CAP]`.

---

## 4. Baseline & benchmark

Sau khi train model cuối, Agent 3 fit một **thang model** trên **cùng tập train**, chấm
**cùng metric** trên Validation và OOT/Test:

| Task | Thang model |
|---|---|
| regression | Mean baseline → Median baseline → Linear Regression → Ridge → Random Forest → LightGBM → XGBoost → *Pipeline tuned* |
| binary | Prior baseline → Logistic Regression → Random Forest → LightGBM → XGBoost → *Pipeline tuned* |
| multiclass | Prior baseline → Logistic Regression (multinomial) → Random Forest → LightGBM → XGBoost → *Pipeline tuned* |

- Tất cả fit **chỉ trên train** → Validation là hold-out thật. *Pipeline tuned* = model +
  hyperparameter pipeline đã chọn, cũng fit trên train để so sánh công bằng. (Model triển
  khai được refit trên train+valid nên valid của nó là in-sample — không dùng cho chart.)
- Model benchmark dùng tham số mặc định hợp lý (không tune).
- **Linear / Ridge / Logistic**: biến category được **one-hot** (gom nhóm hiếm < 0.5%, tối đa
  50 mức, mức lạ bỏ qua); biến số median-impute + chuẩn hoá. Cây dùng mã nhãn gốc.
- Ngân sách thời gian: `BENCHMARK_TIME_BUDGET` (auto-config tự co giãn); model vượt ngân
  sách được ghi "skipped". Bật/tắt: `REG_BENCHMARK_ENABLED`, `CLF_BENCHMARK_ENABLED`
  (áp dụng cho cả binary và multiclass).

**Khuyến nghị model**: model có metric tốt nhất trên Validation; nếu model tuyến tính
(Linear/Ridge cho regression, Logistic cho phân loại) kém ≤ 3% MAE (regression) hoặc
≤ 0.01 AUC (phân loại) thì gợi ý ưu tiên model tuyến tính vì dễ giải thích.

**Output** trong run dir:

| File | Nội dung |
|---|---|
| `<task>_benchmark.json` | bảng metric mọi model × split, chẩn đoán, importance, findings, khuyến nghị |
| `<task>_benchmark_report.md` | báo cáo markdown: bảng so sánh, chart + cách đọc, checklist (cũng nhúng vào `final_report.md`) |
| `charts/reg_*.png` / `clf_*.png` / `mc_*.png` | 7 chart (mỗi chart có bản Validation và OOT/Test khi áp dụng) |

---

## 5. Bảy chart chẩn đoán

Chart 3–7 dùng bản *Pipeline tuned* fit trên train. UI có mục **"Cách đọc chart"** dưới
mỗi chart; nội dung giống cột cuối.

### Regression

| # | Chart | Đọc nhanh |
|---|---|---|
| 1 | Model comparison (MAE / RMSE / R²) | MAE, RMSE thấp; R² cao (có thể âm) |
| 2 | Train vs Validation MAE | gần nhau = tốt; Train ≪ Valid = overfit / leakage / drift |
| 3 | Actual vs Predicted | gần đường y = x; xa dần ở giá trị lớn = yếu nhóm cao |
| 4 | Residual vs Predicted (+ đường trung bình trượt) | ngẫu nhiên quanh 0; hình phễu / cong = sai số theo quy mô / phi tuyến |
| 5 | Residual distribution | tâm lệch 0 = bias; đuôi dài = lỗi cực lớn |
| 6 | MAE theo nhóm target (≤ 5 phân vị) + sai số tương đối | MAE tăng ở nhóm cao → xem log1p / loss khác |
| 7 | Permutation importance (MAE tăng khi đảo feature) | dương lớn = quan trọng; âm = nhiễu |

### Binary

| # | Chart | Đọc nhanh |
|---|---|---|
| 1 | Model comparison (AUC / KS / Brier) | AUC, KS cao; Brier thấp; baseline AUC = 0.5 |
| 2 | Train vs Validation AUC | chênh lớn = overfit / leakage |
| 3 | ROC curves (mọi model) | sát góc trên-trái |
| 4 | Precision-Recall curves | quan trọng khi tỷ lệ Bad thấp |
| 5 | Calibration (reliability) | trên đường chéo = xác suất đáng tin |
| 6 | Bad rate theo decile điểm số | tăng đơn điệu = rank-ordering tốt |
| 7 | Permutation importance (AUC giảm) | như trên |

Model triển khai (refit, đã calibrate) vẫn giữ 9 chart riêng: ROC, PR, KS, Lift, Gain,
bad-rate / avg-score decile, calibration, score distribution.

### Multiclass

| # | Chart | Đọc nhanh |
|---|---|---|
| 1 | Model comparison (macro AUC / Accuracy / Macro F1) | accuracy của Prior baseline = tỷ lệ lớp đa số |
| 2 | Train vs Validation macro AUC | chênh lớn = overfit |
| 3 | Confusion matrix (chuẩn hoá theo lớp thực tế) | đường chéo = recall; ô ngoài chéo đậm = cặp lớp hay nhầm |
| 4 | Precision / Recall / F1 theo từng lớp (kèm số mẫu) | lớp thấp = model chưa học được (hay là lớp hiếm) |
| 5 | ROC one-vs-rest theo từng lớp | lớp sát đường chéo = không nhận diện được |
| 6 | Độ tin cậy dự đoán (top-label) | dưới đường chéo = quá tự tin |
| 7 | Permutation importance (macro AUC giảm) | như trên |

---

## 6. Checklist đánh giá & khuyến nghị

Mỗi mục có mức `OK` / `LƯU Ý` / `CẢNH BÁO` / `NGHIÊM TRỌNG` + hành động đề xuất.
**Ngưỡng là heuristic** (đã chấp nhận tạm, duyệt lại với nghiệp vụ khi cần).

### Regression

| Kiểm tra | Ngưỡng | Hành động |
|---|---|---|
| So với Median baseline | MAE không giảm → nghiêm trọng; giảm < 5% → cảnh báo | xem lại feature, target, cách chia |
| Model có R² < 0 | — | loại model đó |
| Train vs Validation | Valid > 3× Train → cảnh báo; 1.5–3× → lưu ý; Train ≈ baseline → underfit | regularization / leakage / thêm feature |
| Sai số tăng theo target | MAE nhóm cao > 2× nhóm thấp | phân tích nhóm lớn, log1p, loss khác |
| RMSE cao so với MAE | RMSE / MAE > 1.5 (đếm \|residual\| > 3×MAE) | kiểm tra lỗi cực lớn |
| Validation vs OOT/Test | holdout MAE > 1.2× Valid | drift / ổn định theo thời gian |
| Bias | \|mean residual\| > 10% MAE | hiệu chỉnh bias / đổi loss |
| Importance | 1 feature > 50% tổng importance → cảnh báo leakage; ≥ 5/20 âm → lưu ý | kiểm tra leakage, loại biến nhiễu |

### Binary

| Kiểm tra | Ngưỡng |
|---|---|
| Ít Bad trên Validation / OOT | < 30 quan sát Bad |
| Khả năng phân biệt | AUC ≤ 0.55 nghiêm trọng; < 0.65 yếu |
| Train vs Validation | chênh AUC > 0.10 cảnh báo; 0.05–0.10 lưu ý |
| Validation vs OOT | AUC giảm > 0.03 |
| Xác suất vs tỷ lệ Bad | lệch > 20%; Brier không tốt hơn baseline → nghiêm trọng |
| Rank-ordering decile | số lần đảo chiều > ¼ số decile |
| Mất cân bằng | tỷ lệ Bad < 5% → lưu ý dùng PR-AUC / KS |
| Importance | như regression |

### Multiclass

| Kiểm tra | Ngưỡng |
|---|---|
| Lớp ít mẫu | < 30 mẫu trên Validation / OOT, hoặc không có mẫu |
| Khả năng phân biệt | macro AUC ≤ 0.55 / < 0.65 |
| Accuracy ≈ lớp đa số | accuracy ≤ Prior baseline + 0.01 |
| Train vs Validation, Validation vs OOT | như binary (theo macro AUC) |
| Lớp không nhận diện được | recall < 0.2 hoặc F1 < 0.3 |
| Cặp lớp hay nhầm | > 30% mẫu một lớp bị đoán thành lớp khác |
| Quá / thiếu tự tin | \|xác suất top-class trung bình − accuracy\| > 0.1 |
| Mất cân bằng | lớp nhỏ nhất < 5% train |
| Importance | như regression |

---

## 7. Xem kết quả ở đâu

| Nơi | Nội dung |
|---|---|
| UI → tab **Model results** | metric chính theo task, bảng benchmark (Validation / OOT-Test), khuyến nghị model, từng finding, checklist ra quyết định |
| UI → tab **Charts** | 7 chart (Validation và OOT/Test cạnh nhau) + "Cách đọc chart"; binary có thêm 9 chart model triển khai |
| CLI (`main.py`, `test_full_pipeline.py`) | metric theo task, bảng benchmark, model khuyến nghị, checklist `[OK]/[WARN]/[BAD]/[INFO]` |
| `final_report.md` | toàn bộ báo cáo benchmark nhúng sẵn |
