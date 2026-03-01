# MODULE ML — YOUTUBE VIRAL VIDEO PREDICTION

## 1. Tổng quan

Module `ml/` là một **end-to-end binary classification pipeline** được tích hợp trực tiếp vào hệ thống, dự đoán khả năng viral của video YouTube. Bài toán cốt lõi: **Cho một video bất kỳ, dự đoán xác suất video đó trở nên viral (nhãn `is_viral = 1/0`)**, dựa trên metadata, engagement metrics và thống kê kênh.

**Mục tiêu nghiệp vụ:**
- Giúp content creators ra quyết định trước khi đăng bài
- Phân tích điểm mạnh / yếu của từng video qua SHAP explanation
- Dự đoán không chỉ đưa ra nhãn mà còn **giải thích lý do** bằng tiếng Việt

**Cấu trúc thư mục:**
```
ml/
├── __init__.py
├── config.py          # Cấu hình GCP / BigQuery
├── data_loader.py     # Pull dữ liệu từ BigQuery
├── label.py           # Định nghĩa nhãn is_viral
├── features.py        # Feature engineering (36 features)
├── train.py           # Training pipeline + Optuna HPO
├── predict.py         # Production inference + SHAP
├── save_load.py       # Model persistence (JSON + YAML)
└── test_predict.py    # Interactive CLI test end-to-end

models/
├── xgb_viral_v1.json          # Trained XGBoost model
└── feature_config_v1.yaml     # Training config + medians + encoders

plots/                         # ROC curve, PR curve, feature importance
```

---

## 2. Workflow tổng thể

```
BigQuery Intermediate Layer
         │
         │  (3 bảng: int_videos__enhanced,
         │            int_engagement_metrics,
         │            int_channel_summary)
         ▼
┌─────────────────────┐
│   data_loader.py    │  Pull data từ BQ → 3 DataFrames
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     label.py        │  EDA + định nghĩa nhãn is_viral
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    features.py      │  Merge sources + engineer 36 features
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│     train.py        │  Time-split → Optuna HPO → XGBoost → Evaluate
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   save_load.py      │  Lưu model.json + config.yaml
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    predict.py       │  Load model → Inference → SHAP explanation
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  test_predict.py    │  Interactive CLI: nhập tên kênh → dự đoán
└─────────────────────┘
```

---

## 3. Chi tiết từng module

### 3.1. `config.py` — Cấu hình

Quản lý toàn bộ biến môi trường và hằng số cho ML module:

```python
# Kết nối BigQuery
GCP_PROJECT_ID          = "project-8fd99edc-9e20-4b82-b43"
CREDENTIALS_PATH        = "credentials/...json"
BQ_INTERMEDIATE_DATASET = "intermediate"
BQ_LOCATION             = "asia-southeast1"

# 3 bảng nguồn
INT_VIDEOS_ENHANCED_TABLE    = "int_videos__enhanced"
INT_ENGAGEMENT_METRICS_TABLE = "int_engagement_metrics"
INT_CHANNEL_SUMMARY_TABLE    = "int_channel_summary"
```

Tự động resolve relative path của credentials thành absolute path, đảm bảo hoạt động đúng bất kể cwd.

---

### 3.2. `data_loader.py` — Tải dữ liệu từ BigQuery

**Mục tiêu:** Kết nối BigQuery và pull 3 bảng Intermediate layer về Python DataFrame.

| Hàm | Nguồn BQ | Dữ liệu trả về |
|-----|----------|----------------|
| `load_videos_enhanced()` | `int_videos__enhanced` | Metadata video: title, duration, tags, published_at, view_count, like_count, channel_subscribers… |
| `load_engagement_metrics()` | `int_engagement_metrics` | KPIs: engagement_score, like_rate_pct, comment_rate_pct, avg_views_per_day, is_potentially_viral… |
| `load_channel_summary()` | `int_channel_summary` | Stats kênh: subscriber_count, avg_views_per_video, avg_like_rate_pct, upload_freq… |
| `load_all_intermediate_data()` | Cả 3 bảng | `dict{"videos", "engagement", "channels"}` |

Sau khi load, tự động in **DATA LOAD SUMMARY** gồm số lượng videos/channels và phân phối nhãn viral.

**Authentication:** Sử dụng `google.oauth2.service_account.Credentials` với scope `bigquery` read-only.

---

### 3.3. `label.py` — Định nghĩa nhãn viral

**Vấn đề:** YouTube API không cung cấp nhãn "viral" trực tiếp. Cần tự định nghĩa từ các metrics có sẵn, đảm bảo tỷ lệ viral **8–20%** để tránh class imbalance cực đoan.

**Hai chức năng chính:**

**`analyze_label_candidates()` — EDA trước khi gán nhãn:**
- Kiểm tra `is_potentially_viral` có sẵn từ BQ
- Phân tích phân phối `view_count`, `avg_views_per_day`, phân vị 75th/90th/95th/99th
- Đánh giá: nếu viral rate nằm ngoài 8–20% → cần định nghĩa lại
- Render matplotlib charts cho visualization

**`define_viral_label()` — Gán nhãn thực sự:**
- **Multi-strategy**: Kết hợp nhiều điều kiện:
  - `view_ratio`: Tỷ lệ view_count so với trung bình kênh
  - `velocity_score`: Tốc độ tăng view so với subscriber
  - `engagement_score`: Ngưỡng tương tác tổng thể
- Lưu cột `label_strategy` để audit từng video được gán nhãn theo cách nào

---

### 3.4. `features.py` — Feature Engineering

**Mục tiêu:** Biến đổi raw data thành feature matrix 36 chiều nhất quán giữa training và production inference.

**Bước 1 — `merge_all_sources()`:**

Merge 4 DataFrame thành 1 wide DataFrame theo thứ tự:
```
df_labeled (base)
  ← LEFT JOIN df_videos       ON video_id
  ← LEFT JOIN df_engagement   ON video_id
  ← LEFT JOIN df_channels     ON channel_id
```
Tự động resolve duplicate columns từ multiple joins.

**Bước 2 — `engineer_features()`:**

36 features chia 6 nhóm:

| Nhóm | Số features | Ví dụ |
|------|-------------|-------|
| **A — Temporal** | 8 | `published_hour`, `is_weekend`, `is_prime_time` (18–22h), `publish_quarter` |
| **B — Title** | 7 | `title_length`, `title_word_count`, `has_number`, `has_emoji`, `has_caps_word` |
| **C — Content** | 9 | `duration_minutes`, `tag_count`, `is_hd`, `has_caption`, `is_shorts`, `category_id_enc` |
| **D — Channel** | 8 | `subscriber_log`, `channel_age_days`, `upload_freq_per_day`, `channel_avg_views_log` |
| **E — Engagement** | 5 | `avg_views_per_day_log`, `engagement_level_enc`, `like_rate_pct`, `engagement_score` |
| **F — Interaction** | 4 | `view_vs_channel_avg`, `like_rate_vs_channel`, `velocity_score` |

**Các kỹ thuật xử lý:**
- **Log-transform**: Áp dụng cho skewed features (`subscriber_log`, `avg_views_per_day_log`, `channel_avg_views_log`) — giảm ảnh hưởng của outliers
- **LabelEncoder**: Cho categorical (`engagement_level`, `category_id`)
- **Missing value imputation**: Median cho numeric features; fill `0` cho 17 binary features
- **Interaction features**: Tạo features so sánh relative (video vs channel baseline)

**Bước 3 — `fill_missing()`:**

Áp dụng `train_medians` từ config (không tính lại trên dữ liệu mới), đảm bảo **train-serving consistency**.

---

### 3.5. `train.py` — Training Pipeline

**Mục tiêu:** Train XGBoost classifier với Optuna HPO, đánh giá toàn diện trên test set.

**Bước 1 — Time-based Split (không random shuffle):**

```python
# Sort bằng published_at → không có temporal leakage
Train: 70%  (videos cũ nhất)
Val  : 15%  (videos tiếp theo)
Test : 15%  (videos mới nhất)
```

> Tại sao quan trọng: Video tương lai có engagement metrics cao hơn (vì đã tích lũy lâu hơn). Nếu dùng random split, model sẽ bị train trên "future data" → performance ảo.

**Bước 2 — Xử lý class imbalance:**

```python
scale_pos_weight = n_negative / n_positive
# Ví dụ: 1000 non-viral / 120 viral → scale_pos_weight = 8.33
```

**Bước 3 — Hyperparameter Optimization với Optuna:**

- Objective: tối đa hóa **PR-AUC trên validation set**
  (PR-AUC phù hợp hơn ROC-AUC cho imbalanced data)
- Search space:

| Hyperparameter | Range |
|----------------|-------|
| `n_estimators` | 200–1000 |
| `max_depth` | 3–8 |
| `learning_rate` | 0.01–0.3 |
| `subsample` | 0.6–1.0 |
| `colsample_bytree` | 0.6–1.0 |
| `min_child_weight` | 1–10 |
| `gamma` | 0–5 |
| `reg_alpha` | 0–10 |
| `reg_lambda` | 0–10 |

**Bước 4 — Optimal Threshold Tuning:**

Không dùng mặc định 0.5. Tính threshold tối ưu F1-score trên **validation set**, sau đó áp dụng vào test set và production.

**Bước 5 — Evaluation trên Test Set:**

Metrics báo cáo đầy đủ:
- ROC-AUC, PR-AUC (ranking quality)
- Precision, Recall, F1 tại optimal threshold
- Confusion matrix

**Plots xuất ra `plots/`:**
- ROC curve
- Precision-Recall curve
- Feature importance (top 20)
- Confusion matrix heatmap

---

### 3.6. `save_load.py` — Model Persistence

**Mục tiêu:** Lưu/load model kèm đầy đủ metadata theo version, đảm bảo reproducibility.

**Files lưu:**

| File | Format | Nội dung |
|------|--------|----------|
| `models/xgb_viral_{version}.json` | XGBoost native JSON | Trained model weights, booster structure |
| `models/feature_config_{version}.yaml` | YAML UTF-8 | Tất cả metadata cần cho inference |

**Config YAML chứa:**
```yaml
feature_list:        [36 feature names theo đúng thứ tự training]
train_medians:       {feature: median_value, ...}   # Dùng fill missing ở inference
label_encoders:      {feature: {class: index, ...}} # LabelEncoder state
viral_threshold:     0.xx        # Ngưỡng is_potentially_viral
optimal_threshold:   0.xx        # Optimal F1 threshold
viral_rate_train:    0.xx        # Tỷ lệ viral trong training set
scale_pos_weight:    x.xx        # Imbalance ratio
trained_at:          "2026-..."  # Timestamp
model_version:       "v1"
metrics:
  roc_auc:     0.xx
  pr_auc:      0.xx
  f1:          0.xx
  precision:   0.xx
  recall:      0.xx
```

Thiết kế này đảm bảo: **deploy model mới chỉ cần copy 2 files**, không cần refit encoder hay tính lại medians.

**API:**
```python
save_model(model, config, version="v1")   # Lưu
model, config = load_model(version="v1")  # Load
list_saved_models()                        # Liệt kê
```

---

### 3.7. `predict.py` — Production Inference

**Mục tiêu:** Nhận raw YouTube data → trả về prediction + SHAP explanation, không cần data từ BigQuery.

**Pipeline inference:**
```
Raw video dict (từ YouTube API)
    ↓
engineer_features() + fill_missing(train_medians)   # Identical với training
    ↓
model.predict_proba()                               # Viral probability
    ↓
compare vs optimal_threshold                        # 0/1 label
    ↓
shap.TreeExplainer().shap_values()                  # Feature attribution
    ↓
Structured output dict
```

**Output cấu trúc (dict):**

```python
{
    "video_id":           "abc123",
    "viral_probability":  0.82,
    "is_viral_predicted": True,
    "confidence_level":   "HIGH",       # HIGH (>0.7) / MEDIUM / LOW
    "shap_top_positive":  [             # Top 5 features đẩy viral lên
        {"feature": "velocity_score", "shap": 0.34, "value": 2.1,
         "meaning": "Toc do tang view so voi subscriber"},
        ...
    ],
    "shap_top_negative":  [             # Top 5 features kéo viral xuống
        {"feature": "channel_age_days", "shap": -0.15, ...},
        ...
    ],
    "recommendation":     "Video co toc do tang view tot. ...",
    "feature_meanings":   {             # Vietnamese desc cho 40+ features
        "velocity_score": "Toc do tang view so voi subscriber",
        "is_prime_time":  "Dang video trong gio vang (18-22h)",
        ...
    }
}
```

**Mapping feature → nghĩa tiếng Việt:** 40+ features đều có description đầy đủ trong `FEATURE_MEANING` dict, phục vụ hiển thị UI.

---

### 3.8. `test_predict.py` — Interactive CLI Test

**Mục tiêu:** Demo end-to-end từ tên kênh → prediction, không cần viết code.

**Cách chạy:**
```bash
python test_predict.py
python test_predict.py --verbose    # Bật INFO logs
```

**Flow tương tác:**
```
1. Nhập tên kênh: "MrBeast"
2. YouTube API search → Danh sách kênh khớp (tối đa 5)
3. Người dùng chọn kênh
4. Lấy video mới nhất của kênh
5. Gọi predict.py → hiển thị:
   - Viral probability + confidence level
   - Top 5 features giúp viral
   - Top 5 features cản viral
   - Gợi ý cải thiện bằng tiếng Việt
```

---

## 4. Models đã train

| File | Mô tả |
|------|-------|
| `models/xgb_viral_v1.json` | XGBoost model version 1, trained trên toàn bộ data Intermediate layer |
| `models/feature_config_v1.yaml` | Config đi kèm: 36 features, train medians, label encoders, optimal threshold, metrics |

---

## 5. Các điểm kỹ thuật nổi bật

| Điểm | Chi tiết |
|------|----------|
| **No temporal leakage** | Time-based split theo `published_at`, không random shuffle |
| **Imbalance handling** | `scale_pos_weight` tự động + PR-AUC làm tuning objective |
| **Threshold tuning** | Optimal F1 threshold trên validation set, không dùng 0.5 mặc định |
| **Train-serving consistency** | `train_medians` + `label_encoders` lưu trong config YAML, dùng lại khi inference |
| **Explainability** | SHAP TreeExplainer cho mọi prediction — không phải black box |
| **Versioned artifacts** | Model + config theo version (v1, v2, ...), hỗ trợ A/B testing |
| **Bilingual output** | Feature meanings + recommendation bằng tiếng Việt |
| **Standalone modules** | Mỗi file có thể chạy độc lập: `python -m ml.data_loader`, `python -m ml.train`, ... |

---

## 6. Hướng mở rộng

1. **Multi-label prediction**: Không chỉ viral/non-viral, thêm nhãn "trending", "evergreen"
2. **Time-series features**: Đưa velocity curve (tốc độ tăng theo ngày) vào model
3. **NLP features**: Phân tích sentiment title/description với transformer
4. **Thumbnail analysis**: Vision API để extract visual features
5. **Online learning**: Cập nhật model incremental khi có batch data mới
6. **Model monitoring**: Drift detection khi distribution thay đổi theo thời gian
7. **Serving API**: Expose predict endpoint qua FastAPI để Streamlit dashboard gọi trực tiếp

---

**Document Version**: 1.0  
**Last Updated**: February 27, 2026  
**Author**: YouTube Analytics Pipeline Team  
**License**: MIT
