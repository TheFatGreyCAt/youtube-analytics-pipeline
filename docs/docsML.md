## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Cấu trúc thư mục](#2-cấu-trúc-thư-mục)
3. [Quá trình phát triển từng module](#3-quá-trình-phát-triển-từng-module)
   - 3.1 [Data layer — `ml/src/data/`](#31-data-layer--mlsrcdata)
   - 3.2 [Model layer — `ml/src/models/`](#32-model-layer--mlsrcmodels)
   - 3.3 [Pipeline layer — `ml/src/pipeline/`](#33-pipeline-layer--mlsrcpipeline)
4. [Các lỗi gặp phải & cách fix](#4-các-lỗi-gặp-phải--cách-fix)
5. [Training pipeline — từng bước](#5-training-pipeline--từng-bước)
6. [Inference pipeline](#6-inference-pipeline)
7. [Testing](#7-testing)
8. [Dependencies & môi trường](#8-dependencies--môi-trường)

---

## 1. Tổng quan kiến trúc

Hệ thống dự đoán viral YouTube được chia thành **hai giai đoạn** chính:

```
Training (offline)                     Inference (online)
─────────────────────────────          ──────────────────────────────────
BigQuery (3 bảng intermediate)    →    YouTube Data API v3
         ↓                                      ↓
  Feature Engineering                   Feature Engineering
         ↓                                      ↓
  Label Creator                        VideoViralClassifier (B1 + B2)
         ↓                                      ↓
  Model B1 (viral binary)              PredictionExplainer
  Model B2 (time window)                        ↓
         ↓                              VideoReport (output)
  Lưu .pkl vào trained_models/
```

**Hai model chính:**

| Model | Loại | Mục tiêu |
|-------|------|-----------|
| **B1** | Binary Classifier | Video có viral hay không (xác suất) |
| **B2** | Multi-class Classifier | Thời gian viral: `viral_within_7d` / `viral_within_30d` / `not_viral` |

---

## 2. Cấu trúc thư mục

```
ml/
├── requirements.txt           # Dependencies: scikit-learn, pandas, numpy, google-cloud-bigquery, ...
├── __init__.py
├── cache/                     # File cache JSON cho YouTube API + quota tracking
│   ├── _quota_tracker.json
│   ├── channel_*.json
│   ├── search_*.json
│   └── videos_*.json
├── src/
│   ├── data/
│   │   ├── bigquery_loader.py     # Tải 3 bảng intermediate từ BigQuery
│   │   ├── feature_engineer.py    # Tính video features cho training & inference
│   │   └── youtube_client.py      # YouTube API v3 wrapper (quota + cache)
│   ├── models/
│   │   ├── label_creator.py       # Tạo nhãn viral từ dữ liệu BigQuery
│   │   ├── video_classifier.py    # Model B1 + B2 (GradientBoosting / LogisticRegression)
│   │   └── explainer.py           # Giải thích prediction bằng tiếng Việt
│   └── pipeline/
│       ├── viral_system.py        # Main entry point: train() và predict_video()
│       ├── polling_monitor.py     # Background thread polling video mới
│       └── report_generator.py    # Chuẩn hoá output thành VideoReport
├── tests/
│   ├── test_feature_engineer.py
│   ├── test_models.py
│   ├── test_pipeline.py
│   └── test_youtube_client.py
└── trained_models/
    ├── model_b1_viral_classifier.pkl
    ├── model_b2_timewindow_classifier.pkl
    ├── video_fe.pkl
    └── explainer.pkl
```

---

## 3. Quá trình phát triển từng module

### 3.1 Data layer — `ml/src/data/`

#### `bigquery_loader.py`

**Mục tiêu:** Load 3 bảng từ BigQuery intermediate layer về `pandas.DataFrame`.

**Thiết kế quyết định:**
- Dùng class `BigQueryLoader` thay vì hàm đơn lẻ để giữ trạng thái `_cache` in-memory, tránh query BigQuery nhiều lần trong một session.
- Tự động tìm credentials từ biến môi trường `GOOGLE_APPLICATION_CREDENTIALS` hoặc quét thư mục `credentials/*.json`.
- Validate schema (check required columns) ngay sau khi load — fail fast nếu BigQuery thay đổi schema.

**3 bảng được load:**

| Bảng | Mô tả | Cột quan trọng |
|------|--------|----------------|
| `int_channel_summary` | Thống kê tổng hợp cấp kênh | `channel_id`, `total_views`, `subscriber_count` |
| `int_engagement_metrics` | Metrics engagement từng video | `video_id`, `view_count`, `engagement_score` |
| `int_videos__enhanced` | Metadata video nâng cao | `video_id`, `duration_seconds`, `published_at` |

**Vấn đề gặp phải khi coding:** Lúc đầu dùng `pd.read_gbq()` nhưng cần cài thêm `pyarrow` và không handle được credentials tự động. Chuyển sang `google.cloud.bigquery.Client` với `service_account.Credentials` để có kiểm soát hơn.

---

#### `youtube_client.py`

**Mục tiêu:** Wrapper cho YouTube Data API v3 với quota tracking và file cache.

**Quota YouTube API:**
- Miễn phí: **10,000 units/ngày**
- Search: **100 units/call** (đắt nhất)
- Channel stats / Video list: **1 unit/call**

**Thiết kế quyết định:**

- `_FileCache`: Cache response vào file JSON với TTL theo từng loại request (`search=6h`, `channel=24h`, `videos=1h`). Tránh gọi lại API cho cùng một channel/video nhiều lần trong ngày.
- `_QuotaTracker`: Theo dõi quota đã dùng hôm nay, lưu vào `cache/_quota_tracker.json`. Reset tự động sang ngày mới.
- Raise `QuotaExceededError` khi vượt 10,000 units thay vì để API trả HTTP 403 sau này.

**Các method chính:**

| Method | Quota cost | Mô tả |
|--------|-----------|--------|
| `search_channel(name)` | 100 units | Tìm channel ID từ tên |
| `get_channel_stats(channel_id)` | 1 unit | Lấy subscribers, total_views, video_count |
| `get_recent_videos(channel_id, n)` | 1 unit / 50 videos | Lấy N video mới nhất |
| `get_video_stats(video_ids)` | 1 unit / batch 50 | Lấy views/likes/comments |

---

#### `feature_engineer.py`

**Mục tiêu:** Tính feature matrix cho video phục vụ training (từ BigQuery) và inference (từ YouTube API).

**Hai method chính:**

| Method | Nguồn dữ liệu | Dùng cho |
|--------|--------------|----------|
| `transform(engagement_df, video_df)` | BigQuery DataFrames | Training |
| `transform_from_api(video_data, channel_stats)` | YouTube API dicts | Inference |

**Danh sách features được tính:**

```
v1_like_ratio          = like_count / view_count
v2_comment_ratio       = comment_count / view_count
v3_like_comment_ratio  = like_count / (comment_count + 1)
v4_duration_mins       = duration_seconds / 60
v5_relative_views      = view_count / channel_avg_views
v6_engagement_score    = (từ BigQuery)
log_relative_views     = log(view_count) - log(channel_avg_views)
log_views              = log1p(view_count)

# Features từ polling (inference only)
v7_views_at_poll       = views tại thời điểm poll
v8_age_hours           = tuổi video (giờ)
v9_views_per_hour      = views / age_hours
v10_channel_hourly_avg = channel_avg_views / (30 * 24)
v11_velocity_ratio     = views_per_hour / channel_hourly_avg

# Early signal features (từ polling history)
e1_views_6h            = views tại giờ thứ 6
e2_views_24h           = views tại giờ thứ 24
e3_views_48h           = views tại giờ thứ 48
e4_growth_rate_6_24    = (views_24h - views_6h) / views_6h
e5_growth_rate_24_48   = (views_48h - views_24h) / views_24h
```

---

### 3.2 Model layer — `ml/src/models/`

#### `label_creator.py` — `VideoLabelCreator`

**Mục tiêu:** Gán nhãn `is_viral` (0/1) cho video từ dữ liệu BigQuery.

**Thuật toán gán nhãn:**

1. Tính `channel_avg_views` và `channel_std_views` cho mỗi kênh (groupby `channel_id`).
2. Tính `relative_score = (view_count - channel_avg_views) / channel_std_views`.
3. `is_viral = 1` nếu `relative_score > threshold` (mặc định `1.5` std).
4. Gán `time_window_label` dựa trên `views_per_day`:
   - `viral_within_7d`: `views_per_day > channel_daily_avg × 3.0`
   - `viral_within_30d`: `views_per_day > channel_daily_avg × 1.5`
   - `not_viral`: còn lại

**Vì sao dùng relative score thay vì absolute views?**

Kênh lớn (MrBeast) có avg 50M views/video, kênh nhỏ có avg 50K. Nếu dùng ngưỡng tuyệt đối (ví dụ 1M views = viral), kênh lớn sẽ có 0 video viral trong khi kênh nhỏ có quá nhiều. Relative score chuẩn hoá theo từng kênh, giúp model học pattern viral bất kể scale.

---

#### `video_classifier.py` — `VideoViralClassifier`

**Thiết kế:** `VideoViralClassifier` là wrapper gồm hai sub-model:

- **B1 (`_VideoViralB1`):** Binary classifier, output là `P(viral)`.
- **B2 (`_VideoTimeWindowB2`):** Multi-class classifier, output là time window label.

**Luồng predict:**
```
features → B1 → P(viral) > 0.6? → YES → B2 → time_window
                                    NO  → "not_viral"
```

**Thuật toán được chọn:**

| Điều kiện | Algorithm |
|-----------|-----------|
| `viral_rate < 10%` (mất cân bằng nặng) | `GradientBoostingClassifier` + `sample_weight` |
| `len(data) >= 200` | `GradientBoostingClassifier` |
| `len(data) < 200` | `RandomForestClassifier` + `class_weight="balanced"` |

**Evaluation:** Stratified K-Fold Cross Validation (5 folds), đánh giá bằng `F1-Score` thay vì accuracy vì class imbalance.

---

#### `explainer.py` — `PredictionExplainer`

**Mục tiêu:** Giải thích prediction bằng tiếng Việt (human-readable).

**Cơ chế:**
1. `fit()` học distribution (p25/p50/p75/mean) của mỗi feature từ training data.
2. `explain_video()` phân loại từng feature thành `high/medium/low` dựa trên percentile.
3. Map sang template text tiếng Việt trong `_VIDEO_FACTOR_TEMPLATES`.
4. Tính `momentum_score` (0-100) từ tổng hợp các tín hiệu.
5. `project_views()` ước tính projected views 7 ngày và 30 ngày dùng decay model.

---

### 3.3 Pipeline layer — `ml/src/pipeline/`

#### `viral_system.py` — `ViralPredictionSystem`

**Đây là main entry point** của toàn bộ hệ thống.

**Training mode:**
```python
system = ViralPredictionSystem()
system.train()   # load BQ → create labels → feature eng → train B1/B2 → fit explainer
system.save()    # lưu .pkl
```

**Inference mode:**
```python
system = ViralPredictionSystem.load()
report = system.predict_video("MrBeast")
report.print_report()
```

**Lazy initialization:** `YouTubeAPIClient` chỉ được khởi tạo khi `predict_video()` được gọi lần đầu — tránh lãng phí API initialization trong training mode.

---

#### `polling_monitor.py` — `PollingMonitor`

**Mục tiêu:** Background thread poll video stats theo interval định kỳ (mặc định mỗi 6 giờ trong 72 giờ).

Mỗi snapshot được lưu vào `cache/polls/{video_id}.json`. Data này sau đó được feed vào `VideoFeatureEngineer.transform_from_api()` để tính early signal features (`e1_views_6h`, `e4_growth_rate_6_24`, ...).

---

#### `report_generator.py` — `VideoReport`

`VideoReport` là `@dataclass` chuẩn hoá toàn bộ output thành dict/JSON. `print_report()` in terminal-friendly report.

---

## 4. Các lỗi gặp phải & cách fix

### Lỗi 1: `channel_avg_views` bị sai khi tính từ YouTube API

**Vị trí:** `feature_engineer.py` — `transform_from_api()`

**Mô tả lỗi:**

Ban đầu code tính `channel_avg_views` như sau:
```python
# Sai — tính từ API
channel_avg_views = total_views_api / video_count
```

Vấn đề: `total_views` từ YouTube API là **tổng all-time** của kênh (có thể từ 10 năm trước), còn `video_count` là tổng số video đã từng đăng. Kết quả là avg bị inflate giả tạo, khiến `v5_relative_views` của video mới luôn rất thấp → model underpredict viral.

**Fix:**
```python
# Fix: Ưu tiên dùng channel_avg từ BigQuery (chính xác)
bq_avg = self._channel_avg.get(channel_id, 0) if hasattr(self, "_channel_avg") else 0
if bq_avg > 0:
    channel_avg_views = bq_avg  # avg từ int_videos__enhanced (chính xác nhất)
else:
    # Fallback khi không có BQ data
    total_views_api = channel_stats.get("total_views", views) or views
    channel_avg_views = total_views_api / video_count
```

**Bài học:** Luôn ưu tiên data từ data warehouse (BigQuery) hơn API realtime khi cần historical average. API total_views không đại diện cho "avg video gần đây".

---

### Lỗi 2: LR `class_weight="balanced"` gây false positive tràn

**Vị trí:** `video_classifier.py` — `_VideoViralB1.train()`

**Mô tả lỗi:**

Khi tỷ lệ viral rất thấp (< 10%), lúc đầu dùng `LogisticRegression(class_weight="balanced")`. Kết quả:
- Recall ≈ 97% (gần như không bỏ sót video viral)
- Precision ≈ 8-12% (gần như mọi video đều bị predict viral = false positive tràn)
- F1 thực sự rất thấp

**Nguyên nhân:** `class_weight="balanced"` với LR inflate penalty cho class minority quá mạnh, khiến model "thiên vị" predict viral cho hầu hết mọi video.

**Fix:**
```python
if viral_rate < 0.1:
    # Dùng GradientBoosting + sample_weight thay vì LR balanced
    algo = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=5, random_state=42,
    )
    sample_weight = compute_class_weight("balanced", classes=np.unique(y), y=y)
    sample_weight = sample_weight[y]  # per-sample weight
```

**Bài học:** `class_weight="balanced"` trong LR nhạy cảm hơn với imbalance so với tree-based models. GradientBoosting với `sample_weight` kiểm soát tốt hơn trade-off precision/recall.

---

### Lỗi 3: Cross-validation fail với `sample_weight` + sklearn version cũ

**Vị trí:** `video_classifier.py` — `_VideoViralB1.train()`

**Mô tả lỗi:**

Khi truyền `sample_weight` vào `cross_val_predict()`, một số phiên bản sklearn cũ raise:
```
TypeError: fit() got unexpected keyword argument 'sample_weight'
```

**Fix:**
```python
# CV dùng unweighted để tránh incompatibility với sklearn cũ
y_pred = cross_val_predict(self._model, X, y, cv=cv)

# Final fit mới dùng sample_weight
if sample_weight is not None:
    self._model.fit(X, y, model__sample_weight=sample_weight)
```

**Bài học:** CV chỉ dùng cho evaluation (estimate test performance), không cần sample_weight ở đây. Final fit trên toàn data mới cần weighted.

---

### Lỗi 4: Model underestimate kênh lớn (MrBeast, PewDiePie, ...)

**Vị trí:** `viral_system.py` — `predict_video()` và `_apply_absolute_boost()`

**Mô tả lỗi:**

Model B học từ `v5_relative_views = view_count / channel_avg_views`. Kênh lớn như MrBeast có `channel_avg_views ≈ 50M`, nên video đạt 80M views chỉ có `relative_views = 1.6` — thấp hơn nhiều so với kênh nhỏ có avg 50K đạt 500K views (`relative_views = 10`).

Kết quả: Model predict MrBeast có 40% viral dù thực tế video đang có 50K views/giờ — rõ ràng viral.

**Fix — Absolute velocity boost:**
```python
def _apply_absolute_boost(prediction, views_per_hour, abs_views):
    """Bổ sung tín hiệu tuyệt đối để tránh underestimate kênh lớn."""
    prob = prediction.get("probability", 0.0)

    if views_per_hour > 100_000:
        boosted = max(prob, 0.82)      # cực kỳ viral
    elif views_per_hour > 30_000:
        boosted = max(prob, 0.65)      # viral mạnh
    elif views_per_hour > 10_000:
        boosted = max(prob, 0.50)      # tiềm năng viral
    elif views_per_hour > 2_000:
        boosted = max(prob, 0.35)
    
    if abs_views > 5_000_000:
        boosted = max(boosted, 0.60)   # boost thêm nếu raw views đã rất cao
```

**Ngưỡng tham khảo thực tế (toàn YouTube):**
- `>100K views/giờ`: chỉ vài chục video/ngày trên toàn YouTube
- `>30K views/giờ`: top 0.1% viral
- `>10K views/giờ`: viral mạnh

---

### Lỗi 5: `vs_channel_avg` bị sai với video còn non

**Vị trí:** `viral_system.py` — `predict_video()`

**Mô tả lỗi:**

Lúc đầu so sánh video bằng raw `view_count`:
```python
vs_avg_pct = (view_count / channel_avg_views - 1) * 100
```

Video 2 ngày tuổi đạt 100K views so với `channel_avg_views = 500K` (lifetime avg ~90 ngày) → `vs_avg = -80%` (tức là kém hơn avg 80%). Nhưng thực ra video đang có pace rất tốt.

**Fix — Dùng velocity (views/giờ) thay vì raw views:**
```python
channel_hourly_avg = channel_avg / (30 * 24)  # views/giờ kỳ vọng
velocity_ratio = views_per_hour / max(channel_hourly_avg, 1)
vs_avg_pct = (velocity_ratio - 1) * 100

# velocity_ratio=1 → đúng pace trung bình của kênh
# velocity_ratio=2 → nhanh gấp đôi → +100%
```

**Bài học:** Khi so sánh video ở các độ tuổi khác nhau, phải chuẩn hoá theo thời gian (views/giờ), không dùng tổng views tuyệt đối.

---

### Lỗi 6: `matplotlib` crash trên server không có display

**Vị trí:** `label_creator.py`

**Mô tả lỗi:**

```
_tkinter.TclError: no display name and no $DISPLAY environment variable
```

Khi chạy `visualize_distributions()` trên server Linux (Docker container) không có graphical display.

**Fix:**
```python
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — PHẢI đặt trước import pyplot
import matplotlib.pyplot as plt
```

**Bài học:** Backend `Agg` render ra file PNG mà không cần display. Luôn set backend này trong code server-side.

---

### Lỗi 7: `B2` crash khi một class có quá ít samples

**Vị trí:** `video_classifier.py` — `_VideoTimeWindowB2.train()`

**Mô tả lỗi:**

`StratifiedKFold` với `n_splits=5` yêu cầu mỗi class có ít nhất 5 samples. Khi dataset nhỏ và `viral_within_7d` chỉ có 2-3 videos:

```
ValueError: The least populated class in y has only 2 members,
which is too few. The minimum number of groups for any class
cannot be less than n_splits=5.
```

**Fix — Simplify thành binary khi class quá ít:**
```python
min_count = min(np.bincount(y))
if min_count < 3:
    # Gộp 7d và 30d thành "viral"
    simplify_map = {
        "viral_within_7d": "viral",
        "viral_within_30d": "viral",
        "not_viral": "not_viral",
    }
    y_bin = pd.Series(y_raw).map(simplify_map).fillna("not_viral").values
    y = self._le.fit_transform(y_bin)
```

---

### Lỗi 8: CV fail hoàn toàn với edge cases

**Vị trí:** `video_classifier.py` — cả B1 và B2

**Mô tả lỗi:** Với dataset rất nhỏ hoặc class imbalance cực độ, `cross_val_predict` có thể raise nhiều loại lỗi khác nhau.

**Fix — Graceful fallback:**
```python
try:
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred = cross_val_predict(self._model, X, y, cv=cv)
    cv_f1 = f1_score(y, y_pred, zero_division=0)
except Exception as e:
    logger.warning("CV thất bại (%s) — dùng train set evaluation", e)
    self._model.fit(X, y)
    y_pred = self._model.predict(X)
    cv_f1 = f1_score(y, y_pred, zero_division=0)
```

---

## 5. Training pipeline — từng bước

```
ViralPredictionSystem.train()
├── Step 0: BigQueryLoader.load_all()
│   ├── Tải int_channel_summary      (~N kênh)
│   ├── Tải int_engagement_metrics   (~1000+ videos)
│   └── Tải int_videos__enhanced     (~1000+ videos)
│
├── Step 1: VideoLabelCreator.create_labels(engagement_df, video_df)
│   ├── Tính channel baselines (groupby channel_id)
│   ├── Tính relative_score
│   ├── Gán is_viral (0/1)
│   └── Gán time_window_label (viral_within_7d / viral_within_30d / not_viral)
│
├── Step 2: VideoFeatureEngineer
│   ├── fit(engagement_df, video_df)   → học channel_avg, channel_std
│   └── transform(engagement_df, video_df) → DataFrame 8+ features
│
├── Step 3: VideoViralClassifier.train(video_features_df)
│   ├── B1: GradientBoosting / RandomForest
│   │   ├── Stratified 5-fold CV
│   │   └── Final fit với sample_weight (nếu imbalanced)
│   └── B2: LogisticRegression (multi-class)
│       ├── Kiểm tra min class count (simplify nếu cần)
│       └── Stratified K-fold CV
│
└── PredictionExplainer.fit(video_features_df)
    └── Học p25/p50/p75 của mỗi feature
```

**Sample output khi training:**

```
======================================================================
STARTING END-TO-END TRAINING PIPELINE
======================================================================

Step 0: Loading BigQuery data...
  ✅ int_channel_summary — 6 dòng × 7 cột
  ✅ int_engagement_metrics — 300 dòng × 8 cột
  ✅ int_videos__enhanced — 300 dòng × 12 cột

Step 1: Creating video labels...
──────────────────────────────────────────────────
VIDEO LABEL DISTRIBUTION
──────────────────────────────────────────────────
  NOT VIRAL    (label=0): 261 video (87.0%) ████████████████████████████████████
  VIRAL        (label=1):  39 video (13.0%) █████

Step 3: Training Model B (Video Classifier)...
────────────────────────────────────────────────────────────
TRAINING MODEL B1 - VIDEO VIRAL CLASSIFIER
  Videos     : 300
  Features   : 8
  CV F1-Score : 0.412
```

---

## 6. Inference pipeline

```
ViralPredictionSystem.predict_video("MrBeast")
├── YouTubeAPIClient.search_channel("MrBeast")    → channel_id
├── YouTubeAPIClient.get_channel_stats(channel_id) → subscribers, total_views
├── YouTubeAPIClient.get_recent_videos(channel_id, n=1) → video_id mới nhất
├── YouTubeAPIClient.get_video_stats([video_id])   → views, likes, comments, ...
│
├── VideoFeatureEngineer.transform_from_api(video_data, channel_stats)
│   └── Ưu tiên channel_avg từ BigQuery cache (fitted khi train)
│
├── VideoViralClassifier.predict(features)
│   ├── B1: P(viral) → ví dụ 0.43
│   ├── Absolute velocity boost → 0.65 (nếu views/giờ > 30K)
│   └── B2: time_window = "viral_within_30d"
│
├── PredictionExplainer.explain_video(features, prediction)
│   └── Tạo factors & warnings tiếng Việt
│
├── PredictionExplainer.project_views(...)
│   └── Ước tính 7 ngày: ~18M, 30 ngày: ~35M
│
└── VideoReport → print_report()
```

**Sample output:**
```
--------------------------------------------------
VIDEO REPORT
--------------------------------------------------
  Title       : MrBeast's $1,000,000 Challenge
  Channel     : MrBeast
  Published   : 2026-02-24T18:00:00Z
  Age         : 2 ngày 3 giờ

  VIRAL PREDICTION
  Label       : VIRAL WITHIN 30 DAYS
  Probability : 65.0%
  Time window : viral_within_30d
  Confidence  : MEDIUM
```

---

## 7. Testing

Các test case được viết với `unittest` trong `ml/tests/`.

### `test_feature_engineer.py`

| Test | Mô tả |
|------|--------|
| `test_fit_returns_self` | `fit()` trả về `self` (chainable) |
| `test_transform_returns_correct_shape` | Output có đúng số dòng và đủ cột feature |
| `test_no_negative_features` | Ratio features (`f1_efficiency`, `f2_loyalty`, ...) phải >= 0 |
| `test_percentiles_populated_after_fit` | Percentile dict không rỗng sau `fit()` |
| `test_percentile_rank_boundaries` | Giá trị 0 → percentile gần 0, giá trị 1e12 → gần 100 |
| `test_transform_from_api` | Output có đủ cột từ API data |

### `test_models.py`

| Test | Mô tả |
|------|--------|
| `test_creates_is_viral_column` | LabelCreator tạo cột `is_viral` |
| `test_labels_binary` | `is_viral` chỉ có giá trị 0 và 1 |
| `test_train_returns_metrics` | `train()` trả về dict có `b1` và `b2` |
| `test_predict_output_structure` | Output có `will_viral`, `probability`, `time_window`, `confidence` |
| `test_predict_without_train_raises` | Raise `RuntimeError` nếu predict trước khi train |

### `test_pipeline.py`

| Test | Mô tả |
|------|--------|
| `test_to_dict_structure` | `VideoReport.to_dict()` có đủ các key cần thiết |
| `test_will_viral_true` | Prediction flag đúng |
| `test_to_json_valid` | JSON output parse được |
| `test_explain_video_returns_keys` | Explainer trả về `summary` và `factors` |
| `test_project_views_reasonable` | Projected views có cả 7_days và 30_days |
| `test_start_and_stop` | PollingMonitor start/stop không crash |
| `test_get_snapshots_empty_initially` | Snapshots rỗng cho video không tồn tại |

**Chạy tất cả tests:**
```bash
python -m pytest ml/tests/ -v
```

---

## 8. Dependencies & môi trường

**File:** `ml/requirements.txt`

```
scikit-learn>=1.4.0
pandas>=2.2.0
numpy>=1.26.0
matplotlib>=3.8.0
google-cloud-bigquery>=3.23.1
google-auth>=2.23.0
google-auth-httplib2>=0.2.0
db-dtypes>=1.2.0               # pandas BigQuery type support
google-api-python-client>=2.100.0
google-auth-oauthlib>=1.2.0
python-dotenv>=1.0.0
```

**Biến môi trường cần thiết (`.env`):**

```env
YOUTUBE_API_KEY=AIza...
GOOGLE_APPLICATION_CREDENTIALS=credentials/project-xxx.json
```

**Cài đặt:**
```bash
pip install -r ml/requirements.txt
```

**Chạy training:**
```bash
python -c "
from ml.src.pipeline.viral_system import ViralPredictionSystem
system = ViralPredictionSystem()
system.train()
system.save()
"
```

**Chạy inference (interactive CLI):**
```bash
python ml/src/pipeline/viral_system.py
```
