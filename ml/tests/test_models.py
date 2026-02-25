"""
Unit tests cho Video Labels v\u00e0 Model B.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd



def _make_video_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "video_id": [f"v_{i}" for i in range(n)],
        "channel_id": [f"ch_{i%10}" for i in range(n)],
        "view_count": rng.integers(1_000, 10_000_000, n),
        "like_count": rng.integers(100, 200_000, n),
        "comment_count": rng.integers(10, 20_000, n),
        "duration_seconds": rng.integers(60, 3_600, n),
        "engagement_score": rng.uniform(0, 20, n),
    })



class TestVideoLabelCreator(unittest.TestCase):
    def setUp(self):
        from ml.src.models.label_creator import VideoLabelCreator
        self.creator = VideoLabelCreator()
        self.video_df = _make_video_df(200)

    def test_creates_is_viral_column(self):
        result = self.creator.create_labels(self.video_df)
        self.assertIn("is_viral", result.columns)

    def test_labels_binary(self):
        result = self.creator.create_labels(self.video_df)
        unique_vals = set(result["is_viral"].unique())
        self.assertTrue(unique_vals.issubset({0, 1}))

    def test_relative_score_present(self):
        result = self.creator.create_labels(self.video_df)
        self.assertIn("relative_score", result.columns)



# ─── Model B ───────────────────────────────────────────────────────────────────
class TestVideoViralClassifier(unittest.TestCase):
    def setUp(self):
        from ml.src.data.feature_engineer import VideoFeatureEngineer
        from ml.src.models.label_creator import VideoLabelCreator
        from ml.src.models.video_classifier import VideoViralClassifier
        video_df = _make_video_df(200)
        labeled = VideoLabelCreator().create_labels(video_df)
        fe = VideoFeatureEngineer()
        fe.fit(video_df, video_df)
        features = fe.transform(video_df, video_df)
        features["is_viral"] = labeled["is_viral"].values[:len(features)]
        if "time_window_label" in labeled.columns:
            features["time_window_label"] = labeled["time_window_label"].values[:len(features)]
        self.features = features
        self.clf = VideoViralClassifier()

    def test_train_returns_metrics(self):
        result = self.clf.train(self.features)
        self.assertIn("b1", result)
        self.assertIn("b2", result)

    def test_predict_output_structure(self):
        self.clf.train(self.features)
        result = self.clf.predict(self.features.iloc[[0]])
        self.assertIn("will_viral", result)
        self.assertIn("probability", result)
        self.assertIn("time_window", result)
        self.assertIn("confidence", result)
        self.assertGreaterEqual(result["probability"], 0.0)
        self.assertLessEqual(result["probability"], 1.0)

    def test_predict_without_train_raises(self):
        from ml.src.models.video_classifier import VideoViralClassifier
        clf = VideoViralClassifier()
        with self.assertRaises(RuntimeError):
            clf.predict(self.features.iloc[[0]])


if __name__ == "__main__":
    unittest.main()