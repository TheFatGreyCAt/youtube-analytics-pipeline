"""
Unit tests cho pipeline: report_generator (VideoReport), polling_monitor, v\u00e0 explainer.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch




class TestVideoReport(unittest.TestCase):
    def _make_report(self):
        from ml.src.pipeline.report_generator import VideoReport
        return VideoReport(
            video_id="dQw4w9WgXcQ",
            video_title="MrBeast giveaway 2025",
            channel_name="MrBeast",
            published_at="2025-01-01T18:00:00Z",
            video_age="2 ngày 6 giờ",
            prediction={
                "will_viral": True,
                "probability": 0.83,
                "time_window": "viral_within_7d",
                "label": "⚡ VIRAL TRONG TUẦN",
                "confidence": "MEDIUM",
                "has_early_signals": True,
            },
            current_views=2_340_123,
            views_per_hour=43_521,
            vs_channel_avg_pct=340.0,
            channel_percentile=92,
            explanation={
                "summary": "Video đang viral",
                "factors": [{"description": "Like ratio cao"}],
                "warnings": [],
                "momentum_score": 78,
            },
            projected_views={"7_days": "~18M", "30_days": "~35M", "confidence": "MEDIUM"},
        )

    def test_to_dict_structure(self):
        report = self._make_report()
        d = report.to_dict()
        self.assertIn("viral_prediction", d)
        self.assertIn("current_performance", d)
        self.assertIn("early_signals", d)
        self.assertIn("projected_views", d)

    def test_will_viral_true(self):
        report = self._make_report()
        d = report.to_dict()
        self.assertTrue(d["viral_prediction"]["will_viral"])

    def test_to_json_valid(self):
        import json
        report = self._make_report()
        parsed = json.loads(report.to_json())
        self.assertEqual(parsed["channel"], "MrBeast")


# ─── Explainer ─────────────────────────────────────────────────────────────────
class TestPredictionExplainer(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import pandas as pd
        from ml.src.models.explainer import PredictionExplainer
        rng = np.random.default_rng(42)
        self.explainer = PredictionExplainer()
        mock_video_df = pd.DataFrame({
            "v1_like_ratio": rng.uniform(0, 0.1, 50),
            "v5_relative_views": rng.uniform(0.3, 5.0, 50),
        })
        self.explainer.fit(video_features_df=mock_video_df)

    def test_explain_video_returns_keys(self):
        import pandas as pd
        features = pd.DataFrame([{
            "v1_like_ratio": 0.07, "v2_comment_ratio": 0.005,
            "v5_relative_views": 2.5, "v9_views_per_hour": 5000.0,
        }])
        pred = {"will_viral": True, "probability": 0.8, "time_window": "viral_within_7d", "confidence": "HIGH"}
        result = self.explainer.explain_video(features, pred)
        self.assertIn("summary", result)
        self.assertIn("factors", result)

    def test_project_views_reasonable(self):
        from ml.src.models.explainer import PredictionExplainer
        projected = PredictionExplainer.project_views(
            current_views=1_000_000,
            views_per_hour=50_000,
            probability=0.8,
            time_window="viral_within_7d",
        )
        self.assertIn("7_days", projected)
        self.assertIn("30_days", projected)


# ─── Polling Monitor ───────────────────────────────────────────────────────────
class TestPollingMonitor(unittest.TestCase):
    def test_start_and_stop(self):
        import time
        from ml.src.pipeline.polling_monitor import PollingMonitor
        mock_api = MagicMock()
        mock_api.get_video_stats.return_value = {
            "test_vid": {"views": 100_000, "likes": 5_000, "comments": 500}
        }
        monitor = PollingMonitor(mock_api)
        monitor.start("test_vid", interval_hours=0.001, duration_hours=0.01)
        time.sleep(0.1)
        self.assertTrue(monitor.is_monitoring("test_vid"))
        monitor.stop("test_vid")
        time.sleep(0.1)

    def test_get_snapshots_empty_initially(self):
        from ml.src.pipeline.polling_monitor import PollingMonitor
        mock_api = MagicMock()
        monitor = PollingMonitor(mock_api)
        snapshots = monitor.get_snapshots("nonexistent_video")
        self.assertEqual(snapshots, [])


if __name__ == "__main__":
    unittest.main()