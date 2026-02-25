"""
Label Creator — tạo labels cho Video (Model B).
"""
from __future__ import annotations

import logging
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend cho server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class VideoLabelCreator:
    """
    Tạo labels viral cho 1,000 video từ int_engagement_metrics + int_videos__enhanced.

    Label viral = 1 nếu relative_score > threshold (mặc định 1.5 std).
    Label time_window: "fast_viral" / "slow_viral" / "not_viral" (nếu có age data).
    """

    def __init__(
        self,
        relative_threshold: float = 1.5,
        fast_viral_multiplier: float = 3.0,
        slow_viral_multiplier: float = 1.5,
    ) -> None:
        self.relative_threshold = relative_threshold
        self.fast_viral_multiplier = fast_viral_multiplier
        self.slow_viral_multiplier = slow_viral_multiplier

    def create_labels(
        self,
        engagement_df: pd.DataFrame,
        video_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Tạo label viral cho video.

        Args:
            engagement_df: int_engagement_metrics
            video_df:      int_videos__enhanced (optional, để lấy duration)

        Returns:
            DataFrame với cột mới: channel_avg_views, channel_std_views,
            relative_score, is_viral, time_window_label
        """
        df = engagement_df.copy()

        # Merge duration nếu có
        if video_df is not None and "duration_seconds" in video_df.columns:
            extra = video_df[["video_id", "duration_seconds"]].drop_duplicates("video_id")
            df = df.merge(extra, on="video_id", how="left")

        # ── Bước 1: channel baseline ────────────────────────────────────────
        if "channel_id" not in df.columns:
            logger.warning("Không có channel_id — dùng global baseline")
            global_avg = df["view_count"].mean()
            global_std = df["view_count"].std()
            df["channel_avg_views"] = global_avg
            df["channel_std_views"] = global_std
        else:
            channel_stats = df.groupby("channel_id")["view_count"].agg(
                channel_avg_views="mean", channel_std_views="std"
            ).reset_index()
            channel_stats["channel_std_views"] = channel_stats["channel_std_views"].fillna(1)
            df = df.merge(channel_stats, on="channel_id", how="left")

        df["channel_avg_views"] = df["channel_avg_views"].fillna(df["view_count"].mean())
        df["channel_std_views"] = df["channel_std_views"].fillna(1).clip(lower=1)

        # ── Bước 2: relative viral score ───────────────────────────────────
        df["relative_score"] = (
            (df["view_count"] - df["channel_avg_views"]) / df["channel_std_views"]
        ).fillna(0)
        df["is_viral"] = (df["relative_score"] > self.relative_threshold).astype(int)

        # ── Bước 3: time window label (nếu có published_at) ────────────────
        if "published_at" in df.columns:
            df = self._create_time_window_labels(df)
        else:
            logger.info("Không có published_at — bỏ qua time window labels")
            df["time_window_label"] = df["is_viral"].map({1: "viral_within_30d", 0: "not_viral"})

        self._report_distribution(df)
        return df

    def _create_time_window_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Gán nhãn thời gian dựa trên views_per_day so với channel baseline."""
        if "published_at" not in df.columns:
            df["time_window_label"] = "unknown"
            return df

        now = pd.Timestamp.now(tz="UTC")
        df["published_at_ts"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        df["video_age_days"] = (now - df["published_at_ts"]).dt.total_seconds() / 86400
        df["video_age_days"] = df["video_age_days"].clip(lower=0.1)
        df["views_per_day"] = df["view_count"] / df["video_age_days"]
        df["channel_daily_avg"] = df["channel_avg_views"] / 30  # ước tính

        cond_fast = df["views_per_day"] > df["channel_daily_avg"] * self.fast_viral_multiplier
        cond_slow = (df["views_per_day"] > df["channel_daily_avg"] * self.slow_viral_multiplier) & ~cond_fast

        df["time_window_label"] = "not_viral"
        df.loc[cond_slow, "time_window_label"] = "viral_within_30d"
        df.loc[cond_fast, "time_window_label"] = "viral_within_7d"
        return df

    def _report_distribution(self, df: pd.DataFrame) -> None:
        total = len(df)
        viral_count = df["is_viral"].sum()
        not_viral_count = total - viral_count

        print(f"\n{'─'*50}")
        print(f"VIDEO LABEL DISTRIBUTION")
        print(f"{'─'*50}")
        for label, count in [("NOT VIRAL", not_viral_count), ("VIRAL", viral_count)]:
            bar = "█" * int(count / total * 40)
            pct = count / total * 100
            val = 0 if label == "NOT VIRAL" else 1
            print(f"  {label:12s} (label={val}): {count:4d} video ({pct:.1f}%) {bar}")

        if "time_window_label" in df.columns:
            print(f"\n  Time Window Distribution:")
            for label, count in df["time_window_label"].value_counts().items():
                pct = count / total * 100
                print(f"     {label:20s}: {count:4d} ({pct:.1f}%)")

        viral_pct = viral_count / total * 100
        if viral_pct < 5:
            print(f"\n  Warning: only {viral_pct:.1f}% video viral — rất mất cân bằng!")
            print("   Đề xuất: giảm relative_threshold xuống 1.0")
        else:
            print(f"\n  Label balance acceptable nhận được ({viral_pct:.1f}% viral)")
        print(f"{'─'*50}\n")

    def visualize_distributions(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot distribution của labels và key features."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Video Label & Feature Distributions", fontsize=14, fontweight="bold")

        # 1. Label distribution
        label_counts = df["is_viral"].value_counts()
        axes[0, 0].bar(["Not Viral (0)", "Viral (1)"], label_counts.values,
                       color=["#2ecc71", "#e74c3c"])
        axes[0, 0].set_title("Label Distribution (is_viral)")
        axes[0, 0].set_ylabel("Count")
        for i, v in enumerate(label_counts.values):
            axes[0, 0].text(i, v + 5, str(v), ha="center")

        # 2. Relative score dist
        if "relative_score" in df.columns:
            axes[0, 1].hist(df["relative_score"].clip(-3, 6), bins=30, color="#3498db", alpha=0.7)
            axes[0, 1].axvline(self.relative_threshold, color="red", linestyle="--",
                               label=f"Threshold={self.relative_threshold}")
            axes[0, 1].set_title("Relative Score Distribution")
            axes[0, 1].legend()

        # 3. Views distribution (log scale)
        if "view_count" in df.columns:
            axes[0, 2].hist(np.log1p(df["view_count"]), bins=30, color="#9b59b6", alpha=0.7)
            axes[0, 2].set_title("Log(View Count) Distribution")

        # 4. Like ratio
        if "like_count" in df.columns and "view_count" in df.columns:
            ratio = (df["like_count"] / (df["view_count"] + 1)).clip(0, 0.15)
            for viral, color, label in [(0, "#2ecc71", "Not Viral"), (1, "#e74c3c", "Viral")]:
                subset = ratio[df["is_viral"] == viral]
                axes[1, 0].hist(subset, bins=30, alpha=0.5, color=color, label=label)
            axes[1, 0].set_title("Like Ratio by Label")
            axes[1, 0].legend()

        # 5. Engagement score
        if "engagement_score" in df.columns:
            for viral, color, label in [(0, "#2ecc71", "Not Viral"), (1, "#e74c3c", "Viral")]:
                subset = df.loc[df["is_viral"] == viral, "engagement_score"].clip(0, 20)
                axes[1, 1].hist(subset, bins=30, alpha=0.5, color=color, label=label)
            axes[1, 1].set_title("Engagement Score by Label")
            axes[1, 1].legend()

        # 6. Time window (nếu có)
        if "time_window_label" in df.columns:
            tw_counts = df["time_window_label"].value_counts()
            axes[1, 2].bar(tw_counts.index, tw_counts.values, color=["#e74c3c", "#f39c12", "#2ecc71"])
            axes[1, 2].set_title("Time Window Label Distribution")
            axes[1, 2].tick_params(axis="x", rotation=20)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved: {save_path}")
        else:
            plt.show()
        plt.close()