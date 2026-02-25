"""
Report Generator — chuẩn hoá output thành VideoReport.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional


def _fmt_number(n: int | float) -> str:
    n = int(n)
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.0f}K"
    return str(n)


# ─── Video Report ──────────────────────────────────────────────────────────────
@dataclass
class VideoReport:
    video_id: str
    video_title: str
    channel_name: str
    published_at: str
    video_age: str
    prediction: dict
    current_views: int
    views_per_hour: float
    vs_channel_avg_pct: float
    channel_percentile: int
    explanation: dict
    projected_views: dict

    def to_dict(self) -> dict:
        pred = self.prediction
        has_early = pred.get("has_early_signals", False)
        vs_str = (f"+{self.vs_channel_avg_pct:.0f}%" if self.vs_channel_avg_pct >= 0
                  else f"{self.vs_channel_avg_pct:.0f}%")

        return {
            "video_id": self.video_id,
            "video_title": self.video_title,
            "channel": self.channel_name,
            "published_at": self.published_at,
            "video_age": self.video_age,

            "viral_prediction": {
                "will_viral": pred.get("will_viral"),
                "probability": pred.get("probability"),
                "time_window": pred.get("time_window"),
                "label": pred.get("label"),
                "confidence": pred.get("confidence"),
            },

            "current_performance": {
                "views": _fmt_number(self.current_views),
                "views_per_hour": _fmt_number(int(self.views_per_hour)),
                "vs_channel_avg": vs_str,
                "channel_percentile": self.channel_percentile,
            },

            "early_signals": {
                "available": has_early,
                "trend": self._trend_label(),
                "momentum_score": self.explanation.get("momentum_score", 50),
            },

            "explanation": [f["description"] for f in self.explanation.get("factors", [])],
            "summary": self.explanation.get("summary", ""),
            "projected_views": self.projected_views,
            "warnings": self.explanation.get("warnings", []),
            "model_used": "early_signal_model" if has_early else "snapshot_model",
        }

    def print_report(self) -> None:
        d = self.to_dict()
        vp = d["viral_prediction"]
        cp = d["current_performance"]
        es = d["early_signals"]
        pv = d["projected_views"]
        sep = "-" * 50
        print()
        print(sep)
        print("VIDEO REPORT")
        print(sep)
        print(f"  Title       : {self.video_title[:60]}")
        print(f"  Channel     : {self.channel_name}")
        print(f"  Published   : {self.published_at}")
        print(f"  Age         : {self.video_age}")
        print()
        print("  VIRAL PREDICTION")
        print(f"  Label       : {vp['label']}")
        print(f"  Probability : {vp['probability']*100:.1f}%")
        print(f"  Time window : {vp['time_window']}")
        print(f"  Confidence  : {vp['confidence']}")
        print()
        print("  CURRENT PERFORMANCE")
        print(f"  Views       : {cp['views']} (*)")
        print(f"  Views/hour  : {cp['views_per_hour']}")
        print(f"  vs Pace     : {cp['vs_channel_avg']}")
        print(f"  Percentile  : top {100-self.channel_percentile}%")
        print(f"  Early signals: {'yes' if es['available'] else 'no'}")
        print(f"  Momentum    : {es['momentum_score']}/100")
        print()
        print("  PROJECTED VIEWS")
        print(f"  7 days      : {pv.get('7_days', 'N/A')}")
        print(f"  30 days     : {pv.get('30_days', 'N/A')}")
        print(f"  Confidence  : {pv.get('confidence', 'N/A')}")
        print()
        print(f"  Summary: {d['summary']}")
        if d["warnings"]:
            print()
            print("  Warnings:")
            for w in d["warnings"]:
                print(f"    - {w}")
        print()
        print("  (*) Views lay tu YouTube Data API — co the thap hon thuc te")
        print("      do YouTube API co do tre cap nhat so voi trang web.")
        print(sep)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def _trend_label(self) -> str:
        mom = self.explanation.get("momentum_score", 50)
        if mom >= 70:
            return "ACCELERATING"
        if mom >= 50:
            return "STABLE"
        return "SLOWING"