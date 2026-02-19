select
    video_id,
    like_rate_pct,
    comment_rate_pct,
    engagement_score
from {{ ref('int_engagement_metrics') }}
where like_rate_pct > 100
   or comment_rate_pct > 100
   or engagement_score < 0
   or like_rate_pct < 0
   or comment_rate_pct < 0
