select
    video_id,
    view_count,
    like_count,
    comment_count
from {{ ref('stg_youtube__videos') }}
where view_count < 0
   or like_count < 0
   or comment_count < 0
