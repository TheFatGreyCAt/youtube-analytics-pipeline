{{
    config(
        materialized='view',
        schema='intermediate'
    )
}}

with videos as (
    select * from {{ ref('int_videos__enhanced') }}
),

metrics as (
    select
        video_id,
        channel_id,
        title,
        published_at,
        published_date,
        view_count,
        like_count,
        comment_count,
        days_since_published,
        video_length_category,
        channel_name,
        channel_subscribers,
        country_code,
        
        safe_divide(like_count, view_count) * 100 as like_rate_pct,
        safe_divide(comment_count, view_count) * 100 as comment_rate_pct,
        safe_divide((like_count + comment_count * 2), view_count) * 100 as engagement_score,
        
        safe_divide(view_count, nullif(days_since_published, 0)) as avg_views_per_day,
        
        case
            when safe_divide(like_count, view_count) >= 0.05 then 'high'
            when safe_divide(like_count, view_count) >= 0.02 then 'medium'
            else 'low'
        end as engagement_level,
        
        case
            when safe_divide(view_count, nullif(days_since_published, 0)) > 
                 safe_divide(channel_subscribers * 0.1, 1) then true
            else false
        end as is_potentially_viral
        
    from videos
    where days_since_published > 0
)

select * from metrics
