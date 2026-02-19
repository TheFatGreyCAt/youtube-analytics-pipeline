{{
    config(
        materialized='table',
        schema='mart'
    )
}}

with summary as (
    select * from {{ ref('int_channel_summary') }}
),

final as (
    select
        channel_id,
        channel_name,
        country_code,
        subscriber_count,
        channel_created_at,
        
        total_videos_crawled,
        total_views,
        avg_views_per_video,
        total_likes,
        total_comments,
        
        avg_like_rate_pct,
        avg_comment_rate_pct,
        avg_video_duration_seconds,
        
        latest_video_date,
        earliest_video_date,
        date_diff(latest_video_date, earliest_video_date, day) as channel_active_days,
        avg_days_between_uploads,
        
        safe_divide(total_views, subscriber_count) as views_per_subscriber,
        safe_divide(total_videos_crawled, 
            nullif(date_diff(latest_video_date, earliest_video_date, day), 0)
        ) * 7 as videos_per_week,
        
        last_crawled_at,
        current_timestamp() as dbt_updated_at
        
    from summary
)

select * from final
