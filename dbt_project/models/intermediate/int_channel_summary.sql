{{
    config(
        materialized='view'
    )
}}

with videos as (
    select * from {{ ref('int_videos__enhanced') }}
),

channel_agg as (
    select
        channel_id,
        any_value(channel_name) as channel_name,
        any_value(country_code) as country_code,
        any_value(channel_subscribers) as subscriber_count,
        any_value(channel_created_at) as channel_created_at,
        
        count(distinct video_id) as total_videos_crawled,
        sum(view_count) as total_views,
        avg(view_count) as avg_views_per_video,
        sum(like_count) as total_likes,
        sum(comment_count) as total_comments,
        
        avg(safe_divide(like_count, nullif(view_count, 0))) * 100 as avg_like_rate_pct,
        avg(safe_divide(comment_count, nullif(view_count, 0))) * 100 as avg_comment_rate_pct,
        avg(duration_seconds) as avg_video_duration_seconds,
        
        max(published_at) as latest_video_date,
        min(published_at) as earliest_video_date,
        
        date_diff(max(published_at), min(published_at), day) / 
            nullif(count(distinct video_id) - 1, 0) as avg_days_between_uploads,
        
        max(crawled_at) as last_crawled_at
        
    from videos
    group by 1
)

select * from channel_agg
