{{
    config(
        materialized='incremental',
        unique_key='date_channel_key',
        schema='mart',
        partition_by={
            'field': 'metric_date',
            'data_type': 'date',
            'granularity': 'day'
        },
        cluster_by=['channel_id']
    )
}}

with videos as (
    select * from {{ ref('int_engagement_metrics') }}
),

daily_agg as (
    select
        date(published_at) as metric_date,
        channel_id,
        any_value(channel_name) as channel_name,
        any_value(country_code) as country_code,
        
        count(distinct video_id) as videos_published,
        sum(view_count) as total_views,
        sum(like_count) as total_likes,
        sum(comment_count) as total_comments,
        avg(engagement_score) as avg_engagement_score,
        avg(like_rate_pct) as avg_like_rate_pct,
        max(view_count) as max_video_views,
        
        current_timestamp() as dbt_updated_at
        
    from videos
    
    {% if is_incremental() %}
        where date(published_at) > (select max(metric_date) from {{ this }})
    {% endif %}
    
    group by 1, 2
),

final as (
    select
        {{ dbt_utils.generate_surrogate_key(['metric_date', 'channel_id']) }} as date_channel_key,
        *
    from daily_agg
)

select * from final
