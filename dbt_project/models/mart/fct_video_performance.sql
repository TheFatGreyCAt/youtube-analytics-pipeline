{{
    config(
        materialized='incremental',
        unique_key='video_id',
        incremental_strategy='merge',
        schema='mart',
        partition_by={
            'field': 'published_date',
            'data_type': 'date',
            'granularity': 'day'
        },
        cluster_by=['channel_id', 'category_id'],
        merge_update_columns=[
            'view_count', 'like_count', 'comment_count',
            'like_rate_pct', 'comment_rate_pct', 'engagement_score',
            'avg_views_per_day', 'engagement_level', 'is_potentially_viral',
            'dbt_updated_at'
        ]
    )
}}

with metrics as (
    select * from {{ ref('int_engagement_metrics') }}
),

videos as (
    select * from {{ ref('int_videos__enhanced') }}
),

final as (
    select
        m.video_id,
        m.channel_id,
        v.category_id,
        m.title,
        m.published_at,
        m.published_date,
        v.published_year,
        v.published_month,
        m.channel_name,
        m.country_code,
        m.video_length_category,
        v.duration_seconds,
        
        m.view_count,
        m.like_count,
        m.comment_count,
        m.like_rate_pct,
        m.comment_rate_pct,
        m.engagement_score,
        m.avg_views_per_day,
        m.engagement_level,
        m.is_potentially_viral,
        
        v.has_caption,
        v.is_embeddable,
        v.is_made_for_kids,
        v.definition,
        
        v.crawled_at,
        current_timestamp() as dbt_updated_at
        
    from metrics m
    join videos v on m.video_id = v.video_id
    
    {% if is_incremental() %}
        where v.crawled_at > (select max(crawled_at) from {{ this }})
           or m.video_id in (
               select video_id from {{ this }}
               where dbt_updated_at < current_timestamp() - interval 7 day
           )
    {% endif %}
)

select * from final
