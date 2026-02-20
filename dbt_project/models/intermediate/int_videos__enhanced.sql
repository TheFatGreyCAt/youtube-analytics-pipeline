{{
    config(
        materialized='view'
    )
}}

with videos as (
    select * from {{ ref('stg_youtube__videos') }}
),

channels as (
    select * from {{ ref('stg_youtube__channels') }}
),

enhanced as (
    select
        v.video_id,
        v.channel_id,
        v.title,
        v.description,
        v.tags,
        v.category_id,
        v.default_language,
        v.published_at,
        v.view_count,
        v.like_count,
        v.comment_count,
        v.duration_iso8601,
        v.has_caption,
        v.definition,
        v.privacy_status,
        v.is_embeddable,
        v.is_made_for_kids,
        v.crawled_at,
        
        c.channel_name,
        c.subscriber_count as channel_subscribers,
        c.country_code,
        c.channel_created_at,
        
        {{ parse_iso8601_duration('v.duration_iso8601') }} as duration_seconds,
        
        date(v.published_at) as published_date,
        extract(year from v.published_at) as published_year,
        extract(month from v.published_at) as published_month,
        extract(dayofweek from v.published_at) as published_dayofweek,
        extract(hour from v.published_at) as published_hour,
        
        date_diff(current_date(), date(v.published_at), day) as days_since_published,
        
        case
            when {{ parse_iso8601_duration('v.duration_iso8601') }} <= 60 then 'shorts'
            when {{ parse_iso8601_duration('v.duration_iso8601') }} <= 600 then 'short'
            when {{ parse_iso8601_duration('v.duration_iso8601') }} <= 1800 then 'medium'
            else 'long'
        end as video_length_category
        
    from videos v
    left join channels c on v.channel_id = c.channel_id
)

select * from enhanced
