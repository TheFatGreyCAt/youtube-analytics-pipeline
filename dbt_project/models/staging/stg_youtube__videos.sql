{{
    config(
        materialized='view',
        schema='staging'
    )
}}

with source as (
    select * from {{ source('youtube_raw', 'youtube_videos_raw') }}
),

flattened as (
    select
        -- Primary Key
        id as video_id,
        
        -- Snippet fields (BigQuery JSON/STRUCT notation)
        data.snippet.channelId as channel_id,
        data.snippet.title as title,
        data.snippet.description as description,
        data.snippet.tags as tags,
        data.snippet.categoryId as category_id,
        data.snippet.channelTitle as channel_title,
        data.snippet.defaultLanguage as default_language,
        cast(data.snippet.publishedAt as timestamp) as published_at,
        
        -- Statistics fields (cast string to numeric)
        cast(data.statistics.viewCount as int64) as view_count,
        cast(data.statistics.likeCount as int64) as like_count,
        cast(data.statistics.commentCount as int64) as comment_count,
        
        -- Content Details
        data.contentDetails.duration as duration_iso8601,
        cast(data.contentDetails.caption as bool) as has_caption,
        data.contentDetails.definition as definition,
        
        -- Status
        data.status.privacyStatus as privacy_status,
        cast(data.status.embeddable as bool) as is_embeddable,
        cast(data.status.madeForKids as bool) as is_made_for_kids,
        
        -- Metadata
        _crawled_at as crawled_at,
        current_timestamp() as dbt_loaded_at
        
        {{ get_passthrough_columns('youtube__video_passthrough_columns') }}
        
    from source
    where data.status.privacyStatus = 'public'  -- Only public videos
)

select * from flattened
