{{
    config(
        materialized='view'
    )
}}

with source as (
    select * from {{ source('youtube_raw', 'raw_videos') }}
),

flattened as (
    select
        -- Primary Key
        id as video_id,
        
        -- Parse JSON and extract fields
        json_extract_scalar(raw, '$.snippet.channelId') as channel_id,
        json_extract_scalar(raw, '$.snippet.title') as title,
        json_extract_scalar(raw, '$.snippet.description') as description,
        json_extract_array(raw, '$.snippet.tags') as tags,
        json_extract_scalar(raw, '$.snippet.categoryId') as category_id,
        json_extract_scalar(raw, '$.snippet.defaultLanguage') as default_language,
        cast(json_extract_scalar(raw, '$.snippet.publishedAt') as timestamp) as published_at,
        
        -- Statistics fields (cast string to numeric, coalesce NULL values)
        cast(json_extract_scalar(raw, '$.statistics.viewCount') as int64) as view_count,
        coalesce(cast(json_extract_scalar(raw, '$.statistics.likeCount') as int64), 0) as like_count,
        coalesce(cast(json_extract_scalar(raw, '$.statistics.commentCount') as int64), 0) as comment_count,
        
        -- Content Details
        json_extract_scalar(raw, '$.contentDetails.duration') as duration_iso8601,
        cast(json_extract_scalar(raw, '$.contentDetails.caption') as bool) as has_caption,
        json_extract_scalar(raw, '$.contentDetails.definition') as definition,
        
        -- Status
        json_extract_scalar(raw, '$.status.privacyStatus') as privacy_status,
        cast(json_extract_scalar(raw, '$.status.embeddable') as bool) as is_embeddable,
        cast(json_extract_scalar(raw, '$.status.madeForKids') as bool) as is_made_for_kids,
        
        -- Metadata
        cast(ingestion_time as timestamp) as crawled_at,
        current_timestamp() as dbt_loaded_at
        
        {{ get_passthrough_columns('youtube__video_passthrough_columns') }}
        
    from source
    where json_extract_scalar(raw, '$.status.privacyStatus') = 'public'  -- Only public videos
)

select * from flattened
