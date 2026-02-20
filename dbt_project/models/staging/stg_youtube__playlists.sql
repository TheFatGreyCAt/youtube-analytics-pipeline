{{
    config(
        materialized='view'
    )
}}

with source as (
    select * from {{ source('youtube_raw', 'raw_playlists') }}
),

flattened as (
    select
        -- Primary Key
        id as playlist_id,
        channel_id,
        
        -- Snippet fields
        json_extract_scalar(raw, '$.snippet.title') as playlist_name,
        json_extract_scalar(raw, '$.snippet.description') as description,
        cast(json_extract_scalar(raw, '$.snippet.publishedAt') as timestamp) as created_at,
        
        -- Content Details
        cast(json_extract_scalar(raw, '$.contentDetails.itemCount') as int64) as item_count,
        
        -- Status
        json_extract_scalar(raw, '$.status.privacyStatus') as privacy_status,
        
        -- Metadata
        cast(ingestion_time as timestamp) as crawled_at,
        current_timestamp() as dbt_loaded_at
        
    from source
    where json_extract_scalar(raw, '$.status.privacyStatus') = 'public'  -- Only public playlists
)

select * from flattened
