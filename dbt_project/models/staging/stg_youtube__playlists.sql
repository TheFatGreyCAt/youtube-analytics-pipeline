{{
    config(
        materialized='view',
        schema='staging'
    )
}}

with source as (
    select * from {{ source('youtube_raw', 'youtube_playlists_raw') }}
),

flattened as (
    select
        -- Primary Key
        id as playlist_id,
        channel_id,
        
        -- Snippet fields
        data.snippet.title as playlist_name,
        data.snippet.description as description,
        cast(data.snippet.publishedAt as timestamp) as created_at,
        
        -- Content Details
        cast(data.contentDetails.itemCount as int64) as item_count,
        
        -- Status
        data.status.privacyStatus as privacy_status,
        
        -- Metadata
        _crawled_at as crawled_at,
        current_timestamp() as dbt_loaded_at
        
    from source
    where data.status.privacyStatus = 'public'  -- Only public playlists
)

select * from flattened
