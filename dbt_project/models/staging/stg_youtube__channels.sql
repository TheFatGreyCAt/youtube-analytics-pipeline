{{
    config(
        materialized='view'
    )
}}

with source as (
    select * from {{ source('youtube_raw', 'raw_channels') }}
),

flattened as (
    select
        id as channel_id,
        
        json_extract_scalar(raw, '$.items[0].snippet.title') as channel_name,
        json_extract_scalar(raw, '$.items[0].snippet.description') as description,
        cast(json_extract_scalar(raw, '$.items[0].snippet.publishedAt') as timestamp) as channel_created_at,
        json_extract_scalar(raw, '$.items[0].snippet.country') as country_code,
        json_extract_scalar(raw, '$.items[0].snippet.customUrl') as custom_url,
        
        cast(json_extract_scalar(raw, '$.items[0].statistics.subscriberCount') as int64) as subscriber_count,
        cast(json_extract_scalar(raw, '$.items[0].statistics.viewCount') as int64) as total_view_count,
        cast(json_extract_scalar(raw, '$.items[0].statistics.videoCount') as int64) as video_count,
        cast(json_extract_scalar(raw, '$.items[0].statistics.hiddenSubscriberCount') as bool) as has_hidden_subscribers,
        
        json_extract_scalar(raw, '$.items[0].contentDetails.relatedPlaylists.uploads') as uploads_playlist_id,
        
        json_extract_array(raw, '$.items[0].topicDetails.topicIds') as topic_ids,
        
        cast(ingestion_time as timestamp) as crawled_at,
        current_timestamp() as dbt_loaded_at
        
        {{ get_passthrough_columns('youtube__channel_passthrough_columns') }}
        
    from source
)

select * from flattened
