{{
    config(
        materialized='view',
        schema='staging'
    )
}}

with source as (
    select * from {{ source('youtube_raw', 'youtube_channels_raw') }}
),

flattened as (
    select
        id as channel_id,
        
        data.snippet.title as channel_name,
        data.snippet.description as description,
        cast(data.snippet.publishedAt as timestamp) as channel_created_at,
        data.snippet.country as country_code,
        data.snippet.customUrl as custom_url,
        
        cast(data.statistics.subscriberCount as int64) as subscriber_count,
        cast(data.statistics.viewCount as int64) as total_view_count,
        cast(data.statistics.videoCount as int64) as video_count,
        cast(data.statistics.hiddenSubscriberCount as bool) as has_hidden_subscribers,
        
        data.contentDetails.relatedPlaylists.uploads as uploads_playlist_id,
        
        data.topicDetails.topicIds as topic_ids,
        
        _crawled_at as crawled_at,
        current_timestamp() as dbt_loaded_at
        
        {{ get_passthrough_columns('youtube__channel_passthrough_columns') }}
        
    from source
)

select * from flattened
