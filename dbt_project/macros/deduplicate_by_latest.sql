{% macro deduplicate_by_latest(table_name, unique_key, timestamp_field='_crawled_at') %}
    {#
    Deduplicate records by keeping only the latest version
    Usage: {{ deduplicate_by_latest('youtube_videos_raw', 'id') }}
    #}
    
    with source as (
        select * from {{ source('youtube_raw', table_name) }}
    ),
    
    ranked as (
        select 
            *,
            row_number() over (
                partition by {{ unique_key }} 
                order by {{ timestamp_field }} desc
            ) as rn
        from source
    )
    
    select * except(rn)
    from ranked
    where rn = 1

{% endmacro %}
