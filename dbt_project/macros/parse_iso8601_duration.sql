{%- macro parse_iso8601_duration(duration_str) -%}
    {#
    Converts ISO 8601 duration format to total seconds
    Examples:
        PT1H23M45S -> 5025 seconds
        PT45S -> 45 seconds
        PT10M30S -> 630 seconds
    
    Formula: P[n]Y[n]M[n]DT[n]H[n]M[n]S
    We only care about: days, hours, minutes, seconds
    #}
    
    {%- if target.type == 'bigquery' -%}
        CAST(
            COALESCE(
                CAST(REGEXP_EXTRACT({{ duration_str }}, r'(\d+)D') AS INT64) * 86400,
                0
            ) +
            COALESCE(
                CAST(REGEXP_EXTRACT({{ duration_str }}, r'T(\d+)H') AS INT64) * 3600,
                0
            ) +
            COALESCE(
                CAST(REGEXP_EXTRACT({{ duration_str }}, r'(\d+)M(?!S)') AS INT64) * 60,
                0
            ) +
            COALESCE(
                CAST(REGEXP_EXTRACT({{ duration_str }}, r'(\d+)S$') AS INT64),
                0
            )
            AS INT64
        )
    {%- elif target.type == 'postgres' -%}
        CAST(
            COALESCE(
                (regexp_matches({{ duration_str }}, '(\d+)D', 'g'))[1]::INT * 86400,
                0
            ) +
            COALESCE(
                (regexp_matches({{ duration_str }}, 'T(\d+)H', 'g'))[1]::INT * 3600,
                0
            ) +
            COALESCE(
                (regexp_matches({{ duration_str }}, '(\d+)M(?!S)', 'g'))[1]::INT * 60,
                0
            ) +
            COALESCE(
                (regexp_matches({{ duration_str }}, '(\d+)S$', 'g'))[1]::INT,
                0
            )
            AS INT
        )
    {%- endif -%}
{%- endmacro -%}
