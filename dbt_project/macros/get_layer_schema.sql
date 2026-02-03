{%- macro get_layer_schema(layer) -%}
    {#
    Helper macro to get the correct schema name for each layer.
    
    Usage: {{ get_layer_schema('staging') }}
    Returns: staging
    
    Supports:
    - landing: youtube_raw (raw data from Fivetran)
    - staging: staging (cleaned data)
    - intermediate: intermediate (business logic)
    - mart: mart (final tables for BI)
    #}
    
    {%- set layer_schemas = {
        'landing': 'youtube_raw',
        'staging': 'staging',
        'intermediate': 'intermediate',
        'mart': 'mart'
    } -%}
    
    {%- if layer in layer_schemas -%}
        {{ layer_schemas[layer] }}
    {%- else -%}
        {{ exceptions.raise_compiler_error("Invalid layer: " ~ layer ~ ". Must be one of: landing, staging, intermediate, mart") }}
    {%- endif -%}
{%- endmacro -%}
