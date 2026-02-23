{%- macro get_layer_schema(layer) -%}
    {#
    Returns the BigQuery dataset name for a given pipeline layer.
    Usage: {{ get_layer_schema('staging') }}
    #}
    {%- set layer_schemas = {
        'landing':      'youtube_raw',
        'staging':      'staging',
        'intermediate': 'intermediate',
        'mart':         'mart'
    } -%}

    {%- if layer in layer_schemas -%}
        {{ layer_schemas[layer] }}
    {%- else -%}
        {{ exceptions.raise_compiler_error("Invalid layer: " ~ layer ~ ". Must be one of: landing, staging, intermediate, mart") }}
    {%- endif -%}
{%- endmacro -%}
