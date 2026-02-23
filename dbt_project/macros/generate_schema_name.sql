{% macro generate_schema_name(custom_schema_name, node) -%}
    {#
    Override default dbt schema naming.
    Default: <target.schema>_<custom_schema>  e.g. yt_dbt_staging
    This:    <custom_schema>                  e.g. staging, intermediate, mart
    #}
    {%- set default_schema = target.schema -%}
    {%- if custom_schema_name is none -%}
        {{ default_schema }}
    {%- else -%}
        {{ custom_schema_name | trim }}
    {%- endif -%}
{%- endmacro %}
