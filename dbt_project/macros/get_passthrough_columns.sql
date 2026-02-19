{% macro get_passthrough_columns(variable_name) %}
    {%- set passthrough_columns = var(variable_name, []) -%}
    {%- if passthrough_columns -%}
        {%- for column in passthrough_columns -%}
            , {{ column.name }}
            {%- if column.alias %} as {{ column.alias }}{% endif -%}
        {%- endfor -%}
    {%- endif -%}
{% endmacro %}
