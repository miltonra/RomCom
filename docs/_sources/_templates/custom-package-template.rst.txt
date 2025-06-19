{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

    {% block attributes %}
    {% if attributes %}
    .. rubric:: Module Attributes

    .. autosummary::

    {% for item in attributes %}
      {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block modules %}
    {% if modules %}
    Modules
    ------------------

    .. autosummary::
       :template: custom-module-template.rst
       :recursive:

    {% for item in modules %}
       {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}
