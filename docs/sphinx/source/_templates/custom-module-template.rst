{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

    {% block modules %}
    {% if modules %}
   Modules
   ------------------
.. autosummary::
   :toctree: modules
   :template: custom-module-template.rst
   :recursive:
    {% for item in modules %}
       {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}
