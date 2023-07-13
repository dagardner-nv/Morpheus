{{ fullname | escape | underline}}

.. currentmodule:: {{ fullname }}

.. automodule:: {{ fullname }}
   :members:
   :undoc-members:
   :exclude-members: {{ classes | join(", ") }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   .. autodoc2-summary::
      {# :toctree: #}
   {% for item in attributes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block functions %}
   {% if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autodoc2-summary::
      {# :toctree: #}
   {% for item in functions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autodoc2-summary::
      :toctree:
      :template: custom-class-template.rst
   {% for item in classes %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autodoc2-summary::
      {# :toctree: #}
   {% for item in exceptions %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autodoc2-summary::
   :toctree:
   :template: custom-module-template.rst
   :recursive:
{% for item in modules %}
   {# Uncomment the following if toctree titles work with autodoc2-summary #}
   {# {{ item.replace(fullname + ".", "") }} <{{ item }}> #}
   {{ item }}
{%- endfor %}

{% endif %}
{% endblock %}
