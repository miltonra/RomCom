# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from contextlib import suppress

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'RomCom'
copyright = '2024, Robert A. Milton'
author = 'Robert A. Milton'
version = '1.0'
release = version


import os
import sys
sys.path.insert(0, os.path.abspath('../'))


# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
needs_sphinx = '8.0'
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.autosummary', 'sphinx.ext.viewcode',
              "sphinx.ext.mathjax", 'sphinx_copybutton', 'sphinxarg.ext']


# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-warning-control
suppress_warnings = []


# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-the-python-domain
add_module_names = False
modindex_common_prefix = ['rc.']


# https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html
autosummary_generate = True
autosummary_imported_members = False


# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_preserve_defaults = True
autodoc_class_signature = 'separated'
autodoc_type_aliases = {'Store.Path': 'Store.Path', 'Data': 'Data'}
templates_path = ['_templates']
exclude_patterns = []
autodoc_default_options = {'members': True, 'private-members': False, 'inherited-members': True,
                           'show-inheritance': True, 'special-members': '__init__, __call__'
}


# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_css_files = ['pydata-custom.css']
html_permalinks_icon = '§'
html_title = project
html_logo = '_static/logo.svg'
html_favicon = '_static/favicon.png'
html_theme = 'pydata_sphinx_theme'
html_theme_options = {'header_links_before_dropdown': 8, 'header_dropdown_text': 'Extras'}
html_static_path = ['_static']
html_sidebars = {"**": []}
html_show_sourcelink = False