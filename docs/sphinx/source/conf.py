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

# -- General configuration ---------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('../'))
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


needs_sphinx = '8.0'
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.autosummary', 'sphinx.ext.viewcode', "sphinx.ext.mathjax", 'sphinx_copybutton']

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autodoc_preserve_defaults = True
autodoc_class_signature = 'separated'
autodoc_type_aliases = {'Store.Path': 'Store.Path'}

autosummary_generate = True
autosummary_imported_members = False
templates_path = ['_templates']
exclude_patterns = []

add_module_names = False
modindex_common_prefix = ['rc.']

autodoc_default_options = {'members': True, 'private-members': False, 'inherited-members': True, 'show-inheritance': True,
                            'special-members': '__init__, __call__'
}

html_css_files = ['pydata-custom.css']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_permalinks_icon = '§'
html_title = project
html_logo = '_static/logo.svg'
html_favicon = '_static/favicon.svg'
html_theme = 'pydata_sphinx_theme'
html_theme_options = {}
html_static_path = ['_static']
suppress_warnings = []
