# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from contextlib import suppress

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'RomComma'
copyright = '2024, Robert A. Milton'
author = 'Robert A. Milton'
release = '1.0'

# -- General configuration ---------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('../'))
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'sphinx.ext.autosummary', 'sphinx.ext.viewcode', "sphinx.ext.mathjax"]

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

autosummary_generate = True

templates_path = ['_templates']
exclude_patterns = []

add_module_names = False
modindex_common_prefix = ['romcomma.']

autodoc_default_options = { 'members': True, 'undoc-members': True, 'special-members': False, 'private-members': False, 'member-order': 'bysource',
'inherited-members': True, 'show-inheritance': True}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_permalinks_icon = '§'
html_title = project + release
html_theme = 'pydata_sphinx_theme'
html_theme_options = {}
html_static_path = ['_static']
suppress_warnings = []
