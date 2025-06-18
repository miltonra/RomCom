# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from contextlib import suppress

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'RomCom'
copyright = '2025, Robert A. Milton'
author = 'Robert A. Milton'
version = '1.0'
release = version


import os
import sys
sys.path.insert(0, os.path.abspath('../'))


# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
needs_sphinx = '8.3'
extensions = ['sphinx.ext.autodoc', 'autoapi.extension', 'sphinx.ext.napoleon', 'sphinx.ext.viewcode',
              "sphinx.ext.mathjax", 'sphinx_copybutton', 'sphinxarg.ext', 'sphinx.ext.inheritance_diagram',
              'sphinx.ext.graphviz', 'sphinx_design',]

autodoc_typehints_format = 'short'
python_use_unqualified_type_names = True


# Extensions
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-the-python-domain
add_module_names = False
modindex_common_prefix = ['rc.']

# https://sphinx-autoapi.readthedocs.io/en/latest/index.html
autoapi_dirs = ['../../../rc']
autoapi_add_toctree_entry = True
autoapi_root = 'pages/api'
autoapi_template_dir = '_templates'
autoapi_options = [
    'members',
    'special-members',
    'show-inheritance',
    'show-inheritance-diagram',
    'show-module-summary',
    # 'inherited-members', 'imported-members', 'undoc-members', 'private-members',
]
autoapi_keep_files = True

# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
autodoc_typehints = 'description'
autodoc_type_aliases = {'DataFrame': 'rc.base.definitions.Pd.DataFrame', 'Cunt': 'rc.base.definitions.Path',}


# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-warning-control
suppress_warnings = []

# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_css_files = ['pydata-custom.css']
html_permalinks_icon = '§'
html_title = project
html_logo = '_static/MattLogo2.svg'
html_favicon = '_static/MattLogo2.png'
html_theme = 'pydata_sphinx_theme'
html_theme_options = {'header_links_before_dropdown': 8, 'header_dropdown_text': 'Extras',
                      'secondary_sidebar_items': {"**": []}, 'navigation_depth': 5
                      }
html_static_path = ['_static']
# html_sidebars = { '**': ['sidd']}
html_show_sourcelink = False