#  This file is part of the RomCom Python Package <https://github.com/miltonra/RomCom>
#
#  Copyright (C) 2027 Robert A. Milton
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as
#  published by the Free Software Foundation, either version 3 of the
#  License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

""" Configuration file for the Sphinx documentation builder.

    For the full list of built-in configuration values, see the
    `documentation <https: //www.sphinx-doc.org/en/master/usage/configuration.html>`__. """

import os
import sys
sys.path.insert(0, os.path.abspath('../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = 'RomCom'
copyright = '2027, Robert A. Milton'
author = 'Robert A. Milton'
version = '1.0'
release = version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
needs_sphinx = '8.3'
add_module_names = False
modindex_common_prefix = ['rc.']

# -- Extensions --------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-the-python-domain
extensions = ['sphinx.ext.napoleon', 'sphinx.ext.autodoc', 'autoapi.extension',
              "sphinx.ext.mathjax", 'sphinx_copybutton', 'sphinxarg.ext', 'sphinx.ext.inheritance_diagram',
              'sphinx.ext.graphviz', 'sphinx_design', 'sphinx.ext.viewcode', ]

# https://sphinx-autoapi.readthedocs.io/en/latest/index.html
autoapi_dirs = ['../../../rc']
autoapi_add_toctree_entry = True
autoapi_root = 'pages/api'
autoapi_template_dir = '_templates'
autoapi_own_page_level = 'attribute'
autoapi_options = [
    'members',
    'show-inheritance',
    'show-inheritance-diagram',
    'show-module-summary',
    'inherited-members',
    # 'special-members', # 'imported-members', 'undoc-members', 'private-members',
]
autoapi_keep_files = True
autoapi_python_class_content = 'both'

# https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html
autodoc_typehints = 'description'

# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-warning-control
suppress_warnings = []

# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output
html_css_files = ['pydata-custom.css']
html_permalinks_icon = '§'
html_title = project
html_logo = '_static/MattLogo2.svg'
html_favicon = '_static/MattLogo2.png'
html_theme = 'pydata_sphinx_theme'
html_theme_options = {'header_links_before_dropdown': 12, 'header_dropdown_text': 'Extras',
                      'secondary_sidebar_items': {"**": []}, 'navigation_depth': 5,
                      "github_url": "https://github.com/miltonra/RomCom",
                      }
html_static_path = ['_static']
html_show_sourcelink = False
html_sidebars = {
    "**": ["page-toc", "sidebar-nav-bs"]
}