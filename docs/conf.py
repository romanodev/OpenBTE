# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'OpenBTE'
copyright = '2022, Giuseppe Romano'
author = 'Giuseppe Romano'
release = '0.1'


#import os
#import sys
#sys.path.insert(0, os.path.abspath('../openbte'))


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
    'nbsphinx',
    #'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx_toolbox.more_autodoc.autonamedtuple',
    'sphinx_autodoc_typehints',
    'sphinx.ext.napoleon'
]

#autodoc_typehints='none'

autodoc_typehints = 'description'  # show type hints in doc body instead of signature
autoclass_content = 'both'  # get docstring from class level and init simultaneously

napolean_use_rtype = False

always_document_param_types = False

remove_from_toctrees = ["_autosummary/*"]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

pygments_style = None

autosummary_generate = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_book_theme'
#html_theme = 'sphinx_rtd_theme'
html_logo = '_static/openbte_logo.png'

html_static_path = ['_static']

source_suffix = ['.rst', '.ipynb', '.md']

# The main toctree document.
main_doc = 'index'


# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
html_theme_options = {
    'logo_only': True,
    'show_toc_level': 2,
    'repository_url': 'https://github.com/romanodev/OpenBTE',
    'use_repository_button': True,     # add a "link to repository" button
}



