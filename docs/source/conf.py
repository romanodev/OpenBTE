# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import sphinx_rtd_theme


# -- Project information -----------------------------------------------------

project = 'OpenBTE'
copyright = '2020, Giuseppe Romano'
author = 'Giuseppe Romano'

# The full version, including alpha/beta/rc tags
release = '1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['recommonmark','sphinx_rtd_theme','nbsphinx','sphinx.ext.mathjax','sphinx.ext.autodoc','sphinx.ext.viewcode','sphinx.ext.autosummary','sphinx.ext.intersphinx']

pygments_style = None

nbsphinx_codecell_lexer = 'ipython3'

nbsphinx_execute = 'never'

#nbsphinx_prolog = """

#  .. only:: html

#   .. role:: raw-html(raw)
#        :format: html

#   .. nbinfo::

#     Interactive online version:
#     :raw-html:`<a href="https://colab.research.google.com/github/romanodev/OpenBTE/blob/master/docs/source/Tutorial.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg" style="vertical-align:text-bottom"></a>`

#"""

highlight_language = 'python3'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', '**.ipynb_checkpoints']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
