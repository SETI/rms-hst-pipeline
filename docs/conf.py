#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

import datetime
import importlib.metadata
import os
import sys
sys.path.insert(0, os.path.abspath('../src'))

# -- Project information -----------------------------------------------------

project = 'rms-hst-pipeline'
copyright = f'{datetime.date.today().year}, SETI Institute'
author = 'SETI Institute'

try:
    release = importlib.metadata.version('rms-hst-pipeline')
except importlib.metadata.PackageNotFoundError:
    release = '0.0.0'  # fallback for a source checkout that was never installed

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'myst_parser',
]

templates_path = ['_templates']

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = ['.rst', '.md']

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'

add_module_names = False

# -- Extension configuration -------------------------------------------------

# The docstrings in this project use an `Input:` / `Returns:` block rather than Google or
# NumPy style. Napoleon does not recognize `Input:`, so parameter lists render as plain
# text until the docstrings are converted. Both styles stay enabled so that the ones
# already converted render correctly.
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
}

# Not nitpicky yet. The API is undocumented in Sphinx terms, so turning this on now would
# bury real problems under unresolved references. Enable it together with the Sphinx step
# in .github/workflows/run-tests.yml.
nitpicky = False

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

myst_heading_anchors = 2
