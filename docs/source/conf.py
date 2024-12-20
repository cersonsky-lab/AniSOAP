import os, sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
root_doc = "index"

project = "AniSOAP"
copyright = "2023, Cersonsky Lab"
author = "The Cersonsky Lab"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.coverage",
    "sphinx_tabs.tabs",
    "sphinx_gallery.gen_gallery"
]

sphinx_gallery_conf = {
    'examples_dirs': '../../notebooks/',   # path to your example scripts
    'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
    'filename_pattern': '/*',
    'ignore_pattern': r'__init__\.py',
    'example_extensions': {'.py'},
    'within_subsection_order': "ExampleTitleSortKey"
}

sys.path.insert(0, os.path.abspath("../../"))

templates_path = ["_templates"]
exclude_patterns = []

# -- Napoleon settings -------------------------------------------------------
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"

html_static_path = ["_static"]

html_css_files = [
    "custom.css",
]
