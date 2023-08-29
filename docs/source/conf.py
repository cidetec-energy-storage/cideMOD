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
import pathlib
import sys
sys.path.insert(0, pathlib.Path(__file__).parents[2].joinpath('src').resolve().as_posix())


# -- Project information -----------------------------------------------------

project = 'cideMOD'
copyright = '2023, CIDETEC'
author = 'CIDETEC'

# The full version, including alpha/beta/rc tags
release = 'v2.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinxcontrib.bibtex',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# The master toctree document.
master_doc = 'index'

# Bibliography conf
bibtex_bibfiles = ['refs.bib']
bibtex_default_style = 'unsrt'

# autodoc options
autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
    'undoc-members': True,
    'ignore-module-all': True
}
autoclass_content = "both"

# autosummary options
autosummary_generate = True
autosummary_ignore_module_all = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    'preamble': """\\usepackage{etoolbox}
\\patchcmd{\\thebibliography}{\\section*{\\refname}}{}{}{}""",

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'cideMOD.tex', 'cideMOD Documentation',
     author, 'manual'),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'cideMOD', 'cideMOD Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'cideMOD', 'cideMOD Documentation',
     author, 'cideMOD', 'cideMOD Documentation',
     'Miscellaneous'),
]
