# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from importlib.metadata import version as _version
import sys
from unittest.mock import Mock

# Mock _ok module
sys.modules["_ok"] = Mock()

project = "miniscope-io"
copyright = "2023, Jonny"
author = "Jonny, Takuya"
release = _version("mio")
html_title = "mio"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.graphviz",
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinxcontrib.autodoc_pydantic",
    "sphinxcontrib.programoutput",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx_click",
    "sphinx_design",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "rich": ("https://rich.readthedocs.io/en/stable/", None),
    "pyserial": ("https://pyserial.readthedocs.io/en/stable/", None),
}

# ----------
# package settings
# ----------------

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# graphviz
graphviz_output_format = "svg"


# autodoc
# Mock imports for packages we don't have yet - this one is
# for opal kelley stuff we need to figure out the licensing for
autodoc_mock_imports = ["routine"]
autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_json_error_strategy = "coerce"

# todo
todo_include_todos = True
