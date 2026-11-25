"""Sphinx configuration: pages in Markdown (MyST), vignettes as notebooks run at build time with a
cache, the API reference read from the docstrings."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

project = "PySAEM"
author = "The PySAEM authors"
copyright = "2026, the PySAEM authors"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]
myst_enable_extensions = ["dollarmath", "colon_fence"]
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {"members": True}

# a vignette runs once, then again only when its code changes
nb_execution_mode = "cache"
nb_execution_timeout = 900
nb_execution_raise_on_error = True
# the first fit compiles the solver and numba reports it on stderr
nb_output_stderr = "remove"

exclude_patterns = ["_build", "jupyter_execute", "**.ipynb_checkpoints"]

html_theme = "pydata_sphinx_theme"
html_title = "PySAEM"
html_static_path = ["../assets"]
html_theme_options = {
    "logo": {"image_light": "pysaem_logo_light.png", "image_dark": "pysaem_logo_dark.png"},
    "navbar_align": "left",
    "show_toc_level": 2,
}
