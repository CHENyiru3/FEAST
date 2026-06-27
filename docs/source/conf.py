from __future__ import annotations

import sys
from pathlib import Path

# Project root (where setup.py/pyproject.toml lives)
_project_root = Path(__file__).resolve().parents[2]
_src = _project_root / "src"
sys.path.insert(0, str(_src))

project = "FEAST"
copyright = "2025, Yiru CHEN & Maizie Zhou Lab"
author = "Yiru CHEN"
release = "1.0.2"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

# Napoleon settings (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True

# Autodoc defaults
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "imported-members": False,
}
autodoc_typehints = "description"
autosummary_generate = True

# Intersphinx mappings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "anndata": ("https://anndata.readthedocs.io/en/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 3,
    "collapse_navigation": False,
    "sticky_navigation": True,
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]
