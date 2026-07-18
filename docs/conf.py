"""Sphinx configuration for Benchtop documentation."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow importing the package from src/ when building locally
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

project = "Benchtop"
copyright = "2026, Jonah Huggins, Marc Birtwistle"
author = "Jonah Huggins, Marc Birtwistle"
release = "0.1.0"
version = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Avoid failing the build if optional theme assets are missing
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 3,
}
