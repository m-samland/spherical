"""Sphinx configuration for the spherical documentation."""

import os
import sys
from datetime import date
from importlib.metadata import version as package_version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "_ext"))

project = "spherical"
author = "Matthias Samland"
copyright = f"2019-{date.today().year}, {author}"
release = package_version("spherical")
version = ".".join(release.split(".")[:2])

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxarg.ext",
    "spherical_docs",
]

templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["custom.css"]
exclude_patterns = ["_build", "jupyter_execute", "superpowers", "PR_release_*.md", "**.ipynb_checkpoints", "tutorials/runs"]

# -- MyST / notebooks -------------------------------------------------------
# Tutorials are executed locally and committed with outputs; never run them here.
nb_execution_mode = "off"
myst_enable_extensions = ["colon_fence", "deflist", "substitution"]
myst_heading_anchors = 3
# Literal blocks in docstrings are file listings, not Python; code fences name their language.
highlight_language = "none"

# -- API reference ----------------------------------------------------------
autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
# charis is git-only and ships calibration data; mocking it keeps the RTD build light.
autodoc_mock_imports = ["charis"]

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_preprocess_types = True
# Section titles used in existing docstrings that napoleon does not know.
napoleon_custom_sections = [
    "Required Input Files",
    "Generated Output Files",
    "Modified Output Files",
    "Output Dimensions",
    "Scientific Context",
    "Description",
    "Usage",
    "Key Features",
    "Main Components",
    "Header Fields Extracted",
    "Classes",
    "Functions",
    ("Return", "Returns"),
]

def _offline() -> bool:
    return os.environ.get("SPHINX_OFFLINE", "").strip().lower() not in ("", "0", "false", "no")


if _offline():
    intersphinx_mapping = {}
else:
    intersphinx_mapping = {
        "python": ("https://docs.python.org/3", None),
        "numpy": ("https://numpy.org/doc/stable/", None),
        "scipy": ("https://docs.scipy.org/doc/scipy/", None),
        "astropy": ("https://docs.astropy.org/en/stable/", None),
        "astroquery": ("https://astroquery.readthedocs.io/en/latest/", None),
        "matplotlib": ("https://matplotlib.org/stable/", None),
        "pandas": ("https://pandas.pydata.org/docs/", None),
    }

# -- HTML -------------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "spherical"  # the RTD flyout shows the version
html_favicon = "_static/favicon.png"
html_theme_options = {
    "github_url": "https://github.com/m-samland/spherical",
    "use_edit_page_button": True,
    "navigation_with_keys": False,
    # The header is Night in both modes, so one ringed logo serves both.
    "logo": {
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
        "text": "spherical",
        "alt_text": "spherical",
    },
    "header_links_before_dropdown": 5,
}
html_sidebars = {"index": []}  # the landing page has no section sidebar
html_context = {
    "github_user": "m-samland",
    "github_repo": "spherical",
    "github_version": "develop",
    "doc_path": "docs",
}
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True


def _hide_edit_button_on_generated_pages(app, pagename, templatename, context, doctree):
    """Autosummary stubs are build products with no file in the repo to edit."""
    if pagename.startswith("reference/api/generated/"):
        context["theme_use_edit_page_button"] = False


def setup(app):
    app.connect("html-page-context", _hide_edit_button_on_generated_pages)
