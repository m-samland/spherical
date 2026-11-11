"""Sphinx configuration for the spherical documentation."""

import os
import sys
from datetime import date
from importlib.metadata import version as package_version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "_ext"))


def _offline() -> bool:
    return os.environ.get("SPHINX_OFFLINE", "").strip().lower() not in ("", "0", "false", "no")


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

# Every cross-reference must resolve. Offline there are no intersphinx inventories, so
# every numpy or astropy reference would warn; set here, not with -n, which would override it.
nitpicky = not _offline()

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
# Show defaults as written; the repr of a default such as registry=STEP_REGISTRY is not a type.
autodoc_preserve_defaults = True

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
# Off because napoleon merges napoleon_type_aliases into autodoc_type_aliases in place
# (sphinx.ext.napoleon.docstring._lookup_annotation), which breaks annotations using the
# same names. Both "Attributes" sections in the code give their types anyway.
napoleon_attr_annotations = False
# Short type names used in docstrings, mapped to their documented targets.
# Words that docstrings use as types but that name no class.
nitpick_ignore_regex = [("py:class", r"(sequence|scalar|iterable|collection|callable|optional)")]
# TRAP publishes no intersphinx inventory.
nitpick_ignore = [("py:class", "trap.parameters.TrapConfig")]
napoleon_type_aliases = {
    "np.ndarray": "numpy.ndarray",
    "ndarray": "numpy.ndarray",
    "array": "numpy.ndarray",
    # Glossary terms of other projects need their inventories, so they are plain text offline.
    "array_like": "array_like" if _offline() else ":term:`numpy:array_like`",
    "Table": "astropy.table.Table",
    "Table.Row": "astropy.table.Row",
    "Path": "pathlib.Path",
    "pd.DataFrame": "pandas.DataFrame",
    "dataframe": "pandas.DataFrame",
    "path-like": "path-like" if _offline() else ":term:`python:path-like object`",
    "IFSObservation": "spherical.database.ifs_observation.IFSObservation",
    "IRDISObservation": "spherical.database.irdis_observation.IRDISObservation",
    "IFSReductionConfig": "spherical.pipeline.pipeline_config.IFSReductionConfig",
    "IRDISReductionConfig": "spherical.pipeline.pipeline_config.IRDISReductionConfig",
}

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

# -- Sequence strip -----------------------------------------------------------
# The page each pipeline stage links to, on the landing strip, the big-picture map and
# the compact strip of every page with a ``stage:`` front-matter key.
spherical_stage_pages = {
    "archive": "concepts/database",
    "database": "concepts/database",
    "reduction": "concepts/anatomy",
    "trap": "concepts/adi-sdi-trap",
    "products": "concepts/products",
}

# -- Link check ---------------------------------------------------------------
# These hosts refuse automated clients (403, 405 or a hang), so linkcheck cannot tell a
# broken link from a blocked one. Check them by hand in a browser before a release.
linkcheck_ignore = [
    r"https://doi\.org/10\.1051/",  # A&A
    r"https://doi\.org/10\.1117/",  # SPIE
    r"https://ui\.adsabs\.harvard\.edu/",
    r"https://stackoverflow\.com/",
    r"https://hc-dc\.cnrs\.fr/",  # SPHERE Data Center
]
linkcheck_allowed_redirects = {
    r"https://doi\.org/.*": r".*",  # a DOI resolves to the publisher's page
    r"https://spherical-hci\.readthedocs\.io/?": r"https://spherical-hci\.readthedocs\.io/en/latest/",
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
