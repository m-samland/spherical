"""The big-picture map: what each pipeline stage does and what you get from it.

Stdlib only. Stage keys and labels come from ``sequence_strip.STAGES`` so the map
and the landing-page strip name the same stages.
"""

from __future__ import annotations

from collections.abc import Mapping
from html import escape

import sequence_strip

#: Per stage, (what happens, what you get). Short, so they fit a phone column.
STAGE_DETAILS: dict[str, tuple[str, str]] = {
    "archive": ("ESO stores every raw SPHERE frame with its header.", "Raw FITS files and their headers."),
    "database": (
        "spherical groups the headers into sequences and adds star properties.",
        "Tables you can search by target, mode, conditions and quality.",
    ),
    "reduction": (
        "spherical downloads a sequence, calibrates it and finds the star.",
        "Data cubes per frame type, star positions, angles and the stellar PSF.",
    ),
    "trap": (
        "TRAP models the stellar speckles and searches for companions.",
        "Detection maps, contrast curves and candidate lists.",
    ),
    "products": (
        "You read the files for your science.",
        "Companion spectra or photometry, astrometry and detection limits.",
    ),
}


def render_pipeline_map(hrefs: Mapping[str, str]) -> str:
    """Return the map as HTML; each stage name links to ``hrefs[key]``."""
    keys = [key for key, _ in sequence_strip.STAGES]
    missing = [key for key in keys if key not in hrefs]
    if missing:
        raise ValueError(f"no link for pipeline-map stage(s): {missing}")
    unknown = sorted(set(hrefs) - set(keys))
    if unknown:
        raise ValueError(f"unknown pipeline-map stage(s): {unknown}")
    items = []
    for key, label in sequence_strip.STAGES:
        does, gives = STAGE_DETAILS[key]
        items.append(
            '<li class="pipeline-map__stage">'
            f'<a class="pipeline-map__name" href="{escape(hrefs[key], quote=True)}">{escape(label)}</a>'
            f'<p class="pipeline-map__does">{escape(does)}</p>'
            f'<p class="pipeline-map__gives"><span class="pipeline-map__gives-label">You get</span> {escape(gives)}</p>'
            "</li>"
        )
    return f'<ol class="pipeline-map" aria-label="From the archive to science products">{"".join(items)}</ol>'
