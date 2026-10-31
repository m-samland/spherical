"""Observation-sequence strip for the landing page.

Stdlib only, so tests/docs can import it without Sphinx. The strip is HTML,
not SVG: an SVG scaled to a phone shrinks its labels below legibility.
"""

from __future__ import annotations

import posixpath
from collections.abc import Mapping
from html import escape

#: Pipeline stages in order, as (key, label).
STAGES: tuple[tuple[str, str], ...] = (
    ("archive", "Archive"),
    ("database", "Database"),
    ("reduction", "Reduction"),
    ("trap", "TRAP"),
    ("products", "Products"),
)

#: The standard SPHERE sequence as (frame type, relative width); CORO frames carry the science.
FRAMES: tuple[tuple[str, int], ...] = (("FLUX", 1), ("CENTER", 1), ("CORO", 4), ("CENTER", 1), ("FLUX", 1))


def stage_hrefs(found_docs: set[str], stage_pages: Mapping[str, str], from_docname: str, suffix: str) -> dict[str, str]:
    """Link per stage from ``from_docname`` to the page in ``stage_pages``, which must be a known document."""
    unknown = sorted(docname for docname in stage_pages.values() if docname not in found_docs)
    if unknown:
        raise ValueError(f"sequence-strip stage pages name unknown documents: {unknown}")
    start = posixpath.dirname(from_docname) or "."
    return {key: posixpath.relpath(docname + suffix, start) for key, docname in stage_pages.items()}


def render_sequence_strip(hrefs: Mapping[str, str], current: str | None = None, compact: bool = False) -> str:
    """Return the strip as HTML; every stage links to ``hrefs[key]``, ``current`` is highlighted.

    The compact strip, shown at the top of a stage's pages, has no frame row. Its
    highlighted link usually leads to another page, so it is marked as the reader's
    location rather than as the current page.
    """
    keys = [key for key, _ in STAGES]
    unknown = sorted((set(hrefs) | ({current} if current else set())) - set(keys))
    if unknown:
        raise ValueError(f"unknown sequence-strip stage(s): {unknown}")
    missing = [key for key in keys if key not in hrefs]
    if missing:
        raise ValueError(f"no link for sequence-strip stage(s): {missing}")

    stages = []
    for key, label in STAGES:
        current_attr = f' aria-current="{"location" if compact else "page"}"' if key == current else ""
        stages.append(
            f'<li><a class="sequence-strip__stage" href="{escape(hrefs[key], quote=True)}"{current_attr}>'
            f"{escape(label)}</a></li>"
        )
    frames = "".join(
        f'<span class="sequence-strip__frame sequence-strip__frame--{label.lower()}" '
        f'style="flex-grow: {weight}">{label}</span>'
        for label, weight in FRAMES
    )
    if compact:
        return (
            '<nav class="sequence-strip sequence-strip--compact" aria-label="Pipeline stages">'
            f'<ol class="sequence-strip__stages">{"".join(stages)}</ol></nav>'
        )
    sequence = ", ".join(label for label, _ in FRAMES)
    return (
        '<nav class="sequence-strip" aria-label="Pipeline stages">'
        f'<ol class="sequence-strip__stages">{"".join(stages)}</ol>'
        f'<div class="sequence-strip__frames" role="img" aria-label="A SPHERE observing sequence: {sequence}">'
        f"{frames}</div></nav>"
    )
