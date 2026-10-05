"""Observation-sequence strip for the landing page.

Stdlib only, so tests/docs can import it without Sphinx. The strip is HTML,
not SVG: an SVG scaled to a phone shrinks its labels below legibility.
"""

from __future__ import annotations

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


def render_sequence_strip(hrefs: Mapping[str, str], current: str | None = None) -> str:
    """Return the strip as HTML; every stage links to ``hrefs[key]``, ``current`` is highlighted."""
    keys = [key for key, _ in STAGES]
    unknown = sorted((set(hrefs) | ({current} if current else set())) - set(keys))
    if unknown:
        raise ValueError(f"unknown sequence-strip stage(s): {unknown}")
    missing = [key for key in keys if key not in hrefs]
    if missing:
        raise ValueError(f"no link for sequence-strip stage(s): {missing}")

    stages = []
    for key, label in STAGES:
        current_attr = ' aria-current="page"' if key == current else ""
        stages.append(
            f'<li><a class="sequence-strip__stage" href="{escape(hrefs[key], quote=True)}"{current_attr}>'
            f"{escape(label)}</a></li>"
        )
    frames = "".join(
        f'<span class="sequence-strip__frame sequence-strip__frame--{label.lower()}" '
        f'style="flex-grow: {weight}">{label}</span>'
        for label, weight in FRAMES
    )
    sequence = ", ".join(label for label, _ in FRAMES)
    return (
        '<nav class="sequence-strip" aria-label="Pipeline stages">'
        f'<ol class="sequence-strip__stages">{"".join(stages)}</ol>'
        f'<div class="sequence-strip__frames" role="img" aria-label="A SPHERE observing sequence: {sequence}">'
        f"{frames}</div></nav>"
    )
