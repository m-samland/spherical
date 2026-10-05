"""Step diagram for "Anatomy of a reduction", generated from the step registry.

Stdlib only, so tests/docs can import it without Sphinx. HTML rather than SVG,
for the same reason as the sequence strip: labels must stay legible on a phone.
"""

from __future__ import annotations

from collections.abc import Sequence, Set
from dataclasses import dataclass
from html import escape

#: Diagram phases in pipeline order, as (name, steps). Every registered step of
#: both instruments must appear in exactly one phase.
PHASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Get the data", ("download_data",)),
    (
        "Calibrate and build cubes",
        ("reduce_calibration", "extract_cubes", "bundle_output", "irdis_calibration", "preprocess_irdis"),
    ),
    ("Describe the frames", ("compute_frames_info", "cube_header_update")),
    ("Find the star", ("find_centers", "plot_image_center_evolution", "process_extracted_centers")),
    ("Calibrate the flux", ("calibrate_spot_photometry", "calibrate_flux_psf", "spot_to_flux")),
    ("Optional products", ("align_frames",)),
    ("Find companions", ("run_trap_reduction", "run_trap_detection")),
)

#: External package that does the work of a step.
TOOLS: dict[str, str] = {
    "reduce_calibration": "charis",
    "extract_cubes": "charis",
    "run_trap_reduction": "TRAP",
    "run_trap_detection": "TRAP",
}


@dataclass(frozen=True)
class PhaseRow:
    """One phase of the diagram with the steps each instrument runs in it."""

    name: str
    ifs: tuple[str, ...]
    irdis: tuple[str, ...]

    @property
    def shared(self) -> bool:
        return self.ifs == self.irdis


def _phase_of(step: str) -> int:
    for index, (_, steps) in enumerate(PHASES):
        if step in steps:
            return index
    raise ValueError(f"step {step!r} is in no phase of step_diagram.PHASES; add it")


def build_phase_rows(ifs_order: Sequence[str], irdis_order: Sequence[str]) -> list[PhaseRow]:
    """Group both step orders into phases; each lane keeps its instrument's order."""
    for label, order in (("IFS", ifs_order), ("IRDIS", irdis_order)):
        phases = [_phase_of(step) for step in order]
        if phases != sorted(phases):
            raise ValueError(f"{label} step order contradicts the phase order of step_diagram.PHASES: {list(order)}")
    return [
        PhaseRow(
            name=name,
            ifs=tuple(step for step in ifs_order if step in steps),
            irdis=tuple(step for step in irdis_order if step in steps),
        )
        for name, steps in PHASES
    ]


def _lane(steps: Sequence[str], label: str, modifier: str, optional: Set[str]) -> str:
    items = []
    for step in steps:
        classes = "step-diagram__step" + (" step-diagram__step--optional" if step in optional else "")
        tool = f'<span class="step-diagram__tool">{escape(TOOLS[step])}</span>' if step in TOOLS else ""
        note = '<span class="step-diagram__note">off by default</span>' if step in optional else ""
        items.append(f'<li class="{classes}"><code>{escape(step)}</code>{tool}{note}</li>')
    return (
        f'<div class="step-diagram__lane step-diagram__lane--{modifier}">'
        f'<p class="step-diagram__lane-label">{escape(label)}</p>'
        f'<ol class="step-diagram__steps" aria-label="{escape(label, quote=True)}">{"".join(items)}</ol></div>'
    )


def render_step_diagram(rows: Sequence[PhaseRow], optional: Set[str]) -> str:
    """Return the diagram as HTML. ``optional`` names steps that are off by default."""
    phases = []
    for row in rows:
        if row.shared:
            lanes = _lane(row.ifs, "IFS and IRDIS", "shared", optional)
        else:
            lanes = _lane(row.ifs, "IFS", "ifs", optional) + _lane(row.irdis, "IRDIS", "irdis", optional)
        phases.append(
            f'<li class="step-diagram__phase"><p class="step-diagram__phase-name">{escape(row.name)}</p>'
            f'<div class="step-diagram__lanes">{lanes}</div></li>'
        )
    return (
        '<figure class="step-diagram">'
        f'<ol class="step-diagram__phases">{"".join(phases)}</ol>'
        "<figcaption>The steps of a reduction in the order they run, grouped by what they do. "
        "The names are the switches in <code>config.steps</code>.</figcaption></figure>"
    )
