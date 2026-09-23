"""Single source of truth for pipeline step identity, products, and ordering.

Update THIS module (and nowhere else) when a step's output filenames or its
logged ``extra={"step": ...}`` name change: ``should_run``, ``check_output``,
and the ``reduction_status`` aggregator all rely on the names declared here.

Intentionally imports only the standard library so it stays usable without the
pipeline extra (no charis/trap/scipy).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass(frozen=True)
class StepDirs:
    """Runtime directories a step reads/writes. Only the fields a given step
    needs are populated; the rest keep harmless defaults."""

    converted_dir: Path = Path()
    cube_outputdir: Path = Path()
    wavecal_outputdir: Path = Path()
    irdis_calibration_dir: Path = Path()
    trap_result_folder: Path | None = None
    # The frame types the observation actually carries, as the reduction
    # drivers compute them. WAFFLE_MODE is a majority-exposure-time test rather
    # than an existence test, so a waffle sequence can still carry CORO frames,
    # and the steps that write one product per frame type write CORO products
    # for it. Resume has to gate on what the observation has, not on which
    # frame type carries the science.
    available_frame_types: tuple[str, ...] = ("CORO", "CENTER", "FLUX")

    @property
    def additional_outputs(self) -> Path:
        # The step modules write this inside converted_dir (see find_star.py etc.).
        return self.converted_dir / "additional_outputs"


@dataclass(frozen=True)
class StepSpec:
    """Describes one pipeline step.

    Args:
        log_name: The ``extra={"step": ...}`` string the step emits in structured
            logs (may differ from the config-attribute key).
        outputs: Files whose presence means the step is complete, given StepDirs.
        is_final: Marks the terminal step whose completion means the IFS target
            is done.
        internal_guard: True when the step decides skip itself (calibration,
            TRAP reduction) or is inherently idempotent (download); such steps are
            not gated by ``should_run`` and declare no ``outputs``.
        leaf: True when nothing downstream consumes the step's outputs. Such a
            step never starts a ``_forced`` cascade, so forcing it re-runs only
            itself instead of dragging TRAP along, and it is excluded from
            ``check_output``, so an opt-in step nobody enabled does not make a
            finished reduction look incomplete. Being forced *by* an earlier step
            still works.
    """

    log_name: str
    outputs: Callable[[StepDirs], list[Path]]
    is_final: bool = False
    internal_guard: bool = False
    is_trap: bool = False  # TRAP step: excluded from IFS-reduction check_output()
    leaf: bool = False


def target_folder_string(main_id: str, filter_name: str, night_start: str) -> str:
    """Return the ``{target}/{filter}/{date}`` segment identifying one observation.

    The target name has its internal whitespace collapsed and its spaces replaced
    by underscores, so ``"HD  3795"`` and ``"HD 3795"`` name the same directory.

    Args:
        main_id: SIMBAD main identifier of the host.
        filter_name: Observing mode, e.g. ``DB_H23`` or ``OBS_YJ``.
        night_start: Night of the observation as ``YYYY-MM-DD``.
    """
    target = "_".join(str(main_id).split())
    return f"{target}/{filter_name}/{night_start}"


def trap_result_folder(
    reduction_directory: Path | str,
    main_id: str,
    filter_name: str,
    night_start: str,
    instrument: str = "IRDIS",
) -> Path:
    """Return the TRAP result folder for one observation.

    ``{reduction_directory}/{instrument}/trap/{target}/{filter}/{date}``. Both
    IFS and IRDIS use this layout; there is no ``{method}`` segment, matching the
    historical IFS path.

    Public because consumers outside the reduction (multi-epoch analysis, ad-hoc
    inspection) need to find TRAP output, and this module is where the layout is
    defined. Taking plain strings rather than an observation object keeps it
    usable straight from observation-table rows.
    """
    return Path(reduction_directory) / instrument / "trap" / target_folder_string(main_id, filter_name, night_start)


def marker_for(step: str, directory: Path | str) -> Path:
    """Path of a step's zero-byte completion marker inside *directory*."""
    return Path(directory) / f".{step}.done"


def write_marker(step: str, directory: Path | str) -> None:
    """Write a step's completion marker (parent-side, after the step succeeds)."""
    marker = marker_for(step, directory)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.touch()


def _marker_output(step: str, dirs: StepDirs) -> list[Path]:
    directory = {
        "extract_cubes": dirs.cube_outputdir,
        "run_trap_detection": dirs.trap_result_folder,
        "align_frames": dirs.converted_dir,
    }[step]
    return [marker_for(step, directory)]


def _converted(
    *names: str,
    per_frame: tuple[str, ...] = (),
    frame_types: tuple[str, ...] = ("CORO", "CENTER", "FLUX"),
) -> Callable[[StepDirs], list[Path]]:
    """Outputs under ``converted/``.

    ``names`` are literal file names. Each template in ``per_frame`` carries a
    ``{frame}`` marker and expands once per frame type in ``frame_types`` that
    the observation actually carries, lowercased. A sequence with no CORO
    frames therefore declares no CORO products, and one that has them declares
    them whether or not the CORO frames are the science frames.

    The marker is substituted textually rather than through ``str.format``, so
    a file name containing a brace stays literal.
    """
    def resolve(d: StepDirs) -> list[Path]:
        present = [ft for ft in frame_types if ft in d.available_frame_types]
        expanded = list(names)
        for template in per_frame:
            expanded.extend(template.replace("{frame}", ft.lower()) for ft in present)
        return [d.converted_dir / n for n in dict.fromkeys(expanded)]

    return resolve


def _additional(*names: str) -> Callable[[StepDirs], list[Path]]:
    return lambda d: [d.additional_outputs / n for n in names]


_NONE: Callable[[StepDirs], list[Path]] = lambda d: []  # noqa: E731

# Insertion order == canonical pipeline order in ifs_reduction.py, TRAP last.
STEP_REGISTRY: dict[str, StepSpec] = {
    "download_data": StepSpec("download_data", _NONE, internal_guard=True),
    "reduce_calibration": StepSpec("wavelength_calibration", _NONE, internal_guard=True),
    "extract_cubes": StepSpec("extract_cubes", lambda d: _marker_output("extract_cubes", d)),
    "bundle_output": StepSpec(
        "bundle_output",
        # The data cube and its inverse-variance sibling are written together,
        # unconditionally, for every frame type bundled (bundle_output.py), so
        # both gate resume. The parallactic-angle file and the hexagons and
        # residuals variants are written conditionally and are not declared.
        _converted(
            "wavelengths.fits",
            per_frame=("{frame}_cube.fits", "{frame}_ivar_cube.fits"),
        ),
    ),
    "compute_frames_info": StepSpec(
        "frame_info_computation",
        _converted(per_frame=("frames_info_{frame}.csv",)),
    ),
    "cube_header_update": StepSpec("cube_header_update", _NONE),
    "find_centers": StepSpec("fit_centers", _converted("image_centers.fits")),
    "plot_image_center_evolution": StepSpec("plot_center_evolution", _NONE),
    "process_extracted_centers": StepSpec("polynomial_center_fit", _converted("image_centers_fitted_robust.fits")),
    "calibrate_spot_photometry": StepSpec("spot_photometry_calibration", _additional("spot_amplitudes.fits")),
    "calibrate_flux_psf": StepSpec("flux_psf_calibration", _converted("psf_cube_for_postprocessing.fits")),
    "spot_to_flux": StepSpec("spot_to_flux_normalization", _converted("spot_amplitude_variation.fits"), is_final=True),
    # Leaf: the aligned cube feeds nothing downstream. Gated on a marker rather
    # than a filename because the output is named after the science frame type,
    # which the registry cannot know from StepDirs alone. Consequence: deleting
    # the aligned cube does not regenerate it and changing shift_method does not
    # re-run the step; both need force={"align_frames"}. Aligning every frame
    # type would make the names deterministic and retire the marker (#177).
    "align_frames": StepSpec(
        "frame_alignment",
        lambda d: _marker_output("align_frames", d),
        leaf=True,
    ),
    "run_trap_reduction": StepSpec("trap_reduction", _NONE, internal_guard=True, is_trap=True),
    "run_trap_detection": StepSpec("trap_detection", lambda d: _marker_output("run_trap_detection", d), is_trap=True),
}

STEP_ORDER: list[str] = list(STEP_REGISTRY)


# IRDIS registry: reuses shared StepSpec entries verbatim (same log_name and
# outputs → reduction_status / check_output work across instruments), plus two
# IRDIS-only entries. Order is the canonical IRDIS execution sequence.
IRDIS_STEP_REGISTRY: dict[str, StepSpec] = {
    "download_data": STEP_REGISTRY["download_data"],
    "irdis_calibration": StepSpec("irdis_calibration", _NONE, internal_guard=True),
    "preprocess_irdis": StepSpec(
        "preprocess_irdis",
        _converted(
            "wavelengths.fits",
            "badpixel_map.fits",
            per_frame=("{frame}_cube.fits", "{frame}_ivar_cube.fits"),
        ),
    ),
    "cube_header_update": STEP_REGISTRY["cube_header_update"],
    "compute_frames_info": STEP_REGISTRY["compute_frames_info"],
    "find_centers": STEP_REGISTRY["find_centers"],
    "plot_image_center_evolution": STEP_REGISTRY["plot_image_center_evolution"],
    "process_extracted_centers": STEP_REGISTRY["process_extracted_centers"],
    "calibrate_spot_photometry": STEP_REGISTRY["calibrate_spot_photometry"],
    "calibrate_flux_psf": STEP_REGISTRY["calibrate_flux_psf"],
    "spot_to_flux": STEP_REGISTRY["spot_to_flux"],
    "align_frames": STEP_REGISTRY["align_frames"],
    "run_trap_reduction": STEP_REGISTRY["run_trap_reduction"],
    "run_trap_detection": STEP_REGISTRY["run_trap_detection"],
}

IRDIS_STEP_ORDER: list[str] = list(IRDIS_STEP_REGISTRY)


def expected_outputs(
    step: str,
    dirs: StepDirs,
    registry: dict[str, StepSpec] = STEP_REGISTRY,
) -> list[Path]:
    """Files whose presence means *step* is complete for this target."""
    return registry[step].outputs(dirs)


def validate_force(
    force: "bool | set[str]",
    registry: dict[str, StepSpec] = STEP_REGISTRY,
) -> None:
    """Raise ValueError if *force* is a set naming an unknown step."""
    if isinstance(force, set):
        unknown = force - set(registry)
        if unknown:
            raise ValueError(
                f"Unknown step name(s) in force: {sorted(unknown)}. "
                f"Valid names: {sorted(registry)}"
            )


def _forced(
    step: str,
    force: "bool | set[str]",
    step_order: list[str] = STEP_ORDER,
    registry: dict[str, StepSpec] = STEP_REGISTRY,
) -> bool:
    """True if *step* must recompute: force=True, or *step* is at/after the
    earliest force-named non-leaf step in step_order (cascade).

    Leaf steps never start a cascade — nothing downstream depends on them, so
    forcing one should re-run only itself. A leaf named in *force* is still
    forced, and a leaf at or after a forced non-leaf step is still forced.
    """
    if force is True:
        return True
    if not force:  # False or empty set
        return False
    if step in force:
        return True
    # Callers may pass an instrument's step_order against the default registry,
    # so a name the registry does not know counts as non-leaf: that is the
    # behaviour every such call site had before leaves existed.
    cascade_starters = [s for s in force if s not in registry or not registry[s].leaf]
    if not cascade_starters:
        return False
    first = min(step_order.index(s) for s in cascade_starters)
    return step_order.index(step) >= first


def should_run(
    step: str,
    enabled: bool,
    dirs: StepDirs,
    force: "bool | set[str]",
    logger,
    step_order: list[str] = STEP_ORDER,
    registry: dict[str, StepSpec] = STEP_REGISTRY,
) -> bool:
    """Decide whether to run *step*: skip (resume) when its outputs already
    exist, unless forced. Not used for ``internal_guard`` steps."""
    if not enabled:
        return False
    if _forced(step, force, step_order=step_order, registry=registry):
        return True
    outs = expected_outputs(step, dirs, registry=registry)
    if outs and all(p.exists() for p in outs):
        logger.info(
            f"{step}: outputs present — skipping",
            extra={"step": registry[step].log_name, "status": "skipped_complete"},
        )
        return False
    return True
