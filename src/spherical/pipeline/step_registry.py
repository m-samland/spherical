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
    """

    log_name: str
    outputs: Callable[[StepDirs], list[Path]]
    is_final: bool = False
    internal_guard: bool = False
    is_trap: bool = False  # TRAP step: excluded from IFS-reduction check_output()


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
    }[step]
    return [marker_for(step, directory)]


def _converted(*names: str) -> Callable[[StepDirs], list[Path]]:
    return lambda d: [d.converted_dir / n for n in names]


def _additional(*names: str) -> Callable[[StepDirs], list[Path]]:
    return lambda d: [d.additional_outputs / n for n in names]


_NONE: Callable[[StepDirs], list[Path]] = lambda d: []  # noqa: E731

# Insertion order == canonical pipeline order in ifs_reduction.py, TRAP last.
STEP_REGISTRY: dict[str, StepSpec] = {
    "download_data": StepSpec("download_data", _NONE, internal_guard=True),
    "reduce_calibration": StepSpec("wavelength_calibration", _NONE, internal_guard=True),
    "extract_cubes": StepSpec("extract_cubes", lambda d: _marker_output("extract_cubes", d)),
    "bundle_output": StepSpec("bundle_output", _converted("coro_cube.fits", "center_cube.fits", "wavelengths.fits")),
    "compute_frames_info": StepSpec(
        "frame_info_computation",
        _converted("frames_info_coro.csv", "frames_info_center.csv", "frames_info_flux.csv"),
    ),
    "cube_header_update": StepSpec("cube_header_update", _NONE),
    "find_centers": StepSpec("fit_centers", _converted("image_centers.fits")),
    "plot_image_center_evolution": StepSpec("plot_center_evolution", _NONE),
    "process_extracted_centers": StepSpec("polynomial_center_fit", _converted("image_centers_fitted_robust.fits")),
    "calibrate_spot_photometry": StepSpec("spot_photometry_calibration", _additional("spot_amplitudes.fits")),
    "calibrate_flux_psf": StepSpec("flux_psf_calibration", _converted("psf_cube_for_postprocessing.fits")),
    "spot_to_flux": StepSpec("spot_to_flux_normalization", _converted("spot_amplitude_variation.fits"), is_final=True),
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
            "coro_cube.fits",
            "center_cube.fits",
            "flux_cube.fits",
            "coro_ivar_cube.fits",
            "center_ivar_cube.fits",
            "flux_ivar_cube.fits",
            "wavelengths.fits",
            "badpixel_map.fits",
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
) -> bool:
    """True if *step* must recompute: force=True, or *step* is at/after the
    earliest force-named step in step_order (cascade)."""
    if force is True:
        return True
    if not force:  # False or empty set
        return False
    first = min(step_order.index(s) for s in force)
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
    if _forced(step, force, step_order=step_order):
        return True
    outs = expected_outputs(step, dirs, registry=registry)
    if outs and all(p.exists() for p in outs):
        logger.info(
            f"{step}: outputs present — skipping",
            extra={"step": registry[step].log_name, "status": "skipped_complete"},
        )
        return False
    return True
