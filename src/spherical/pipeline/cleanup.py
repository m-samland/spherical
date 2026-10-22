"""Remove reduction products that a finished observation no longer needs.

``cleanup_pipeline_products`` works stage by stage, for IFS and IRDIS.

``raw``
    The observation's raw frames under ``raw_directory``, found by DP.ID, so the
    frames of other observations in a shared calibration folder are kept.
``calibration``
    The IFS charis wavelength calibration or the IRDIS master calibrations.
``extracted``
    IFS only, the extracted cubes of single frames.
``cubes``
    The bundled cubes in ``converted/``, removed only after the TRAP detection finished.

Nothing is removed before the bundled cubes exist. Resumed runs skip download,
calibration and extraction while the cubes exist, and skip the whole observation
once the cubes were removed, until something is forced (#218).
"""

from __future__ import annotations

from glob import glob
from pathlib import Path
from typing import NamedTuple

import numpy as np

from spherical.pipeline.science_frames import configured_frame_types, frame_types_present
from spherical.pipeline.step_registry import (
    IRDIS_STEP_REGISTRY,
    STEP_REGISTRY,
    StepDirs,
    cubes_cleaned_marker,
    expected_outputs,
    marker_for,
    target_folder_string,
    trap_result_folder,
)
from spherical.pipeline.steps.download_data import _dp_id_from_path

#: The stages ``cleanup_pipeline_products`` can remove, in pipeline order.
STAGES = ("raw", "calibration", "extracted", "cubes")

_RAW_KEYS = {
    "IFS": ("CORO", "CENTER", "FLUX", "WAVECAL"),
    "IRDIS": ("CORO", "CENTER", "FLUX", "FLAT", "BG_SCIENCE"),
}


def validate_stages(stages) -> None:
    """Raise ValueError if *stages* names a stage not in ``STAGES``."""
    unknown = set(stages) - set(STAGES)
    if unknown:
        raise ValueError(f"Unknown cleanup stage(s): {sorted(unknown)}. Valid stages: {list(STAGES)}")


class _Layout(NamedTuple):
    instrument: str
    label: str
    cube_outputdir: Path
    converted_dir: Path
    calibration_dir: Path
    trap_marker: Path
    cubes_ready: bool


def _layout(observation, config) -> _Layout:
    """Where the reduction of *observation* keeps each stage, as the drivers lay it out."""
    row = observation.observation
    instrument = str(row["INSTRUMENT"][0]).upper()
    main_id, band, night = str(row["MAIN_ID"][0]), str(row["FILTER"][0]), str(row["NIGHT_START"][0])
    label = target_folder_string(main_id, band, night)
    reduction = Path(config.directories.reduction_directory)
    observation_dir = reduction / instrument / "observation" / label
    if instrument == "IFS":
        cube_outputdir = observation_dir / config.extraction.method
        calibration_dir = reduction / "IFS/calibration" / band / str(observation.frames["WAVECAL"]["DP.ID"][0])[6:]
        cube_step, registry = "bundle_output", STEP_REGISTRY
    else:
        cube_outputdir = observation_dir
        calibration_dir = reduction / "IRDIS/calibration" / band / night
        cube_step, registry = "preprocess_irdis", IRDIS_STEP_REGISTRY
    converted_dir = cube_outputdir / "converted"
    dirs = StepDirs(
        converted_dir=converted_dir,
        cube_outputdir=cube_outputdir,
        available_frame_types=frame_types_present(
            observation, candidates=configured_frame_types(config.preprocessing.frame_types_to_extract)
        ),
    )
    cubes_ready = cubes_cleaned_marker(converted_dir).exists() or all(
        p.exists() for p in expected_outputs(cube_step, dirs, registry=registry)
    )
    trap_marker = marker_for(
        "run_trap_detection", trap_result_folder(reduction, main_id, band, night, instrument=instrument)
    )
    return _Layout(instrument, label, cube_outputdir, converted_dir, calibration_dir, trap_marker, cubes_ready)


def _raw_files(observation, layout: _Layout, config) -> list[Path]:
    ids = set()
    for key in _RAW_KEYS[layout.instrument]:
        table = observation.frames.get(key)
        if table is not None and len(table) > 0:
            ids.update(str(x) for x in np.array(table["DP.ID"]).flatten())
    pattern = Path(config.directories.raw_directory) / layout.instrument / "**" / "SPHER.*.fits*"
    return [Path(p) for p in glob(str(pattern), recursive=True) if _dp_id_from_path(p) in ids]


def _calibration_files(observation, layout: _Layout, config) -> list[Path]:
    if not layout.calibration_dir.is_dir():
        return []
    return [p for p in layout.calibration_dir.rglob("*") if p.is_file()]


def _extracted_files(observation, layout: _Layout, config) -> list[Path]:
    if layout.instrument != "IFS":
        return []
    files = [
        p
        for frame_type in ("CORO", "CENTER", "FLUX")
        for p in (layout.cube_outputdir / frame_type).glob("SPHER.*cube*.fits")
    ]
    marker = marker_for("extract_cubes", layout.cube_outputdir)
    return files + ([marker] if marker.exists() else [])


def _cube_files(observation, layout: _Layout, config) -> list[Path]:
    converted = layout.converted_dir
    files = sorted({*converted.glob("*_cube.fits"), *converted.glob("*_cube_aligned.fits")})
    marker = marker_for("align_frames", converted)
    return files + ([marker] if marker.exists() else [])


_STAGE_FILES = {
    "raw": _raw_files,
    "calibration": _calibration_files,
    "extracted": _extracted_files,
    "cubes": _cube_files,
}


def cleanup_pipeline_products(
    observations,
    config,
    stages=frozenset({"extracted", "calibration"}),
    dry_run=True,
) -> tuple[int, float]:
    """Remove the chosen stages of finished reductions.

    Parameters
    ----------
    observations : IFSObservation, IRDISObservation or list of them
        Observations whose products to remove.
    config : IFSReductionConfig or IRDISReductionConfig
        The configuration the observations were reduced with.
    stages : set of str
        Any of ``"raw"``, ``"calibration"``, ``"extracted"`` and ``"cubes"``.
    dry_run : bool
        Only report what would be removed. Default True.

    Returns
    -------
    tuple of (int, float)
        Number of files and megabytes removed, or that would be removed.

    Notes
    -----
    An observation is cleaned only once its bundled cubes exist, and ``cubes``
    only once its TRAP detection finished. Removed files are not regenerated by a
    resumed run. Force the steps that make them to bring them back, see Output
    products in the documentation.
    """
    validate_stages(stages)
    if not isinstance(observations, list):
        observations = [observations]
    verb = "Would remove" if dry_run else "Removed"
    total_files, total_bytes = 0, 0
    for observation in observations:
        layout = _layout(observation, config)
        if not layout.cubes_ready:
            print(f"{layout.label}: the cubes are not built yet, nothing removed")
            continue
        todo = [stage for stage in STAGES if stage in stages]
        if "cubes" in todo and not layout.trap_marker.exists():
            print(f"{layout.label}: TRAP detection has not finished, keeping the cubes")
            todo.remove("cubes")
        files = [f for stage in todo for f in _STAGE_FILES[stage](observation, layout, config)]
        n_bytes = sum(f.stat().st_size for f in files)
        if not dry_run:
            for f in files:
                f.unlink()
            for directory in sorted({f.parent for f in files}, key=lambda d: len(d.parts), reverse=True):
                try:
                    directory.rmdir()
                except OSError:
                    pass  # not empty
            if "cubes" in todo:
                cubes_cleaned_marker(layout.converted_dir).touch()
        print(f"{layout.label}: {verb.lower()} {len(files)} files ({n_bytes / 1e6:.1f} MB) of {', '.join(todo) or 'no stage'}")
        total_files += len(files)
        total_bytes += n_bytes
    print(f"{verb} {total_files} files, {total_bytes / 1e9:.2f} GB in total")
    return total_files, total_bytes / 1e6
