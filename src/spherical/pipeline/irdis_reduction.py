"""VLT/SPHERE IRDIS Data Reduction Pipeline — Phase 1 (download-only skeleton).

Parallel orchestrator to :mod:`spherical.pipeline.ifs_reduction`. Phase 1 wires
only the shared ``download_data`` step so a user can pull an IRDIS DBI/CI
dataset from the ESO archive without hitting the IFS orchestrator's
unconditional WAVECAL access, which crashes for IRDIS observations.

Later phases add ``irdis_calibration`` and ``preprocess_irdis`` (replacing the
charis-specific IFS extraction path) and generalize the shared downstream
steps.

See Also
--------
spherical.pipeline.ifs_reduction : IFS orchestrator (canonical step sequence)
spherical.pipeline.pipeline_config.IRDISReductionConfig : Configuration
"""
from __future__ import annotations

import copy
import os
import time
import traceback
from glob import glob
from os import path
from pathlib import Path
from typing import Union

import matplotlib

from spherical.database.irdis_observation import IRDISObservation
from spherical.pipeline.logging_utils import (
    PipelineLoggerAdapter,
    get_pipeline_log_context,
    get_pipeline_logger,
    remove_queue_listener,
)
from spherical.pipeline.pipeline_config import IRDISReductionConfig, defaultIRDISReduction
from spherical.pipeline.step_registry import (
    IRDIS_STEP_ORDER,
    IRDIS_STEP_REGISTRY,
    StepDirs,
    _forced,
    should_run,
)
from spherical.pipeline.steps.download_data import (
    download_data_for_observation,
    update_observation_file_paths,
)
from spherical.pipeline.steps.irdis_calibration import run_irdis_calibration
from spherical.pipeline.steps.irdis_preprocess import run_irdis_preprocess

matplotlib.use(backend="Agg")


def execute_irdis_target(
    observation: IRDISObservation,
    config: IRDISReductionConfig | None = None,
) -> None:
    """Execute the IRDIS reduction pipeline for a single observation.

    Phase 1 wires only the ``download_data`` step. All other flags on
    ``config.steps`` are ignored: enabling them at this stage would either
    no-op (no IRDIS step is implemented yet) or reach the shared IFS
    downstream steps, which are not yet instrument-generalized.

    Parameters
    ----------
    observation : IRDISObservation
        IRDIS observation object (metadata + frame tables). Must expose
        ``.observation`` (astropy Table with ``INSTRUMENT``, ``MAIN_ID``,
        ``FILTER``, ``NIGHT_START``) and ``.frames`` (dict of Tables keyed
        by frame type — at minimum ``CORO``/``CENTER``/``FLUX``/``FLAT``/
        ``BG_SCIENCE`` for the download step to pick from).
    config : IRDISReductionConfig, optional
        If ``None``, ``defaultIRDISReduction()`` is instantiated.

    Returns
    -------
    None
        Writes downloaded raw files under
        ``config.directories.raw_directory`` and a log/crash-report under
        ``{reduction_directory}/IRDIS/observation/{target}/{filter}/{date}/``.
    """
    if config is None:
        config = defaultIRDISReduction()

    config.apply_resources()

    if config.steps.all_steps_disabled():
        return None

    steps = config.steps
    raw_directory = config.directories.raw_directory
    reduction_directory = config.directories.reduction_directory

    start = time.time()
    observation = copy.copy(observation)

    target_name = str(observation.observation["MAIN_ID"][0])
    target_name = "_".join(target_name.split())
    obs_band = str(observation.observation["FILTER"][0])
    date = str(observation.observation["NIGHT_START"][0])
    observation.target_name = target_name  # type: ignore[attr-defined]
    observation.obs_band = obs_band  # type: ignore[attr-defined]
    observation.date = date  # type: ignore[attr-defined]

    name_mode_date = f"{target_name}/{obs_band}/{date}"
    outputdir = Path(str(reduction_directory)) / "IRDIS/observation" / name_mode_date
    outputdir.mkdir(parents=True, exist_ok=True)

    context = get_pipeline_log_context(observation)
    logger = get_pipeline_logger(name_mode_date, outputdir, verbose=True)
    logger = PipelineLoggerAdapter(logger, context)

    logger.info(
        "IRDIS pipeline session started",
        extra={"step": "session_start", "status": "started"},
    )

    try:
        if steps.download_data:
            download_data_for_observation(
                raw_directory=str(raw_directory),
                observation=observation,
                eso_username=config.preprocessing.eso_username,
                store_password=config.preprocessing.store_password,
                logger=logger,
            )

        # Hydrate FILE column on observation.frames so downstream steps can
        # locate raw files. Deferred to only run when a step that actually
        # reads raw files is enabled.
        if steps.irdis_calibration or steps.preprocess_irdis:
            existing_file_paths = glob(
                os.path.join(str(raw_directory) or "", "**", "SPHER.*.fits"),
                recursive=True,
            )
            update_observation_file_paths(
                existing_file_paths,
                observation,
                logger=logger,
                used_keys=("CORO", "CENTER", "FLUX", "FLAT", "BG_SCIENCE"),
            )

        # Calibration outputs live under
        # {reduction}/IRDIS/calibration/{filter}/{date}/. Phase 3 uses the
        # science-observation date as the calibration group id; a future
        # multi-observation master-calibration scheme will change the key.
        calib_outputdir = (
            Path(str(reduction_directory)) / "IRDIS/calibration" / obs_band / date
        )
        # dirs is prepared here so later phases can gate via should_run
        # without re-computing directories.
        converted_dir = outputdir / "converted"
        dirs = StepDirs(
            converted_dir=converted_dir,
            cube_outputdir=outputdir,
            irdis_calibration_dir=calib_outputdir,
        )

        if steps.irdis_calibration:
            run_irdis_calibration(
                observation=observation,
                calibration_config=config.calibration,
                calib_outputdir=calib_outputdir,
                logger=logger,
                force=_forced(
                    "irdis_calibration", steps.force, step_order=IRDIS_STEP_ORDER
                ),
            )

        if should_run(
            "preprocess_irdis",
            steps.preprocess_irdis,
            dirs,
            steps.force,
            logger,
            step_order=IRDIS_STEP_ORDER,
            registry=IRDIS_STEP_REGISTRY,
        ):
            run_irdis_preprocess(
                observation=observation,
                config=config,
                calib_outputdir=calib_outputdir,
                converted_outputdir=converted_dir,
                logger=logger,
            )

        elapsed_min = (time.time() - start) / 60.0
        logger.info(f"IRDIS pipeline finished in {elapsed_min:.2f} minutes.")
        return None

    except Exception:
        logger.exception("IRDIS pipeline execution failed.")
        crash_report_path = outputdir / "crash_report.txt"
        with open(crash_report_path, "w") as f:
            f.write(f"An error occurred during the reduction of {name_mode_date}.\n\n")
            traceback.print_exc(file=f)
        logger.info(f"Crash report saved to {crash_report_path}")
        return None

    finally:
        remove_queue_listener()


def output_directory_path(
    reduction_directory: str | Path,
    observation: Union[IRDISObservation, "object"],
) -> str:
    """Return the canonical IRDIS reduction output directory.

    No ``{method}`` path segment (IRDIS has no extraction method), matching
    the layout described in the design spec §2:
    ``{reduction_directory}/IRDIS/observation/{target}/{filter}/{date}/``.
    """
    target_name = "_".join(str(observation.observation["MAIN_ID"][0]).split())
    obs_band = str(observation.observation["FILTER"][0])
    date = str(observation.observation["NIGHT_START"][0])
    return path.join(
        str(reduction_directory),
        "IRDIS/observation",
        target_name,
        obs_band,
        date,
    ) + "/"


def check_output(
    reduction_directory: str | Path,
    observation_object_list: list,
) -> tuple[list[bool], list[list[str]]]:
    """Verify completeness of IRDIS reduction outputs per observation.

    Mirrors :func:`spherical.pipeline.ifs_reduction.check_output` but
    iterates the IRDIS step registry. No ``method`` parameter — IRDIS has
    no extraction method.

    Parameters
    ----------
    reduction_directory : str or Path
        Root of the reduction tree (contains ``IRDIS/observation/...``).
    observation_object_list : list
        IRDIS observation objects.

    Returns
    -------
    reduced : list of bool
        Per-observation completion flag.
    missing_files_reduction : list of list of str
        Per-observation list of missing output paths (relative to
        ``cube_outputdir``).
    """
    from spherical.pipeline.step_registry import (
        IRDIS_STEP_REGISTRY,
        StepDirs,
        expected_outputs,
    )

    reduced = []
    missing_files_reduction = []

    for observation in observation_object_list:
        outputdir = Path(output_directory_path(reduction_directory, observation))
        converted_dir = outputdir / "converted"
        dirs = StepDirs(
            converted_dir=converted_dir,
            cube_outputdir=outputdir,
        )
        missing_files: list[str] = []
        for step, spec in IRDIS_STEP_REGISTRY.items():
            if spec.internal_guard or spec.is_trap:
                continue
            for p in expected_outputs(step, dirs, registry=IRDIS_STEP_REGISTRY):
                if not p.exists():
                    missing_files.append(str(p.relative_to(converted_dir.parent)))
        reduced.append(len(missing_files) == 0)
        missing_files_reduction.append(missing_files)

    return reduced, missing_files_reduction
