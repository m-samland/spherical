"""``preprocessing.frame_types_to_extract`` means one thing on both instruments (#181, #178).

The config is validated when it is built, and both drivers and both completeness
checks narrow the observation's frame types to the configured ones the same way.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from astropy.table import Table

from spherical.pipeline.pipeline_config import PreprocConfig


class TestPreprocConfigValidation:
    def test_an_unknown_name_raises_at_construction(self):
        with pytest.raises(ValueError, match="CENTRE"):
            PreprocConfig(frame_types_to_extract=["CORO", "CENTRE"])

    def test_an_unknown_name_raises_on_merge(self):
        with pytest.raises(ValueError, match="WAVECAL"):
            PreprocConfig().merge(frame_types_to_extract=["CORO", "WAVECAL"])

    def test_lowercase_names_are_accepted(self):
        config = PreprocConfig(frame_types_to_extract=["coro", "center"])
        assert config.frame_types_to_extract == ["coro", "center"]


def _ifs_observation():
    frame = Table({"DP.ID": ["SPHER.2020-01-01T00:00:00.000"]})
    return SimpleNamespace(
        observation=Table({
            "INSTRUMENT": ["ifs"], "MAIN_ID": ["TARGET"], "FILTER": ["OBS_H"],
            "NIGHT_START": ["2020-01-01"], "WAFFLE_MODE": [False],
        }),
        frames={"CORO": frame, "CENTER": frame, "FLUX": frame, "WAVECAL": frame},
    )


def test_ifs_extraction_receives_the_configured_frame_types(tmp_path):
    """Extraction was hard-coded to all three types, so narrowing saved nothing."""
    pytest.importorskip("charis")
    from spherical.pipeline import ifs_reduction as ir
    from spherical.pipeline.pipeline_config import defaultIFSReduction

    config = defaultIFSReduction()
    config.directories.reduction_directory = tmp_path
    config.directories.raw_directory = tmp_path / "raw"
    config.steps.disable_all_ifs_steps()
    config.steps = config.steps.merge(extract_cubes=True)
    config.preprocessing = config.preprocessing.merge(frame_types_to_extract=["flux", "coro"])

    with patch.object(ir, "update_observation_file_paths"), patch.object(
        ir, "extract_cubes_with_multiprocessing"
    ) as extract:
        ir.execute_target(observation=_ifs_observation(), config=config)

    assert list(extract.call_args.kwargs["frame_types_to_extract"]) == ["CORO", "FLUX"]


def _irdis_observation():
    frame = Table({"DP.ID": ["SPHER.2024-01-01T00:00:00.000"]})
    return SimpleNamespace(
        observation=Table({
            "INSTRUMENT": ["irdis"], "MAIN_ID": ["TARGET"], "FILTER": ["DB_H23"],
            "NIGHT_START": ["2024-01-01"], "WAFFLE_MODE": [False],
        }),
        frames={"CORO": frame, "CENTER": frame, "FLUX": frame, "FLAT": frame, "BG_SCIENCE": frame},
        filter="DB_H23", target_name=None, obs_band=None, date=None,
    )


def test_irdis_driver_honours_the_configured_frame_types(tmp_path):
    """IRDIS read only the observation, so the shared setting was inert there."""
    from spherical.pipeline.irdis_reduction import execute_irdis_target
    from spherical.pipeline.pipeline_config import defaultIRDISReduction

    config = defaultIRDISReduction()
    config.directories.reduction_directory = tmp_path / "reduction"
    config.directories.raw_directory = tmp_path / "data"
    config.steps.disable_all_ifs_steps()
    config.steps.disable_all_irdis_steps()
    config.steps = config.steps.merge(cube_header_update=True)
    config.preprocessing = config.preprocessing.merge(frame_types_to_extract=["CENTER", "CORO"])

    with patch(
        "spherical.pipeline.irdis_reduction.update_observation_file_paths"
    ), patch(
        "spherical.pipeline.irdis_reduction.run_cube_header_update"
    ) as header_update:
        execute_irdis_target(observation=_irdis_observation(), config=config)

    assert header_update.call_args.kwargs["frame_types_to_extract"] == ["CORO", "CENTER"]


def test_irdis_check_output_respects_a_narrowed_frame_type_config(tmp_path):
    from spherical.pipeline.irdis_reduction import check_output, output_directory_path
    from spherical.pipeline.step_registry import IRDIS_STEP_REGISTRY, StepDirs, expected_outputs

    observation = _irdis_observation()
    converted = Path(output_directory_path(str(tmp_path), observation)) / "converted"
    dirs = StepDirs(
        converted_dir=converted, cube_outputdir=converted.parent,
        available_frame_types=("CORO", "CENTER"),
    )
    for step, spec in IRDIS_STEP_REGISTRY.items():
        if spec.internal_guard or spec.is_trap or spec.leaf:
            continue
        for p in expected_outputs(step, dirs, registry=IRDIS_STEP_REGISTRY):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()

    reduced, missing = check_output(
        str(tmp_path), [observation], frame_types_to_extract=["CENTER", "CORO"]
    )
    assert reduced == [True], missing

    reduced_all, missing_all = check_output(str(tmp_path), [observation])
    assert reduced_all == [False]
    assert any("flux_cube.fits" in m for m in missing_all[0])
