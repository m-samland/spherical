"""Tests for the IRDIS Phase 1 orchestrator (download-only skeleton)."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from astropy.table import Table


def _make_irdis_observation(tmp_path):
    """Build a minimal IRDISObservation-like stand-in for orchestrator tests.

    Uses SimpleNamespace to avoid needing real IRDISObservation construction
    (which requires large table fixtures). The orchestrator only touches the
    attributes accessed below.
    """
    obs_row = Table(
        {
            "INSTRUMENT": ["irdis"],
            "MAIN_ID": ["TEST_TARGET"],
            "FILTER": ["DB_H23"],
            "NIGHT_START": ["2024-01-01"],
        }
    )
    frames = {
        "CORO": Table({"DP.ID": ["SPHER.2024-01-01T00:00:00.000"]}),
        "CENTER": Table({"DP.ID": ["SPHER.2024-01-01T00:01:00.000"]}),
        "FLUX": Table({"DP.ID": ["SPHER.2024-01-01T00:02:00.000"]}),
        "FLAT": Table({"DP.ID": ["SPHER.2024-01-01T00:03:00.000"]}),
        "BG_SCIENCE": Table({"DP.ID": ["SPHER.2024-01-01T00:04:00.000"]}),
    }
    return SimpleNamespace(
        observation=obs_row,
        frames=frames,
        filter="DB_H23",
        target_name=None,
        obs_band=None,
        date=None,
    )


def test_execute_irdis_target_download_only_calls_download(tmp_path):
    from spherical.pipeline.irdis_reduction import execute_irdis_target
    from spherical.pipeline.pipeline_config import defaultIRDISReduction

    observation = _make_irdis_observation(tmp_path)

    config = defaultIRDISReduction()
    config.directories.base_path = tmp_path
    config.directories.raw_directory = tmp_path / "data"
    config.directories.reduction_directory = tmp_path / "reduction"
    config.steps.disable_all_ifs_steps()
    config.steps.disable_all_irdis_steps()
    config.steps = config.steps.merge(download_data=True)

    with patch(
        "spherical.pipeline.irdis_reduction.download_data_for_observation"
    ) as mocked:
        execute_irdis_target(observation=observation, config=config)

    mocked.assert_called_once()
    kwargs = mocked.call_args.kwargs
    assert kwargs["raw_directory"] == str(tmp_path / "data")
    passed_obs = kwargs["observation"]
    assert passed_obs.observation["MAIN_ID"][0] == "TEST_TARGET"
    assert passed_obs.observation["FILTER"][0] == "DB_H23"
    assert passed_obs.target_name == "TEST_TARGET"
    assert passed_obs.obs_band == "DB_H23"
    assert passed_obs.date == "2024-01-01"


def test_execute_targets_dispatches_irdis_observation(tmp_path):
    from spherical.pipeline.ifs_reduction import execute_targets
    from spherical.pipeline.pipeline_config import defaultIRDISReduction

    observation = _make_irdis_observation(tmp_path)
    config = defaultIRDISReduction()
    config.directories.base_path = tmp_path
    config.directories.raw_directory = tmp_path / "data"
    config.directories.reduction_directory = tmp_path / "reduction"
    config.steps.disable_all_ifs_steps()
    config.steps = config.steps.merge(download_data=True)

    with patch(
        "spherical.pipeline.ifs_reduction.execute_irdis_target"
    ) as irdis_target, patch(
        "spherical.pipeline.ifs_reduction.execute_target"
    ) as ifs_target:
        execute_targets(observations=observation, config=config)

    irdis_target.assert_called_once()
    ifs_target.assert_not_called()


def test_execute_targets_rejects_ifs_config_for_irdis_observation(tmp_path):
    from spherical.pipeline.ifs_reduction import execute_targets
    from spherical.pipeline.pipeline_config import IFSReductionConfig

    observation = _make_irdis_observation(tmp_path)
    with pytest.raises(ValueError, match="IRDIS observation.*IFSReductionConfig"):
        execute_targets(observations=observation, config=IFSReductionConfig())


def test_execute_targets_rejects_irdis_config_for_ifs_observation(tmp_path):
    from spherical.pipeline.ifs_reduction import execute_targets
    from spherical.pipeline.pipeline_config import defaultIRDISReduction

    ifs_obs = _make_irdis_observation(tmp_path)
    ifs_obs.observation["INSTRUMENT"][0] = "ifs"

    with pytest.raises(ValueError, match="IFS observation.*IRDISReductionConfig"):
        execute_targets(observations=ifs_obs, config=defaultIRDISReduction())


def test_execute_targets_none_config_still_dispatches(tmp_path):
    from spherical.pipeline.ifs_reduction import execute_targets

    observation = _make_irdis_observation(tmp_path)
    with patch(
        "spherical.pipeline.ifs_reduction.execute_irdis_target"
    ) as irdis_target:
        execute_targets(observations=observation, config=None)
    irdis_target.assert_called_once()
    assert irdis_target.call_args.kwargs["config"] is None


def test_output_directory_path_no_method_segment(tmp_path):
    from spherical.pipeline.irdis_reduction import output_directory_path

    observation = _make_irdis_observation(tmp_path)
    path = output_directory_path(str(tmp_path / "reduction"), observation)
    assert "IRDIS/observation" in path
    assert "TEST_TARGET" in path
    assert "DB_H23" in path
    assert "2024-01-01" in path
    assert "optext" not in path
    assert "apphot" not in path


def test_irdis_template_is_valid_python():
    """Guard against syntax errors in the Phase 1 driver template."""
    import ast
    from pathlib import Path

    template = Path(__file__).parent.parent / "examples" / "irdis_reduction_template.py"
    assert template.exists(), f"Template missing: {template}"
    ast.parse(template.read_text())


def test_import_spherical_without_pipeline_extra():
    """CLAUDE.md constraint: ``import spherical`` and ``spherical.database``
    must succeed with only base deps. This runs in the pipeline-installed
    test env, but the import itself must not touch pipeline modules eagerly.
    """
    import importlib

    import spherical  # noqa: F401
    importlib.import_module("spherical.database")


def test_check_output_reports_missing_for_new_dir(tmp_path):
    from spherical.pipeline.irdis_reduction import check_output

    observation = _make_irdis_observation(tmp_path)
    reduced, missing = check_output(str(tmp_path / "reduction"), [observation])
    assert reduced == [False]
    assert any("coro_cube.fits" in m for m in missing[0])


def test_check_output_reports_complete_when_all_outputs_exist(tmp_path):
    from pathlib import Path

    from spherical.pipeline.irdis_reduction import check_output, output_directory_path

    observation = _make_irdis_observation(tmp_path)
    converted_dir = Path(output_directory_path(str(tmp_path / "reduction"), observation))
    converted_dir.mkdir(parents=True, exist_ok=True)
    (converted_dir / "additional_outputs").mkdir(parents=True, exist_ok=True)

    for name in (
        "coro_cube.fits", "center_cube.fits", "flux_cube.fits",
        "coro_ivar_cube.fits", "center_ivar_cube.fits", "flux_ivar_cube.fits",
        "wavelengths.fits", "badpixel_map.fits",
        "image_centers.fits",
        "image_centers_fitted_robust.fits",
        "psf_cube_for_postprocessing.fits",
        "spot_amplitude_variation.fits",
        "frames_info_coro.csv", "frames_info_center.csv", "frames_info_flux.csv",
    ):
        (converted_dir / name).write_text("x")
    (converted_dir / "additional_outputs" / "spot_amplitudes.fits").write_text("x")

    reduced, missing = check_output(str(tmp_path / "reduction"), [observation])
    assert reduced == [True], f"missing: {missing}"
    assert missing == [[]]


def test_execute_irdis_target_calls_calibration_step(tmp_path):
    """Task 5 wiring: with irdis_calibration enabled, the orchestrator invokes
    run_irdis_calibration after download."""
    from spherical.pipeline.irdis_reduction import execute_irdis_target
    from spherical.pipeline.pipeline_config import defaultIRDISReduction

    observation = _make_irdis_observation(tmp_path)
    config = defaultIRDISReduction()
    config.directories.base_path = tmp_path
    config.directories.raw_directory = tmp_path / "data"
    config.directories.reduction_directory = tmp_path / "reduction"
    config.steps.disable_all_ifs_steps()
    config.steps.disable_all_irdis_steps()
    config.steps = config.steps.merge(download_data=False, irdis_calibration=True)

    with patch(
        "spherical.pipeline.irdis_reduction.download_data_for_observation"
    ), patch(
        "spherical.pipeline.irdis_reduction.update_observation_file_paths"
    ), patch(
        "spherical.pipeline.irdis_reduction.run_irdis_calibration"
    ) as run_calib:
        execute_irdis_target(observation=observation, config=config)

    run_calib.assert_called_once()


def test_execute_irdis_target_calls_preprocess_step(tmp_path):
    """Task 7 wiring: with preprocess_irdis enabled and no outputs on disk,
    the orchestrator invokes run_irdis_preprocess after run_irdis_calibration."""
    from spherical.pipeline.irdis_reduction import execute_irdis_target
    from spherical.pipeline.pipeline_config import defaultIRDISReduction

    observation = _make_irdis_observation(tmp_path)
    config = defaultIRDISReduction()
    config.directories.base_path = tmp_path
    config.directories.raw_directory = tmp_path / "data"
    config.directories.reduction_directory = tmp_path / "reduction"
    config.steps.disable_all_ifs_steps()
    config.steps.disable_all_irdis_steps()
    config.steps = config.steps.merge(
        download_data=False, irdis_calibration=True, preprocess_irdis=True,
    )

    with patch(
        "spherical.pipeline.irdis_reduction.download_data_for_observation"
    ), patch(
        "spherical.pipeline.irdis_reduction.update_observation_file_paths"
    ), patch(
        "spherical.pipeline.irdis_reduction.run_irdis_calibration"
    ), patch(
        "spherical.pipeline.irdis_reduction.run_irdis_preprocess"
    ) as run_pre:
        execute_irdis_target(observation=observation, config=config)

    run_pre.assert_called_once()


def test_execute_irdis_target_skips_preprocess_when_outputs_exist(tmp_path):
    """should_run gate: preprocess_irdis is not called when all 8 outputs
    already live in converted/."""
    from spherical.pipeline.irdis_reduction import execute_irdis_target, output_directory_path
    from spherical.pipeline.pipeline_config import defaultIRDISReduction

    observation = _make_irdis_observation(tmp_path)
    config = defaultIRDISReduction()
    config.directories.base_path = tmp_path
    config.directories.raw_directory = tmp_path / "data"
    config.directories.reduction_directory = tmp_path / "reduction"
    config.steps.disable_all_ifs_steps()
    config.steps.disable_all_irdis_steps()
    config.steps = config.steps.merge(
        download_data=False, irdis_calibration=False, preprocess_irdis=True,
    )

    from pathlib import Path

    converted = Path(output_directory_path(str(tmp_path / "reduction"), observation)) / "converted"
    converted.mkdir(parents=True, exist_ok=True)
    for name in (
        "coro_cube.fits", "center_cube.fits", "flux_cube.fits",
        "coro_ivar_cube.fits", "center_ivar_cube.fits", "flux_ivar_cube.fits",
        "wavelengths.fits", "badpixel_map.fits",
    ):
        (converted / name).write_text("stub")

    with patch(
        "spherical.pipeline.irdis_reduction.download_data_for_observation"
    ), patch(
        "spherical.pipeline.irdis_reduction.update_observation_file_paths"
    ), patch(
        "spherical.pipeline.irdis_reduction.run_irdis_preprocess"
    ) as run_pre:
        execute_irdis_target(observation=observation, config=config)

    run_pre.assert_not_called()
