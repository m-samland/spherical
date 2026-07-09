"""Unit tests for the resume/skip step registry and force logic."""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from spherical.pipeline import step_registry as sr
from spherical.pipeline.pipeline_config import PipelineStepsConfig


def _dirs(tmp_path: Path) -> sr.StepDirs:
    return sr.StepDirs(
        converted_dir=tmp_path / "converted",
        cube_outputdir=tmp_path / "cubes",
        wavecal_outputdir=tmp_path / "wavecal",
        trap_result_folder=tmp_path / "trap",
    )


def test_step_order_covers_config_steps_and_trap_last():
    # Every registry key is a real PipelineStepsConfig boolean.
    cfg = PipelineStepsConfig()
    for step in sr.STEP_REGISTRY:
        assert hasattr(cfg, step), step
    # TRAP steps are the final two, in reduction->detection order.
    assert sr.STEP_ORDER[-2:] == ["run_trap_reduction", "run_trap_detection"]
    # Exactly one final step.
    finals = [s for s, spec in sr.STEP_REGISTRY.items() if spec.is_final]
    assert finals == ["spot_to_flux"]


def test_expected_outputs_locations(tmp_path):
    d = _dirs(tmp_path)
    assert sr.expected_outputs("bundle_output", d) == [
        d.converted_dir / "coro_cube.fits",
        d.converted_dir / "center_cube.fits",
        d.converted_dir / "wavelengths.fits",
    ]
    # additional_outputs lives INSIDE converted_dir (matches the step modules).
    assert sr.expected_outputs("calibrate_spot_photometry", d) == [
        d.converted_dir / "additional_outputs" / "spot_amplitudes.fits",
    ]
    # Side-effect steps declare no outputs.
    assert sr.expected_outputs("cube_header_update", d) == []


def test_marker_roundtrip(tmp_path):
    d = _dirs(tmp_path)
    marker = sr.expected_outputs("extract_cubes", d)[0]
    assert not marker.exists()
    sr.write_marker("extract_cubes", d.cube_outputdir)
    assert marker.exists()


def test_forced_cascade():
    force = {"extract_cubes"}
    assert sr._forced("extract_cubes", force) is True
    assert sr._forced("run_trap_reduction", force) is True   # downstream
    assert sr._forced("reduce_calibration", force) is False  # upstream
    assert sr._forced("download_data", force) is False       # upstream
    assert sr._forced("spot_to_flux", True) is True
    assert sr._forced("spot_to_flux", False) is False
    assert sr._forced("spot_to_flux", set()) is False


def test_validate_force_rejects_unknown():
    with pytest.raises(ValueError):
        sr.validate_force({"extract_cube"})  # typo
    sr.validate_force({"extract_cubes"})  # ok, no raise
    sr.validate_force(True)
    sr.validate_force(False)


def test_should_run_disabled_even_when_forced(tmp_path):
    log = MagicMock()
    assert sr.should_run("extract_cubes", False, _dirs(tmp_path), True, log) is False


def test_should_run_forced_ignores_existing(tmp_path):
    d = _dirs(tmp_path)
    sr.write_marker("extract_cubes", d.cube_outputdir)  # outputs exist
    log = MagicMock()
    assert sr.should_run("extract_cubes", True, d, True, log) is True


def test_should_run_skips_when_complete_and_logs(tmp_path):
    d = _dirs(tmp_path)
    d.converted_dir.mkdir(parents=True)
    (d.converted_dir / "image_centers.fits").touch()  # find_centers output
    log = MagicMock()
    assert sr.should_run("find_centers", True, d, False, log) is False
    # Skip is logged with the canonical log_name and the new status.
    _, kwargs = log.info.call_args
    assert kwargs["extra"] == {"step": "fit_centers", "status": "skipped_complete"}


def test_should_run_runs_when_output_missing(tmp_path):
    d = _dirs(tmp_path)
    d.converted_dir.mkdir(parents=True)
    log = MagicMock()
    assert sr.should_run("find_centers", True, d, False, log) is True


def test_should_run_side_effect_step_always_runs(tmp_path):
    log = MagicMock()
    assert sr.should_run("cube_header_update", True, _dirs(tmp_path), False, log) is True
