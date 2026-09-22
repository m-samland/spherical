"""Unit tests for the resume/skip step registry and force logic."""
from pathlib import Path
from types import SimpleNamespace
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


def test_config_force_defaults_false():
    assert PipelineStepsConfig().force is False


def test_config_force_merge_set():
    cfg = PipelineStepsConfig().merge(force={"extract_cubes"})
    assert cfg.force == {"extract_cubes"}


def test_overwrite_fields_removed():
    cfg = PipelineStepsConfig()
    for gone in ("overwrite_calibration", "overwrite_bundle", "overwrite_preprocessing", "overwrite_trap"):
        assert not hasattr(cfg, gone), gone


def test_check_output_uses_registry_and_real_additional_dir(tmp_path, monkeypatch):
    # ifs_reduction imports charis at module level; the CI pipeline job has none.
    pytest.importorskip("charis")

    from spherical.pipeline import ifs_reduction as ir

    # Point output_directory_path at a converted dir we control.
    converted = tmp_path / "IFS/observation/T/OBS_H/2020-01-01/optext/converted"
    converted.mkdir(parents=True)
    monkeypatch.setattr(ir, "output_directory_path", lambda rd, obs, method='optext': str(converted) + "/")

    # check_output reads WAFFLE_MODE to know which frame type is the science one.
    observation = SimpleNamespace(observation={"WAFFLE_MODE": [False]})

    # Create every registry output for a "complete" target.
    dirs = sr.StepDirs(converted_dir=converted, cube_outputdir=converted.parent, wavecal_outputdir=tmp_path)
    for step, spec in sr.STEP_REGISTRY.items():
        if spec.internal_guard or spec.is_trap:
            continue
        for p in sr.expected_outputs(step, dirs):
            p.parent.mkdir(parents=True, exist_ok=True)
            p.touch()

    reduced, missing = ir.check_output(str(tmp_path), [observation])
    assert reduced == [True]
    assert missing == [[]]


class TestScienceIdentifierOutputs:
    """Frame-type-dependent outputs follow WAFFLE_MODE, not a fixed 'coro'.

    A continuous-waffle observation has no CORO frames, so preprocess never
    writes coro_cube.fits and frame_info never writes frames_info_coro.csv.
    Declaring those unconditionally kept such a target permanently 'incomplete'
    and re-ran the step on every invocation.
    """

    def test_default_identifier_is_coro(self):
        assert sr.StepDirs().science_identifier == "coro"

    def test_bundle_output_follows_the_identifier(self, tmp_path):
        d = sr.StepDirs(converted_dir=tmp_path, science_identifier="center")
        assert sr.expected_outputs("bundle_output", d) == [
            tmp_path / "center_cube.fits",
            tmp_path / "wavelengths.fits",
        ]

    def test_compute_frames_info_follows_the_identifier(self, tmp_path):
        coro = sr.StepDirs(converted_dir=tmp_path)
        assert [p.name for p in sr.expected_outputs("compute_frames_info", coro)] == [
            "frames_info_coro.csv",
            "frames_info_center.csv",
            "frames_info_flux.csv",
        ]
        waffle = sr.StepDirs(converted_dir=tmp_path, science_identifier="center")
        assert [p.name for p in sr.expected_outputs("compute_frames_info", waffle)] == [
            "frames_info_center.csv",
            "frames_info_flux.csv",
        ]

    def test_preprocess_irdis_follows_the_identifier(self, tmp_path):
        waffle = sr.StepDirs(converted_dir=tmp_path, science_identifier="center")
        names = [p.name for p in sr.expected_outputs(
            "preprocess_irdis", waffle, registry=sr.IRDIS_STEP_REGISTRY
        )]
        assert "coro_cube.fits" not in names
        assert "coro_ivar_cube.fits" not in names
        assert names.count("center_cube.fits") == 1
        assert "center_ivar_cube.fits" in names
        assert "flux_cube.fits" in names
        assert "badpixel_map.fits" in names

    def test_waffle_target_resumes_without_any_coro_product(self, tmp_path):
        """The case the center->coro symlink used to paper over."""
        converted = tmp_path / "converted"
        converted.mkdir()
        dirs = sr.StepDirs(converted_dir=converted, science_identifier="center")
        for p in sr.expected_outputs(
            "preprocess_irdis", dirs, registry=sr.IRDIS_STEP_REGISTRY
        ):
            p.touch()

        assert not sr.should_run(
            "preprocess_irdis", True, dirs, set(), MagicMock(),
            step_order=sr.IRDIS_STEP_ORDER, registry=sr.IRDIS_STEP_REGISTRY,
        )
        assert not (converted / "coro_cube.fits").exists()
