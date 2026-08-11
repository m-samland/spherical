"""Tests for the IRDIS step registry, ordering, and order-parametrized guards."""
from __future__ import annotations

from pathlib import Path

import pytest

from spherical.pipeline.step_registry import (
    IRDIS_STEP_ORDER,
    IRDIS_STEP_REGISTRY,
    STEP_REGISTRY,
    StepDirs,
    _forced,
    expected_outputs,
    should_run,
    validate_force,
)


class TestIRDISStepRegistry:
    def test_registry_has_new_entries(self):
        assert "irdis_calibration" in IRDIS_STEP_REGISTRY
        assert "preprocess_irdis" in IRDIS_STEP_REGISTRY

    def test_irdis_calibration_is_internal_guard_no_outputs(self):
        spec = IRDIS_STEP_REGISTRY["irdis_calibration"]
        assert spec.internal_guard is True
        assert spec.outputs(StepDirs()) == []

    def test_preprocess_irdis_outputs_match_spec(self):
        spec = IRDIS_STEP_REGISTRY["preprocess_irdis"]
        dirs = StepDirs(converted_dir=Path("/tmp/converted"))
        outs = [p.name for p in spec.outputs(dirs)]
        assert "coro_cube.fits" in outs
        assert "center_cube.fits" in outs
        assert "flux_cube.fits" in outs
        assert "coro_ivar_cube.fits" in outs
        assert "wavelengths.fits" in outs
        assert "badpixel_map.fits" in outs

    def test_shared_steps_are_the_same_specs(self):
        """Spec §6: reuse shared StepSpec entries so aggregators see the same log_name."""
        for shared in (
            "download_data",
            "cube_header_update",
            "find_centers",
            "process_extracted_centers",
            "spot_to_flux",
            "run_trap_reduction",
            "run_trap_detection",
        ):
            assert IRDIS_STEP_REGISTRY[shared] is STEP_REGISTRY[shared], (
                f"{shared} must be the SAME StepSpec object as the IFS one"
            )

    def test_irdis_final_step_is_spot_to_flux(self):
        assert IRDIS_STEP_REGISTRY["spot_to_flux"].is_final is True

    def test_irdis_order_places_irdis_steps_before_downstream(self):
        idx = {s: IRDIS_STEP_ORDER.index(s) for s in IRDIS_STEP_ORDER}
        assert idx["download_data"] < idx["irdis_calibration"] < idx["preprocess_irdis"]
        assert idx["preprocess_irdis"] < idx["cube_header_update"]
        assert idx["preprocess_irdis"] < idx["find_centers"]
        assert idx["spot_to_flux"] < idx["run_trap_reduction"]


class TestStepDirsIRDISField:
    def test_new_field_default(self):
        dirs = StepDirs()
        assert dirs.irdis_calibration_dir == Path()

    def test_new_field_populated(self, tmp_path):
        dirs = StepDirs(irdis_calibration_dir=tmp_path / "calib")
        assert dirs.irdis_calibration_dir == tmp_path / "calib"


class TestForcedIsInstrumentAware:
    def test_default_step_order_is_ifs(self):
        assert _forced("find_centers", {"extract_cubes"}) is True
        assert _forced("download_data", {"extract_cubes"}) is False

    def test_irdis_order_cascades_from_irdis_calibration(self):
        assert _forced("preprocess_irdis", {"irdis_calibration"}, step_order=IRDIS_STEP_ORDER) is True
        assert _forced("find_centers", {"irdis_calibration"}, step_order=IRDIS_STEP_ORDER) is True
        assert _forced("download_data", {"irdis_calibration"}, step_order=IRDIS_STEP_ORDER) is False

    def test_force_true_always_true(self):
        assert _forced("download_data", True, step_order=IRDIS_STEP_ORDER) is True

    def test_force_false_always_false(self):
        assert _forced("preprocess_irdis", False, step_order=IRDIS_STEP_ORDER) is False


class TestShouldRunIRDIS:
    def test_defaults_still_use_ifs_registry(self, tmp_path):
        import logging

        dirs = StepDirs(converted_dir=tmp_path)
        assert should_run("find_centers", True, dirs, False, logging.getLogger()) is True

    def test_irdis_registry_skips_when_outputs_present(self, tmp_path):
        import logging

        outs_dir = tmp_path / "converted"
        outs_dir.mkdir()
        for name in ("coro_cube.fits", "center_cube.fits", "flux_cube.fits",
                     "coro_ivar_cube.fits", "center_ivar_cube.fits",
                     "flux_ivar_cube.fits", "wavelengths.fits", "badpixel_map.fits"):
            (outs_dir / name).write_text("x")
        dirs = StepDirs(converted_dir=outs_dir)
        assert should_run(
            "preprocess_irdis", True, dirs, False,
            logging.getLogger(), registry=IRDIS_STEP_REGISTRY,
        ) is False


class TestValidateForceIRDIS:
    def test_ifs_default_rejects_irdis_step(self):
        with pytest.raises(ValueError, match="Unknown step"):
            validate_force({"preprocess_irdis"})

    def test_irdis_registry_accepts_irdis_step(self):
        validate_force({"preprocess_irdis"}, registry=IRDIS_STEP_REGISTRY)


class TestExpectedOutputsIRDIS:
    def test_expected_outputs_uses_registry_kwarg(self, tmp_path):
        dirs = StepDirs(converted_dir=tmp_path)
        outs = expected_outputs("preprocess_irdis", dirs, registry=IRDIS_STEP_REGISTRY)
        names = [p.name for p in outs]
        assert "coro_cube.fits" in names
