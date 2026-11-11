"""TRAP must refuse to start when the preprocessing products are missing.

Running TRAP with the preprocessing steps switched off used to fail deep inside
the reduction with a bare traceback on the first `fits.getdata` call, which said
nothing about the real cause (issue #139). The guard checked here turns that
into an explicit, actionable error naming the missing files.

The required set is the same for IFS and IRDIS: both registries write these
products into `converted/` (see `step_registry.py`).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

# run_trap imports trap at module level; the CI pipeline job installs no trap.
pytest.importorskip("trap")


REQUIRED = (
    "wavelengths.fits",
    "coro_cube.fits",
    "frames_info_coro.csv",
    "image_centers_fitted_robust.fits",
    "psf_cube_for_postprocessing.fits",
)


def _populate(directory: Path, names) -> None:
    for name in names:
        path = directory / name
        if path.suffix == ".csv":
            path.write_text("DEROT ANGLE\n0.0\n")
        else:
            fits.writeto(path, np.zeros((2, 2), dtype=np.float32))


class TestVerifyTrapInputs:
    def test_missing_products_raise_naming_the_files(self, tmp_path: Path):
        from spherical.pipeline.run_trap import _verify_trap_inputs

        with pytest.raises(FileNotFoundError) as excinfo:
            _verify_trap_inputs(tmp_path, "coro")

        message = str(excinfo.value)
        assert "wavelengths.fits" in message
        assert "preprocessing" in message.lower()

    def test_complete_directory_passes(self, tmp_path: Path):
        from spherical.pipeline.run_trap import _verify_trap_inputs

        _populate(tmp_path, REQUIRED)
        _verify_trap_inputs(tmp_path, "coro")  # must not raise

    def test_legacy_psf_filename_satisfies_the_psf_requirement(self, tmp_path: Path):
        """run_trap falls back to master_flux_calibrated_psf_frames.fits, so
        either name must count as present."""
        from spherical.pipeline.run_trap import _verify_trap_inputs

        _populate(tmp_path, [n for n in REQUIRED if n != "psf_cube_for_postprocessing.fits"])
        _populate(tmp_path, ["master_flux_calibrated_psf_frames.fits"])
        _verify_trap_inputs(tmp_path, "coro")  # must not raise

    def test_only_the_relevant_cube_is_required(self, tmp_path: Path):
        """With continuous satellite spots run_trap reads the center products,
        so a missing coro cube must not block it."""
        from spherical.pipeline.run_trap import _verify_trap_inputs

        _populate(tmp_path, [n for n in REQUIRED if not n.startswith(("coro_", "frames_info_"))])
        _populate(tmp_path, ["center_cube.fits", "frames_info_center.csv"])
        _verify_trap_inputs(tmp_path, "center")  # must not raise
