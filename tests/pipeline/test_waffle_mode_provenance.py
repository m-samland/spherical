"""WAFFLE_MODE provenance: the flag travels in the cube header, never in a symlink.

The science frame type ("center" in a continuous-waffle sequence, "coro"
otherwise) is a property of the observation. It used to be guessed from files on
disk, either by symlinking center_cube.fits onto coro_cube.fits or by comparing
frame counts. Both are gone; the flag is stamped as a header card so a standalone
re-run can read it without an observation object.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy.io import fits

from spherical.pipeline.steps.align_frames import WAFFLE_KEYWORD


def _converted_with_cubes(tmp_path, names=("coro", "center", "flux")):
    converted = tmp_path / "converted"
    converted.mkdir(parents=True)
    for name in names:
        fits.writeto(
            converted / f"{name}_cube.fits",
            np.zeros((2, 2, 4, 4), dtype=np.float32),
            overwrite=True,
        )
    return converted


class TestCubeHeaderUpdateStampsWaffleMode:
    @pytest.mark.parametrize("waffle", [True, False])
    def test_keyword_is_written_to_every_frame_type(self, tmp_path, waffle):
        from spherical.pipeline.steps.cube_header_update import run_cube_header_update

        converted = _converted_with_cubes(tmp_path)
        run_cube_header_update(
            frame_types_to_extract=["CORO", "CENTER", "FLUX"],
            converted_dir=str(converted),
            continuous_satellite_spots=waffle,
            logger=MagicMock(),
        )
        for name in ("coro", "center", "flux"):
            header = fits.getheader(converted / f"{name}_cube.fits")
            assert header[WAFFLE_KEYWORD] is waffle, name

    def test_omitting_the_flag_writes_no_keyword(self, tmp_path):
        from spherical.pipeline.steps.cube_header_update import run_cube_header_update

        converted = _converted_with_cubes(tmp_path, names=("coro",))
        run_cube_header_update(
            frame_types_to_extract=["CORO"],
            converted_dir=str(converted),
            logger=MagicMock(),
        )
        assert WAFFLE_KEYWORD not in fits.getheader(converted / "coro_cube.fits")

    def test_a_missing_cube_does_not_abort_the_others(self, tmp_path):
        from spherical.pipeline.steps.cube_header_update import run_cube_header_update

        converted = _converted_with_cubes(tmp_path, names=("center",))
        run_cube_header_update(
            frame_types_to_extract=["CORO", "CENTER"],
            converted_dir=str(converted),
            continuous_satellite_spots=True,
            logger=MagicMock(),
        )
        assert fits.getheader(converted / "center_cube.fits")[WAFFLE_KEYWORD] is True


class TestNoSymlinksRemainInTheTree:
    def test_no_pipeline_module_creates_a_symlink(self):
        """Aliasing two frame types onto one file makes in-place header updates
        rewrite the wrong cube. Nothing in the pipeline may do it."""
        pipeline = Path("src/spherical/pipeline")
        offenders = [
            str(p) for p in pipeline.rglob("*.py")
            if "os.symlink" in p.read_text() or ".symlink_to(" in p.read_text()
        ]
        assert offenders == []
