"""cleanup_pipeline_products removes chosen stages for IFS and IRDIS (#218)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from astropy.table import Table

from spherical.pipeline.cleanup import cleanup_pipeline_products
from spherical.pipeline.pipeline_config import defaultIFSReduction, defaultIRDISReduction

WAVECAL_ID = "SPHER.2019-12-31T12:00:00.000"
OTHER_ID = "SPHER.2018-01-01T00:00:00.000"  # a calibration frame of another observation
FRAMES = {
    "ifs": {
        "CORO": ["SPHER.2020-01-01T00:00:00.000"],
        "CENTER": ["SPHER.2020-01-01T00:01:00.000"],
        "FLUX": ["SPHER.2020-01-01T00:02:00.000"],
        "WAVECAL": [WAVECAL_ID],
    },
    "irdis": {
        "CORO": ["SPHER.2020-01-01T00:00:00.000"],
        "CENTER": ["SPHER.2020-01-01T00:01:00.000"],
        "FLUX": ["SPHER.2020-01-01T00:02:00.000"],
        "FLAT": ["SPHER.2020-01-01T10:00:00.000"],
        "BG_SCIENCE": ["SPHER.2020-01-01T00:03:00.000"],
    },
}
BAND = {"ifs": "OBS_H", "irdis": "DB_K12"}
CUBES = ["coro_cube.fits", "center_cube.fits", "flux_cube.fits",
         "coro_ivar_cube.fits", "center_ivar_cube.fits", "flux_ivar_cube.fits", "wavelengths.fits"]


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * 10)
    return path


def _setup(tmp_path, instrument, cubes=True, trap=True):
    """Build an observation stand-in, a config and a finished reduction tree."""
    observation = SimpleNamespace(
        observation=Table({
            "INSTRUMENT": [instrument], "MAIN_ID": ["HD 1"], "FILTER": [BAND[instrument]],
            "NIGHT_START": ["2020-01-01"], "WAFFLE_MODE": [False],
        }),
        frames={key: Table({"DP.ID": ids}) for key, ids in FRAMES[instrument].items()},
    )
    config = defaultIFSReduction() if instrument == "ifs" else defaultIRDISReduction()
    config.directories.raw_directory = tmp_path / "raw"
    config.directories.reduction_directory = tmp_path / "red"

    inst, band = instrument.upper(), BAND[instrument]
    raw = tmp_path / "raw" / inst
    red = tmp_path / "red" / inst
    p = {}
    p["science_raw"] = _touch(raw / "science/HD_1" / band / "2020-01-01/CORO/SPHER.2020-01-01T00:00:00.000.fits")
    if instrument == "ifs":
        p["calib_raw"] = _touch(raw / "calibration" / band / f"WAVECAL/{WAVECAL_ID}.fits")
        p["other_calib_raw"] = _touch(raw / "calibration" / band / f"WAVECAL/{OTHER_ID}.fits")
        p["calibration"] = _touch(red / "calibration" / band / WAVECAL_ID[6:] / "lensletflat.fits")
        p["other_calibration"] = _touch(red / "calibration" / band / OTHER_ID[6:] / "lensletflat.fits")
        cube_outputdir = red / "observation/HD_1" / band / "2020-01-01/optext"
        p["extracted"] = _touch(cube_outputdir / "CORO/SPHER.2020-01-01T00:00:00.000cube_resampled_DIT_000.fits")
        p["extract_marker"] = _touch(cube_outputdir / ".extract_cubes.done")
    else:
        p["calib_raw"] = _touch(raw / "calibration" / band / "FLAT/SPHER.2020-01-01T10:00:00.000.fits")
        p["other_calib_raw"] = _touch(raw / "calibration" / band / f"FLAT/{OTHER_ID}.fits")
        p["calibration"] = _touch(red / "calibration" / band / "2020-01-01/master_flat.fits")
        p["other_calibration"] = _touch(red / "calibration" / band / "2019-06-01/master_flat.fits")
        cube_outputdir = red / "observation/HD_1" / band / "2020-01-01"
        _touch(cube_outputdir / "converted/badpixel_map.fits")
    converted = cube_outputdir / "converted"
    p["converted"] = converted
    if cubes:
        for name in CUBES:
            _touch(converted / name)
    p["aligned"] = _touch(converted / "coro_cube_aligned.fits")
    p["align_marker"] = _touch(converted / ".align_frames.done")
    p["psf"] = _touch(converted / "psf_cube_for_postprocessing.fits")
    if trap:
        _touch(red / "trap/HD_1" / band / "2020-01-01/.run_trap_detection.done")
    return observation, config, p


@pytest.mark.parametrize("instrument", ["ifs", "irdis"])
def test_default_stages_remove_calibration_and_extracted_cubes(tmp_path, instrument):
    observation, config, p = _setup(tmp_path, instrument)

    n_files, size_mb = cleanup_pipeline_products([observation], config, dry_run=False)

    assert not p["calibration"].exists()
    assert p["other_calibration"].exists()
    if instrument == "ifs":
        assert not p["extracted"].exists()
        assert not p["extract_marker"].exists()
    assert p["science_raw"].exists()
    assert (p["converted"] / "coro_cube.fits").exists()
    # IFS: the calibration file, the extracted cube and .extract_cubes.done. IRDIS: master_flat.fits.
    assert n_files == (3 if instrument == "ifs" else 1)
    assert size_mb > 0


@pytest.mark.parametrize("instrument", ["ifs", "irdis"])
def test_raw_removes_only_this_observations_frames(tmp_path, instrument):
    observation, config, p = _setup(tmp_path, instrument)

    cleanup_pipeline_products([observation], config, stages={"raw"}, dry_run=False)

    assert not p["science_raw"].exists()
    assert not p["calib_raw"].exists()
    assert p["other_calib_raw"].exists()


def test_raw_removes_compressed_frames(tmp_path):
    observation, config, p = _setup(tmp_path, "ifs")
    compressed = _touch(p["science_raw"].with_name("SPHER.2020-01-01T00:01:00.000.fits.Z"))

    cleanup_pipeline_products([observation], config, stages={"raw"}, dry_run=False)

    assert not compressed.exists()


@pytest.mark.parametrize("instrument", ["ifs", "irdis"])
def test_cubes_stage_removes_cubes_and_leaves_the_marker(tmp_path, instrument):
    observation, config, p = _setup(tmp_path, instrument)
    converted = p["converted"]

    cleanup_pipeline_products([observation], config, stages={"cubes"}, dry_run=False)

    for name in ("coro_cube.fits", "flux_ivar_cube.fits"):
        assert not (converted / name).exists()
    assert not p["aligned"].exists()
    assert not p["align_marker"].exists()
    assert (converted / "wavelengths.fits").exists()
    assert p["psf"].exists()
    assert (converted / ".cubes.cleaned").exists()


def test_cubes_are_kept_until_trap_detection_finished(tmp_path, capsys):
    observation, config, p = _setup(tmp_path, "irdis", trap=False)

    cleanup_pipeline_products([observation], config, stages={"cubes", "calibration"}, dry_run=False)

    assert (p["converted"] / "coro_cube.fits").exists()
    assert not (p["converted"] / ".cubes.cleaned").exists()
    assert not p["calibration"].exists()
    assert "TRAP detection has not finished" in capsys.readouterr().out


def test_nothing_is_removed_before_the_cubes_exist(tmp_path):
    observation, config, p = _setup(tmp_path, "ifs", cubes=False)

    n_files, _ = cleanup_pipeline_products([observation], config, stages={"raw", "calibration"}, dry_run=False)

    assert n_files == 0
    assert p["science_raw"].exists()
    assert p["calibration"].exists()


def test_an_observation_never_reduced_is_left_alone(tmp_path, capsys):
    observation, config, _ = _setup(tmp_path, "irdis")
    config.directories.reduction_directory = tmp_path / "empty"

    assert cleanup_pipeline_products([observation], config, stages={"raw"}, dry_run=False) == (0, 0.0)
    assert "not built yet" in capsys.readouterr().out


def test_cleaned_cubes_still_allow_cleaning_the_rest_and_a_repeat(tmp_path):
    observation, config, p = _setup(tmp_path, "ifs")
    cleanup_pipeline_products([observation], config, stages={"cubes"}, dry_run=False)

    cleanup_pipeline_products([observation], config, stages={"cubes", "raw"}, dry_run=False)

    assert not p["science_raw"].exists()
    assert (p["converted"] / ".cubes.cleaned").exists()


def test_dry_run_removes_nothing(tmp_path):
    observation, config, p = _setup(tmp_path, "ifs")

    n_files, _ = cleanup_pipeline_products(
        [observation], config, stages={"raw", "calibration", "extracted", "cubes"}, dry_run=True)

    assert n_files > 0
    assert p["science_raw"].exists() and p["calibration"].exists() and p["extracted"].exists()
    assert (p["converted"] / "coro_cube.fits").exists()
    assert not (p["converted"] / ".cubes.cleaned").exists()


def test_an_unknown_stage_raises(tmp_path):
    observation, config, _ = _setup(tmp_path, "ifs")
    with pytest.raises(ValueError, match="wavecal"):
        cleanup_pipeline_products([observation], config, stages={"wavecal"})
