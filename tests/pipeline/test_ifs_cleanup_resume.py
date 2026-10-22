"""IFS resume after cleanup (#218). Needs charis, which ifs_reduction imports."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from astropy.table import Table

pytest.importorskip("charis")

from spherical.pipeline import ifs_reduction as ir  # noqa: E402
from spherical.pipeline.pipeline_config import defaultIFSReduction  # noqa: E402


def _observation():
    frame = Table({"DP.ID": ["SPHER.2020-01-01T00:00:00.000"]})
    return SimpleNamespace(
        observation=Table({
            "INSTRUMENT": ["ifs"], "MAIN_ID": ["TARGET"], "FILTER": ["OBS_H"],
            "NIGHT_START": ["2020-01-01"], "WAFFLE_MODE": [False],
        }),
        frames={"CORO": frame, "CENTER": frame, "FLUX": frame, "WAVECAL": frame},
    )


def _config(tmp_path, **steps):
    config = defaultIFSReduction()
    config.directories.reduction_directory = tmp_path
    config.directories.raw_directory = tmp_path / "raw"
    config.steps.disable_all_ifs_steps()
    config.steps = config.steps.merge(**steps)
    return config


def _converted(tmp_path):
    converted = tmp_path / "IFS/observation/TARGET/OBS_H/2020-01-01/optext/converted"
    converted.mkdir(parents=True, exist_ok=True)
    return converted


def _write_cubes(converted):
    for name in ("wavelengths.fits", "coro_cube.fits", "center_cube.fits", "flux_cube.fits",
                 "coro_ivar_cube.fits", "center_ivar_cube.fits", "flux_ivar_cube.fits"):
        (converted / name).write_text("stub")


_UPSTREAM = dict(download_data=True, reduce_calibration=True, extract_cubes=True)


def _patched():
    return (
        patch.object(ir, "download_data_for_observation"),
        patch.object(ir, "update_observation_file_paths"),
        patch.object(ir, "run_wavelength_calibration"),
        patch.object(ir, "extract_cubes_with_multiprocessing"),
    )


def test_cleaned_cubes_skip_the_observation(tmp_path):
    config = _config(tmp_path, **_UPSTREAM)
    marker = _converted(tmp_path) / ".cubes.cleaned"
    marker.touch()

    p_dl, p_hy, p_cal, p_ex = _patched()
    with p_dl as download, p_hy, p_cal as calib, p_ex as extract:
        ir.execute_target(observation=_observation(), config=config)

    download.assert_not_called()
    calib.assert_not_called()
    extract.assert_not_called()
    assert marker.exists()


def test_forcing_a_later_step_rebuilds_cleaned_cubes(tmp_path):
    config = _config(tmp_path, force={"align_frames"}, **_UPSTREAM)
    marker = _converted(tmp_path) / ".cubes.cleaned"
    marker.touch()

    p_dl, p_hy, p_cal, p_ex = _patched()
    with p_dl as download, p_hy, p_cal as calib, p_ex as extract:
        ir.execute_target(observation=_observation(), config=config)

    assert not marker.exists()
    download.assert_called_once()
    calib.assert_called_once()
    extract.assert_called_once()


def test_present_cubes_skip_download_calibration_and_extraction(tmp_path):
    config = _config(tmp_path, **_UPSTREAM)
    _write_cubes(_converted(tmp_path))

    p_dl, p_hy, p_cal, p_ex = _patched()
    with p_dl as download, p_hy as hydrate, p_cal as calib, p_ex as extract:
        ir.execute_target(observation=_observation(), config=config)

    download.assert_not_called()
    hydrate.assert_not_called()
    calib.assert_not_called()
    extract.assert_not_called()


def test_forcing_extraction_runs_the_upstream_chain(tmp_path):
    config = _config(tmp_path, force={"extract_cubes"}, **_UPSTREAM)
    _write_cubes(_converted(tmp_path))

    p_dl, p_hy, p_cal, p_ex = _patched()
    with p_dl as download, p_hy, p_cal as calib, p_ex as extract:
        ir.execute_target(observation=_observation(), config=config)

    download.assert_called_once()
    calib.assert_called_once()
    extract.assert_called_once()


def test_steps_cleanup_runs_after_a_successful_reduction(tmp_path):
    config = _config(tmp_path, download_data=True, cleanup={"extracted", "cubes"})

    with patch.object(ir, "download_data_for_observation"), patch.object(
        ir, "cleanup_pipeline_products") as cleanup:
        ir.execute_target(observation=_observation(), config=config)

    cleanup.assert_called_once()
    assert cleanup.call_args.kwargs["stages"] == {"extracted"}
    assert cleanup.call_args.kwargs["dry_run"] is False


def test_check_output_counts_a_cleaned_observation_as_complete(tmp_path):
    (_converted(tmp_path) / ".cubes.cleaned").touch()

    reduced, missing = ir.check_output(str(tmp_path), [_observation()])

    assert reduced == [True]
    assert missing == [[]]


def test_rebuilding_cleaned_cubes_from_kept_extracted_cubes_needs_no_raw_frames(tmp_path):
    config = _config(tmp_path, force={"align_frames"}, **_UPSTREAM)
    converted = _converted(tmp_path)
    (converted / ".cubes.cleaned").touch()
    (converted.parent / ".extract_cubes.done").touch()

    p_dl, p_hy, p_cal, p_ex = _patched()
    with p_dl as download, p_hy as hydrate, p_cal as calib, p_ex as extract:
        ir.execute_target(observation=_observation(), config=config)

    download.assert_not_called()
    hydrate.assert_not_called()
    calib.assert_not_called()
    extract.assert_not_called()


def test_a_calibration_only_run_still_downloads_and_calibrates(tmp_path):
    config = _config(tmp_path, download_data=True, reduce_calibration=True)

    p_dl, p_hy, p_cal, p_ex = _patched()
    with p_dl as download, p_hy, p_cal as calib, p_ex:
        ir.execute_target(observation=_observation(), config=config)

    download.assert_called_once()
    calib.assert_called_once()
