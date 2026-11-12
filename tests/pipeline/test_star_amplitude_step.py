"""The calibrate_star_amplitude step: inputs, fallbacks, block mask and driver wiring."""

from __future__ import annotations

import logging
from dataclasses import asdict
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table

from spherical.pipeline.pipeline_config import PreprocConfig, defaultIRDISReduction
from spherical.pipeline.step_registry import force_needs_cubes
from spherical.pipeline.steps.star_amplitude_calibration import STEP_NAME, run_star_amplitude_calibration

LOGGER = logging.getLogger("test_star_amplitude_step")
KEY = "TEST_TARGET/OBS_YJ/2024-01-01"
N_SCIENCE = 20
MJD0 = 60000.0


WFS_CONFIG = ("Wide", "FILT_M", "OPEN")
DEFAULT_SPARTA_CONFIG = {0: (*WFS_CONFIG, "OPEN"), 1: (*WFS_CONFIG, "ND_2.0"), 2: (*WFS_CONFIG, "OPEN")}


def _converted(tmp_path, wfs_flux, wfs_science, blocks=(0, 0, 1, 1), decisions=None, segments=None, flux=None,
               science="coro", sparta_config=None):
    """Write a minimal converted folder with two channels.

    Block ``b`` of the FLUX frames starts ``0.05 * b`` days after ``MJD0``; the science frames follow the first
    block one minute apart. ``flux_r3`` is 100 times the FLUX-frame WFS unless ``flux`` is given.
    ``sparta_config`` maps a segment to ``(wfs_mode, spatial_filter, wfs_filter, nd_filter)`` for
    ``conditions/sparta_config.csv``, by default segments 0 to 2 with one WFS configuration.
    """
    converted = tmp_path / "converted"
    (converted / "additional_outputs").mkdir(parents=True)
    (converted / "conditions").mkdir()
    blocks = np.asarray(blocks)
    n_flux = len(blocks)
    wfs_flux = np.asarray(wfs_flux, dtype=float)
    dit = np.concatenate([np.arange((blocks == b).sum()) for b in np.unique(blocks)])
    origfile = [f"FLUX_{b}.fits" for b in blocks]
    flux_times = MJD0 + 0.05 * blocks + dit * 1e-4
    pd.DataFrame({"ORIGFILE": origfile, "DIT INDEX": dit, "MJD START": flux_times}).to_csv(
        converted / "frames_info_flux.csv", index=False)
    science_times = MJD0 + 0.002 + np.arange(N_SCIENCE) / 1440.0
    pd.DataFrame({"MJD START": science_times}).to_csv(converted / f"frames_info_{science}.csv", index=False)
    fits.writeto(converted / "wavelengths.fits", np.array([1000.0, 1100.0]))

    flux = np.broadcast_to(100 * wfs_flux if flux is None else np.asarray(flux, dtype=float), (2, n_flux))
    decisions = ["kept"] * n_flux if decisions is None else list(decisions)
    rows = [dict(frame=f, channel=c, origfile=origfile[f], block=int(blocks[f]), flux_r3=flux[c, f], sharpness=1.0,
                 decision=decisions[f], **{"DIT INDEX": int(dit[f]), "MJD": flux_times[f]})
            for c in range(2) for f in range(n_flux)]
    pd.DataFrame(rows).to_csv(converted / "additional_outputs" / "flux_frame_vetting.csv", index=False)

    segment_flux, segment_science = (0, 0) if segments is None else segments
    for name, values, segment in (("flux", wfs_flux, segment_flux), (science, wfs_science, segment_science)):
        values = np.asarray(values, dtype=float)
        pd.DataFrame({"visloop_flux_subap": values, "flag_visloop": ["direct"] * len(values),
                      "segment": [segment] * len(values)}).to_csv(converted / "conditions" / f"matched_{name}.csv",
                                                                 index=False)
    sparta_config = DEFAULT_SPARTA_CONFIG if sparta_config is None else sparta_config
    pd.DataFrame([{"dp_id": f"SPARTA_{segment}", "segment": segment,
                   **dict(zip(("wfs_mode", "spatial_filter", "wfs_filter", "nd_filter"), values))}
                  for segment, values in sparta_config.items()]).to_csv(converted / "conditions" / "sparta_config.csv",
                                                                        index=False)
    return converted


def _run(converted, waffle=False, **parameters):
    reduction_parameters = {**asdict(PreprocConfig()), **parameters}
    return run_star_amplitude_calibration(converted, reduction_parameters, waffle, "IFS", KEY, logger=LOGGER)


def _read(converted):
    with fits.open(converted / "star_amplitude.fits") as hdus:
        return hdus[0].data.astype(float), hdus["BLOCKS"].data.astype(int).tolist(), dict(hdus[0].header)


def _statuses(caplog):
    return [r.status for r in caplog.records if getattr(r, "step", None) == STEP_NAME]


def _cloud():
    return np.linspace(1.0, 0.6, N_SCIENCE)


def test_stable_sequence_keeps_amplitude_one(tmp_path, caplog):
    converted = _converted(tmp_path, np.ones(4), np.ones(N_SCIENCE))
    with caplog.at_level(logging.INFO):
        assert _run(converted) == "constant"
    amplitude, blocks, header = _read(converted)
    assert amplitude.shape == (2, N_SCIENCE) and np.all(amplitude == 1)
    assert blocks == [1, 1]
    assert header["MODE"] == "constant" and header["REASON"] == "stable"
    assert "star_amplitude_mode" in _statuses(caplog) and _statuses(caplog)[-1] == "success"


def test_floor_acts_on_measured_frames_and_leaves_a_neighbouring_gap_frame_finite(tmp_path):
    science = _cloud()
    science[5], science[6] = 0.01, np.nan
    converted = _converted(tmp_path, [1.0, 1.0, 0.6, 0.6], science)
    assert _run(converted) == "wfs"
    amplitude, _, header = _read(converted)
    assert np.isnan(amplitude[:, 5]).all()
    assert np.isfinite(amplitude[:, 6]).all()
    assert np.isfinite(np.delete(amplitude, 5, axis=1)).all()
    assert header["NFLOOR"] == 1 and header["NFILLED"] == 1


def test_cloud_into_the_closing_block_follows_the_wfs(tmp_path):
    converted = _converted(tmp_path, [1.0, 1.0, 0.6, 0.6], _cloud())
    assert _run(converted) == "wfs"
    amplitude, blocks, header = _read(converted)
    assert amplitude.shape == (2, N_SCIENCE)
    np.testing.assert_allclose(amplitude, np.broadcast_to(_cloud() / 0.8, (2, N_SCIENCE)), rtol=1e-6)
    assert blocks == [1, 1]
    assert header["MODE"] == "wfs" and header["REASON"] == "variable"
    with fits.open(converted / "star_amplitude.fits") as hdus:
        np.testing.assert_allclose(hdus["CHANNELS"].data["scale"], [100.0, 100.0])
        np.testing.assert_allclose(hdus["CHANNELS"].data["level"], [80.0, 80.0])


def test_missing_conditions_fall_back_to_constant_with_a_warning(tmp_path, caplog):
    converted = _converted(tmp_path, [1.0, 1.0, 0.6, 0.6], _cloud())
    for path in (converted / "conditions").iterdir():
        path.unlink()
    (converted / "conditions").rmdir()
    assert _run(converted) == "constant"
    amplitude, blocks, header = _read(converted)
    assert amplitude.shape == (2, N_SCIENCE) and np.all(amplitude == 1)
    assert blocks == [1, 1]
    assert header["MODE"] == "constant" and header["REASON"] == "star_amplitude_no_wfs"
    assert "star_amplitude_no_wfs" in _statuses(caplog)


def test_flux_wfs_in_an_nd_segment_counts(tmp_path, caplog):
    converted = _converted(tmp_path, [1.0, 1.0, 0.6, 0.6], _cloud(), segments=(1, 0))
    with caplog.at_level(logging.INFO):
        assert _run(converted) == "wfs"
    amplitude, _, header = _read(converted)
    np.testing.assert_allclose(amplitude, np.broadcast_to(_cloud() / 0.8, (2, N_SCIENCE)), rtol=1e-6)
    assert header["REASON"] == "variable" and header["SEGMENT"] == 0
    assert "star_amplitude_no_sparta_config" not in _statuses(caplog)


def test_without_sparta_config_every_segment_counts(tmp_path, caplog):
    converted = _converted(tmp_path, [1.0, 1.0, 0.6, 0.6], _cloud(), segments=(1, 0))
    (converted / "conditions" / "sparta_config.csv").unlink()
    with caplog.at_level(logging.INFO):
        assert _run(converted) == "wfs"
    _, _, header = _read(converted)
    assert header["REASON"] == "variable" and header["SEGMENT"] == 0
    assert _statuses(caplog).count("star_amplitude_no_sparta_config") == 1


def test_flux_wfs_in_another_segment_is_no_wfs(tmp_path, caplog):
    sparta_config = {0: (*WFS_CONFIG, "OPEN"), 1: ("Narrow", *WFS_CONFIG[1:], "ND_2.0")}
    converted = _converted(tmp_path, [1.0, 1.0, 0.6, 0.6], _cloud(), segments=(1, 0), sparta_config=sparta_config)
    assert _run(converted) == "constant"
    amplitude, blocks, header = _read(converted)
    assert amplitude.shape == (2, N_SCIENCE) and np.all(amplitude == 1)
    assert blocks == [1, 1]
    assert header["REASON"] == "star_amplitude_no_wfs" and header["SEGMENT"] == 0
    assert "star_amplitude_no_wfs" in _statuses(caplog)


def test_stale_vetting_table_is_not_read(tmp_path, caplog):
    converted = _converted(tmp_path, [1.0, 1.0, 0.6, 0.6], _cloud())
    assert _run(converted, flux_frame_vetting=False) == "constant"
    amplitude, blocks, header = _read(converted)
    assert amplitude.shape == (2, N_SCIENCE) and np.all(amplitude == 1)
    assert blocks == []
    assert header["MODE"] == "constant" and header["REASON"] == "star_amplitude_no_vetting"
    assert "star_amplitude_no_vetting" in _statuses(caplog)
    assert not (converted / "additional_outputs" / "star_amplitude_blocks.csv").exists()


def test_frame_count_mismatch_falls_back_to_constant(tmp_path, caplog):
    converted = _converted(tmp_path, [1.0, 1.0, 0.6, 0.6], _cloud())
    matched = converted / "conditions" / "matched_coro.csv"
    pd.read_csv(matched).iloc[:-1].to_csv(matched, index=False)
    assert _run(converted) == "constant"
    amplitude, blocks, header = _read(converted)
    assert amplitude.shape == (2, N_SCIENCE) and np.all(amplitude == 1)
    assert blocks == [1, 1]
    assert header["REASON"] == "star_amplitude_frame_mismatch"
    assert "star_amplitude_frame_mismatch" in _statuses(caplog)


def test_frames_below_the_floor_are_nan(tmp_path, caplog):
    science = _cloud()
    science[7] = 0.05
    converted = _converted(tmp_path, [1.0, 1.0, 0.6, 0.6], science)
    assert _run(converted) == "wfs"
    amplitude, blocks, header = _read(converted)
    assert amplitude.shape == (2, N_SCIENCE)
    assert np.isnan(amplitude[:, 7]).all() and np.isfinite(np.delete(amplitude, 7, axis=1)).all()
    assert blocks == [1, 1]
    assert header["MODE"] == "wfs" and header["NFLOOR"] == 1
    assert "star_amplitude_below_floor" in _statuses(caplog)


def test_terrible_block_is_left_out_of_the_psf(tmp_path, caplog):
    converted = _converted(tmp_path, np.ones(4), np.ones(N_SCIENCE), flux=[100.0, 100.0, 60.0, 60.0])
    assert _run(converted) == "constant"
    amplitude, blocks, header = _read(converted)
    assert amplitude.shape == (2, N_SCIENCE) and np.all(amplitude == 1)
    assert blocks == [1, 0]
    assert header["MODE"] == "constant" and header["REASON"] == "stable"
    assert "star_amplitude_block_left_out" in _statuses(caplog)


def test_block_emptied_by_the_vetting_has_no_psf_block(tmp_path):
    decisions = ["kept", "kept", "frame_rejected", "frame_rejected", "kept", "kept"]
    converted = _converted(tmp_path, np.ones(6), np.ones(N_SCIENCE), blocks=(0, 0, 1, 1, 2, 2), decisions=decisions)
    assert _run(converted) == "constant"
    amplitude, blocks, header = _read(converted)
    assert amplitude.shape == (2, N_SCIENCE) and np.all(amplitude == 1)
    assert blocks == [1, 1]
    assert header["REASON"] == "stable"
    table = pd.read_csv(converted / "additional_outputs" / "star_amplitude_blocks.csv").drop_duplicates("block")
    assert table["psf_block"].tolist() == [0, -1, 1]
    assert table["usable"].tolist() == [True, False, True]


def test_block_excluded_by_hand(tmp_path):
    converted = _converted(tmp_path, np.ones(4), np.ones(N_SCIENCE))
    assert _run(converted, star_amplitude_exclude_blocks={KEY: [1]}) == "constant"
    amplitude, blocks, header = _read(converted)
    assert amplitude.shape == (2, N_SCIENCE) and np.all(amplitude == 1)
    assert blocks == [1, 0]
    assert header["REASON"] == "stable"
    table = pd.read_csv(converted / "additional_outputs" / "star_amplitude_blocks.csv").drop_duplicates("block")
    assert table["excluded_by_hand"].tolist() == [False, True]


def test_continuous_waffle_spot_mode_keeps_the_spot_shape(tmp_path):
    wfs = _cloud()
    converted = _converted(tmp_path, [1.0, 1.0, 0.6, 0.6], wfs, science="center")
    spots = np.broadcast_to(wfs * (1 + 0.1 * np.sin(np.arange(N_SCIENCE))), (2, N_SCIENCE))
    fits.writeto(converted / "spot_amplitude_variation.fits", spots)
    assert _run(converted, waffle=True, star_amplitude_waffle_mode="spots") == "spots"
    amplitude, blocks, header = _read(converted)
    assert amplitude.shape == (2, N_SCIENCE)
    k = amplitude / spots
    np.testing.assert_allclose(k, k[:, :1] * np.ones((1, N_SCIENCE)), rtol=1e-6)
    assert not np.allclose(amplitude, wfs / 0.8)
    assert blocks == [1, 1]
    assert header["MODE"] == "spots" and header["REASON"] == "variable" and header["WAFMODE"] == "spots"
    with fits.open(converted / "star_amplitude.fits") as hdus:
        assert "spot_level_scatter" in hdus["CHANNELS"].columns.names


def test_forcing_the_step_on_a_cleaned_observation_needs_no_cubes():
    assert not force_needs_cubes({"calibrate_star_amplitude"})
    assert not force_needs_cubes({"observing_conditions", "calibrate_star_amplitude"})


def test_forcing_the_step_on_a_cleaned_irdis_observation_does_not_rebuild(tmp_path):
    from spherical.pipeline import irdis_reduction as ir

    frame = Table({"DP.ID": ["SPHER.2024-01-01T00:00:00.000"]})
    observation = SimpleNamespace(
        observation=Table({"INSTRUMENT": ["irdis"], "MAIN_ID": ["TEST TARGET"], "FILTER": ["DB_H23"],
                           "NIGHT_START": ["2024-01-01"], "WAFFLE_MODE": [False]}),
        frames={"CORO": frame, "CENTER": frame, "FLUX": frame, "FLAT": frame, "BG_SCIENCE": frame},
        filter="DB_H23",
    )
    converted = tmp_path / "reduction/IRDIS/observation/TEST_TARGET/DB_H23/2024-01-01/converted"
    converted.mkdir(parents=True)
    (converted / ".cubes.cleaned").touch()
    config = defaultIRDISReduction()
    config.directories.raw_directory = tmp_path / "raw"
    config.directories.reduction_directory = tmp_path / "reduction"
    config.steps.enable_all_irdis_steps()
    config.steps.force = {"calibrate_star_amplitude"}

    with patch.object(ir, "download_data_for_observation") as download, patch.object(
        ir, "run_irdis_preprocess") as preprocess, patch.object(ir, "run_observing_conditions") as conditions, patch.object(
        ir, "run_star_amplitude_calibration") as amplitude:
        ir.execute_irdis_target(observation=observation, config=config)

    download.assert_not_called()
    preprocess.assert_not_called()
    conditions.assert_not_called()
    amplitude.assert_called_once()
    assert amplitude.call_args.args[0] == converted
    assert amplitude.call_args.args[3:] == ("IRDIS", "TEST_TARGET/DB_H23/2024-01-01")
