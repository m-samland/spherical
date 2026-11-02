"""SPARTA file reader and reductions (spherical.conditions.sparta)."""

import subprocess
import sys

import numpy as np
import pytest
from astropy.io import fits
from astropy.time import Time

from spherical.conditions.sparta import (
    SAMPLE_STAMP,
    SPARTA_TABLES,
    derive_atmospheric,
    dtts_photometry,
    read_sparta_file,
    wfs_gain,
)


def _unix_to_mjd(seconds):
    return Time(np.asarray(seconds, dtype=float), format="unix").mjd


MODE = "GAIN_1000_FREQ_1380Hz"
T0 = 1506589251


def _table(name, seconds, **columns):
    seconds = np.asarray(seconds, dtype=float)
    cols = [
        fits.Column(name="Sec", format="J", array=np.floor(seconds).astype(">i4")),
        fits.Column(name="USec", format="J", array=np.round((seconds % 1) * 1e6).astype(">i4")),
    ]
    for column, values in columns.items():
        cols.append(fits.Column(name=column, format="D", array=np.asarray(values, dtype=">f8")))
    return fits.BinTableHDU.from_columns(cols, name=name)


def _dtts_table(images, seconds):
    n = len(seconds)
    cols = [
        fits.Column(name="Sec", format="J", array=np.floor(seconds).astype(">i4")),
        fits.Column(name="USec", format="J", array=np.zeros(n, dtype=">i4")),
        fits.Column(name="WindowSizeX", format="J", array=np.full(n, 32, dtype=">i4")),
        fits.Column(name="WindowSizeY", format="J", array=np.full(n, 32, dtype=">i4")),
        fits.Column(name="Pixels", format="1024E", array=np.asarray(images, dtype=">f4").reshape(n, 1024)),
    ]
    return fits.BinTableHDU.from_columns(cols, name="IRPixelAvgFrame")


def _write(path, hdus, spatial_filter="SMALL"):
    primary = fits.PrimaryHDU()
    primary.header["HIERARCH ESO AOS VISWFS MODE"] = MODE
    if spatial_filter is not None:
        primary.header["HIERARCH ESO INS4 OPTI22 NAME"] = spatial_filter
    primary.header["HIERARCH ESO INS4 FILT3 NAME"] = "LP_780"
    primary.header["HIERARCH ESO INS4 FILT2 NAME"] = "OPEN"
    primary.header["HIERARCH ESO OBS ID"] = 1837295
    fits.HDUList([primary, *hdus]).writeto(path)
    return path


def _full_hdus(n_dtts=4):
    return [
        _table(
            "AtmPerfParams",
            [T0, T0 + 20, T0 + 40],
            R0=[0.1] * 3,
            T0=[0] * 3,
            L0=[25] * 3,
            WindSpeed=[10] * 3,
            StrehlRatio=[0.8] * 3,
        ),
        _table("VisLoopParams", [T0, T0 + 10, T0 + 20], Flux_avg=[17000.0] * 3, Flux_std=[1.0] * 3),
        _table("IRLoopParams", [T0, T0 + 30], Flux_avg=[1.0, 2.0], Flux_std=[0.1, 0.1]),
        _dtts_table(_dtts_star(n_dtts), T0 + 2.0 * np.arange(1, n_dtts + 1)),
    ]


def _dtts_star(n):
    """Images on a flat background of 5 with a 2x2 star of 10 above it, so the photometric sum is 40."""
    images = np.full((n, 32, 32), 5.0)
    images[:, 15:17, 15:17] += 10.0
    return images


def test_tables_have_sample_intervals(tmp_path):
    hdus = [_table("VisLoopParams", [0, 10, 20], Flux_avg=[1.0] * 3, Flux_std=[0.0] * 3)]
    result = read_sparta_file(_write(tmp_path / "a.fits", hdus))
    vis = result.tables["VisLoopParams"]
    assert len(vis) == 3
    np.testing.assert_allclose(vis["mjd_end"] - vis["mjd_start"], 10 / 86400, rtol=1e-6)
    stamp = _unix_to_mjd([0, 10, 20])
    expected_start = {"start": stamp, "mid": stamp - 5 / 86400, "end": stamp - 10 / 86400}[SAMPLE_STAMP["VisLoopParams"]]
    np.testing.assert_allclose(vis["mjd_start"], expected_start, atol=1e-9)


def test_wfs_flux_is_gain_normalised(tmp_path):
    result = read_sparta_file(_write(tmp_path / "a.fits", _full_hdus()))
    np.testing.assert_allclose(result.tables["VisLoopParams"]["flux_subap"], 1.0)
    assert result.config["wfs_gain"] == 1000.0
    assert wfs_gain("GAIN_1000_FREQ_1380Hz") == 1000.0
    assert np.isnan(wfs_gain("weird"))
    assert np.isnan(wfs_gain(None))


def test_dtts_photometry_subtracts_border_background():
    image = np.full((32, 32), 5.0)
    image[15:18, 15:18] += 100.0
    result = dtts_photometry(image[None])
    assert result["dtts_sum"].iloc[0] == pytest.approx(900, abs=1e-9)
    assert result["dtts_peak"].iloc[0] == pytest.approx(100, rel=0.01)


def test_dtts_peak_of_a_smooth_peak_is_the_vertex():
    yy, xx = np.mgrid[:32, :32]
    image = 1000.0 * np.exp(-((xx - 15.4) ** 2 + (yy - 16.0) ** 2) / (2 * 3.0**2))
    peak = dtts_photometry(image[None])["dtts_peak"].iloc[0]
    assert peak == pytest.approx(1000.0, rel=0.02)


def test_derive_atmospheric_follows_vigan():
    import pandas as pd

    atm = pd.DataFrame({"R0": [0.1, 1.2, 0.1], "WindSpeed": [10.0, 10.0, 10.0], "StrehlRatio": [0.8, 0.8, 0.99]})
    out = derive_atmospheric(atm)
    assert out["tau0"].iloc[0] == pytest.approx(0.00314)
    assert out["seeing"].iloc[0] == pytest.approx(np.degrees(500e-9 / 0.1) * 3600)
    assert np.isnan(out["r0_clean"].iloc[1])
    assert np.isnan(out["strehl_clean"].iloc[2])
    assert out["strehl_clean"].iloc[0] == 0.8


def test_empty_and_single_row_tables(tmp_path):
    hdus = [
        _table("AtmPerfParams", [T0], R0=[0.1], T0=[0], L0=[25], WindSpeed=[10], StrehlRatio=[0.8]),
        _table("IRLoopParams", [], Flux_avg=[], Flux_std=[]),
    ]
    result = read_sparta_file(_write(tmp_path / "a.fits", hdus))
    assert result.tables["IRLoopParams"].empty
    assert list(result.tables["IRLoopParams"].columns) == ["mjd_start", "mjd_end", "Flux_avg", "Flux_std"]
    atm = result.tables["AtmPerfParams"]
    assert len(atm) == 1
    np.testing.assert_allclose(atm["mjd_end"] - atm["mjd_start"], 20 / 86400, rtol=1e-6)


def test_missing_extension_and_header_key(tmp_path):
    hdus = _full_hdus()[:3]
    result = read_sparta_file(_write(tmp_path / "a.fits", hdus, spatial_filter=None))
    assert "IRPixelAvgFrame" in result.missing
    assert result.config["spatial_filter"] is None
    assert result.dtts.empty
    assert list(result.dtts.columns) == ["mjd_start", "mjd_end", "dtts_sum", "dtts_peak"]
    assert result.dtts_images.shape[0] == 0


def test_full_file_reads_every_table(tmp_path):
    result = read_sparta_file(_write(tmp_path / "SPHER.x.fits", _full_hdus()))
    assert result.dp_id == "SPHER.x"
    assert set(result.tables) == set(SPARTA_TABLES)
    assert result.missing == []
    assert result.config["ob_id"] == 1837295
    assert len(result.dtts) == 4
    assert result.dtts_images.shape == (4, 32, 32)
    np.testing.assert_allclose(result.dtts["dtts_sum"], 40.0)


def test_conditions_package_is_base_importable():
    probe = (
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] == 'scipy' or name.startswith('spherical.pipeline'):\n"
        "            raise ImportError(name)\n"
        "sys.meta_path.insert(0, Block())\n"
        "import spherical.conditions, spherical.conditions.sparta\n"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_non_positive_flux_samples_become_nan(tmp_path):
    star = np.zeros((3, 32, 32))
    star[:, 15:17, 15:17] = [[[10.0]], [[0.0]], [[10.0]]]
    hdus = [
        _table("VisLoopParams", [T0, T0 + 10, T0 + 20], Flux_avg=[17000.0, 0.0, 17000.0], Flux_std=[1.0] * 3),
        _table("IRLoopParams", [T0, T0 + 30], Flux_avg=[-1.0, 2.0], Flux_std=[0.1, 0.1]),
        _dtts_table(star, T0 + 2.0 * np.arange(1, 4)),
    ]
    result = read_sparta_file(_write(tmp_path / "a.fits", hdus))
    vis = result.tables["VisLoopParams"]
    assert vis["Flux_avg"].isna().tolist() == [False, True, False]
    assert vis["flux_subap"].isna().tolist() == [False, True, False]
    assert vis["Flux_std"].notna().all()
    assert result.tables["IRLoopParams"]["Flux_avg"].isna().tolist() == [True, False]
    assert result.dtts["dtts_sum"].isna().tolist() == [False, True, False]
    assert result.dtts["dtts_peak"].isna().tolist() == [False, True, False]
