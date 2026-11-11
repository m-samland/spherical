"""Background choice for observation objects (#250).

Daytime ``DARK,BACKGROUND`` frames are the default background, SKY frames are an opt-in,
and a SKY file taken at the science pointing is never used, because it contains the star.
HR 8799 BB_H 2017-10-12 has such a pair: one SKY file at a 10" offset and one back on target.

These build small astropy Tables directly, so no real data or network access is needed.
"""

import numpy as np
import pytest
from astropy.table import Table, vstack

from spherical.database.ifs_observation import IFSObservation
from spherical.database.irdis_observation import IRDISObservation

NIGHT = "2017-10-12"
MJD = 58039.2
RA, DEC = 346.869784, 21.13502
OFFSET_DEG = 10 / 3600


def _files(dpr_types, mode_key, mode_value, mjd0=MJD, ra=RA, dec=DEC):
    n = len(dpr_types)
    return Table(
        {
            "DPR_TYPE": list(dpr_types),
            "MJD_OBS": mjd0 + np.arange(n) * 0.001,
            "EXPTIME": [2.0] * n,
            "ND_FILTER": ["OPEN"] * n,
            "NIGHT_START": [NIGHT] * n,
            "RA": np.full(n, ra),
            "DEC": np.full(n, dec),
            mode_key: [mode_value] * n,
        }
    )


def _sequence(mode_key, mode_value, sky_offsets_deg):
    science = _files(["OBJECT,FLUX", "OBJECT,CENTER", "OBJECT", "OBJECT", "OBJECT,CENTER"], mode_key, mode_value)
    skies = [
        _files(["SKY"], mode_key, mode_value, mjd0=MJD + 0.01 * (i + 1), ra=RA + offset, dec=DEC + offset)
        for i, offset in enumerate(sky_offsets_deg)
    ]
    files = vstack([science, *skies])
    files.sort("MJD_OBS")
    return files


def _calibration(mode_key, mode_value):
    day = _files(["DARK,BACKGROUND"] * 3 + ["DARK"] * 2, mode_key, mode_value, mjd0=MJD + 0.5)
    return {"BACKGROUND": day}


def _observation(mode_value):
    return Table(
        {
            "MAIN_ID": ["HD 218396"],
            "NIGHT_START": [NIGHT],
            "FILTER": [mode_value],
            "WAFFLE_MODE": [False],
        }
    )


IRDIS = pytest.param(IRDISObservation, "DB_FILTER", "DB_H23", id="irdis")
IFS = pytest.param(IFSObservation, "IFS_MODE", "OBS_H", id="ifs")


@pytest.mark.parametrize(("cls", "mode_key", "mode_value"), [IRDIS, IFS])
def test_daytime_background_is_the_default(cls, mode_key, mode_value):
    files = _sequence(mode_key, mode_value, [OFFSET_DEG])

    obs = cls(_observation(mode_value), files, _calibration(mode_key, mode_value))

    assert set(obs.frames["BG_SCIENCE"]["DPR_TYPE"]) == {"DARK,BACKGROUND"}
    assert len(obs.frames["BG_SCIENCE"]) == 3


@pytest.mark.parametrize(("cls", "mode_key", "mode_value"), [IRDIS, IFS])
def test_prefer_sky_uses_off_target_sky(cls, mode_key, mode_value):
    files = _sequence(mode_key, mode_value, [OFFSET_DEG])

    obs = cls(_observation(mode_value), files, _calibration(mode_key, mode_value), prefer_sky=True)

    assert list(obs.frames["BG_SCIENCE"]["DPR_TYPE"]) == ["SKY"]


@pytest.mark.parametrize(("cls", "mode_key", "mode_value"), [IRDIS, IFS])
def test_sky_at_the_science_pointing_is_rejected(cls, mode_key, mode_value):
    files = _sequence(mode_key, mode_value, [OFFSET_DEG, 0.0])

    obs = cls(_observation(mode_value), files, _calibration(mode_key, mode_value), prefer_sky=True)

    sky = obs.frames["BG_SCIENCE"]
    assert len(sky) == 1
    assert sky["RA"][0] == pytest.approx(RA + OFFSET_DEG)


@pytest.mark.parametrize(("cls", "mode_key", "mode_value"), [IRDIS, IFS])
def test_only_on_target_sky_falls_back_to_daytime_background(cls, mode_key, mode_value):
    files = _sequence(mode_key, mode_value, [0.0])

    obs = cls(_observation(mode_value), files, _calibration(mode_key, mode_value), prefer_sky=True)

    assert set(obs.frames["BG_SCIENCE"]["DPR_TYPE"]) == {"DARK,BACKGROUND"}


@pytest.mark.parametrize(("cls", "mode_key", "mode_value"), [IRDIS, IFS])
def test_on_target_sky_is_not_used_as_fallback(cls, mode_key, mode_value):
    """Without daytime backgrounds the order falls through to DARK, skipping the on-target SKY."""
    files = _sequence(mode_key, mode_value, [0.0])
    calibration = _calibration(mode_key, mode_value)
    calibration["BACKGROUND"] = calibration["BACKGROUND"][calibration["BACKGROUND"]["DPR_TYPE"] == "DARK"]

    obs = cls(_observation(mode_value), files, calibration)

    assert set(obs.frames["BG_SCIENCE"]["DPR_TYPE"]) == {"DARK"}
