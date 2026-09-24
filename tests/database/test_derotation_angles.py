"""Parallactic and derotation angles across the sidereal-time wrap (#167).

HR 8799 (RA 23h08m, Dec +21.2 deg) on 2016-11-18 UT transits while the local sidereal
time wraps from 24h to 0h. The hour angle ``LST - RA`` must stay continuous through that
wrap, or the derotation angle jumps by 360 deg and the table's ``ROTATION`` reports
343 deg instead of 17 deg.
"""

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import Angle

from spherical.database.metadata import compute_angles, parallactic_angle

PARANAL_LAT = -24.6268


def _hr8799_frames(times):
    """``frames_info`` as ``compute_times`` leaves it: times are ``datetime64[ns]``."""
    times = pd.to_datetime(np.asarray(times))
    n = len(times)
    return pd.DataFrame(
        {
            "MJD": np.full(n, 57710.0),
            "INS4 DROT2 RA": np.full(n, 230818.7775),
            "INS4 DROT2 DEC": np.full(n, 211341.84),
            "TEL GEOLON": np.full(n, -70.4045),
            "TEL GEOLAT": np.full(n, PARANAL_LAT),
            "TEL GEOELEV": np.full(n, 2648.0),
            "TIME START": times,
            "TIME": times,
            "TIME END": times,
            "SEQ ARM": ["IRDIS"] * n,
            "INS4 DROT2 MODE": ["ELEV"] * n,
            "INS4 DROT2 POSANG": np.zeros(n),
        }
    )


def test_parallactic_angle_is_periodic_in_hour_angle():
    """+0.9 h and -23.1 h are the same hour angle and must give the same angle."""
    dec = Angle(21.2, u.deg)
    lat = Angle(PARANAL_LAT, u.deg)
    pa = parallactic_angle(Angle([0.9, -23.1], u.hourangle), dec, lat)
    assert np.isclose(pa[0].value, pa[1].value)


def test_derotation_angle_is_continuous_across_the_lst_wrap():
    times = pd.date_range("2016-11-18T00:39:23", "2016-11-18T02:08:26", periods=40)
    frames = _hr8799_frames(times)

    compute_angles(frames)

    derot = frames["DEROT ANGLE"].to_numpy()
    assert np.max(np.abs(np.diff(derot))) < 5.0
    assert abs(derot[-1] - derot[0]) < 30.0
    ha = frames["HOUR ANGLE"].to_numpy()
    assert np.all((ha >= -12) & (ha < 12))


def test_parallactic_angle_matches_the_eso_header():
    """TEL PARANG START of three files of the real sequence, compared modulo 360 deg."""
    times = np.array(
        ["2016-11-18T00:39:23.4864", "2016-11-18T00:49:22.3532", "2016-11-18T01:07:26.6513"]
    )
    frames = _hr8799_frames(times)

    compute_angles(frames)

    header = np.array([167.688, 164.684, 159.432])
    diff = (frames["PARANG START"].to_numpy() - header + 180.0) % 360.0 - 180.0
    assert np.all(np.abs(diff) < 0.1)
