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
            "DP.ID": [f"SPHER.{i}" for i in range(n)],
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


def test_a_declination_just_south_of_the_equator_keeps_its_sign():
    """-00d05m30s is ``INS4 DROT2 DEC = -530.0``; its degree part is -0, which must not drop the sign (#229)."""
    frames = _hr8799_frames(pd.date_range("2016-11-18T00:39:23", periods=2, freq="10min"))
    frames["INS4 DROT2 DEC"] = -530.0

    compute_angles(frames)

    assert np.allclose(frames["DEC"].to_numpy(), -(5 / 60 + 30 / 3600))


def test_field_tracking_does_not_rotate():
    """In field tracking North is set by POSANG, not by the parallactic angle (#187)."""
    frames = _hr8799_frames(pd.date_range("2016-11-18T00:39:23", "2016-11-18T02:08:26", periods=10))
    frames["INS4 DROT2 MODE"] = "SKY"
    frames["INS4 DROT2 POSANG"] = 10.0

    compute_angles(frames)

    assert np.allclose(frames["DEROT ANGLE"].to_numpy(), 10.0 - 1.75)


def test_field_tracking_drift_correction_does_not_flip():
    """PZ Tel 2014-08-05, four IRDIS files 3.5 min apart (headers from the archive).

    The pupil-tracking drift term ALT - 2 DROT2 crosses 90 deg here and flipped the
    derotation angle by 180 deg; the field-tracking term stays within 0.2 deg of 0 (#187).
    """
    times = pd.to_datetime(
        ["2014-08-06T04:47:35.964", "2014-08-06T04:51:08.826", "2014-08-06T04:54:41.128", "2014-08-06T04:58:13.512"]
    )
    frames = pd.DataFrame(
        {
            "MJD": [56875.199722, 56875.202185, 56875.204643, 56875.207101],
            "DP.ID": [f"SPHER.{i}" for i in range(4)],
            "INS4 DROT2 RA": np.full(4, 185415.87),
            "INS4 DROT2 DEC": np.full(4, -500936.31),
            "TEL GEOLON": np.full(4, -70.4045),
            "TEL GEOLAT": np.full(4, PARANAL_LAT),
            "TEL GEOELEV": np.full(4, 2648.0),
            "TIME START": times,
            "TIME": times,
            "TIME END": times,
            "SEQ ARM": ["IRDIS"] * 4,
            "INS4 DROT2 MODE": ["SKY"] * 4,
            "INS4 DROT2 POSANG": np.full(4, 329.23),
            "INS4 DROT2 BEGIN": [-16.5899, -17.3841, -18.1643, -18.9447],
            "TEL ALT": [54.173, 53.694, 53.211, 52.723],
            "TEL PARANG START": [56.715, 57.826, 58.907, 59.969],
        }
    )

    compute_angles(frames)

    assert np.all(np.abs(frames["DEROT ANGLE"].to_numpy() - (329.23 - 1.75)) < 0.2)
