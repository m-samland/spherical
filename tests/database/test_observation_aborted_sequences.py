"""Construction of observation objects for sequences with no coronagraphic frames.

A sequence aborted before the coronagraphic frames keeps ``WAFFLE_MODE=False``, because
``PRIMARY_SCIENCE`` is decided by an exposure-time contest that FLUX can win. Such an
observation still has to build, so that flux-only and centre-only sequences stay usable
for download and manual analysis.

These build small astropy Tables directly and pass an empty calibration dictionary, so no
real data or network access is needed.
"""

import numpy as np
import pytest
from astropy.table import Table

from spherical.database.ifs_observation import IFSObservation
from spherical.database.irdis_observation import IRDISObservation

NIGHT = "2026-06-16"


def _science_files(dpr_types, mode_key, mode_value):
    n = len(dpr_types)
    return Table(
        {
            "DPR_TYPE": list(dpr_types),
            "MJD_OBS": 61208.0 + np.arange(n) * 0.001,
            "EXPTIME": [2.0] * n,
            "ND_FILTER": ["ND_1.0"] * n,
            "NIGHT_START": [NIGHT] * n,
            mode_key: [mode_value] * n,
        }
    )


def _observation(mode_value, waffle_mode):
    return Table(
        {
            "MAIN_ID": ["HD 144667"],
            "NIGHT_START": [NIGHT],
            "FILTER": [mode_value],
            "WAFFLE_MODE": [waffle_mode],
        }
    )


IRDIS = pytest.param(IRDISObservation, "DB_FILTER", "DB_K12", id="irdis")
IFS = pytest.param(IFSObservation, "IFS_MODE", "OBS_H", id="ifs")


@pytest.mark.parametrize(("cls", "mode_key", "mode_value"), [IRDIS, IFS])
def test_flux_only_sequence_builds_with_empty_center_split(cls, mode_key, mode_value):
    """HD 144667: the sequence was aborted after a single flux frame."""
    files = _science_files(["OBJECT,FLUX"], mode_key, mode_value)

    obs = cls(_observation(mode_value, waffle_mode=False), files, {})

    assert len(obs.frames["CORO"]) == 0
    assert len(obs.frames["CENTER"]) == 0
    assert len(obs.frames["FLUX"]) == 1
    assert len(obs.frames["CENTER_BEFORE"]) == 0
    assert len(obs.frames["CENTER_AFTER"]) == 0


@pytest.mark.parametrize(("cls", "mode_key", "mode_value"), [IRDIS, IFS])
def test_center_and_flux_without_coronagraphy_builds(cls, mode_key, mode_value):
    """HD 106036: one centre frame and two flux frames, no coronagraphic frames."""
    files = _science_files(
        ["OBJECT,FLUX", "OBJECT,FLUX", "OBJECT,CENTER"], mode_key, mode_value
    )

    obs = cls(_observation(mode_value, waffle_mode=False), files, {})

    assert len(obs.frames["CORO"]) == 0
    assert len(obs.frames["CENTER"]) == 1
    assert len(obs.frames["CENTER_BEFORE"]) == 0
    assert len(obs.frames["CENTER_AFTER"]) == 0


@pytest.mark.parametrize(("cls", "mode_key", "mode_value"), [IRDIS, IFS])
def test_empty_center_split_keeps_science_file_columns(cls, mode_key, mode_value):
    """The empty split must stay a column-compatible table, not a bare ``Table()``."""
    files = _science_files(["OBJECT,FLUX"], mode_key, mode_value)

    obs = cls(_observation(mode_value, waffle_mode=False), files, {})

    assert obs.frames["CENTER_BEFORE"].colnames == files.colnames
    assert obs.frames["CENTER_AFTER"].colnames == files.colnames


@pytest.mark.parametrize(("cls", "mode_key", "mode_value"), [IRDIS, IFS])
def test_coronagraphic_sequence_still_splits_centers(cls, mode_key, mode_value):
    """The guard must not disturb a normal sequence bracketed by centre frames."""
    files = _science_files(
        ["OBJECT,CENTER", "OBJECT", "OBJECT", "OBJECT,CENTER"], mode_key, mode_value
    )

    obs = cls(_observation(mode_value, waffle_mode=False), files, {})

    assert len(obs.frames["CORO"]) == 2
    assert len(obs.frames["CENTER_BEFORE"]) == 1
    assert len(obs.frames["CENTER_AFTER"]) == 1
    assert obs.frames["CENTER_BEFORE"]["MJD_OBS"][0] < obs.frames["CORO"]["MJD_OBS"][0]
    assert obs.frames["CENTER_AFTER"]["MJD_OBS"][0] > obs.frames["CORO"]["MJD_OBS"][-1]
