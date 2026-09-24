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
from spherical.database.observation_table import select_primary_science_frames
from spherical.database.sphere_database import USABLE_MIN_EXPTIME_SCI

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


# ---------------------------------------------------------------------------
# PRIMARY_SCIENCE contest (#179): FLUX frames are never science, so the contest
# runs between CORO and CENTER, and FLUX is only the label of last resort.
# ---------------------------------------------------------------------------


def _group(frames):
    """Observation group from ``(dpr_type, exptime, ndit)`` tuples."""
    return Table(
        {
            "DPR_TYPE": [f[0] for f in frames],
            "EXPTIME": [float(f[1]) for f in frames],
            "NDIT": [int(f[2]) for f in frames],
        }
    )


def _science_minutes(rows):
    """``TOTAL_EXPTIME_SCI`` as ``compute_basic_metadata`` derives it from the primary rows."""
    return round(float(np.sum(rows["EXPTIME"] * rows["NDIT"])) / 60, 3)


CORO, CENTER, FLUX = "OBJECT", "OBJECT,CENTER", "OBJECT,FLUX"


def test_flux_only_sequence_keeps_flux_label():
    kind, rows, _ = select_primary_science_frames(_group([(FLUX, 2.0, 1)]), "NDIT")
    assert kind == "FLUX"
    assert len(rows) == 1


def test_center_beats_more_flux_exposure_without_coro():
    """HD 106036: 2 FLUX frames outweigh 1 CENTER cube, but only CENTER is science."""
    group = _group([(FLUX, 2.0, 60), (FLUX, 2.0, 60), (CENTER, 4.0, 15)])
    kind, rows, exptimes = select_primary_science_frames(group, "NDIT")
    assert kind == "CENTER"
    assert list(rows["DPR_TYPE"]) == [CENTER]
    assert _science_minutes(rows) == 1.0
    assert exptimes["TOTAL_EXPTIME_FLUX"] == 4.0


def test_single_cube_waffle_counts_its_full_exposure():
    """NCENTER=1 with a large NDIT is a real sequence, and must clear the usable cut."""
    group = _group([(FLUX, 2.0, 10), (CENTER, 4.0, 100)])
    kind, rows, _ = select_primary_science_frames(group, "NDIT")
    assert kind == "CENTER"
    assert _science_minutes(rows) >= USABLE_MIN_EXPTIME_SCI


def test_coronagraphic_sequence_with_one_center_stays_coro():
    group = _group([(CENTER, 4.0, 1)] + [(CORO, 32.0, 4)] * 10 + [(FLUX, 2.0, 1)])
    kind, _, _ = select_primary_science_frames(group, "NDIT")
    assert kind == "CORO"


def test_center_outweighing_stray_coro_frames_is_waffle():
    group = _group([(CORO, 16.0, 1)] * 2 + [(CENTER, 16.0, 4)] * 10)
    kind, _, _ = select_primary_science_frames(group, "NDIT")
    assert kind == "CENTER"


def test_flux_over_center_over_coro_picks_center_and_excludes_flux():
    group = _group([(FLUX, 8.0, 100), (CENTER, 16.0, 10), (CORO, 16.0, 2)])
    kind, rows, _ = select_primary_science_frames(group, "NDIT")
    assert kind == "CENTER"
    assert set(rows["DPR_TYPE"]) == {CENTER}
    assert _science_minutes(rows) == round(16.0 * 10 / 60, 3)


def test_coro_wins_a_tie_with_center():
    group = _group([(CENTER, 10.0, 6), (CORO, 10.0, 6)])
    kind, _, _ = select_primary_science_frames(group, "NDIT")
    assert kind == "CORO"


def test_all_zero_exposure_returns_no_science_rows():
    """The caller skips the row when the primary set is empty (observation_table.py guard)."""
    group = _group([(CORO, 0.0, 1), (CENTER, 0.0, 1), (FLUX, 0.0, 1)])
    _, rows, _ = select_primary_science_frames(group, "NDIT")
    assert len(rows) == 0


@pytest.mark.parametrize(("cls", "mode_key", "mode_value"), [IRDIS, IFS])
def test_check_frames_reports_missing_calibrations_instead_of_crashing(cls, mode_key, mode_value):
    """No calibration table leaves ``frames["FLAT"]`` as ``None``, which is missing, not a TypeError."""
    files = _science_files(["OBJECT,FLUX"], mode_key, mode_value)
    obs = cls(_observation(mode_value, waffle_mode=False), files, {})

    with pytest.raises(FileNotFoundError, match="FLAT"):
        obs.check_frames()


def test_nan_center_exposure_does_not_beat_coro():
    """A NaN total must not win: ``x >= nan`` is False, which used to hand the win to CENTER."""
    group = _group([(CORO, 10.0, 1), (CENTER, np.nan, 1)])
    kind, _, _ = select_primary_science_frames(group, "NDIT")
    assert kind == "CORO"
