"""Flux-setup flags and ``HCI_READY`` for mixed flux DIT (#173).

Mixed flux DIT no longer blocks ``HCI_READY``, because the pipeline keeps every
unsaturated flux cube and scales each frame by its own DIT and ND (#171, #172). Having no
flux cube still blocks. These build small astropy Tables directly, for IRDIS and IFS
alike, so no real data or network access is needed.
"""

import numpy as np
import pytest
from astropy.table import Table

from spherical.database.observation_table import compute_hci_ready, evaluate_observation_flags

# file_table gives every file a DB_FILTER column, "N/A" on IFS, so it is no instrument test.
IRDIS = pytest.param("NAXIS3", "DB_H23", id="irdis")
IFS = pytest.param("NAXIS3", "N/A", id="ifs")


def _files(flux_setups, ndit_key, db_filter):
    """One CENTER, one CORO and one file per ``(EXPTIME, NDIT, ND_FILTER)`` flux setup."""
    setups = [(2.0, 4, "OPEN", "OBJECT,CENTER"), (16.0, 20, "OPEN", "OBJECT")]
    setups += [(*setup, "OBJECT,FLUX") for setup in flux_setups]
    exptime, ndit, nd_filter, dpr_type = zip(*setups)
    return Table({
        "DPR_TYPE": dpr_type,
        "EXPTIME": exptime,
        ndit_key: ndit,
        "ND_FILTER": nd_filter,
        "DB_FILTER": [db_filter] * len(setups),
        "WAFFLE_AMP": [0.05] * len(setups),
    })


def _ready(flags, derotator_flag=False, polarimetry=False):
    return compute_hci_ready({**flags, "DEROTATOR_FLAG": derotator_flag}, polarimetry)


@pytest.mark.parametrize(("ndit_key", "db_filter"), [IRDIS, IFS])
def test_homogeneous_flux_is_ready_with_unit_spread(ndit_key, db_filter):
    flags = evaluate_observation_flags(
        _files([(0.837, 90, "ND_2.0"), (0.837, 90, "ND_2.0")], ndit_key, db_filter), ndit_key)

    assert not flags["FLUX_DIT_FLAG"] and not flags["FLUX_ND_FLAG"]
    assert flags["FLUX_DIT_SPREAD"] == 1.0
    assert (flags["DIT_FLUX"], flags["NDIT_FLUX"], flags["ND_FILTER_FLUX"]) == (0.837, 90, "ND_2.0")
    assert _ready(flags)


@pytest.mark.parametrize(("ndit_key", "db_filter"), [IRDIS, IFS])
def test_mixed_flux_dit_and_nd_is_ready_and_reported(ndit_key, db_filter):
    """pi Men 2015-12-19: 90 x 0.837 s ND_2.0 before the sequence, 8 x 8 s ND_1.0 after."""
    flags = evaluate_observation_flags(
        _files([(0.837, 90, "ND_2.0"), (8.0, 8, "ND_1.0")], ndit_key, db_filter), ndit_key)

    assert flags["FLUX_DIT_FLAG"] and flags["FLUX_ND_FLAG"]
    assert flags["FLUX_DIT_SPREAD"] == pytest.approx(8.0 / 0.837)
    # The setup with the most integration time (75 s against 64 s) is recorded.
    assert (flags["DIT_FLUX"], flags["NDIT_FLUX"], flags["ND_FILTER_FLUX"]) == (0.837, 90, "ND_2.0")
    assert _ready(flags)


@pytest.mark.parametrize(("ndit_key", "db_filter"), [IRDIS, IFS])
def test_mixed_nd_at_one_dit_sets_only_the_nd_flag(ndit_key, db_filter):
    flags = evaluate_observation_flags(
        _files([(4.0, 10, "ND_1.0"), (4.0, 10, "ND_2.0")], ndit_key, db_filter), ndit_key)

    assert flags["FLUX_ND_FLAG"] and not flags["FLUX_DIT_FLAG"]
    assert flags["FLUX_DIT_SPREAD"] == 1.0
    assert _ready(flags)


@pytest.mark.parametrize(("ndit_key", "db_filter"), [IRDIS, IFS])
def test_no_flux_still_blocks(ndit_key, db_filter):
    flags = evaluate_observation_flags(_files([], ndit_key, db_filter), ndit_key)

    assert flags["FLUX_FLAG"] and not flags["FLUX_ND_FLAG"]
    assert np.isnan(flags["FLUX_DIT_SPREAD"])
    assert not _ready(flags)


@pytest.mark.parametrize("blocker", ["CENTER_DIT_FLAG", "CORO_DIT_FLAG", "CENTER_FLAG"])
def test_center_and_coro_flags_still_block(blocker):
    flags = evaluate_observation_flags(_files([(4.0, 10, "ND_1.0")], "NAXIS3", "DB_H23"), "NAXIS3")

    assert _ready(flags)
    assert not _ready({**flags, blocker: True})


def test_derotator_flag_blocks_unless_polarimetry():
    flags = evaluate_observation_flags(_files([(4.0, 10, "ND_1.0")], "NAXIS3", "DB_H23"), "NAXIS3")

    assert not _ready(flags, derotator_flag=True)
    assert _ready(flags, derotator_flag=True, polarimetry=True)
