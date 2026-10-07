"""SIMBAD matches kept for the target table (#217)."""
import numpy as np
from astropy.table import MaskedColumn, Table

from spherical.database.target_table import select_stellar_matches


def _results(plx, flux_j=None, plx_mask=None):
    n = len(plx)
    return Table({
        "flux_j": MaskedColumn(flux_j if flux_j is not None else [8.0] * n, mask=[False] * n),
        "pmra": MaskedColumn([1.0] * n, mask=[False] * n),
        "pmdec": MaskedColumn([1.0] * n, mask=[False] * n),
        "plx_value": MaskedColumn(plx, mask=plx_mask if plx_mask is not None else [False] * n),
    })


def test_default_keeps_every_positive_parallax_in_mas():
    """0.0005 mas is 2 Mpc: the default must not act as a distance cut."""
    kept = select_stellar_matches(_results([0.0005, 0.6, 50.0, 0.0, -0.3]), J_mag_limit=14.0)
    assert kept["plx_value"].tolist() == [0.0005, 0.6, 50.0]


def test_parallax_limit_is_in_mas():
    kept = select_stellar_matches(_results([0.6, 1.5, 50.0]), J_mag_limit=14.0, parallax_limit=1.0)
    assert kept["plx_value"].tolist() == [1.5, 50.0]


def test_missing_parallax_and_faint_j_are_dropped():
    results = _results([5.0, 5.0, 5.0], flux_j=[8.0, 15.0, 9.0], plx_mask=[False, False, True])
    kept = select_stellar_matches(results, J_mag_limit=14.0)
    assert np.asarray(kept["flux_j"]).tolist() == [8.0]
