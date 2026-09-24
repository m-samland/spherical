"""HEALPix indexing used to group observations by sky position.

The reference indices below were generated independently with ``healpy``
(the previous implementation, GPL, dropped in favour of ``astropy-healpix``)
using the RING scheme at ``HEALPIX_NSIDE``:

    healpy.ang2pix(2**15, np.pi / 2 - np.deg2rad(dec), np.deg2rad(ra))

Both libraries implement the same HEALPix standard, so these values pin the
swap to a second implementation rather than to our own output.
"""

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from spherical.database.target_table import HEALPIX_NSIDE, compute_healpix_indices

# (name, ra_deg, dec_deg, healpy RING index at nside=2**15)
REFERENCE_POSITIONS = [
    ("Vega", 279.23473479, 38.78368896, 2407042338),
    ("Sgr A*", 266.41683, -29.00781, 9566714599),
    ("Polaris", 37.95456067, 89.26410897, 531697),
    ("origin", 0.0, 0.0, 6442254336),
    ("near south pole", 123.456789, -89.9, 12884891761),
]


def test_nside_is_unchanged():
    # The index values pinned below are only meaningful at this resolution.
    assert HEALPIX_NSIDE == 2**15


@pytest.mark.parametrize(
    "ra_deg, dec_deg, expected_index",
    [pytest.param(ra, dec, idx, id=name) for name, ra, dec, idx in REFERENCE_POSITIONS],
)
def test_matches_healpy_reference_indices(ra_deg, dec_deg, expected_index):
    coordinates = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg)

    assert compute_healpix_indices(coordinates) == expected_index


def test_accepts_array_valued_coordinates():
    ra = [ra for _, ra, _, _ in REFERENCE_POSITIONS]
    dec = [dec for _, _, dec, _ in REFERENCE_POSITIONS]
    expected = [idx for _, _, _, idx in REFERENCE_POSITIONS]

    indices = compute_healpix_indices(SkyCoord(ra=ra * u.deg, dec=dec * u.deg))

    assert np.array_equal(indices, expected)


def test_northern_and_southern_declinations_are_not_swapped():
    """Guards the colatitude convention: RING indices increase from north to south.

    An earlier implementation passed ``dec + pi/2`` as the colatitude instead of
    ``pi/2 - dec``, which mirrored every position across the equator.
    """
    npix = 12 * HEALPIX_NSIDE**2

    north = compute_healpix_indices(SkyCoord(ra=10.0 * u.deg, dec=89.9 * u.deg))
    south = compute_healpix_indices(SkyCoord(ra=10.0 * u.deg, dec=-89.9 * u.deg))

    assert north < npix / 100
    assert south > npix * 99 / 100
