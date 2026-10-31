"""Per-channel PSF guess for IRDIS flux cubes (#170, Defect A).

IRDIS FLUX cubes are never cropped, so the real cross-channel offset between the two
detector halves survives into them. A single guess taken on the wavelength median
cannot be right for both channels.
"""
import logging

import numpy as np

from spherical.pipeline.steps.find_star import (
    guess_position_psf,
    guess_positions_per_channel,
    star_centers_from_PSF_img_cube,
)

LOGGER = logging.getLogger("test")

# pi Men measured star positions (x, y) in the two H23 channels
PI_MEN_H2 = (518, 490)
PI_MEN_H3 = (520, 481)


def _two_channel_cube(pos_ch0, pos_ch1, size=1024, peak=1000.0):
    """IRDIS-like (2, size, size) cube with the star at a different position
    in each channel."""
    yy, xx = np.mgrid[0:size, 0:size]
    planes = []
    for (x, y) in (pos_ch0, pos_ch1):
        planes.append(peak * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.0 ** 2)))
    return np.stack(planes)


def test_per_channel_guess_finds_each_channel_separately():
    cube = _two_channel_cube(PI_MEN_H2, PI_MEN_H3)
    guesses = guess_positions_per_channel(cube)
    assert len(guesses) == 2
    assert abs(guesses[0][1] - 518) <= 2 and abs(guesses[0][0] - 490) <= 2
    assert abs(guesses[1][1] - 520) <= 2 and abs(guesses[1][0] - 481) <= 2


def test_wavelength_median_guess_is_wrong_for_at_least_one_channel():
    """Regression witness for the old single-guess path."""
    cube = _two_channel_cube(PI_MEN_H2, PI_MEN_H3)
    cy, cx = guess_position_psf(cube)
    per_channel = guess_positions_per_channel(cube)
    matches = sum(abs(cy - g[0]) <= 2 and abs(cx - g[1]) <= 2 for g in per_channel)
    assert matches <= 1


def test_coincident_channels_give_the_same_guess():
    cube = _two_channel_cube((300, 400), (300, 400))
    guesses = guess_positions_per_channel(cube)
    assert guesses[0] == guesses[1]


def test_bad_pixel_mask_applies_to_its_own_channel():
    """A hot pixel flagged only in channel 1 must not steer channel 0's guess away."""
    cube = _two_channel_cube(PI_MEN_H2, PI_MEN_H3)
    cube[1, 200, 200] = 1e6
    mask = np.zeros(cube.shape, dtype=bool)
    mask[1, 200, 200] = True
    guesses = guess_positions_per_channel(cube, bad_pixel_mask=mask)
    assert abs(guesses[1][1] - 520) <= 2 and abs(guesses[1][0] - 481) <= 2
    assert abs(guesses[0][1] - 518) <= 2 and abs(guesses[0][0] - 490) <= 2


def test_star_centers_accepts_one_guess_per_channel():
    """With a guess per channel each channel's stamp is cut around its own star.

    The offset is set beyond the 30 px stamp half-size, so a single shared guess
    cannot place both stars in their stamps.
    """
    cube = _two_channel_cube((518, 490), (520, 440))
    centers, _ = star_centers_from_PSF_img_cube(
        cube=cube.copy(), wave=np.array([1593.0, 1667.0]), pixel=12.25, logger=LOGGER,
        guess_center_yx=[(490, 518), (440, 520)],
        fit_background=False, mask_deviating=False,
    )
    assert np.allclose(centers[0], (518, 490), atol=0.5)
    assert np.allclose(centers[1], (520, 440), atol=0.5)


def test_star_centers_still_accepts_a_single_guess():
    cube = _two_channel_cube((518, 490), (518, 490))
    centers, _ = star_centers_from_PSF_img_cube(
        cube=cube.copy(), wave=np.array([1593.0, 1667.0]), pixel=12.25, logger=LOGGER,
        guess_center_yx=(490, 518),
        fit_background=False, mask_deviating=False,
    )
    assert np.allclose(centers, [(518, 490), (518, 490)], atol=0.5)
