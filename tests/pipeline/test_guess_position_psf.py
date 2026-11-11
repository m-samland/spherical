"""Per-channel PSF guess for IRDIS flux cubes (#170, Defect A).

IRDIS FLUX cubes are never cropped, so the real cross-channel offset between the two
detector halves survives into them. A single guess taken on the wavelength median
cannot be right for both channels.
"""
import logging

import numpy as np

from spherical.pipeline.steps.find_star import (
    guess_position_psf,
    guess_positions_joint,
    guess_positions_per_channel,
    star_centers_from_PSF_img_cube,
)
from spherical.pipeline.steps.flux_psf_calibration import irdis_guess_smoothing_px, read_channel_offset

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


# --- joint seeding from the cross-channel offset (#296) ---------------------

K12_OFFSET_XY = (2.4, -13.8)
K1_STAR = (505, 492)
K2_STAR = (507, 478)


def test_joint_guess_places_the_hot_pixel_channel_from_the_other_channel():
    cube = _two_channel_cube(K1_STAR, K2_STAR)
    cube[1, 352, 246] = 5000.0
    assert guess_positions_per_channel(cube)[1] == (352, 246)
    guesses, independent = guess_positions_joint(cube, K12_OFFSET_XY)
    assert independent[1] == (352, 246)
    assert guesses[0] == (492, 505)
    assert guesses[1] == (478, 507)


def test_joint_guess_survives_a_hot_pixel_far_brighter_than_the_star_after_smoothing():
    cube = _two_channel_cube(K1_STAR, K2_STAR)
    cube[0] *= 0.3
    cube[0, 632, 168] = 1e5
    guesses, independent = guess_positions_joint(cube, K12_OFFSET_XY, smooth_sigma_px=2.0)
    assert independent[0] == (632, 168)
    assert guesses == [(492, 505), (478, 507)]


def test_joint_guess_matches_the_independent_one_on_a_clean_frame():
    cube = _two_channel_cube(K1_STAR, K2_STAR)
    guesses, independent = guess_positions_joint(cube, K12_OFFSET_XY, smooth_sigma_px=2.0)
    assert guesses == independent


def test_smoothing_lets_the_star_win_over_a_brighter_single_pixel():
    cube = _two_channel_cube(K1_STAR, K1_STAR)[:1]
    cube[0, 300, 300] = 3000.0
    assert guess_position_psf(cube) == (300, 300)
    assert guess_position_psf(cube, smooth_sigma_px=2.0) == (492, 505)


def test_smoothing_ignores_excluded_pixels_next_to_the_star():
    cube = _two_channel_cube(K1_STAR, K1_STAR)[:1]
    mask = np.zeros(cube.shape, dtype=bool)
    mask[0, 492, 506:510] = True
    cube[0, 492, 506:510] = 1e6
    assert guess_position_psf(cube, bad_pixel_mask=mask[0], smooth_sigma_px=2.0) == (492, 505)


def test_guess_smoothing_is_a_lambda_over_d_fwhm():
    sigma = irdis_guess_smoothing_px(np.array([2110.0, 2251.0]), 12.25)
    assert 1.8 < sigma < 2.1  # lambda/D = 56 mas = 4.6 px at 2180 nm


def test_missing_channel_offset_falls_back_with_a_warning(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger="test"):
        assert read_channel_offset(tmp_path, LOGGER) is None
    assert any(r.status == "flux_guess_independent" for r in caplog.records)


def test_measured_channel_offset_is_read(tmp_path):
    from astropy.io import fits

    fits.writeto(tmp_path / "cross_channel_offset.fits", np.array(K12_OFFSET_XY, dtype=np.float32))
    assert np.allclose(read_channel_offset(tmp_path, LOGGER), K12_OFFSET_XY)


def test_joint_guess_follows_an_offset_of_the_opposite_sign():
    cube = _two_channel_cube(K2_STAR, K1_STAR)
    cube[1, 352, 246] = 5000.0
    guesses, _ = guess_positions_joint(cube, (-K12_OFFSET_XY[0], -K12_OFFSET_XY[1]))
    assert guesses == [(478, 507), (492, 505)]


def test_non_finite_channel_offset_falls_back_with_a_warning(tmp_path, caplog):
    from astropy.io import fits

    fits.writeto(tmp_path / "cross_channel_offset.fits", np.array([np.nan, -13.8], dtype=np.float32))
    with caplog.at_level(logging.WARNING, logger="test"):
        assert read_channel_offset(tmp_path, LOGGER) is None
    assert any(r.status == "flux_guess_independent" for r in caplog.records)
