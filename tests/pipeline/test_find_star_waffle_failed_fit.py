"""star_centers_from_waffle_img_cube: a spot fit that leaves its search box is a failed fit (#286).

The ``mask_deviating`` refit has no bounds, so where the pixels do not constrain a spot the fit can run far
off, as in the off-frame fits of #278, and its position and amplitude used to be stored as a measurement.
Where such a fit ends depends on noise and on platform numerics, so the fitter of one spot is made to end
there instead of relying on a synthetic image to send it there.
"""
from __future__ import annotations

import numpy as np
import pytest

import spherical.pipeline.steps.find_star as find_star
from spherical.pipeline.steps.find_star import (
    WAFFLE_SPOT_FREQUENCY as FREQ,
)
from spherical.pipeline.steps.find_star import (
    star_centers_from_waffle_img_cube,
)

PIXEL = 12.25          # IRDIS mas/px
WAVE = 2110.0          # nm
LOD = WAVE * 1e-9 / 7.99 * 180 / np.pi * 3600 * 1000 / PIXEL
SIGMA = LOD / 2.355
N = 240
TRUE_CENTER = (120.3, 119.7)
GUESS = np.array([[120, 120]])


def _frame(seed=0):
    """Four unit-peak waffle spots with noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:N, :N]
    img = rng.normal(0, 0.02, (N, N))
    for s in range(4):
        cx = TRUE_CENTER[0] + FREQ * LOD * np.cos(np.pi / 2 * s)
        cy = TRUE_CENTER[1] + FREQ * LOD * np.sin(np.pi / 2 * s)
        img += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * SIGMA ** 2))
    return img[None, :, :]


def _fit(img, mask_deviating=True):
    spot_centers, _, img_centers, amplitudes = star_centers_from_waffle_img_cube(
        img, wave=np.array([WAVE]), waffle_orientation='+', center_guess=GUESS, pixel=PIXEL,
        orientation_offset=0., fit_background=True, fit_symmetric_gaussian=False, mask_deviating=mask_deviating,
        deviation_threshold=0.8, high_pass=False, save_plot=False, save_path=None)
    return spot_centers[0], img_centers[0], amplitudes[0]


def _fitter_ending_spot_0_at(monkeypatch, x_mean, on_call):
    """The real fitter, except that the ``on_call``-th fit of spot 0 ends at ``x_mean`` in box coordinates."""
    real = find_star.fitting.LevMarLSQFitter
    instances = []

    class Fitter(real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.spot = len(instances)
            self.calls = 0
            instances.append(self)

        def __call__(self, *args, **kwargs):
            model = super().__call__(*args, **kwargs)
            self.calls += 1
            if self.spot == 0 and self.calls == on_call:
                gaussian = model[0] if model.n_submodels > 1 else model
                gaussian.x_mean = x_mean
            return model

    monkeypatch.setattr(find_star.fitting, "LevMarLSQFitter", Fitter)


@pytest.mark.parametrize("x_mean", [-102.0, -2.5, 18.5, 74.0])
def test_refit_leaving_its_box_is_stored_as_nan(monkeypatch, x_mean):
    reference_centers, _, reference_amplitudes = _fit(_frame())
    _fitter_ending_spot_0_at(monkeypatch, x_mean, on_call=2)
    spot_centers, img_center, amplitudes = _fit(_frame())

    assert np.all(np.isnan(spot_centers[0]))
    assert np.isnan(amplitudes[0])
    np.testing.assert_allclose(spot_centers[1:], reference_centers[1:], atol=1e-6)
    np.testing.assert_allclose(amplitudes[1:], reference_amplitudes[1:], rtol=1e-6)
    assert np.all(np.isnan(img_center))


def test_first_fit_stopped_on_its_bound_is_stored_as_nan(monkeypatch):
    _fitter_ending_spot_0_at(monkeypatch, -2.0, on_call=1)
    spot_centers, _, amplitudes = _fit(_frame(), mask_deviating=False)
    assert np.all(np.isnan(spot_centers[0])) and np.isnan(amplitudes[0])
    assert np.all(np.isfinite(spot_centers[1:]))


@pytest.mark.parametrize("x_mean", [-1.5, 17.5])
def test_fit_within_the_margin_is_kept(monkeypatch, x_mean):
    _fitter_ending_spot_0_at(monkeypatch, x_mean, on_call=2)
    spot_centers, _, amplitudes = _fit(_frame())
    assert np.all(np.isfinite(spot_centers[0])) and np.isfinite(amplitudes[0])


def test_good_spots_are_kept():
    spot_centers, img_center, amplitudes = _fit(_frame())
    assert np.all(np.isfinite(spot_centers))
    assert np.all(np.isfinite(amplitudes))
    assert np.hypot(*(img_center - np.array(TRUE_CENTER))) < 0.1
