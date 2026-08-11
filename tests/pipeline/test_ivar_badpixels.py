"""Tests for the local-baseline inverse-variance bad-pixel detector."""

import numpy as np
import pytest

# scipy arrives with the pipeline extra only; skip cleanly in the CI `test` env
# instead of erroring at collection.
pytest.importorskip("scipy")

from spherical.pipeline.ivar_badpixels import bad_pixel_mask_from_ivar


def _flat_field(shape=(40, 40), value=1e-3):
    return np.full(shape, value)


class TestBadPixelMaskFromIvar:
    def test_flat_field_flags_nothing(self):
        assert not bad_pixel_mask_from_ivar(_flat_field()).any()

    def test_isolated_deficit_is_flagged(self):
        ivar = _flat_field()
        ivar[20, 25] = 1e-5  # 1% of the local baseline
        mask = bad_pixel_mask_from_ivar(ivar)
        assert mask[20, 25]
        assert mask.sum() == 1

    def test_marginal_deficit_below_threshold_is_kept(self):
        """The point of thresholding: a 50% dip is noise, not a dead spaxel."""
        ivar = _flat_field()
        ivar[20, 25] = 5e-4
        assert not bad_pixel_mask_from_ivar(ivar, ratio_threshold=0.2).any()
        assert bad_pixel_mask_from_ivar(ivar, ratio_threshold=0.6)[20, 25]

    def test_zero_threshold_reduces_to_exact_zero_mask(self):
        """The IFS default since charis issue 013: ``ratio_threshold=0`` disables
        the soft test, so only exact zeros / non-finite values are flagged and a
        real deficit (a moiré trough) is kept."""
        ivar = _flat_field()
        ivar[20, 25] = 1e-5  # 1% of baseline — a genuine soft deficit
        ivar[10, 10] = 0.0   # exact zero — a masked spaxel
        mask = bad_pixel_mask_from_ivar(ivar, ratio_threshold=0.0)
        assert mask[10, 10]
        assert not mask[20, 25]
        assert mask.sum() == 1

    def test_in_field_footprint_catches_exact_zero_clusters(self):
        """A cluster of exact zeros larger than ``filter_size`` collapses the local
        median baseline to ~0, so the default gate misses its interior. An explicit
        ``in_field`` footprint flags every in-field exact zero regardless."""
        ivar = _flat_field((40, 40))
        ivar[15:24, 15:24] = 0.0  # 9x9 in-field zero cluster, wider than filter_size

        default = bad_pixel_mask_from_ivar(ivar, ratio_threshold=0.0)
        assert not default[19, 19]  # cluster interior leaks past the baseline gate

        in_field = np.ones((40, 40), dtype=bool)
        hardened = bad_pixel_mask_from_ivar(ivar, ratio_threshold=0.0, in_field=in_field)
        assert hardened[15:24, 15:24].all()

    def test_in_field_never_flags_outside_the_footprint(self):
        """Out-of-field exact zeros (the unilluminated border) stay unflagged even
        though they are ``ivar <= 0`` — the reduction footprint handles them."""
        ivar = _flat_field((40, 40))
        ivar[:5, :] = 0.0  # a border strip of exact zeros
        in_field = np.ones((40, 40), dtype=bool)
        in_field[:5, :] = False  # ...declared out-of-field

        mask = bad_pixel_mask_from_ivar(ivar, ratio_threshold=0.0, in_field=in_field)
        assert not mask[:5, :].any()

    def test_in_field_broadcasts_over_leading_axes(self):
        """A 2-D footprint applies to every wavelength/frame plane."""
        ivar = np.stack([_flat_field((30, 30)) for _ in range(4)])
        ivar[:, 10, 10] = 0.0
        in_field = np.ones((30, 30), dtype=bool)
        mask = bad_pixel_mask_from_ivar(ivar, ratio_threshold=0.0, in_field=in_field)
        assert mask.shape == ivar.shape
        assert mask[:, 10, 10].all()

    def test_zero_and_nonfinite_always_flagged(self):
        ivar = _flat_field()
        ivar[10, 10] = 0.0
        ivar[11, 11] = np.nan
        ivar[12, 12] = -1.0
        mask = bad_pixel_mask_from_ivar(ivar)
        assert mask[10, 10] and mask[11, 11] and mask[12, 12]

    def test_smooth_radial_gradient_is_not_flagged(self):
        """Regression against a global threshold: ivar drops by two orders of
        magnitude towards the star over the halo scale, which a global cut would
        flag wholesale while a local baseline tracks it."""
        yy, xx = np.indices((80, 80))
        r = np.hypot(yy - 40, xx - 40)
        ivar = 1e-3 * (1.0 - 0.99 * np.exp(-(r ** 2) / (2 * 20.0 ** 2)))
        assert not bad_pixel_mask_from_ivar(ivar).any()
        # the same field with a global cut would flag the core
        assert (ivar < 0.2 * np.median(ivar)).sum() > 100

    def test_deficit_on_a_gradient_is_still_found(self):
        yy, xx = np.indices((80, 80))
        r = np.hypot(yy - 40, xx - 40)
        ivar = 1e-3 * (1.0 - 0.99 * np.exp(-(r ** 2) / (2 * 20.0 ** 2)))
        ivar[45, 52] *= 0.01
        mask = bad_pixel_mask_from_ivar(ivar)
        assert mask[45, 52]

    def test_unilluminated_border_is_never_flagged(self):
        """The resampled IFS cube is square but the lenslet array is not."""
        ivar = np.full((60, 60), 1e-18)
        ivar[15:45, 15:45] = 1e-3
        mask = bad_pixel_mask_from_ivar(ivar)
        assert not mask[:10, :10].any()

    def test_per_channel_masks_are_independent(self):
        """A dead detector pixel kills one (lenslet, wavelength) pair, so the
        mask must stay 3-D rather than collapse over wavelength."""
        ivar = np.stack([_flat_field() for _ in range(4)])
        ivar[0, 20, 25] = 1e-6
        ivar[2, 30, 31] = 1e-6
        mask = bad_pixel_mask_from_ivar(ivar)
        assert mask.shape == ivar.shape
        assert mask[0, 20, 25] and not mask[1, 20, 25]
        assert mask[2, 30, 31] and not mask[0, 30, 31]
        assert mask.sum() == 2

    def test_four_dimensional_input_keeps_shape(self):
        ivar = np.stack([np.stack([_flat_field() for _ in range(3)]) for _ in range(2)])
        ivar[1, 2, 20, 25] = 1e-6
        mask = bad_pixel_mask_from_ivar(ivar)
        assert mask.shape == ivar.shape
        assert mask[1, 2, 20, 25]
        assert mask.sum() == 1

    def test_all_dead_plane_does_not_crash(self):
        mask = bad_pixel_mask_from_ivar(np.zeros((20, 20)))
        assert mask.all()

    @pytest.mark.parametrize("size", [2, 4, 1])
    def test_rejects_even_or_tiny_filter_size(self, size):
        with pytest.raises(ValueError, match="odd integer"):
            bad_pixel_mask_from_ivar(_flat_field(), filter_size=size)

    def test_rejects_one_dimensional_input(self):
        with pytest.raises(ValueError, match="at least 2 dimensions"):
            bad_pixel_mask_from_ivar(np.ones(10))
