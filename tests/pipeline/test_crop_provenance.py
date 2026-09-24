"""Crop provenance cards: who writes them, who can read them back.

With ``crop=True`` a product is delivered in crop coordinates. Nothing in the
data says so, so these cards are the only record. A consumer that assumes
detector coordinates is wrong by the crop origin, which is hundreds of pixels,
and no shape check catches it.
"""
from __future__ import annotations

import numpy as np
import pytest
from astropy.io import fits

from spherical.pipeline.crop_provenance import (
    CROP_APPLIED,
    CROP_KEYWORDS,
    copy_crop_cards,
    crop_cards_from_cube,
    read_crop_origins,
    stamp_crop_cards,
)

ORIGINS = np.array([[352, 397], [354, 383]])


class TestStampAndRead:
    def test_a_stamped_header_reads_back_as_written(self):
        h = stamp_crop_cards(fits.Header(), ORIGINS, 257)
        x0, y0 = read_crop_origins(h)
        assert list(x0) == [352, 354]
        assert list(y0) == [397, 383]
        assert h["HIERARCH SPHERICAL CROP SIZE"] == 257

    def test_an_uncropped_product_is_stamped_as_uncropped(self):
        """False and absent are different answers, so FLUX still gets a card."""
        h = stamp_crop_cards(fits.Header(), None, 257)
        assert h[CROP_APPLIED] is False
        assert read_crop_origins(h) is None
        assert "HIERARCH SPHERICAL CROP SIZE" not in h

    def test_a_header_with_no_cards_reads_as_uncropped(self):
        assert read_crop_origins(fits.Header()) is None

    def test_none_is_not_an_origin_of_zero(self):
        """A caller subtracting the origin must subtract nothing, not zero."""
        assert read_crop_origins(stamp_crop_cards(fits.Header(), None, 257)) is None

    def test_channels_do_not_get_swapped(self):
        """The two IRDIS channels have different origins; an x/y or channel
        swap here offsets every measured centre by tens of pixels."""
        h = stamp_crop_cards(fits.Header(), np.array([[10, 20], [30, 40]]), 101)
        x0, y0 = read_crop_origins(h)
        assert (list(x0), list(y0)) == ([10, 30], [20, 40])


class TestCopy:
    def test_cards_survive_the_copy(self):
        src = stamp_crop_cards(fits.Header(), ORIGINS, 257)
        dst = copy_crop_cards(fits.Header(), src)
        assert read_crop_origins(dst) is not None
        assert all(k in dst for k in CROP_KEYWORDS)

    def test_absent_cards_are_not_invented(self):
        """IFS has no crop cards. Copying must not fabricate CROP APPLIED=False,
        which would assert something the source never said."""
        dst = copy_crop_cards(fits.Header(), fits.Header())
        assert len(dst) == 0

    def test_only_crop_cards_are_copied(self):
        src = stamp_crop_cards(fits.Header(), ORIGINS, 257)
        src["HIERARCH SPHERICAL FILTER"] = "DB_K12"
        dst = copy_crop_cards(fits.Header(), src)
        assert "HIERARCH SPHERICAL FILTER" not in dst


class TestCardsFromCube:
    def _cube(self, path, origins):
        fits.writeto(
            path,
            np.zeros((2, 1, 4, 4), dtype=np.float32),
            header=stamp_crop_cards(fits.Header(), origins, 257),
            overwrite=True,
        )

    def test_reads_the_center_cube(self, tmp_path):
        self._cube(tmp_path / "center_cube.fits", ORIGINS)
        assert read_crop_origins(crop_cards_from_cube(tmp_path)) is not None

    def test_falls_back_to_the_coro_cube(self, tmp_path):
        """A waffle sequence is centred on CENTER frames, but a reduction
        without them still has to answer."""
        self._cube(tmp_path / "coro_cube.fits", ORIGINS)
        x0, _ = read_crop_origins(crop_cards_from_cube(tmp_path))
        assert list(x0) == [352, 354]

    def test_no_cube_yields_no_cards_rather_than_an_error(self, tmp_path):
        """Older reductions and IFS both land here; neither is a failure."""
        assert len(crop_cards_from_cube(tmp_path)) == 0

    def test_an_uncropped_cube_yields_an_explicit_false(self, tmp_path):
        self._cube(tmp_path / "center_cube.fits", None)
        cards = crop_cards_from_cube(tmp_path)
        assert cards[CROP_APPLIED] is False


class TestProductsCarryTheCards:
    """The gap this module closes: products that are cropped, or that hold
    coordinates measured on cropped data, but said nothing about it."""

    @pytest.fixture
    def converted(self, tmp_path):
        fits.writeto(
            tmp_path / "center_cube.fits",
            np.zeros((2, 2, 8, 8), dtype=np.float32),
            header=stamp_crop_cards(fits.Header(), ORIGINS, 257),
            overwrite=True,
        )
        return tmp_path

    def test_centre_files_written_with_copied_cards_are_readable(self, converted):
        """What find_star and process_centers now do, at the file level."""
        centers = np.zeros((2, 2, 2), dtype=np.float32)
        fits.writeto(
            converted / "image_centers.fits",
            centers,
            header=crop_cards_from_cube(converted),
            overwrite=True,
        )
        h = fits.getheader(converted / "image_centers.fits")
        x0, y0 = read_crop_origins(h)
        assert (list(x0), list(y0)) == ([352, 354], [397, 383])

    def test_a_cropped_centre_maps_back_onto_the_detector(self, converted):
        """The whole point: recovering the detector coordinate needs the cards."""
        x0, y0 = read_crop_origins(fits.getheader(converted / "center_cube.fits"))
        cropped = np.array([[127.4, 127.2], [128.0, 127.9]])
        detector = np.stack([cropped[:, 0] + x0, cropped[:, 1] + y0], axis=-1)
        assert np.allclose(detector[0], [479.4, 524.2])
        assert np.allclose(detector[1], [482.0, 510.9])
