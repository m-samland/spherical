"""Waffle-spot search-box placement and the self-correcting seed (#144).

The waffle fit only ever sees the seed through the four integer search-box
centres, so these are the units that decide whether a frame is fit correctly.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy")

from spherical.pipeline.steps.find_star import (
    WAFFLE_SPOT_FREQUENCY,
    frames_to_plot,
    refine_center_seed,
    seed_boxes_would_move,
    waffle_spot_box_centers,
)

PIXEL = 12.25          # IRDIS mas/px
WAVE = 2110.0          # nm, K1
LOD = WAVE * 1e-9 / 7.99 * 180 / np.pi * 3600 * 1000 / PIXEL


def _true_spot(center_xy, s, orient=0.0, freq=WAFFLE_SPOT_FREQUENCY):
    """Where the spot physically is, for a given waffle frequency."""
    r = freq * LOD
    return (center_xy[0] + r * np.cos(orient + np.pi / 2 * s),
            center_xy[1] + r * np.sin(orient + np.pi / 2 * s))


class TestBoxPlacement:
    def test_boxes_land_on_spots_at_the_physical_radius(self):
        """Spots sit at 10*sqrt(2)*lambda/D. The boxes must too.

        An integer seed isolates the frequency constant from the ``int()``
        truncation, leaving at most 1 px of placement error per axis. The old
        ``0.97`` fudge put the boxes 1.9 px inward at K1, which this catches.
        """
        center = (480.0, 534.0)
        boxes = waffle_spot_box_centers(center, LOD, orient=0.0)
        assert boxes.shape == (4, 2)
        for s, (bx, by) in enumerate(boxes):
            tx, ty = _true_spot(center, s)
            assert abs(bx - tx) <= 1.0, f"spot {s}: box x {bx} vs spot {tx:.2f}"
            assert abs(by - ty) <= 1.0, f"spot {s}: box y {by} vs spot {ty:.2f}"

    def test_boxes_are_integers(self):
        """The fit depends on the seed only through these integers."""
        boxes = waffle_spot_box_centers((480.7, 534.2), LOD, orient=0.0)
        assert boxes.dtype.kind == "i"

    def test_sub_pixel_seed_change_leaves_boxes_untouched(self):
        """Truncation means a small seed change is a genuine no-op."""
        a = waffle_spot_box_centers((480.10, 534.10), LOD, orient=0.0)
        b = waffle_spot_box_centers((480.20, 534.15), LOD, orient=0.0)
        assert np.array_equal(a, b)

    def test_whole_pixel_seed_change_moves_every_box(self):
        a = waffle_spot_box_centers((480.0, 524.0), LOD, orient=0.0)
        b = waffle_spot_box_centers((480.0, 534.0), LOD, orient=0.0)
        assert np.array_equal(b[:, 1] - a[:, 1], np.full(4, 10))

    def test_x_orientation_rotates_the_pattern(self):
        plus = waffle_spot_box_centers((480.0, 534.0), LOD, orient=0.0)
        cross = waffle_spot_box_centers((480.0, 534.0), LOD, orient=np.pi / 4)
        assert not np.array_equal(plus, cross)
        # same radius from the seed, just rotated
        rp = np.hypot(*(plus - np.array([480, 534])).T)
        rc = np.hypot(*(cross - np.array([480, 534])).T)
        assert np.allclose(rp, rc, atol=1.5)


class TestSeedGuard:
    """The guard must be exact, not a tuned tolerance.

    Boxes are placed at ``int(int(seed) + radius*cos)``. For positive pixel
    coordinates an integer shift of ``int(seed)`` shifts every box by exactly
    that integer, so the boxes move iff the truncated seed moves.
    """

    def test_sub_pixel_refinement_is_a_no_op(self):
        old = np.array([[480.0, 524.7], [482.5, 511.4]])
        new = np.array([[480.4, 524.1], [482.9, 511.9]])
        assert seed_boxes_would_move(old, new) is False

    def test_whole_pixel_refinement_moves_boxes(self):
        old = np.array([[480.0, 524.7], [482.5, 511.4]])
        new = np.array([[480.7, 534.2], [483.2, 521.2]])
        assert seed_boxes_would_move(old, new) is True

    def test_a_single_channel_moving_is_enough(self):
        old = np.array([[480.0, 524.7], [482.5, 511.4]])
        new = np.array([[480.1, 524.8], [482.6, 515.0]])
        assert seed_boxes_would_move(old, new) is True

    def test_agrees_with_actually_placing_the_boxes(self):
        """Cross-check the shortcut against the real placement."""
        rng = np.random.default_rng(0)
        for _ in range(200):
            old = np.array([[480.0, 524.7]]) + rng.normal(0, 3, (1, 2))
            new = old + rng.normal(0, 3, (1, 2))
            moved = not np.array_equal(
                waffle_spot_box_centers(old[0], LOD, 0.0),
                waffle_spot_box_centers(new[0], LOD, 0.0),
            )
            assert seed_boxes_would_move(old, new) is moved

    def test_nan_refinement_never_triggers_a_refit(self):
        old = np.array([[480.0, 524.7], [482.5, 511.4]])
        new = np.array([[np.nan, np.nan], [482.6, 511.5]])
        assert seed_boxes_would_move(old, new) is False


class TestRefineCenterSeed:
    def test_uses_the_median_over_frames(self):
        seed = np.array([[480.0, 524.7]])
        centers = np.array([[[490.0, 530.0], [492.0, 534.0], [491.0, 532.0]]])
        refined = refine_center_seed(centers, seed)
        assert refined == pytest.approx(np.array([[491.0, 532.0]]))

    def test_ignores_nan_frames(self):
        seed = np.array([[480.0, 524.7]])
        centers = np.array([[[490.0, 530.0], [np.nan, np.nan], [492.0, 534.0]]])
        refined = refine_center_seed(centers, seed)
        assert refined == pytest.approx(np.array([[491.0, 532.0]]))

    def test_falls_back_to_the_seed_when_every_frame_failed(self):
        """A refit seeded from garbage is worse than no refit."""
        seed = np.array([[480.0, 524.7], [482.5, 511.4]])
        centers = np.full((2, 3, 2), np.nan)
        refined = refine_center_seed(centers, seed)
        assert refined == pytest.approx(seed)

    def test_falls_back_per_channel(self):
        seed = np.array([[480.0, 524.7], [482.5, 511.4]])
        centers = np.full((2, 3, 2), np.nan)
        centers[1] = np.array([[490.0, 530.0], [490.0, 530.0], [490.0, 530.0]])
        refined = refine_center_seed(centers, seed)
        assert refined[0] == pytest.approx(seed[0])
        assert refined[1] == pytest.approx(np.array([490.0, 530.0]))


class TestFramesToPlot:
    def test_subsamples_evenly_and_keeps_the_ends(self):
        sel = frames_to_plot(100, 5)
        assert len(sel) == 5
        assert sel[0] == 0 and sel[-1] == 99
        assert sorted(set(sel)) == list(sel)

    def test_plots_every_frame_when_there_are_fewer_than_asked(self):
        assert list(frames_to_plot(3, 10)) == [0, 1, 2]

    def test_none_means_all(self):
        assert list(frames_to_plot(4, None)) == [0, 1, 2, 3]

    def test_zero_disables_plotting(self):
        assert list(frames_to_plot(100, 0)) == []

    def test_single_frame(self):
        assert list(frames_to_plot(1, 10)) == [0]


# --------------------------------------------------------------------------
# End-to-end: a stale nominal seed must not survive into the measured centre.
# --------------------------------------------------------------------------

CROP_X0, CROP_Y0 = 380, 425      # puts the K-band nominal near the middle of a
FRAME = 220                      # small synthetic frame, so the test stays fast


def _waffle_frame(center_xy, lod, size=FRAME, seed=0):
    """Four unit-peak waffle spots on an exponential halo."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    img = np.zeros((size, size))
    sigma = lod / 2.355
    for s in range(4):
        cx, cy = _true_spot(center_xy, s)
        img += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    r = np.hypot(xx - center_xy[0], yy - center_xy[1])
    # normalised so the halo at the spot radius is 0.3x the spot peak
    r_spot = WAFFLE_SPOT_FREQUENCY * lod
    img += 0.3 * np.exp(-r / 40.0) / np.exp(-r_spot / 40.0)
    return img + rng.normal(0, 0.01, img.shape)


def _write_observation(tmp_path, true_centers, n_frames=3):
    from astropy.io import fits

    waves = np.array([2110.0, 2251.0])
    lods = waves * 1e-9 / 7.99 * 180 / np.pi * 3600 * 1000 / PIXEL
    cube = np.empty((2, n_frames, FRAME, FRAME), dtype=np.float32)
    for ch in range(2):
        for i in range(n_frames):
            cube[ch, i] = _waffle_frame(true_centers[ch], lods[ch], seed=10 * ch + i)

    header = fits.Header()
    header["HIERARCH SPHERICAL CROP APPLIED"] = True
    header["HIERARCH SPHERICAL CROP SIZE"] = FRAME
    for ch in range(2):
        header[f"HIERARCH SPHERICAL CROP X0 CH{ch}"] = CROP_X0
        header[f"HIERARCH SPHERICAL CROP Y0 CH{ch}"] = CROP_Y0
    fits.writeto(tmp_path / "center_cube.fits", cube, header, overwrite=True)
    fits.writeto(tmp_path / "wavelengths.fits", waves, overwrite=True)

    import pandas as pd
    pd.DataFrame({
        "OCS WAFFLE ORIENT": ["+"] * n_frames,
        "INS COMB IFLT": ["DB_K12"] * n_frames,
    }).to_csv(tmp_path / "frames_info_center.csv", index=False)


def _observation():
    from unittest.mock import MagicMock

    obs = MagicMock()
    obs.observation = {"INSTRUMENT": ["IRDIS"], "FILTER": ["DB_K12"], "WAFFLE_MODE": [True]}
    obs.frames = {"CORO": None}
    return obs


def _cropped_nominal():
    from spherical.pipeline.steps.irdis_preprocess import NOMINAL_STAR_POSITIONS_K_BAND

    nominal = np.array(NOMINAL_STAR_POSITIONS_K_BAND, dtype=float)
    nominal[:, 0] -= CROP_X0
    nominal[:, 1] -= CROP_Y0
    return nominal


class TestTwoPassSeedEndToEnd:
    def test_recovers_a_star_that_moved_away_from_the_nominal(self, tmp_path, caplog):
        """The Beta Pic failure mode: the coronagraph is ~9 px off the nominal.

        A single pass walks the spots to the edge of their box and biases the
        centre. The re-seeded pass has to land on the truth.
        """
        import logging

        from astropy.io import fits

        from spherical.pipeline.steps.find_star import fit_centers_in_parallel

        truth = _cropped_nominal() + np.array([0.3, 9.0])
        _write_observation(tmp_path, truth)

        with caplog.at_level(logging.INFO):
            fit_centers_in_parallel(str(tmp_path), _observation(), ncpu=1, n_center_plots=0)

        centers = fits.getdata(str(tmp_path / "image_centers.fits"))
        assert centers.shape == (2, 3, 2)
        for ch in range(2):
            measured = np.nanmedian(centers[ch], axis=0)
            assert measured == pytest.approx(truth[ch], abs=0.25), (
                f"ch{ch}: measured {measured} vs truth {truth[ch]}"
            )
        assert "refitting" in caplog.text

    def test_no_refit_when_the_nominal_is_already_right(self, tmp_path, caplog):
        import logging

        from spherical.pipeline.steps.find_star import fit_centers_in_parallel

        # Offsets chosen so the measured centre stays in the same integer bin
        # as the nominal, which is exactly what the guard keys on.
        truth = _cropped_nominal() + np.array([0.25, 0.0])
        _write_observation(tmp_path, truth)

        with caplog.at_level(logging.INFO):
            fit_centers_in_parallel(str(tmp_path), _observation(), ncpu=1, n_center_plots=0)

        assert "no refit needed" in caplog.text

    def test_plot_count_is_honoured(self, tmp_path):
        from spherical.pipeline.steps.find_star import fit_centers_in_parallel

        _write_observation(tmp_path, _cropped_nominal() + np.array([0.25, 0.0]), n_frames=5)
        fit_centers_in_parallel(str(tmp_path), _observation(), ncpu=1, n_center_plots=2)

        pdfs = sorted((tmp_path / "center_plots").glob("CENTER_img_*.pdf"))
        assert len(pdfs) == 2
        assert pdfs[0].name == "CENTER_img_000.pdf"
        assert pdfs[-1].name == "CENTER_img_004.pdf"
