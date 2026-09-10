"""Tests for the center-evolution plot, in particular the two frame grids.

``image_centers.fits`` always holds the raw CENTER-frame measurements. What
the two fitted files hold depends on the branch ``process_centers`` took:

* waffle sequence -> CENTER-frame centers, same length as the raw array;
* coronagraphic sequence -> DMS-propagated CORO-frame centers, a different
  length entirely.

The plot used to derive its loop bound from the raw array and index the
fitted ones with it, which raised IndexError whenever there were more CENTER
than CORO frames (issue #129) and silently dropped frames otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits

from spherical.pipeline.steps.plot_center_evolution import (
    _build_center_series,
    _residuals_from_median,
    _time_bases,
    _wavelength_legend_entries,
    run_image_center_evolution_plot,
)


def _positions(n_frames, value, n_wave=2):
    """A (n_wave, n_frames, 2) center array with every entry set to ``value``."""
    return np.full((n_wave, n_frames, 2), float(value), dtype=np.float32)


def _times(n, start="2023-06-13T02:00:00"):
    return pd.date_range(start, periods=n, freq="60s").strftime("%Y-%m-%dT%H:%M:%S").tolist()


def _write_inputs(tmp_path, raw, fitted, robust, n_center, n_coro=None, wavelengths=None,
                  outliers=None):
    fits.writeto(tmp_path / "image_centers.fits", raw, overwrite=True)
    fits.writeto(tmp_path / "image_centers_fitted.fits", fitted, overwrite=True)
    fits.writeto(tmp_path / "image_centers_fitted_robust.fits", robust, overwrite=True)
    if wavelengths is None:
        wavelengths = np.linspace(2110.0, 2251.0, raw.shape[0])
    fits.writeto(tmp_path / "wavelengths.fits", np.asarray(wavelengths, dtype=float), overwrite=True)
    pd.DataFrame({"TIME": _times(n_center)}).to_csv(tmp_path / "frames_info_center.csv", index=False)
    if n_coro is not None:
        pd.DataFrame({"TIME": _times(n_coro, start="2023-06-13T02:30:00")}).to_csv(
            tmp_path / "frames_info_coro.csv", index=False
        )
    if outliers is not None:
        (tmp_path / "additional_outputs").mkdir(exist_ok=True)
        fits.writeto(tmp_path / "additional_outputs" / "center_outlier_frames.fits",
                     np.asarray(outliers, dtype=np.int32), overwrite=True)


class TestBuildCenterSeries:
    """The series builder pairs each position array with its own time base."""

    def test_aligned_arrays_all_use_center_times(self):
        center_minutes = np.arange(5.0)
        series = _build_center_series(
            raw=_positions(5, 1), fitted=_positions(5, 2), robust=_positions(5, 3),
            center_minutes=center_minutes, coro_minutes=None,
        )
        assert len(series) == 3
        for entry in series:
            np.testing.assert_array_equal(entry.minutes, center_minutes)

    def test_duplicate_position_arrays_are_drawn_once(self):
        """The waffle branch writes ``fitted`` as a copy of the raw centers.

        Drawing it twice puts one marker exactly under another and implies a
        fit that never happened.
        """
        raw = _positions(5, 1)
        series = _build_center_series(
            raw=raw, fitted=raw.copy(), robust=_positions(5, 3),
            center_minutes=np.arange(5.0), coro_minutes=None,
        )
        assert [entry.marker for entry in series] == ["+", "x"]

    def test_near_identical_arrays_are_drawn_once(self):
        """The copies differ by float32 round-trip noise on a real reduction.

        bet Pic 2014-12-07 has ``image_centers_fitted.fits`` matching
        ``image_centers.fits`` to 3e-5 px, which exact equality misses, so both
        markers were still drawn on top of each other.
        """
        raw = _positions(5, 1)
        almost = raw + 3e-5
        series = _build_center_series(
            raw=raw, fitted=almost, robust=_positions(5, 3),
            center_minutes=np.arange(5.0), coro_minutes=None,
        )
        assert [entry.marker for entry in series] == ["+", "x"]

    def test_genuinely_different_arrays_are_both_kept(self):
        """A real fit moves the centers far more than the rounding tolerance."""
        raw = _positions(5, 1)
        series = _build_center_series(
            raw=raw, fitted=raw + 0.01, robust=_positions(5, 3),
            center_minutes=np.arange(5.0), coro_minutes=None,
        )
        assert [entry.marker for entry in series] == ["+", "o", "x"]

    def test_propagated_arrays_use_coro_times(self):
        """Propagated centers describe CORO frames, so they carry CORO times."""
        center_minutes = np.arange(2.0)
        coro_minutes = np.arange(10.0)
        series = _build_center_series(
            raw=_positions(2, 1), fitted=_positions(10, 2), robust=_positions(10, 2),
            center_minutes=center_minutes, coro_minutes=coro_minutes,
        )
        by_marker = {entry.marker: entry for entry in series}
        np.testing.assert_array_equal(by_marker["+"].minutes, center_minutes)
        np.testing.assert_array_equal(by_marker["o"].minutes, coro_minutes)

    def test_identical_propagated_arrays_are_drawn_once(self):
        """The DMS branch writes the same propagated array to both fitted files."""
        propagated = _positions(10, 2)
        series = _build_center_series(
            raw=_positions(2, 1), fitted=propagated, robust=propagated.copy(),
            center_minutes=np.arange(2.0), coro_minutes=np.arange(10.0),
        )
        assert [entry.marker for entry in series] == ["+", "o"]

    def test_propagated_without_coro_times_has_no_time_base(self):
        """Missing frames_info_coro.csv must not borrow the CENTER timestamps."""
        series = _build_center_series(
            raw=_positions(2, 1), fitted=_positions(10, 2), robust=_positions(10, 3),
            center_minutes=np.arange(2.0), coro_minutes=None,
        )
        by_marker = {entry.marker: entry for entry in series}
        assert by_marker["o"].minutes is None
        assert by_marker["x"].minutes is None

    @pytest.mark.parametrize("n_center,n_coro", [(5, 3), (2, 10), (4, 4)])
    def test_every_series_time_base_matches_its_frame_count(self, n_center, n_coro):
        """The invariant behind #129: no series may be indexed past its length."""
        series = _build_center_series(
            raw=_positions(n_center, 1), fitted=_positions(n_coro, 2), robust=_positions(n_coro, 3),
            center_minutes=np.arange(float(n_center)), coro_minutes=np.arange(float(n_coro)),
        )
        for entry in series:
            assert len(entry.minutes) == entry.positions.shape[1], entry.label


class TestTimeBases:
    """Frame ordering within a sequence must not change the time base.

    The usual SPHERE sequence is FLUX, CENTER, CORO, CENTER, FLUX, so CENTER
    comes first; specialized waffle sequences can put CORO first. Both are
    covered below, because the propagated series' colours depend on the two
    grids sharing an origin regardless of which one opens the sequence.

    IFS reductions write frames_info_coro.csv too, so an aligned reduction has
    those timestamps available and must still ignore them — folding them into
    the origin would shift an IFS colorbar that has never included them.
    """

    # (center_start, coro_start, offset in minutes of the later one)
    CENTER_FIRST = ("2015-09-24T03:00:00", "2015-09-24T03:10:00", 10.0)
    CORO_FIRST = ("2015-09-24T03:30:00", "2015-09-24T03:00:00", 30.0)

    def _paths(self, tmp_path, center_start, coro_start):
        pd.DataFrame({"TIME": _times(4, start=center_start)}).to_csv(
            tmp_path / "frames_info_center.csv", index=False
        )
        pd.DataFrame({"TIME": _times(60, start=coro_start)}).to_csv(
            tmp_path / "frames_info_coro.csv", index=False
        )
        return str(tmp_path / "frames_info_center.csv"), str(tmp_path / "frames_info_coro.csv")

    @pytest.mark.parametrize("ordering", [CENTER_FIRST, CORO_FIRST])
    def test_aligned_origin_is_the_first_center_frame(self, tmp_path, ordering):
        """Whatever the sequence order, an aligned reduction ignores CORO."""
        center_start, coro_start, _ = ordering
        center_path, coro_path = self._paths(tmp_path, center_start, coro_start)
        center_minutes, coro_minutes = _time_bases(center_path, coro_path, propagated=False)
        assert coro_minutes is None
        assert center_minutes[0] == 0.0

    def test_propagated_grids_share_one_origin_center_first(self, tmp_path):
        """The usual sequence: CENTER opens, so it defines the origin."""
        center_start, coro_start, offset = self.CENTER_FIRST
        center_path, coro_path = self._paths(tmp_path, center_start, coro_start)
        center_minutes, coro_minutes = _time_bases(center_path, coro_path, propagated=True)
        assert center_minutes[0] == 0.0
        assert coro_minutes[0] == offset

    def test_propagated_grids_share_one_origin_coro_first(self, tmp_path):
        """A waffle-style sequence where CORO opens instead."""
        center_start, coro_start, offset = self.CORO_FIRST
        center_path, coro_path = self._paths(tmp_path, center_start, coro_start)
        center_minutes, coro_minutes = _time_bases(center_path, coro_path, propagated=True)
        assert coro_minutes[0] == 0.0
        assert center_minutes[0] == offset

    def test_missing_coro_file_falls_back_to_center_origin(self, tmp_path):
        center_path, _ = self._paths(tmp_path, *self.CENTER_FIRST[:2])
        center_minutes, coro_minutes = _time_bases(
            center_path, str(tmp_path / "does_not_exist.csv"), propagated=True
        )
        assert coro_minutes is None
        assert center_minutes[0] == 0.0


class TestWavelengthLegendEntries:
    """Marker size encodes wavelength, which the legend never explained."""

    def test_irdis_lists_both_channels(self):
        entries = _wavelength_legend_entries(np.array([2110.0, 2251.0]), np.array([20.0, 300.0]))
        assert [label for _, label in entries] == ["2.11 µm", "2.25 µm"]
        assert [size for size, _ in entries] == [20.0, 300.0]

    def test_ifs_is_summarised_to_three_channels(self):
        """39 legend entries would be unreadable, so show the span."""
        wavelengths = np.linspace(950.0, 1650.0, 39)
        sizes = np.linspace(20.0, 300.0, 39)
        entries = _wavelength_legend_entries(wavelengths, sizes)
        assert len(entries) == 3
        assert [label for _, label in entries] == ["0.95 µm", "1.30 µm", "1.65 µm"]
        assert [size for size, _ in entries] == [20.0, sizes[19], 300.0]

    def test_single_channel(self):
        entries = _wavelength_legend_entries(np.array([1600.0]), np.array([20.0]))
        assert [label for _, label in entries] == ["1.60 µm"]


class TestResidualsFromMedian:
    """Both channels share a y scale once the detector offset is removed."""

    def test_subtracts_the_per_channel_median(self):
        positions = np.zeros((2, 4, 2), dtype=np.float32)
        positions[0, :, 0] = [10.0, 11.0, 12.0, 13.0]   # ch0 x, median 11.5
        positions[1, :, 1] = [100.0, 100.0, 102.0, 102.0]  # ch1 y, median 101.0
        residuals = _residuals_from_median(positions)
        np.testing.assert_allclose(residuals[0, :, 0], [-1.5, -0.5, 0.5, 1.5])
        np.testing.assert_allclose(residuals[1, :, 1], [-1.0, -1.0, 1.0, 1.0])

    def test_ignores_nans(self):
        """A failed center fit leaves NaN, which must not poison the median."""
        positions = np.zeros((1, 3, 2), dtype=np.float32)
        positions[0, :, 0] = [5.0, np.nan, 7.0]
        residuals = _residuals_from_median(positions)
        np.testing.assert_allclose(residuals[0, [0, 2], 0], [-1.0, 1.0])
        assert np.isnan(residuals[0, 1, 0])


class TestRunPlot:
    """End-to-end: the step must render rather than log a failure."""

    def _assert_rendered(self, tmp_path, caplog):
        assert (tmp_path / "center_plots" / "center_evolution_time_colorbar.pdf").exists()
        assert (tmp_path / "center_plots" / "center_evolution_timeseries.pdf").exists()
        assert "Failed to create center evolution plot" not in caplog.text

    def test_more_center_than_coro_frames_still_renders(self, tmp_path, caplog):
        """Reproduces issue #129: 5 CENTER frames, 3 propagated CORO frames."""
        _write_inputs(
            tmp_path, raw=_positions(5, 1), fitted=_positions(3, 2), robust=_positions(3, 2),
            n_center=5, n_coro=3,
        )
        run_image_center_evolution_plot(str(tmp_path))
        self._assert_rendered(tmp_path, caplog)

    def test_more_coro_than_center_frames_still_renders(self, tmp_path, caplog):
        """The ordinary coronagraphic case: 2 CENTER frames, 10 CORO frames."""
        _write_inputs(
            tmp_path, raw=_positions(2, 1), fitted=_positions(10, 2), robust=_positions(10, 2),
            n_center=2, n_coro=10,
        )
        run_image_center_evolution_plot(str(tmp_path))
        self._assert_rendered(tmp_path, caplog)

    def test_missing_coro_frame_info_still_renders(self, tmp_path, caplog):
        """A reduction without frames_info_coro.csv falls back, it does not fail."""
        _write_inputs(
            tmp_path, raw=_positions(2, 1), fitted=_positions(10, 2), robust=_positions(10, 3),
            n_center=2, n_coro=None,
        )
        run_image_center_evolution_plot(str(tmp_path))
        self._assert_rendered(tmp_path, caplog)

    def test_aligned_frame_grids_still_render(self, tmp_path, caplog):
        """Regression: the waffle / IFS case is unaffected."""
        _write_inputs(
            tmp_path, raw=_positions(6, 1), fitted=_positions(6, 2), robust=_positions(6, 3),
            n_center=6, n_coro=None,
        )
        run_image_center_evolution_plot(str(tmp_path))
        self._assert_rendered(tmp_path, caplog)

    def test_timeseries_renders_with_flagged_frames_marked(self, tmp_path, caplog):
        """The outlier index file is optional input to the time series."""
        _write_inputs(
            tmp_path, raw=_positions(20, 1), fitted=_positions(20, 2), robust=_positions(20, 3),
            n_center=20, n_coro=None, outliers=[[3, 7, -1], [11, -1, -1]],
        )
        run_image_center_evolution_plot(str(tmp_path))
        self._assert_rendered(tmp_path, caplog)

    def test_timeseries_renders_without_wavelengths(self, tmp_path, caplog):
        """An older reduction may have no wavelengths.fits next to the centers."""
        _write_inputs(
            tmp_path, raw=_positions(6, 1), fitted=_positions(6, 2), robust=_positions(6, 3),
            n_center=6,
        )
        (tmp_path / "wavelengths.fits").unlink()
        run_image_center_evolution_plot(str(tmp_path))
        self._assert_rendered(tmp_path, caplog)

    def test_ifs_reduction_keeps_its_marker_stacking(self, tmp_path, caplog, monkeypatch):
        """An IFS reduction draws frame-major in the order 'o', 'x', '+'.

        Grouping the draw calls by series instead would put every robust marker
        on top of every raw one, restacking a plot that has looked the same
        since the IFS pipeline was written.
        """
        import matplotlib.axes

        n_wave, n_frames = 39, 4
        rng = np.random.default_rng(0)
        _write_inputs(
            tmp_path,
            raw=(128 + rng.normal(0, 0.3, (n_wave, n_frames, 2))).astype(np.float32),
            fitted=(128 + rng.normal(0, 0.1, (n_wave, n_frames, 2))).astype(np.float32),
            robust=(128 + rng.normal(0, 0.05, (n_wave, n_frames, 2))).astype(np.float32),
            n_center=n_frames, n_coro=60,
        )

        markers: list[str] = []
        real_scatter = matplotlib.axes.Axes.scatter

        def recording_scatter(self, x, y, **kwargs):
            # The wavelength legend draws empty proxy handles with no marker set.
            if kwargs.get("marker") is not None:
                markers.append(kwargs["marker"])
            return real_scatter(self, x, y, **kwargs)

        monkeypatch.setattr(matplotlib.axes.Axes, "scatter", recording_scatter)
        run_image_center_evolution_plot(str(tmp_path))

        self._assert_rendered(tmp_path, caplog)
        assert markers == ["o", "x", "+"] * n_frames
