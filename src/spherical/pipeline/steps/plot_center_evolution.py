"""
Plot Image Center Evolution Step

Parameters
----------
converted_dir : str
    Directory where the output files are stored and written.
"""
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from spherical.pipeline.logging_utils import optional_logger

# Drawn for a position array whose frame timestamps are unavailable, so it is
# visibly not part of the time colour scale.
_UNTIMED_COLOR = "0.5"

# Draw order, back to front. Frame-major within it, which is what the step has
# always done, so an aligned reduction (IFS, or a waffle sequence) keeps exactly
# the stacking it had.
_MARKER_DRAW_ORDER = ("o", "x", "+")

# Two center arrays agreeing this closely are the same measurement written
# twice. A float32 round trip through FITS costs ~3e-5 px, which exact equality
# would call a difference, and nothing in SPHERE astrometry is meaningful at
# 1e-4 px.
_DUPLICATE_TOLERANCE_PX = 1e-4


@dataclass(frozen=True)
class _CenterSeries:
    """One set of center positions sharing a single frame grid and time base.

    Attributes
    ----------
    label : str
        Legend entry.
    marker : str
        Matplotlib marker.
    positions : numpy.ndarray
        ``(n_wave, n_frames, 2)`` center positions in (x, y) pixels.
    minutes : numpy.ndarray or None
        Elapsed minutes per frame, ``(n_frames,)``, or None when the frame
        timestamps for this grid are not on disk.
    alpha : float
        Marker transparency.
    """

    label: str
    marker: str
    positions: np.ndarray
    minutes: np.ndarray | None
    alpha: float = 0.6


def _build_center_series(raw, fitted, robust, center_minutes, coro_minutes):
    """Pair each center array with the time base of the frames it describes.

    ``image_centers.fits`` always holds CENTER-frame measurements. The two
    fitted arrays hold either the same CENTER frames (waffle sequence, where
    the satellite spots give the center directly) or DMS-propagated CORO
    frames (coronagraphic sequence). Those two grids have unrelated lengths,
    so each series carries its own timestamps and its own frame count instead
    of borrowing the raw array's.

    Position arrays that duplicate one already kept are dropped: the waffle
    branch writes ``fitted`` as a copy of the raw centers and the DMS branch
    writes ``robust`` as a copy of ``fitted``, and drawing either twice implies
    a fit that never happened. The comparison carries a tolerance because a
    float32 round trip through FITS leaves the two copies differing by ~3e-5 px.

    Parameters
    ----------
    raw, fitted, robust : numpy.ndarray
        ``(n_wave, n_frames, 2)`` center positions.
    center_minutes : numpy.ndarray
        Elapsed minutes for the CENTER frames.
    coro_minutes : numpy.ndarray or None
        Elapsed minutes for the CORO frames, or None if unavailable.

    Returns
    -------
    list of _CenterSeries
    """
    propagated = fitted.shape[1] != raw.shape[1]
    fitted_minutes = coro_minutes if propagated else center_minutes

    candidates = [
        _CenterSeries(
            "Measured (CENTER)" if propagated else "Original Data",
            "+", raw, center_minutes, 0.6,
        ),
        _CenterSeries(
            "DMS-propagated (CORO)" if propagated else "1st Fit (fitted)",
            "o", fitted, fitted_minutes, 0.6,
        ),
        _CenterSeries(
            "DMS-propagated, robust (CORO)" if propagated else "2nd Fit (robust)",
            "x", robust, fitted_minutes, 0.9,
        ),
    ]

    series: list[_CenterSeries] = []
    for candidate in candidates:
        if candidate.positions.shape[1] == 0:
            continue
        if any(_same_positions(candidate.positions, kept.positions) for kept in series):
            continue
        series.append(candidate)
    return series


def _same_positions(a, b):
    """True when two center arrays are the same measurement written twice."""
    if a.shape != b.shape:
        return False
    return bool(np.allclose(a, b, rtol=0.0, atol=_DUPLICATE_TOLERANCE_PX, equal_nan=True))


def _wavelength_legend_entries(wavelengths, sizes):
    """Marker sizes paired with the wavelength each one stands for.

    Marker area ramps with wavelength channel, which is the only thing telling
    the IRDIS channels apart in the scatter plot and was never in the legend.
    IFS has 39 channels, so only the first, middle and last are listed rather
    than a legend taller than the figure.

    Parameters
    ----------
    wavelengths : numpy.ndarray
        Channel wavelengths in nanometres, as stored in ``wavelengths.fits``.
    sizes : numpy.ndarray
        Marker size per channel, same length as ``wavelengths``.

    Returns
    -------
    list of (float, str)
        Marker size and its label, in micron.
    """
    n_wave = len(wavelengths)
    indices = range(n_wave) if n_wave <= 3 else (0, n_wave // 2, n_wave - 1)
    return [(float(sizes[i]), f"{wavelengths[i] / 1000.0:.2f} µm") for i in indices]


def _residuals_from_median(positions):
    """Center positions with each channel's median subtracted.

    The IRDIS channels sit ~13 px apart on the detector, so subtracting the
    per-channel median is what lets both share one axis in the time series.

    Parameters
    ----------
    positions : numpy.ndarray
        ``(n_wave, n_frames, 2)`` center positions.

    Returns
    -------
    numpy.ndarray
        Same shape, in pixels relative to each channel's median.
    """
    return positions - np.nanmedian(positions, axis=1)[:, None, :]


def _elapsed_minutes(time_strings, start_time):
    """Minutes elapsed since ``start_time`` for a column of timestamps."""
    times = pd.to_datetime(time_strings)
    return ((times - start_time).dt.total_seconds() / 60.0).to_numpy()


def _optional_fits(path):
    """Return the data in a FITS file, or None if it is not there."""
    return fits.getdata(path) if os.path.exists(path) else None


def _plot_center_timeseries(series, wavelengths, outlier_frames, output_path):
    """Draw x and y against time, one panel each, and save to ``output_path``.

    The scatter plot puts x against y and encodes time as colour, which buries
    a slow drift. Here time is an axis, so a drift is a slope, the frame-to-
    frame jitter is the width of the band, and a failed center fit is a spike.

    Positions are shown relative to each channel's median so that channels
    sitting far apart on the detector share one scale. Series are drawn against
    their own timestamps, which is what lets the measured CENTER frames and the
    DMS-propagated CORO frames appear together despite being different grids.

    Parameters
    ----------
    series : list of _CenterSeries
        Position arrays with their time bases, from `_build_center_series`.
    wavelengths : numpy.ndarray or None
        Channel wavelengths in nanometres, used to label the channels.
    outlier_frames : numpy.ndarray or None
        ``(n_wave, k)`` frame indices flagged by the center fit, padded with -1.
        Marked on the first series, which holds the raw measurements.
    output_path : str
        Where to write the figure.
    """
    timed = [entry for entry in series if entry.minutes is not None]
    if not timed:
        return

    n_wave = timed[0].positions.shape[0]
    channel_colors = plt.cm.viridis(np.linspace(0, 0.9, n_wave))
    # A line per series, so the marker tells you which array a point came from.
    styles = {"+": dict(marker=".", linestyle="none", markersize=4),
              "o": dict(marker="none", linestyle="-", linewidth=1.0),
              "x": dict(marker="none", linestyle="--", linewidth=1.0)}

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for entry in timed:
        residuals = _residuals_from_median(entry.positions)
        style = styles.get(entry.marker, styles["+"])
        for ch in range(entry.positions.shape[0]):
            for axis_idx, ax in enumerate(axes):
                ax.plot(entry.minutes, residuals[ch, :, axis_idx],
                        color=channel_colors[ch % n_wave], alpha=0.8, **style)

    raw = timed[0]
    if outlier_frames is not None:
        residuals = _residuals_from_median(raw.positions)
        for ch in range(min(len(outlier_frames), raw.positions.shape[0])):
            flagged = np.asarray(outlier_frames[ch])
            flagged = flagged[(flagged >= 0) & (flagged < raw.positions.shape[1])]
            if flagged.size == 0:
                continue
            for axis_idx, ax in enumerate(axes):
                ax.plot(raw.minutes[flagged], residuals[ch, flagged, axis_idx],
                        marker="o", linestyle="none", markersize=7,
                        markerfacecolor="none", markeredgecolor="crimson",
                        label="_flagged" if ch else "flagged by center fit")

    # Scale to the cleanest series available. A failed fit throws the center by
    # several pixels, which would otherwise compress the drift and the jitter —
    # the things this plot exists to show — into a flat line. The robust series
    # has those frames replaced, so prefer it; the DMS branch has no robust
    # series but its propagated track carries no spikes either, so there every
    # series counts and the dither range stays on screen.
    robust_series = [entry for entry in timed if entry.marker == "x"]
    reference = [_residuals_from_median(entry.positions) for entry in (robust_series or timed)]
    all_residuals = [_residuals_from_median(entry.positions) for entry in timed]
    for axis_idx, ax in enumerate(axes):
        finite = np.concatenate([r[:, :, axis_idx].ravel() for r in reference])
        finite = finite[np.isfinite(finite)]
        if finite.size:
            # A percentile rather than the max: outlier replacement by local
            # median does not always fully recover a long bad run.
            span = max(float(np.percentile(np.abs(finite), 99)) * 1.3, 0.25)
            ax.set_ylim(-span, span)
            hidden = int(sum(np.sum(np.abs(r[:, :, axis_idx]) > span) for r in all_residuals))
            if hidden:
                ax.annotate(f"{hidden} points beyond ±{span:.2f} px", xy=(0.995, 0.03),
                            xycoords="axes fraction", ha="right", va="bottom",
                            fontsize=8, color="crimson")
        ax.axhline(0.0, color="0.7", linewidth=0.8, zorder=0)
        ax.set_ylabel(f"Δ{'xy'[axis_idx]} from median (px)")
        ax.grid(alpha=0.2)
    axes[1].set_xlabel("Elapsed Time (minutes)")
    axes[0].set_title("Center Position vs Time")

    # Same first/middle/last summary the scatter legend uses, so 39 IFS channels
    # do not produce 39 legend entries.
    channel_idx = range(n_wave) if n_wave <= 3 else (0, n_wave // 2, n_wave - 1)
    handles = [
        Line2D([0], [0], color=channel_colors[i], linewidth=2,
               label=(f"{wavelengths[i] / 1000.0:.2f} µm" if wavelengths is not None
                      else f"channel {i}"))
        for i in channel_idx
    ]
    handles += [Line2D([0], [0], color="0.4", label=entry.label, **styles.get(entry.marker, styles["+"]))
                for entry in timed]
    if outlier_frames is not None:
        handles.append(Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
                              markeredgecolor="crimson", label="flagged by center fit"))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=min(len(handles), 4), frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _read_frame_times(path):
    """Return the TIME column of a frames_info CSV, or None if unusable."""
    if not os.path.exists(path):
        return None
    frame_info = pd.read_csv(path)
    if "TIME" not in frame_info.columns or len(frame_info) == 0:
        return None
    return pd.to_datetime(frame_info["TIME"])


def _time_bases(frame_info_center_path, frame_info_coro_path, propagated):
    """Elapsed minutes for each frame grid that the plot will actually use.

    The CORO timestamps matter only when the fitted arrays sit on the CORO
    frame grid. An aligned reduction (IFS, or a waffle sequence) ignores them
    even though ``frames_info_coro.csv`` is on disk, so its time origin stays
    the first CENTER frame. When both grids are drawn they share one origin,
    which keeps a single colorbar meaningful across the two series.

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray or None)
        Elapsed minutes for the CENTER frames, and for the CORO frames when
        they are in use.
    """
    center_times = _read_frame_times(frame_info_center_path)
    coro_times = _read_frame_times(frame_info_coro_path) if propagated else None
    start_time = center_times.min()
    if coro_times is not None:
        start_time = min(start_time, coro_times.min())
    return (
        _elapsed_minutes(center_times, start_time),
        None if coro_times is None else _elapsed_minutes(coro_times, start_time),
    )


@optional_logger
def run_image_center_evolution_plot(converted_dir: str, logger) -> None:
    """Create visualization of star center position evolution across wavelength and time.

    Creates a scatter plot showing the evolution of star center positions
    across wavelength and observation time. What is compared depends on which
    branch ``process_extracted_centers`` took:

    * **Waffle sequence (or IFS).** The fitted arrays describe the same CENTER
      frames as the raw measurements, so the plot compares raw against fitted
      against robust, frame by frame, on one time base.
    * **Coronagraphic sequence.** The fitted arrays hold DMS-propagated CORO
      positions on a different frame grid. The measured CENTER positions and
      the propagated CORO positions are then drawn as two independent series,
      each coloured by its own timestamps against a shared time axis.

    Required Input Files
    -------------------
    From previous steps:
    - converted_dir/image_centers.fits
        Raw star center positions from waffle spot fitting (CENTER frames)
    - converted_dir/image_centers_fitted.fits
        First-pass fits, or DMS-propagated CORO centers
    - converted_dir/image_centers_fitted_robust.fits
        Robust fits, or DMS-propagated CORO centers
    - converted_dir/frames_info_center.csv
        CENTER frame information including timestamps
    - converted_dir/frames_info_coro.csv
        CORO frame timestamps; only needed when the fitted arrays are
        DMS-propagated, and the propagated series is drawn untimed without it.

    Generated Output Files
    ---------------------
    In converted_dir/center_plots/:
    - center_evolution_time_colorbar.pdf
        Scatter plot showing center position evolution with time colorbar
    - center_evolution_timeseries.pdf
        x and y against time, one panel each, relative to each channel's
        median, with frames flagged by the center fit ringed

    Parameters
    ----------
    converted_dir : str
        Directory containing the input files and where outputs will be written.

    Returns
    -------
    None
        This function writes a visualization plot to disk and does not return
        a value.

    Notes
    -----
    - Marker size increases with wavelength
    - Color indicates elapsed time since the start of the observation
    - Uses PiYG colormap for time visualization; a series whose timestamps are
      unavailable is drawn in neutral gray
    - Maintains equal aspect ratio for proper spatial representation

    Examples
    --------
    >>> run_image_center_evolution_plot(
    ...     converted_dir="/path/to/converted"
    ... )
    """
    logger.info("Starting center evolution plot step.", extra={"step": "plot_center_evolution", "status": "started"})
    logger.debug(f"Parameters: converted_dir={converted_dir}")
    try:
        image_centers_path = os.path.join(converted_dir, 'image_centers.fits')
        image_centers_fitted_path = os.path.join(converted_dir, 'image_centers_fitted.fits')
        image_centers_fitted2_path = os.path.join(converted_dir, 'image_centers_fitted_robust.fits')
        frame_info_center_path = os.path.join(converted_dir, 'frames_info_center.csv')
        frame_info_coro_path = os.path.join(converted_dir, 'frames_info_coro.csv')
        for f in [image_centers_path, image_centers_fitted_path, image_centers_fitted2_path, frame_info_center_path]:
            if not os.path.exists(f):
                logger.warning(f"Missing required file: {f}", extra={"step": "plot_center_evolution", "status": "failed"})
        image_centers = fits.getdata(image_centers_path)
        image_centers_fitted = fits.getdata(image_centers_fitted_path)
        image_centers_fitted2 = fits.getdata(image_centers_fitted2_path)
        # Both are optional: a reduction from an older version may have neither,
        # and each only adds annotation to the plots.
        wavelengths = _optional_fits(os.path.join(converted_dir, 'wavelengths.fits'))
        if wavelengths is not None:
            wavelengths = np.asarray(wavelengths).ravel()
            if len(wavelengths) != image_centers.shape[0]:
                logger.warning(
                    f"wavelengths.fits has {len(wavelengths)} channels but the centers have "
                    f"{image_centers.shape[0]}; omitting the wavelength legend.",
                    extra={"step": "plot_center_evolution", "status": "info"},
                )
                wavelengths = None
        outlier_frames = _optional_fits(
            os.path.join(converted_dir, 'additional_outputs', 'center_outlier_frames.fits')
        )
        plot_dir = os.path.join(converted_dir, 'center_plots/')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
            logger.debug(f"Created plot output directory: {plot_dir}")

        propagated = image_centers_fitted.shape[1] != image_centers.shape[1]
        center_minutes, coro_minutes = _time_bases(
            frame_info_center_path, frame_info_coro_path, propagated
        )

        series = _build_center_series(
            raw=image_centers,
            fitted=image_centers_fitted,
            robust=image_centers_fitted2,
            center_minutes=center_minutes,
            coro_minutes=coro_minutes,
        )
        for entry in series:
            if entry.minutes is not None and len(entry.minutes) != entry.positions.shape[1]:
                logger.warning(
                    f"'{entry.label}': {entry.positions.shape[1]} frames but "
                    f"{len(entry.minutes)} timestamps; drawing it untimed.",
                    extra={"step": "plot_center_evolution", "status": "info"},
                )
        series = [
            entry if entry.minutes is None or len(entry.minutes) == entry.positions.shape[1]
            else _CenterSeries(entry.label, entry.marker, entry.positions, None, entry.alpha)
            for entry in series
        ]

        timed = [entry.minutes for entry in series if entry.minutes is not None]
        vmin = min(float(m.min()) for m in timed) if timed else 0.0
        vmax = max(float(m.max()) for m in timed) if timed else 1.0
        if vmax <= vmin:  # single frame, or all frames at one timestamp
            vmax = vmin + 1.0
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.cm.PiYG

        fig, ax = plt.subplots(figsize=(8, 6))
        # Frame-major, markers back to front within each frame. Series no longer
        # share a frame count, so each one is simply skipped once it runs out —
        # that bound is what used to be taken from the raw array for all three.
        drawn = sorted(series, key=lambda entry: _MARKER_DRAW_ORDER.index(entry.marker))
        for frame_idx in range(max(entry.positions.shape[1] for entry in drawn)):
            for entry in drawn:
                if frame_idx >= entry.positions.shape[1]:
                    continue
                sizes = np.linspace(20, 300, entry.positions.shape[0])
                color = _UNTIMED_COLOR if entry.minutes is None else cmap(norm(entry.minutes[frame_idx]))
                ax.scatter(entry.positions[:, frame_idx, 0], entry.positions[:, frame_idx, 1],
                           s=sizes, marker=entry.marker, color=color, alpha=entry.alpha)
        # Legend follows the draw order, which is the order it has always been in.
        legend_elements = [
            Line2D([0], [0], marker=entry.marker, color='gray', linestyle='None',
                   markersize=10, label=entry.label)
            for entry in drawn
        ]
        marker_legend = ax.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=len(legend_elements) or 1,
            title='Marker Meaning',
            frameon=False
        )
        # Marker area encodes the wavelength channel. Without this second legend
        # the two IRDIS clusters look unexplained.
        if wavelengths is not None:
            sizes = np.linspace(20, 300, image_centers.shape[0])
            size_handles = [
                ax.scatter([], [], s=size, color='gray', alpha=0.6, label=label)
                for size, label in _wavelength_legend_entries(wavelengths, sizes)
            ]
            ax.add_artist(marker_legend)
            ax.legend(
                handles=size_handles,
                loc='upper center',
                bbox_to_anchor=(0.5, -0.32),
                ncol=len(size_handles),
                title='Marker Size (wavelength)',
                frameon=False,
                labelspacing=1.4,
                borderpad=1.0,
            )
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label('Elapsed Time (minutes)')
        ax.set_xlabel('X Center Position')
        ax.set_ylabel('Y Center Position')
        ax.set_title('Center Position Evolution per Wavelength and Time')
        ax.set_aspect('equal')
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.25)
        output_path = os.path.join(plot_dir, 'center_evolution_time_colorbar.pdf')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Center evolution plot written to: {output_path}")

        timeseries_path = os.path.join(plot_dir, 'center_evolution_timeseries.pdf')
        _plot_center_timeseries(series, wavelengths, outlier_frames, timeseries_path)
        logger.info(f"Center time series plot written to: {timeseries_path}")
    except Exception:
        logger.exception("Failed to create center evolution plot.", extra={"step": "plot_center_evolution", "status": "failed"})
        return
    logger.info("Finished center evolution plot step.", extra={"step": "plot_center_evolution", "status": "success"})
