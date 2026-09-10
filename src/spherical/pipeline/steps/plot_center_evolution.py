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
    a fit that never happened.

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
        if any(np.array_equal(candidate.positions, kept.positions) for kept in series):
            continue
        series.append(candidate)
    return series


def _elapsed_minutes(time_strings, start_time):
    """Minutes elapsed since ``start_time`` for a column of timestamps."""
    times = pd.to_datetime(time_strings)
    return ((times - start_time).dt.total_seconds() / 60.0).to_numpy()


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
        legend_elements = [
            Line2D([0], [0], marker=entry.marker, color='gray', linestyle='None',
                   markersize=10, label=entry.label)
            for entry in series
        ]
        ax.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, -0.15),
            ncol=len(legend_elements) or 1,
            title='Marker Meaning',
            frameon=False
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
    except Exception:
        logger.exception("Failed to create center evolution plot.", extra={"step": "plot_center_evolution", "status": "failed"})
        return
    logger.info("Finished center evolution plot step.", extra={"step": "plot_center_evolution", "status": "success"})
