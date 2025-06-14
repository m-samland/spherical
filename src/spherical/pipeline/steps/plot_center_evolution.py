"""
Plot Image Center Evolution Step

Parameters
----------
converted_dir : str
    Directory where the output files are stored and written.
"""
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D


def run_image_center_evolution_plot(converted_dir: str) -> None:
    """Create visualization of star center position evolution across wavelength and time.

    This is the eighth step in the SPHERE/IFS data reduction pipeline. It creates
    a scatter plot showing the evolution of star center positions across different
    wavelengths and observation times, comparing raw measurements with both first-pass
    and robust polynomial fits.

    Required Input Files
    -------------------
    From previous steps:
    - converted_dir/image_centers.fits
        Raw star center positions from waffle spot fitting
    - converted_dir/image_centers_fitted.fits
        First-pass polynomial fits to center positions
    - converted_dir/image_centers_fitted_robust.fits
        Second-pass robust polynomial fits after outlier rejection
    - converted_dir/frames_info_center.csv
        Frame information including timestamps

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
    - Creates a scatter plot with three types of markers:
        * '+' for raw measurements
        * 'o' for first-pass polynomial fits
        * 'x' for robust polynomial fits
    - Marker size increases with wavelength
    - Color indicates elapsed time since start of observation
    - Uses PiYG colormap for time visualization
    - Includes legend and time colorbar
    - Creates output directory if it doesn't exist
    - Maintains equal aspect ratio for proper spatial representation

    Examples
    --------
    >>> run_image_center_evolution_plot(
    ...     converted_dir="/path/to/converted"
    ... )
    """
    image_centers = fits.getdata(os.path.join(converted_dir, 'image_centers.fits'))
    image_centers_fitted = fits.getdata(os.path.join(converted_dir, 'image_centers_fitted.fits'))
    image_centers_fitted2 = fits.getdata(os.path.join(converted_dir, 'image_centers_fitted_robust.fits'))
    plot_dir = os.path.join(converted_dir, 'center_plots/')
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    frame_info_center = pd.read_csv(os.path.join(converted_dir, 'frames_info_center.csv'))
    time_strings = frame_info_center["TIME"]
    times = pd.to_datetime(time_strings)
    start_time = times.min()
    elapsed_minutes = (times - start_time).dt.total_seconds() / 60.0
    norm = Normalize(vmin=elapsed_minutes.min(), vmax=elapsed_minutes.max())
    cmap = plt.cm.PiYG
    colors = cmap(norm(elapsed_minutes))
    n_wavelengths = image_centers.shape[0]
    n_frames = image_centers.shape[1]
    sizes = np.linspace(20, 300, n_wavelengths)
    fig, ax = plt.subplots(figsize=(8, 6))
    for frame_idx in range(n_frames):
        color = colors[frame_idx]
        ax.scatter(image_centers_fitted[:, frame_idx, 0], image_centers_fitted[:, frame_idx, 1],
                s=sizes, marker='o', color=color, alpha=0.6)
        ax.scatter(image_centers_fitted2[:, frame_idx, 0], image_centers_fitted2[:, frame_idx, 1],
                s=sizes, marker='x', color=color, alpha=0.9)
        ax.scatter(image_centers[:, frame_idx, 0], image_centers[:, frame_idx, 1],
                s=sizes, marker='+', color=color, alpha=0.6)
    legend_elements = [
        Line2D([0], [0], marker='o', color='gray', linestyle='None', markersize=10, label='1st Fit (fitted)'),
        Line2D([0], [0], marker='x', color='gray', linestyle='None', markersize=10, label='2nd Fit (robust)'),
        Line2D([0], [0], marker='+', color='gray', linestyle='None', markersize=10, label='Original Data'),
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=3,
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
    plt.savefig(os.path.join(plot_dir, 'center_evolution_time_colorbar.pdf'), bbox_inches='tight')
    plt.close()
