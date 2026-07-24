"""Module for creating mosaics of FITS images from TRAP results.

This module provides functionality to create mosaic visualizations of TRAP (Template-based
Reduction of Astronomical data Pipeline) results, including support for different template
types and integration with observation metadata.
"""

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from matplotlib.figure import Figure
from matplotlib.patches import Circle

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


TemplateType = Literal["flat", "L-type", "T-type"]

TEMPLATE_PATTERNS = {
    "flat": "template_matching/normalized_detection_image_flat.fits",
    "L-type": "template_matching/normalized_detection_image_L-type.fits",
    "T-type": "template_matching/normalized_detection_image_T-type.fits",
}

CANDIDATE_PATTERNS = {
    "flat": "template_matching/validated_companion_table_short_flat.csv",
    "L-type": "template_matching/validated_companion_table_short_L-type.csv",
    "T-type": "template_matching/validated_companion_table_short_T-type.csv",
}

# Default color scheme for candidates (easily distinguishable colors)
DEFAULT_CANDIDATE_COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange  
    # '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    # '#bcbd22',  # olive
    '#17becf'   # cyan
]


# Shared infrastructure functions for mosaic plotting

def setup_mosaic_grid(
    n_combinations: int,
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = 300
) -> Tuple[Figure, np.ndarray, int, int]:
    """Set up the matplotlib grid for mosaic plotting.
    
    Args:
        n_combinations: Number of subplot combinations to display
        figsize: Figure size in inches. If None, automatically calculated
        dpi: DPI for the figure
        
    Returns:
        Tuple of (figure, axes_array, n_rows, n_cols)
    """
    # Calculate grid dimensions
    n_cols = int(np.ceil(np.sqrt(n_combinations)))
    n_rows = int(np.ceil(n_combinations / n_cols))
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        # Base size per subplot, with minimum and maximum limits
        subplot_size = max(4, min(8, 40 / max(n_rows, n_cols)))
        figsize = (int(n_cols * subplot_size), int(n_rows * subplot_size))
    
    # Create figure
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=figsize,
        dpi=dpi,
        constrained_layout=False
    )
    axes = np.atleast_2d(axes)
    
    return fig, axes, n_rows, n_cols


def setup_subplot_title(
    target: str,
    obs_mode: str, 
    date: str,
    obs_table: Optional[Table] = None,
    figsize: Optional[Tuple[int, int]] = None
) -> str:
    """Generate subplot title with observation information.
    
    Args:
        target: Target name
        obs_mode: Observation mode
        date: Observation date
        obs_table: Optional observation table for additional info
        figsize: Figure size for font scaling
        
    Returns:
        Formatted title string
    """
    # Build basic title
    title = f"{target}\n{obs_mode}\n{date}"
    
    # Get observation info if table is available
    if obs_table is not None:
        exptime, rotation, fwhm = get_observation_info(obs_table, target, obs_mode, date)
        title += f"\nEXPTIME: {exptime:.1f}m\nROT: {rotation:.1f}°\nFWHM: {fwhm:.2f}\""
        
    return title


def setup_compact_subplot_title(
    target: str,
    obs_mode: str, 
    date: str,
    obs_table: Optional[Table] = None,
    figsize: Optional[Tuple[int, int]] = None
) -> str:
    """Generate compact subplot title for combined mosaics.
    
    Args:
        target: Target name
        obs_mode: Observation mode
        date: Observation date
        obs_table: Optional observation table for additional info
        figsize: Figure size for font scaling
        
    Returns:
        Formatted compact title string
    """
    # More compact title format for combined mosaics
    title = f"{target} | {obs_mode} | {date}"
    
    # Optionally add observation info on new line if available
    if obs_table is not None:
        exptime, rotation, fwhm = get_observation_info(obs_table, target, obs_mode, date)
        if exptime > 0:  # Only add if we found valid data
            title += f"\nEXP: {exptime:.0f}m | ROT: {rotation:.0f}° | FWHM: {fwhm:.2f}\""
        
    return title


def cleanup_mosaic_subplots(
    fig: Figure,
    axes: np.ndarray,
    n_combinations: int,
    n_rows: int,
    n_cols: int,
    main_title: str,
    figsize: Optional[Tuple[int, int]] = None
) -> None:
    """Clean up empty subplots and add main title.
    
    Args:
        fig: Matplotlib figure
        axes: Axes array
        n_combinations: Number of actual combinations plotted
        n_rows: Number of rows in grid
        n_cols: Number of columns in grid
        main_title: Main title for the figure
        figsize: Figure size for font scaling
    """
    # Remove empty subplots
    for idx in range(n_combinations, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axes[row, col])
    
    # Auto-calculate font sizes based on figure size
    if figsize is None:
        figsize = tuple(fig.get_size_inches().astype(int))
    base_font_scale = min(figsize) / 20
    suptitle_fontsize = max(12, int(18 * base_font_scale))
    
    # Add main title
    fig.suptitle(
        main_title,
        fontsize=suptitle_fontsize,
        y=0.98
    )
    
    # Apply tight layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])


def save_mosaic_figure(
    fig: Figure,
    output_path: Optional[Path],
    dpi: int = 300,
    filename_suffix: Optional[str] = None,
    auto_scale: bool = False
) -> None:
    """Save mosaic figure with appropriate filename modifications.
    
    Args:
        fig: Matplotlib figure to save
        output_path: Optional path to save the figure
        dpi: DPI for saving
        filename_suffix: Additional suffix to add to filename
        auto_scale: Whether auto-scaling was used (adds _autoscale suffix)
    """
    if output_path is None:
        return
        
    # Build the final output path with modifications
    stem = output_path.stem
    suffix = output_path.suffix
    
    # Add suffixes
    final_stem = stem
    if filename_suffix:
        final_stem += f"_{filename_suffix}"
    if auto_scale:
        final_stem += "_autoscale"
        
    modified_output_path = output_path.parent / f"{final_stem}{suffix}"
    
    # Configure save parameters based on file format
    save_kwargs = {"bbox_inches": "tight", "dpi": dpi}
    
    # Optimize PDF output for searchable text
    if suffix.lower() == '.pdf':
        save_kwargs.update({
            "format": "pdf",
            "metadata": {
                "Title": f"TRAP Results Mosaic - {final_stem}",
                "Author": "visualize_trap_results",
                "Subject": "Astronomical data visualization from TRAP pipeline",
                "Creator": "matplotlib",
                "Producer": "visualize_trap_results package"
            }
        })
        # Use higher DPI for PDF to ensure text quality
        if dpi < 300:
            save_kwargs["dpi"] = 300
            logger.info("Increased DPI to 300 for PDF output to ensure text quality")
    
    fig.savefig(str(modified_output_path), **save_kwargs)


def load_observation_metadata(
    observation_table_path: Optional[Path] = None,
    observation_table: Optional[Table] = None
) -> Optional[Table]:
    """Load observation table if provided.
    
    Args:
        observation_table_path: Path to observation table
        observation_table: Pre-loaded observation table
        
    Returns:
        Observation table or None
    """
    obs_table = observation_table
    if obs_table is None and observation_table_path is not None:
        obs_table = load_observation_table(observation_table_path)
    return obs_table


def get_mosaic_file_combinations(
    base_path: Path,
    template_type: TemplateType,
    file_type: Literal["fits", "csv"] = "fits",
    combinations: Optional[List[Tuple[str, str, str]]] = None
) -> Dict[Tuple[str, str, str], Optional[Path]]:
    """Get file combinations for mosaic plotting.

    Args:
        base_path: Root path to search
        template_type: Template type to look for
        file_type: Type of files to find ("fits" or "csv")
        combinations: Restrict discovery to these (target, obs_mode, date)
            tuples. If None, all combinations under base_path are used. Used by
            the batched plotters to render one batch at a time.

    Returns:
        Dictionary mapping (target, obs_mode, date) to file paths
    """
    if file_type == "fits":
        pattern = TEMPLATE_PATTERNS[template_type]
    else:  # csv
        pattern = CANDIDATE_PATTERNS[template_type]

    results = {}
    if combinations is None:
        combinations = get_all_combinations(base_path)

    for target, obs_mode, date in combinations:
        file_path = base_path / target / obs_mode / date / pattern
            
        results[(target, obs_mode, date)] = file_path if file_path.exists() else None
    
    return results


def load_observation_table(table_path: Path) -> Table:
    """Load the observation lookup table.
    
    Args:
        table_path: Path to the FITS table containing observation metadata
        
    Returns:
        Astropy Table with observation metadata including MAIN_ID, FILTER, NIGHT_START,
        TOTAL_EXPTIME_SCI, and ROTATION columns
    """
    return Table.read(table_path)


def normalize_target_name(name: str) -> str:
    """Normalize target name for comparison by handling underscores and spaces.
    
    This function ensures consistent comparison between target names in the directory
    structure and the observation table by:
    1. Replacing underscores with spaces
    2. Normalizing multiple spaces into single spaces
    
    Args:
        name: Target name to normalize (e.g., "HD_123456" or "HD  123456")
        
    Returns:
        Normalized target name (e.g., "HD 123456")
    """
    return " ".join(name.replace("_", " ").split())


def get_observation_info(
    table: Table,
    target: str,
    obs_mode: str,
    date: str
) -> Tuple[float, float, float]:
    """Get exposure time, rotation, and FWHM for a specific observation.
    
    Args:
        table: Observation lookup table containing MAIN_ID, FILTER, NIGHT_START,
              TOTAL_EXPTIME_SCI, ROTATION, and MEAN_FWHM columns
        target: Target name (with underscores)
        obs_mode: Observation mode (filter)
        date: Observation date
        
    Returns:
        Tuple of (exposure_time, rotation, mean_fwhm) in seconds, degrees, and arcseconds respectively.
        Returns (0.0, 0.0, 0.0) if no matching observation is found.
        
    Note:
        Prints debug information if no matching observation is found, including
        similar target names from the table.
    """
    # Normalize target name for comparison
    target_spaces = normalize_target_name(target)
    
    # Normalize all target names in the table
    normalized_targets = np.array([normalize_target_name(t) for t in table["MAIN_ID"]])
    
    # Find matching row
    mask = (
        (normalized_targets == target_spaces) &
        (table["FILTER"] == obs_mode) &
        (table["NIGHT_START"] == date)
    )
    
    if not np.any(mask):
        # Only log warning if observation info was requested but not found
        logger.warning(f"Observation info not found for {target}/{obs_mode}/{date}")
        return 0.0, 0.0, 0.0
    
    row = table[mask][0]
    return row["TOTAL_EXPTIME_SCI"], row["ROTATION"], row["MEAN_FWHM"]


def get_all_combinations(base_path: Path) -> List[Tuple[str, str, str]]:
    """Get all unique target/obs_mode/date combinations from the directory structure.
    
    Args:
        base_path: Root path to start searching from
        
    Returns:
        List of tuples containing (target_name, obs_mode, date), sorted alphabetically
        by target name, observation mode, and date
    """
    combinations = set()
    
    for target_dir in base_path.iterdir():
        if not target_dir.is_dir():
            continue
            
        for obs_mode_dir in target_dir.iterdir():
            if not obs_mode_dir.is_dir():
                continue
                
            for date_dir in obs_mode_dir.iterdir():
                if not date_dir.is_dir():
                    continue
                    
                combinations.add((
                    target_dir.name,
                    obs_mode_dir.name,
                    date_dir.name
                ))
    
    # Sort results by target name, observation mode, and date
    return sorted(combinations)


def find_fits_files(
    base_path: Path,
    template_type: TemplateType = "flat"
) -> Dict[Tuple[str, str, str], Optional[Path]]:
    """Find all matching FITS files in the directory structure.
    
    Args:
        base_path: Root path to start searching from
        template_type: Type of template to look for ("flat", "L-type", or "T-type")
        
    Returns:
        Dictionary mapping (target, obs_mode, date) tuples to their FITS file paths.
        Missing files are mapped to None.
        
    Raises:
        ValueError: If no matching FITS files are found
    """
    pattern = TEMPLATE_PATTERNS[template_type]
    results = {}
    
    # Get all possible combinations
    combinations = get_all_combinations(base_path)
    
    # Check each combination
    for target, obs_mode, date in combinations:
        fits_path = base_path / target / obs_mode / date / pattern
        results[(target, obs_mode, date)] = fits_path if fits_path.exists() else None
    
    return results


def load_candidate_table(csv_path: Path) -> List[Dict]:
    """Load the validated companion table CSV file using pandas.
    
    Args:
        csv_path: Path to the CSV file containing candidate detection data
        
    Returns:
        List of dictionaries with candidate data including candidate_id, x, y, 
        wavelength, contrast, uncertainty, and norm_snr_fit_free columns
    """
    candidates = []
    if csv_path.exists():
        try:
            # Use pandas to read CSV - much more efficient and robust
            df = pd.read_csv(csv_path)
            
            # Convert DataFrame to list of dictionaries
            candidates = df.to_dict('records')
            
            # Convert numeric fields to ensure they're proper floats
            numeric_fields = ['x', 'y', 'norm_snr_fit_free', 'wavelength', 'contrast', 'uncertainty']
            for candidate in candidates:
                for field in numeric_fields:
                    if field in candidate and pd.notna(candidate[field]):
                        candidate[field] = float(candidate[field])
                    elif field in candidate:
                        candidate[field] = None  # Handle NaN values explicitly
                        
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            logger.warning(f"Could not parse CSV file {csv_path}: {e}")
        except Exception as e:
            logger.warning(f"Error reading CSV file {csv_path}: {e}")
            
    return candidates


def filter_candidates_by_snr(
    candidates: List[Dict],
    snr_min: Optional[float] = None,
    snr_max: Optional[float] = None
) -> List[Dict]:
    """Filter candidates by SNR range.
    
    Args:
        candidates: List of candidate dictionaries from CSV
        snr_min: Minimum SNR threshold. If None, no lower limit
        snr_max: Maximum SNR threshold. If None, no upper limit
        
    Returns:
        Filtered list of candidates within the SNR range
    """
    if snr_min is None and snr_max is None:
        return candidates
    
    filtered_candidates = []
    for candidate in candidates:
        # Check if candidate has SNR field
        if 'norm_snr_fit_free' not in candidate:
            continue
            
        snr = candidate['norm_snr_fit_free']
        if snr is None or not np.isfinite(snr):
            continue
            
        # Apply SNR filters
        if snr_min is not None and snr < snr_min:
            continue
        if snr_max is not None and snr > snr_max:
            continue
            
        filtered_candidates.append(candidate)
    
    return filtered_candidates


def get_unique_candidates(candidates: List[Dict]) -> List[Tuple[str, float, float, float]]:
    """Extract unique candidates with their positions and SNR.
    
    Args:
        candidates: List of candidate dictionaries from CSV
        
    Returns:
        List of tuples containing (candidate_id, x, y, norm_snr_fit_free)
        with one entry per unique candidate_id
    """
    unique_candidates = {}
    for candidate in candidates:
        candidate_id = candidate['candidate_id']
        if candidate_id not in unique_candidates:
            unique_candidates[candidate_id] = (
                candidate_id,
                candidate['x'],
                candidate['y'], 
                candidate['norm_snr_fit_free']
            )
    return list(unique_candidates.values())


def plot_detection_mosaic(
    base_path: Path,
    template_type: TemplateType = "flat",
    output_path: Optional[Path] = None,
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = 100,
    vmin: float = -5,
    vmax: float = 5,
    cmap: str = "viridis",
    observation_table_path: Optional[Path] = None,
    observation_table: Optional[Table] = None,
    show_candidates: bool = True,
    candidate_circle_radius: float = 5.0,
    candidate_colors: Optional[List[str]] = None,
    candidate_text_size: Optional[int] = None,
    auto_scale: bool = False,
    individual_scaling: bool = False,
    percentile_low: float = 0.2,
    percentile_high: float = 99.9,
    margin_fraction: float = 0.0,
    snr_min: Optional[float] = None,
    snr_max: Optional[float] = None,
    combinations: Optional[List[Tuple[str, str, str]]] = None
) -> Figure:
    """Create a mosaic plot of FITS images.
    
    This function creates a mosaic visualization of TRAP results, showing detection
    images for all targets in a grid layout. Each subplot includes the target name,
    observation mode, date, and optionally exposure time and rotation from the
    observation table. Detected planet candidates can be overlaid as colored circles
    with SNR labels, using consistent colors that match the spectrum plots.
    
    Args:
        base_path: Root path containing the FITS files
        template_type: Type of template to plot ("flat", "L-type", or "T-type")
        output_path: Optional path to save the figure
        figsize: Figure size in inches. If None, automatically calculated based on number of subplots
        dpi: DPI for the figure
        vmin: Minimum value for the color scale (in sigma). Ignored if auto_scale=True or individual_scaling=True
        vmax: Maximum value for the color scale (in sigma). Ignored if auto_scale=True or individual_scaling=True
        cmap: Colormap to use
        observation_table_path: Path to the observation lookup table
        observation_table: Pre-loaded observation table
        show_candidates: Whether to overlay detected planet candidates (default: True)
        candidate_circle_radius: Radius of candidate circles in pixels (default: 5.0)
        candidate_colors: List of colors to use for candidates. If None, uses DEFAULT_CANDIDATE_COLORS
        candidate_text_size: Font size for SNR labels. If None, automatically calculated
        auto_scale: Whether to automatically determine global vmin/vmax from all data (default: False)
        individual_scaling: Whether to scale each subplot individually with its own colorbar (default: False)
        percentile_low: Lower percentile for auto-scaling (default: 1.0)
        percentile_high: Upper percentile for auto-scaling (default: 99.0)
        margin_fraction: Fraction of data range to add as margin (default: 0.1)
        snr_min: Minimum SNR threshold for candidate filtering. If None, no lower limit (default: None)
        snr_max: Maximum SNR threshold for candidate filtering. If None, no upper limit (default: None)
        
    Returns:
        The matplotlib Figure object
        
    Raises:
        ValueError: If no matching FITS files are found
    """
    # Load observation table if provided
    obs_table = load_observation_metadata(observation_table_path, observation_table)
    
    # Use default colors if not provided
    if candidate_colors is None:
        candidate_colors = DEFAULT_CANDIDATE_COLORS
    
    # Find all matching FITS files
    fits_files = get_mosaic_file_combinations(base_path, template_type, "fits", combinations=combinations)
    if not fits_files:
        raise ValueError(
            f"No matching FITS files found in {base_path} "
            f"for template type {template_type}"
        )
    
    # Auto-scale color limits if requested (global scaling only)
    if auto_scale and not individual_scaling:
        all_data = []
        for fits_path in fits_files.values():
            if fits_path is not None and fits_path.exists():
                try:
                    with fits.open(fits_path) as hdul:
                        data = np.squeeze(hdul[0].data)
                        # Only include finite values for statistics
                        finite_data = data[np.isfinite(data)]
                        if len(finite_data) > 0:
                            all_data.extend(finite_data.flatten())
                except Exception as e:
                    logger.warning(f"Could not read {fits_path} for auto-scaling: {e}")
                    
        if all_data:
            all_data = np.array(all_data)
            # Use configurable percentiles to avoid outliers
            plow, phigh = np.percentile(all_data, [percentile_low, percentile_high])
            # Add configurable margin to avoid saturation
            data_range = phigh - plow
            vmin = plow - margin_fraction * data_range
            vmax = phigh + margin_fraction * data_range
        else:
            logger.warning("No valid data found for auto-scaling, using default limits")
    
    # Set up mosaic grid
    fig, axes, n_rows, n_cols = setup_mosaic_grid(len(fits_files), figsize, dpi)
    n_files = len(fits_files)
    
    # Auto-calculate font sizes based on figure size
    if figsize is None:
        figsize = tuple(fig.get_size_inches().astype(int))
    base_font_scale = min(figsize) / 20  # Scale factor based on smaller dimension
    title_fontsize = max(8, int(12 * base_font_scale))
    
    if candidate_text_size is None:
        candidate_text_size = max(6, int(8 * base_font_scale))
    
    # Plot each FITS file
    for idx, ((target, obs_mode, date), fits_path) in enumerate(fits_files.items()):
        row = idx // n_cols
        col = idx % n_cols
        
        if fits_path is not None:
            # Read FITS file
            with fits.open(fits_path) as hdul:
                data = hdul[0].data
                # Squeeze out extra dimensions if present
                data = np.squeeze(data)
        else:
            # Create a blank image for missing files
            data = np.zeros((207, 207))  # Assuming standard size
            data.fill(np.nan)  # Fill with NaN to make it obvious
        
        # Calculate individual scaling if requested
        if individual_scaling and fits_path is not None:
            finite_data = data[np.isfinite(data)]
            if len(finite_data) > 0:
                # Use configurable percentiles for this individual image
                plow, phigh = np.percentile(finite_data, [percentile_low, percentile_high])
                data_range = phigh - plow
                subplot_vmin = plow - margin_fraction * data_range
                subplot_vmax = phigh + margin_fraction * data_range
            else:
                # Fallback to global defaults if no finite data
                subplot_vmin, subplot_vmax = vmin, vmax
        else:
            # Use global scaling
            subplot_vmin, subplot_vmax = vmin, vmax
            
        # Plot image
        im = axes[row, col].imshow(
            data,
            origin="lower",
            cmap=cmap,
            vmin=subplot_vmin,
            vmax=subplot_vmax
        )
        
        # Overlay planet candidates if requested
        if show_candidates and fits_path is not None:
            csv_path = fits_path.parent / CANDIDATE_PATTERNS[template_type].split('/')[-1]
            candidates = load_candidate_table(csv_path)
            if candidates:
                # Apply SNR filtering if specified
                candidates = filter_candidates_by_snr(candidates, snr_min, snr_max)
                if candidates:  # Check if any candidates remain after filtering
                    unique_candidates = get_unique_candidates(candidates)
                    
                    # Create a consistent mapping of candidate IDs to colors
                    candidate_id_list = sorted(set(cand_id for cand_id, _, _, _ in unique_candidates))
                    
                    for candidate_id, x, y, snr in unique_candidates:
                        # Get color index based on candidate ID position in sorted list
                        color_idx = candidate_id_list.index(candidate_id) % len(candidate_colors)
                        candidate_color = candidate_colors[color_idx]
                        
                        # Draw colored circle at candidate position
                        circle = Circle(
                            (x, y), 
                            candidate_circle_radius, 
                            color=candidate_color, 
                            fill=False, 
                            linewidth=2
                        )
                        axes[row, col].add_patch(circle)
                        
                        # Add SNR label in matching color text
                        axes[row, col].text(
                            x + candidate_circle_radius + 2, 
                            y, 
                            f"SNR={snr:.1f}",
                            color=candidate_color,
                            fontsize=candidate_text_size,
                            verticalalignment='center'
                        )
        
        # Get observation info if table is available
        title = setup_subplot_title(target, obs_mode, date, obs_table, figsize)
            
        axes[row, col].set_title(
            title,
            fontsize=title_fontsize
        )
        axes[row, col].axis("off")
        
        # Add colorbar based on scaling mode
        if individual_scaling:
            # Add colorbar to each subplot when using individual scaling
            plt.colorbar(
                im,
                ax=axes[row, col],
                label="SNR (σ)",
                shrink=0.8  # Make colorbar smaller to fit better
            )
        elif idx == n_files - 1:
            # Add single colorbar to the last subplot for global scaling
            plt.colorbar(
                im,
                ax=axes[row, col],
                label="Signal-to-Noise Ratio (σ)"
            )
    
    # Clean up empty subplots and add main title
    cleanup_mosaic_subplots(
        fig, axes, n_files, n_rows, n_cols, 
        f"Template Type: {template_type}", figsize
    )
    
    # Save figure if output path is provided
    save_mosaic_figure(fig, output_path, dpi, auto_scale=auto_scale)
    
    return fig


def plot_candidate_spectra(
    candidates: List[Dict], 
    ax: Optional[plt.Axes] = None,
    colors: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> Tuple[Figure, plt.Axes]:
    """Plot contrast spectra for a list of candidates.
    
    This function plots the contrast spectrum vs wavelength for each candidate
    in the provided list. Each candidate gets a different color for easy
    distinction. The function uses matplotlib's step function with where='mid'
    which is appropriate for spectral data measured in discrete bins.
    
    Args:
        candidates: List of candidate dictionaries containing wavelength, 
                   contrast, uncertainty, and candidate_id
        ax: Optional existing axes to plot on. If None, creates new figure
        colors: Optional list of colors to use for candidates. If None, uses default scheme
        figsize: Figure size if creating new figure
        
    Returns:
        Tuple of (figure, axes) objects
        
    Raises:
        ValueError: If no candidates with spectral data are found
    """
    # Group candidates by candidate_id to organize spectra
    candidate_spectra = {}
    for candidate in candidates:
        candidate_id = candidate['candidate_id']
        if candidate_id not in candidate_spectra:
            candidate_spectra[candidate_id] = {
                'wavelengths': [],
                'contrasts': [],
                'uncertainties': []
            }
        
        # Only add if we have the necessary spectral data columns
        # Allow NaN values but require the columns to exist
        if (all(key in candidate for key in ['wavelength', 'contrast', 'uncertainty']) and
            candidate['wavelength'] is not None and 
            candidate['contrast'] is not None and 
            candidate['uncertainty'] is not None):
            
            # Check if values are finite (not NaN/inf) before adding
            wavelength = candidate['wavelength']
            contrast = candidate['contrast']
            uncertainty = candidate['uncertainty']
            
            # Convert to float and check if finite
            try:
                wavelength = float(wavelength)
                contrast = float(contrast)
                uncertainty = float(uncertainty)
                
                # Only add if all values are finite
                if (np.isfinite(wavelength) and 
                    np.isfinite(contrast) and 
                    np.isfinite(uncertainty)):
                    candidate_spectra[candidate_id]['wavelengths'].append(wavelength)
                    candidate_spectra[candidate_id]['contrasts'].append(contrast)
                    candidate_spectra[candidate_id]['uncertainties'].append(uncertainty)
            except (ValueError, TypeError):
                # Skip entries that can't be converted to float
                continue
    
    # Filter out candidates without spectral data
    candidate_spectra = {
        cid: data for cid, data in candidate_spectra.items() 
        if len(data['wavelengths']) > 0
    }
    
    if not candidate_spectra:
        raise ValueError("No candidates with spectral data found")
    
    # Create figure if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Use default colors if not provided
    if colors is None:
        colors = DEFAULT_CANDIDATE_COLORS
    
    # Plot each candidate spectrum
    for idx, (candidate_id, spectrum_data) in enumerate(candidate_spectra.items()):
        # Convert to numpy arrays and sort by wavelength
        wavelengths = np.array(spectrum_data['wavelengths'])
        contrasts = np.array(spectrum_data['contrasts'])
        uncertainties = np.array(spectrum_data['uncertainties'])
        
        # Sort by wavelength
        sort_idx = np.argsort(wavelengths)
        wavelengths = wavelengths[sort_idx]
        contrasts = contrasts[sort_idx]
        uncertainties = uncertainties[sort_idx]
        
        # Get color for this candidate
        color = colors[idx % len(colors)]
        
        # Plot spectrum using step function
        ax.step(
            wavelengths, contrasts, 
            where='mid', 
            color=color, 
            linewidth=2,
            label=f'Candidate {candidate_id}'
        )
        
        # Add error bars
        ax.errorbar(
            wavelengths, contrasts, yerr=uncertainties,
            fmt='none',  # No markers, just error bars
            color=color,
            alpha=0.7,
            capsize=3
        )
    
    # Set labels and formatting
    ax.set_xlabel('Wavelength (micron)', fontsize=12)
    ax.set_ylabel('Contrast Spectrum', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add legend if multiple candidates
    if len(candidate_spectra) > 1:
        ax.legend(fontsize=10)
    
    # Set reasonable y-limits to avoid extreme outliers
    if len(candidate_spectra) > 0:
        all_contrasts = []
        for spectrum_data in candidate_spectra.values():
            all_contrasts.extend(spectrum_data['contrasts'])
        
        if all_contrasts:
            contrasts = np.array(all_contrasts)
            # Use percentiles to set reasonable limits
            p5, p95 = np.percentile(contrasts, [5, 95])
            contrast_range = p95 - p5
            ylim = (p5 - 0.1 * contrast_range, p95 + 0.1 * contrast_range)
            ax.set_ylim(*ylim)
    
    return fig, ax


def plot_spectrum_mosaic(
    base_path: Path,
    template_type: TemplateType = "flat",
    output_path: Optional[Path] = None,
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = 300,
    observation_table_path: Optional[Path] = None,
    observation_table: Optional[Table] = None,
    colors: Optional[List[str]] = None,
    match_detection_layout: bool = False,
    snr_min: Optional[float] = None,
    snr_max: Optional[float] = None,
    combinations: Optional[List[Tuple[str, str, str]]] = None
) -> Figure:
    """Create a mosaic plot of candidate contrast spectra.
    
    This function creates a mosaic visualization showing contrast spectra for 
    all candidates found in each target/observation combination. Each subplot
    shows the spectra of candidates detected in that specific observation,
    with the same layout and labeling as the FITS mosaic plots.
    
    Args:
        base_path: Root path containing the CSV candidate files
        template_type: Type of template to plot ("flat", "L-type", or "T-type")
        output_path: Optional path to save the figure. "_spectrum" is automatically added to the filename
        figsize: Figure size in inches. If None, automatically calculated based on number of subplots
        dpi: DPI for the figure
        observation_table_path: Path to the observation lookup table
        observation_table: Pre-loaded observation table
        colors: Optional list of colors to use for candidates. If None, uses default scheme
        match_detection_layout: If True, use the same grid layout as detection mosaic (based on FITS files),
                               showing blank subplots where CSV files are missing. If False, only show
                               subplots for existing CSV files (compact layout). Default: False
        snr_min: Minimum SNR threshold for candidate filtering. If None, no lower limit (default: None)
        snr_max: Maximum SNR threshold for candidate filtering. If None, no upper limit (default: None)
        
    Returns:
        The matplotlib Figure object
        
    Raises:
        ValueError: If no matching CSV files are found
    """
    # Load observation table if provided
    obs_table = load_observation_metadata(observation_table_path, observation_table)
    
    # Determine layout mode and get file combinations
    if match_detection_layout:
        # Use FITS files to determine the grid layout (same as detection mosaic)
        fits_files = get_mosaic_file_combinations(base_path, template_type, "fits", combinations=combinations)
        if not fits_files:
            raise ValueError(f"No FITS files found in {base_path} for template type {template_type}")

        # Get CSV files for the same combinations
        csv_files = get_mosaic_file_combinations(base_path, template_type, "csv", combinations=combinations)
        
        # Use all FITS combinations for layout, but track which have CSV data
        layout_combinations = list(fits_files.keys())
        plot_layout = "detection_match"
    else:
        # Compact layout: only use existing CSV files
        csv_files = get_mosaic_file_combinations(base_path, template_type, "csv", combinations=combinations)
        if not csv_files:
            raise ValueError(f"No CSV files found in {base_path} for template type {template_type}")
        
        # Filter to only existing CSV files
        layout_combinations = [(k, v) for k, v in csv_files.items() if v is not None and v.exists()]
        layout_combinations = [k for k, v in layout_combinations]
        plot_layout = "compact"
    
    # Load candidate data from existing CSV files
    candidate_files = {}
    for target, obs_mode, date in layout_combinations:
        csv_path = csv_files.get((target, obs_mode, date))
        
        if csv_path is not None and csv_path.exists():
            candidates = load_candidate_table(csv_path)
            
            if candidates:
                # Apply SNR filtering if specified
                candidates = filter_candidates_by_snr(candidates, snr_min, snr_max)
                
                if candidates:  # Check if any candidates remain after filtering
                    # Check for spectral data columns
                    spectral_columns = ['wavelength', 'contrast', 'uncertainty']
                    available_columns = list(candidates[0].keys())
                    missing_columns = [col for col in spectral_columns if col not in available_columns]
                    
                    if missing_columns:
                        logger.warning(f"Missing spectral columns {missing_columns} in {target}/{obs_mode}/{date}")
                        continue
                    
                    # Count unique candidates
                    unique_candidate_ids = set()
                    for candidate in candidates:
                        if 'candidate_id' in candidate:
                            unique_candidate_ids.add(candidate['candidate_id'])
                    
                    # Only include if we have candidates with spectral data
                    spectral_candidates = [
                        c for c in candidates 
                        if all(key in c for key in ['wavelength', 'contrast', 'uncertainty'])
                    ]
                    if spectral_candidates:
                        candidate_files[(target, obs_mode, date)] = spectral_candidates
                        logger.info(f"Processing {target}/{obs_mode}/{date}: {len(unique_candidate_ids)} candidates")
                    else:
                        logger.warning(f"No spectral data found in {target}/{obs_mode}/{date}")
        else:
            if match_detection_layout:
                # For detection layout matching, we expect some missing CSV files
                # Only log if explicitly requested or in debug mode
                pass
            else:
                logger.warning(f"CSV file not found for {target}/{obs_mode}/{date}")
    
    # Check if we have any data to plot
    if not candidate_files and plot_layout == "compact":
        raise ValueError(
            f"No candidate files with spectral data found in {base_path} "
            f"for template type {template_type}. "
            f"Check that CSV files contain 'wavelength', 'contrast', and 'uncertainty' columns."
        )
    elif not layout_combinations:
        raise ValueError(f"No combinations found in {base_path} for template type {template_type}")
    
    logger.info(f"Found {len(candidate_files)} observations with spectral candidates ({plot_layout} layout)")
    
    # Set up mosaic grid based on layout mode
    if match_detection_layout:
        # Use all combinations for grid size (matching detection mosaic)
        fig, axes, n_rows, n_cols = setup_mosaic_grid(len(layout_combinations), figsize, dpi)
        n_files = len(layout_combinations)
    else:
        # Use only combinations with data for grid size (compact)
        fig, axes, n_rows, n_cols = setup_mosaic_grid(len(candidate_files), figsize, dpi)
        n_files = len(candidate_files)
    
    # Auto-calculate font sizes based on figure size
    if figsize is None:
        figsize = tuple(fig.get_size_inches().astype(int))
    base_font_scale = min(figsize) / 20
    title_fontsize = max(8, int(12 * base_font_scale))
    
    # Use default colors if not provided
    if colors is None:
        colors = DEFAULT_CANDIDATE_COLORS
    
    # Plot each observation pair
    if match_detection_layout:
        # Use the same order as FITS files (matching detection mosaic layout)
        for idx, (target, obs_mode, date) in enumerate(layout_combinations):
            row = idx // n_cols
            col_detection = idx % n_cols
            col_spectrum = col_detection + 1
            
            # Check if we have candidate data for this combination
            candidates = candidate_files.get((target, obs_mode, date))
            
            if candidates:
                try:
                    # Plot candidate spectra for this observation
                    plot_candidate_spectra(
                        candidates, 
                        ax=axes[row, col_spectrum],
                        colors=colors
                    )
                except ValueError:
                    # No spectral data - create empty plot
                    axes[row, col_spectrum].text(
                        0.5, 0.5, 'No spectral data',
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=axes[row, col_spectrum].transAxes,
                        fontsize=12
                    )
                    axes[row, col_spectrum].set_xlabel('Wavelength (micron)')
                    axes[row, col_spectrum].set_ylabel('Contrast Spectrum')
            else:
                # No CSV file or no data - create blank plot (similar to missing FITS)
                axes[row, col_spectrum].text(
                    0.5, 0.5, 'No candidate data',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=axes[row, col_spectrum].transAxes,
                    fontsize=12,
                    color='gray'
                )
                axes[row, col_spectrum].set_xlabel('Wavelength (micron)')
                axes[row, col_spectrum].set_ylabel('Contrast Spectrum')
                axes[row, col_spectrum].grid(True, alpha=0.3)
            
            # Get observation info if table is available
            title = setup_subplot_title(target, obs_mode, date, obs_table, figsize)
                
            axes[row, col_detection].set_title(
                title,
                fontsize=title_fontsize
            )
            axes[row, col_detection].axis("off")
            
            # === SPECTRUM PANEL (RIGHT) ===
            spectrum_ax = axes[row, col_spectrum]
            
            if csv_path is not None and csv_path.exists():
                candidates = load_candidate_table(csv_path)
                if candidates:
                    # Check for spectral data
                    spectral_candidates = [
                        c for c in candidates 
                        if all(key in c for key in ['wavelength', 'contrast', 'uncertainty'])
                    ]
                    
                    if spectral_candidates:
                        try:
                            plot_candidate_spectra(
                                spectral_candidates, 
                                ax=spectrum_ax,
                                colors=colors
                            )
                            # Remove legend to avoid clutter
                            legend = spectrum_ax.get_legend()
                            if legend:
                                legend.remove()
                        except ValueError:
                            # No valid spectral data
                            spectrum_ax.text(
                                0.5, 0.5, 'No spectral data',
                                horizontalalignment='center',
                                verticalalignment='center',
                                transform=spectrum_ax.transAxes,
                                fontsize=12
                            )
                            spectrum_ax.set_xlabel('Wavelength (micron)')
                            spectrum_ax.set_ylabel('Contrast Spectrum')
                            spectrum_ax.grid(True, alpha=0.3)
                    else:
                        # No spectral columns
                        spectrum_ax.text(
                            0.5, 0.5, 'No spectral data',
                            horizontalalignment='center',
                            verticalalignment='center',
                            transform=spectrum_ax.transAxes,
                            fontsize=12
                        )
                        spectrum_ax.set_xlabel('Wavelength (micron)')
                        spectrum_ax.set_ylabel('Contrast Spectrum')
                        spectrum_ax.grid(True, alpha=0.3)
                else:
                    # Empty CSV
                    spectrum_ax.text(
                        0.5, 0.5, 'No candidate data',
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=spectrum_ax.transAxes,
                        fontsize=12,
                        color='gray'
                    )
                    spectrum_ax.set_xlabel('Wavelength (micron)')
                    spectrum_ax.set_ylabel('Contrast Spectrum')
                    spectrum_ax.grid(True, alpha=0.3)
            else:
                # No CSV file
                spectrum_ax.text(
                    0.5, 0.5, 'No candidate data',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=spectrum_ax.transAxes,
                    fontsize=12,
                    color='gray'
                )
                spectrum_ax.set_xlabel('Wavelength (micron)')
                spectrum_ax.set_ylabel('Contrast Spectrum')
                spectrum_ax.grid(True, alpha=0.3)
            
            spectrum_ax.set_title(f"Candidate Spectra\n{title}", fontsize=title_fontsize)
    else:
        # Compact layout: only plot combinations with data
        for idx, ((target, obs_mode, date), candidates) in enumerate(candidate_files.items()):
            row = idx // n_cols
            col = idx % n_cols
            
            try:
                # Plot candidate spectra for this observation
                plot_candidate_spectra(
                    candidates, 
                    ax=axes[row, col],
                    colors=colors
                )
            except ValueError:
                # No spectral data - create empty plot
                axes[row, col].text(
                    0.5, 0.5, 'No spectral data',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=axes[row, col].transAxes,
                    fontsize=12
                )
                axes[row, col].set_xlabel('Wavelength (micron)')
                axes[row, col].set_ylabel('Contrast Spectrum')
            
            # Get observation info if table is available
            title = setup_subplot_title(target, obs_mode, date, obs_table, figsize)
                
            axes[row, col].set_title(
                title,
                fontsize=title_fontsize
            )
            
            # Remove legend for individual subplots to avoid clutter
            legend = axes[row, col].get_legend()
            if legend:
                legend.remove()
    
    # Clean up empty subplots and add main title
    cleanup_mosaic_subplots(
        fig, axes, n_files, n_rows, n_cols, 
        f"Candidate Contrast Spectra - Template Type: {template_type}", figsize
    )
    
    # Save figure if output path is provided
    save_mosaic_figure(fig, output_path, dpi)
    
    return fig


def plot_combined_mosaic(
    base_path: Path,
    template_type: TemplateType = "flat",
    output_path: Optional[Path] = None,
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = 300,
    vmin: float = -5,
    vmax: float = 5,
    cmap: str = "viridis",
    observation_table_path: Optional[Path] = None,
    observation_table: Optional[Table] = None,
    show_candidates: bool = True,
    candidate_circle_radius: float = 5.0,
    candidate_colors: Optional[List[str]] = None,
    candidate_text_size: Optional[int] = None,
    auto_scale: bool = False,
    individual_scaling: bool = False,
    percentile_low: float = 0.2,
    percentile_high: float = 99.9,
    margin_fraction: float = 0.0,
    snr_min: Optional[float] = None,
    snr_max: Optional[float] = None,
    combinations: Optional[List[Tuple[str, str, str]]] = None
) -> Figure:
    """Create a combined mosaic plot showing detection images and candidate spectra side by side.
    
    This function creates a mosaic visualization where each observation gets two panels:
    - Left panel: Detection image (from FITS file)
    - Right panel: Candidate contrast spectra (from CSV file)
    
    The layout uses an even number of columns to maintain consistency, with pairs of
    panels for each observation. Missing files result in blank panels.
    
    Args:
        base_path: Root path containing the FITS and CSV files
        template_type: Type of template to plot ("flat", "L-type", or "T-type")
        output_path: Optional path to save the figure. "_combined" is automatically added to the filename
        figsize: Figure size in inches. If None, automatically calculated based on number of subplots
        dpi: DPI for the figure
        vmin: Minimum value for the color scale (in sigma). Ignored if auto_scale=True or individual_scaling=True
        vmax: Maximum value for the color scale (in sigma). Ignored if auto_scale=True or individual_scaling=True
        cmap: Colormap to use for detection images
        observation_table_path: Path to the observation lookup table
        observation_table: Pre-loaded observation table
        show_candidates: Whether to overlay detected planet candidates on detection images (default: True)
        candidate_circle_radius: Radius of candidate circles in pixels (default: 5.0)
        candidate_colors: List of colors to use for candidates. If None, uses DEFAULT_CANDIDATE_COLORS
        candidate_text_size: Font size for SNR labels. If None, automatically calculated
        auto_scale: Whether to automatically determine global vmin/vmax from all data (default: False)
        individual_scaling: Whether to scale each subplot individually with its own colorbar (default: False)
        percentile_low: Lower percentile for auto-scaling (default: 0.2)
        percentile_high: Upper percentile for auto-scaling (default: 99.9)
        margin_fraction: Fraction of data range to add as margin (default: 0.0)
        snr_min: Minimum SNR threshold for candidate filtering. If None, no lower limit (default: None)
        snr_max: Maximum SNR threshold for candidate filtering. If None, no upper limit (default: None)
        
    Returns:
        The matplotlib Figure object
        
    Raises:
        ValueError: If no observations are found
    """
    # Load observation table if provided
    obs_table = load_observation_metadata(observation_table_path, observation_table)
    
    # Use default colors if not provided
    if candidate_colors is None:
        candidate_colors = DEFAULT_CANDIDATE_COLORS
    
    # Get all combinations and file paths
    fits_files = get_mosaic_file_combinations(base_path, template_type, "fits", combinations=combinations)
    csv_files = get_mosaic_file_combinations(base_path, template_type, "csv", combinations=combinations)

    # Use all combinations that have either FITS or CSV files
    all_combinations = set(fits_files.keys()) | set(csv_files.keys())
    if not all_combinations:
        raise ValueError(f"No observations found in {base_path} for template type {template_type}")
    
    # Sort combinations for consistent ordering
    sorted_combinations = sorted(all_combinations)
    n_observations = len(sorted_combinations)
    
    # Calculate grid dimensions - we need 2 columns per observation (detection + spectrum)
    # Use even number of columns for consistency
    n_cols_per_obs = 2
    n_cols_total = n_cols_per_obs * min(3, n_observations)  # Max 3 observations per row (was 4)
    n_rows = int(np.ceil(n_observations * n_cols_per_obs / n_cols_total))
    
    # Auto-calculate figure size if not provided
    if figsize is None:
        # Make it wider since we have pairs of plots, but also taller for titles
        subplot_width = max(4, min(8, 30 / (n_cols_total // 2)))  # Increased base width
        subplot_height = max(5, min(8, 25 / n_rows))  # Increased base height for titles
        figsize = (int(n_cols_total * subplot_width), int(n_rows * subplot_height))
    
    # Create figure
    fig, axes = plt.subplots(
        n_rows, n_cols_total,
        figsize=figsize,
        dpi=dpi,
        constrained_layout=False
    )
    axes = np.atleast_2d(axes)
    
    # Auto-scale color limits if requested (global scaling only for FITS images)
    if auto_scale and not individual_scaling:
        all_data = []
        for combination in sorted_combinations:
            fits_path = fits_files.get(combination)
            if fits_path is not None and fits_path.exists():
                try:
                    with fits.open(fits_path) as hdul:
                        data = np.squeeze(hdul[0].data)
                        finite_data = data[np.isfinite(data)]
                        if len(finite_data) > 0:
                            all_data.extend(finite_data.flatten())
                except Exception as e:
                    logger.warning(f"Could not read {fits_path} for auto-scaling: {e}")
                    
        if all_data:
            all_data = np.array(all_data)
            plow, phigh = np.percentile(all_data, [percentile_low, percentile_high])
            data_range = phigh - plow
            vmin = plow - margin_fraction * data_range
            vmax = phigh + margin_fraction * data_range
        else:
            logger.warning("No valid data found for auto-scaling, using default limits")
    
    # Auto-calculate font sizes based on figure size
    base_font_scale = min(figsize) / 25  # Reduced scaling factor for combined plots
    title_fontsize = max(6, int(10 * base_font_scale))  # Smaller titles for combined plots
    
    if candidate_text_size is None:
        candidate_text_size = max(5, int(7 * base_font_scale))  # Smaller candidate text
    
    # Plot each observation pair
    for obs_idx, (target, obs_mode, date) in enumerate(sorted_combinations):
        # Calculate subplot positions
        panel_start_idx = obs_idx * n_cols_per_obs
        row = panel_start_idx // n_cols_total
        col_detection = panel_start_idx % n_cols_total
        col_spectrum = col_detection + 1
        
        # Skip if we've exceeded the grid
        if row >= n_rows or col_spectrum >= n_cols_total:
            break
        
        # Get file paths
        fits_path = fits_files.get((target, obs_mode, date))
        csv_path = csv_files.get((target, obs_mode, date))
        
        # Generate compact title for both panels
        title = setup_compact_subplot_title(target, obs_mode, date, obs_table, figsize)
        
        # === DETECTION IMAGE PANEL (LEFT) ===
        detection_ax = axes[row, col_detection]
        
        if fits_path is not None and fits_path.exists():
            # Read and plot FITS file
            with fits.open(fits_path) as hdul:
                data = np.squeeze(hdul[0].data)
            
            # Calculate scaling
            if individual_scaling:
                finite_data = data[np.isfinite(data)]
                if len(finite_data) > 0:
                    plow, phigh = np.percentile(finite_data, [percentile_low, percentile_high])
                    data_range = phigh - plow
                    subplot_vmin = plow - margin_fraction * data_range
                    subplot_vmax = phigh + margin_fraction * data_range
                else:
                    subplot_vmin, subplot_vmax = vmin, vmax
            else:
                subplot_vmin, subplot_vmax = vmin, vmax
            
            # Plot image
            im = detection_ax.imshow(
                data,
                origin="lower",
                cmap=cmap,
                vmin=subplot_vmin,
                vmax=subplot_vmax
            )
            
            # Overlay candidates if requested
            if show_candidates and csv_path is not None and csv_path.exists():
                candidates = load_candidate_table(csv_path)
                if candidates:
                    # Apply SNR filtering if specified
                    if snr_min is not None or snr_max is not None:
                        candidates = filter_candidates_by_snr(candidates, snr_min, snr_max)
                    unique_candidates = get_unique_candidates(candidates)
                    candidate_id_list = sorted(set(cand_id for cand_id, _, _, _ in unique_candidates))
                    
                    for candidate_id, x, y, snr in unique_candidates:
                        color_idx = candidate_id_list.index(candidate_id) % len(candidate_colors)
                        candidate_color = candidate_colors[color_idx]
                        
                        circle = Circle(
                            (x, y), 
                            candidate_circle_radius, 
                            color=candidate_color, 
                            fill=False, 
                            linewidth=2
                        )
                        detection_ax.add_patch(circle)
                        
                        detection_ax.text(
                            x + candidate_circle_radius + 2, 
                            y, 
                            f"SNR={snr:.1f}",
                            color=candidate_color,
                            fontsize=candidate_text_size,
                            verticalalignment='center'
                        )
            
            # Add colorbar for individual scaling
            if individual_scaling:
                plt.colorbar(im, ax=detection_ax, label="SNR (σ)", shrink=0.8)
        else:
            # Create blank detection image
            data = np.zeros((207, 207))
            data.fill(np.nan)
            detection_ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
            detection_ax.text(
                0.5, 0.5, 'No detection data',
                horizontalalignment='center',
                verticalalignment='center',
                transform=detection_ax.transAxes,
                fontsize=12,
                color='gray'
            )
        
        detection_ax.set_title(f"Detection Image\n{title}", fontsize=title_fontsize)
        detection_ax.axis("off")
        
        # === SPECTRUM PANEL (RIGHT) ===
        spectrum_ax = axes[row, col_spectrum]
        
        if csv_path is not None and csv_path.exists():
            candidates = load_candidate_table(csv_path)
            if candidates:
                # Apply SNR filtering if specified
                if snr_min is not None or snr_max is not None:
                    candidates = filter_candidates_by_snr(candidates, snr_min, snr_max)
                # Check for spectral data
                spectral_candidates = [
                    c for c in candidates 
                    if all(key in c for key in ['wavelength', 'contrast', 'uncertainty'])
                ]
                
                if spectral_candidates:
                    try:
                        plot_candidate_spectra(
                            spectral_candidates, 
                            ax=spectrum_ax,
                            colors=candidate_colors
                        )
                        # Remove legend to avoid clutter
                        legend = spectrum_ax.get_legend()
                        if legend:
                            legend.remove()
                    except ValueError:
                        # No valid spectral data
                        spectrum_ax.text(
                            0.5, 0.5, 'No spectral data',
                            horizontalalignment='center',
                            verticalalignment='center',
                            transform=spectrum_ax.transAxes,
                            fontsize=12
                        )
                        spectrum_ax.set_xlabel('Wavelength (micron)')
                        spectrum_ax.set_ylabel('Contrast Spectrum')
                        spectrum_ax.grid(True, alpha=0.3)
                else:
                    # No spectral columns
                    spectrum_ax.text(
                        0.5, 0.5, 'No spectral data',
                        horizontalalignment='center',
                        verticalalignment='center',
                        transform=spectrum_ax.transAxes,
                        fontsize=12
                    )
                    spectrum_ax.set_xlabel('Wavelength (micron)')
                    spectrum_ax.set_ylabel('Contrast Spectrum')
                    spectrum_ax.grid(True, alpha=0.3)
            else:
                # Empty CSV
                spectrum_ax.text(
                    0.5, 0.5, 'No candidate data',
                    horizontalalignment='center',
                    verticalalignment='center',
                    transform=spectrum_ax.transAxes,
                    fontsize=12,
                    color='gray'
                )
                spectrum_ax.set_xlabel('Wavelength (micron)')
                spectrum_ax.set_ylabel('Contrast Spectrum')
                spectrum_ax.grid(True, alpha=0.3)
        else:
            # No CSV file
            spectrum_ax.text(
                0.5, 0.5, 'No candidate data',
                horizontalalignment='center',
                verticalalignment='center',
                transform=spectrum_ax.transAxes,
                fontsize=12,
                color='gray'
            )
            spectrum_ax.set_xlabel('Wavelength (micron)')
            spectrum_ax.set_ylabel('Contrast Spectrum')
            spectrum_ax.grid(True, alpha=0.3)
    
    # Add single colorbar for global scaling
    if not individual_scaling and sorted_combinations:
        # Add colorbar to the last detection image that was plotted
        last_obs_idx = min(len(sorted_combinations) - 1, (n_rows * n_cols_total // n_cols_per_obs) - 1)
        last_panel_start = last_obs_idx * n_cols_per_obs
        last_row = last_panel_start // n_cols_total
        last_col = last_panel_start % n_cols_total
        
        if last_row < n_rows and last_col < n_cols_total:
            # Get the last detection image that was actually plotted
            for obs_idx in range(len(sorted_combinations) - 1, -1, -1):
                panel_start_idx = obs_idx * n_cols_per_obs
                row = panel_start_idx // n_cols_total
                col_detection = panel_start_idx % n_cols_total
                
                if row < n_rows and col_detection < n_cols_total:
                    target, obs_mode, date = sorted_combinations[obs_idx]
                    fits_path = fits_files.get((target, obs_mode, date))
                    if fits_path is not None and fits_path.exists():
                        # Add colorbar to this detection plot
                        detection_ax = axes[row, col_detection]
                        images = [child for child in detection_ax.get_children() if hasattr(child, 'get_array')]
                        if images:
                            plt.colorbar(images[0], ax=detection_ax, label="Signal-to-Noise Ratio (σ)")
                        break
    
    # Remove empty subplots
    total_panels_used = len(sorted_combinations) * n_cols_per_obs
    for idx in range(total_panels_used, n_rows * n_cols_total):
        row = idx // n_cols_total
        col = idx % n_cols_total
        if row < n_rows and col < n_cols_total:
            fig.delaxes(axes[row, col])
    
    # Add main title with more space
    suptitle_fontsize = max(12, int(16 * base_font_scale))  # Smaller main title
    fig.suptitle(
        f"Combined Detection and Spectrum Mosaic - Template Type: {template_type}",
        fontsize=suptitle_fontsize,
        y=0.97  # Move down slightly to give more space
    )
    
    # Apply constrained layout instead of tight_layout for better handling
    try:
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave more space at top
    except UserWarning:
        # If tight_layout fails, use manual spacing
        plt.subplots_adjust(
            left=0.05, right=0.95, 
            top=0.92, bottom=0.08,
            hspace=0.4, wspace=0.3  # More spacing between subplots
        )
    
    # Save figure if output path is provided
    save_mosaic_figure(fig, output_path, dpi)
    
    return fig


# Batching wrapper functions

def _build_batch_filename(
    content: str,
    template_type: TemplateType,
    suffix: Optional[str],
    batch_idx: int,
    total_batches: int,
    output_format: str,
) -> str:
    """Build the filename for one batch mosaic.

    Mirrors the single-file naming (content_mosaic_template[_suffix]) and
    appends the batch marker and requested extension, so batched output honors
    --format and --suffix the same way single-file output does.
    """
    stem = f"{content}_mosaic_{template_type}"
    if suffix:
        stem += f"_{suffix}"
    return f"{stem}_batch_{batch_idx + 1:02d}_of_{total_batches:02d}.{output_format}"


def plot_detection_mosaic_batched(
    base_path: Path,
    template_type: TemplateType = "flat",
    batch_size: int = 30,
    output_dir: Optional[Path] = None,
    output_format: str = "png",
    suffix: Optional[str] = None,
    **kwargs
) -> List[Figure]:
    """Create batched detection mosaic plots.
    
    Args:
        base_path: Root path containing the FITS files
        template_type: Type of template to plot ("flat", "L-type", or "T-type")
        batch_size: Maximum observations per mosaic (default: 30)
        output_dir: Directory to save batched figures. If None, no files are saved
        output_format: File extension/format for saved batches (default: "png")
        suffix: Optional filename suffix so subset runs do not overwrite
        **kwargs: All other arguments passed to plot_detection_mosaic()

    Returns:
        List of matplotlib Figure objects
    """
    # Get all valid FITS combinations
    fits_files = get_mosaic_file_combinations(base_path, template_type, "fits")
    valid_combinations = [
        combo for combo, fits_path in fits_files.items()
        if fits_path is not None and fits_path.exists()
    ]

    if not valid_combinations:
        raise ValueError(
            f"No template-matched FITS files found in {base_path} "
            f"for template type {template_type}"
        )

    # Sort and create batches
    sorted_combinations = sorted(valid_combinations)
    batches = [
        sorted_combinations[i:i + batch_size]
        for i in range(0, len(sorted_combinations), batch_size)
    ]

    logger.info(f"Creating {len(batches)} batches with up to {batch_size} observations each")

    figures = []
    for batch_idx, batch_combinations in enumerate(batches):
        # Generate output path for this batch
        batch_output = None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            batch_filename = _build_batch_filename(
                "detection", template_type, suffix, batch_idx, len(batches), output_format
            )
            batch_output = output_dir / batch_filename
        
        logger.info(f"Processing batch {batch_idx + 1}/{len(batches)} ({len(batch_combinations)} observations)")
        
        fig = _plot_detection_mosaic_for_batch(
            base_path, template_type, batch_combinations, batch_output, 
            batch_idx + 1, len(batches), **kwargs
        )
        figures.append(fig)
    
    return figures


def plot_spectrum_mosaic_batched(
    base_path: Path,
    template_type: TemplateType = "flat",
    batch_size: int = 30,
    output_dir: Optional[Path] = None,
    output_format: str = "png",
    suffix: Optional[str] = None,
    **kwargs
) -> List[Figure]:
    """Create batched spectrum mosaic plots.

    Args:
        base_path: Root path containing the CSV files
        template_type: Type of template to plot ("flat", "L-type", or "T-type")
        batch_size: Maximum observations per mosaic (default: 30)
        output_dir: Directory to save batched figures. If None, no files are saved
        output_format: File extension/format for saved batches (default: "png")
        suffix: Optional filename suffix so subset runs do not overwrite
        **kwargs: All other arguments passed to plot_spectrum_mosaic()

    Returns:
        List of matplotlib Figure objects
    """
    # For spectrum mosaics, we determine valid combinations based on FITS files
    # but only include those that also have CSV data
    fits_files = get_mosaic_file_combinations(base_path, template_type, "fits")
    csv_files = get_mosaic_file_combinations(base_path, template_type, "csv")
    
    valid_combinations = []
    for combo, fits_path in fits_files.items():
        if fits_path is not None and fits_path.exists():
            csv_path = csv_files.get(combo)
            if csv_path is not None and csv_path.exists():
                valid_combinations.append(combo)
    
    if not valid_combinations:
        raise ValueError(
            f"No observations with both FITS and CSV files found in {base_path} "
            f"for template type {template_type}"
        )
    
    # Sort and create batches
    sorted_combinations = sorted(valid_combinations)
    batches = [
        sorted_combinations[i:i + batch_size] 
        for i in range(0, len(sorted_combinations), batch_size)
    ]
    
    logger.info(f"Creating {len(batches)} spectrum batches with up to {batch_size} observations each")
    
    figures = []
    for batch_idx, batch_combinations in enumerate(batches):
        # Generate output path for this batch
        batch_output = None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            batch_filename = _build_batch_filename(
                "spectrum", template_type, suffix, batch_idx, len(batches), output_format
            )
            batch_output = output_dir / batch_filename
        
        logger.info(f"Processing spectrum batch {batch_idx + 1}/{len(batches)} ({len(batch_combinations)} observations)")
        
        fig = _plot_spectrum_mosaic_for_batch(
            base_path, template_type, batch_combinations, batch_output,
            batch_idx + 1, len(batches), **kwargs
        )
        figures.append(fig)
    
    return figures


def plot_combined_mosaic_batched(
    base_path: Path,
    template_type: TemplateType = "flat",
    batch_size: int = 30,
    output_dir: Optional[Path] = None,
    output_format: str = "png",
    suffix: Optional[str] = None,
    **kwargs
) -> List[Figure]:
    """Create batched combined mosaic plots.

    Args:
        base_path: Root path containing the FITS and CSV files
        template_type: Type of template to plot ("flat", "L-type", or "T-type")
        batch_size: Maximum observations per mosaic (default: 30)
        output_dir: Directory to save batched figures. If None, no files are saved
        output_format: File extension/format for saved batches (default: "png")
        suffix: Optional filename suffix so subset runs do not overwrite
        **kwargs: All other arguments passed to plot_combined_mosaic()

    Returns:
        List of matplotlib Figure objects
    """
    # For combined mosaics, we use the same logic as the original function
    fits_files = get_mosaic_file_combinations(base_path, template_type, "fits")
    valid_combinations = [
        combo for combo, fits_path in fits_files.items() 
        if fits_path is not None and fits_path.exists()
    ]
    
    if not valid_combinations:
        raise ValueError(
            f"No template-matched FITS files found in {base_path} "
            f"for template type {template_type}"
        )
    
    # Sort and create batches
    sorted_combinations = sorted(valid_combinations)
    batches = [
        sorted_combinations[i:i + batch_size] 
        for i in range(0, len(sorted_combinations), batch_size)
    ]
    
    logger.info(f"Creating {len(batches)} combined batches with up to {batch_size} observations each")
    
    figures = []
    for batch_idx, batch_combinations in enumerate(batches):
        # Generate output path for this batch
        batch_output = None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            batch_filename = _build_batch_filename(
                "combined", template_type, suffix, batch_idx, len(batches), output_format
            )
            batch_output = output_dir / batch_filename
        
        logger.info(f"Processing combined batch {batch_idx + 1}/{len(batches)} ({len(batch_combinations)} observations)")
        
        fig = _plot_combined_mosaic_for_batch(
            base_path, template_type, batch_combinations, batch_output,
            batch_idx + 1, len(batches), **kwargs
        )
        figures.append(fig)
    
    return figures


def _plot_detection_mosaic_for_batch(
    base_path: Path,
    template_type: TemplateType,
    batch_combinations: List[Tuple[str, str, str]],
    output_path: Optional[Path],
    batch_num: int,
    total_batches: int,
    **kwargs
) -> Figure:
    """Plot detection mosaic for a specific batch."""
    fig = plot_detection_mosaic(
        base_path=base_path,
        template_type=template_type,
        output_path=output_path,
        combinations=batch_combinations,
        **kwargs
    )
    fig.suptitle(
        f"Detection Mosaic - Template Type: {template_type} (Batch {batch_num}/{total_batches})",
        fontsize=16,
    )
    return fig


def _plot_spectrum_mosaic_for_batch(
    base_path: Path,
    template_type: TemplateType,
    batch_combinations: List[Tuple[str, str, str]],
    output_path: Optional[Path],
    batch_num: int,
    total_batches: int,
    **kwargs
) -> Figure:
    """Plot spectrum mosaic for a specific batch."""
    fig = plot_spectrum_mosaic(
        base_path=base_path,
        template_type=template_type,
        output_path=output_path,
        combinations=batch_combinations,
        **kwargs
    )
    fig.suptitle(
        f"Spectrum Mosaic - Template Type: {template_type} (Batch {batch_num}/{total_batches})",
        fontsize=16,
    )
    return fig


def _plot_combined_mosaic_for_batch(
    base_path: Path,
    template_type: TemplateType,
    batch_combinations: List[Tuple[str, str, str]],
    output_path: Optional[Path],
    batch_num: int,
    total_batches: int,
    **kwargs
) -> Figure:
    """Plot combined mosaic for a specific batch."""
    fig = plot_combined_mosaic(
        base_path=base_path,
        template_type=template_type,
        output_path=output_path,
        combinations=batch_combinations,
        **kwargs
    )
    fig.suptitle(
        f"Combined Mosaic - Template Type: {template_type} (Batch {batch_num}/{total_batches})",
        fontsize=16,
    )
    return fig