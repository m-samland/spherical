"""
Cleanup utilities for SPHERE IFS data reduction pipeline.

This module provides utility functions to check pipeline completion status
and clean up intermediate files after successful reductions. These functions
are designed to be called manually for maintenance and storage management.
"""

from pathlib import Path
from typing import Union

from spherical.database.ifs_observation import IFSObservation
from spherical.database.irdis_observation import IRDISObservation
from spherical.pipeline.pipeline_config import IFSReductionConfig
from spherical.pipeline.toolbox import make_target_folder_string


def check_cube_building_success(
    observation: Union[IFSObservation, IRDISObservation],
    config: IFSReductionConfig,
    method: str = 'optext',
    frame_types_to_check: list[str] | None = None,
    check_optional_files: bool = True
) -> tuple[bool, list[str]]:
    """
    Check if cube building step completed successfully for a specific target.

    Verifies that all expected output files from the bundle_output and 
    cube_header_update steps exist for the given observation. This includes
    bundled data cubes, inverse variance cubes, wavelength solution, and
    metadata files.

    Parameters
    ----------
    observation : IFSObservation or IRDISObservation
        Observation object to check completion for.
    config : IFSReductionConfig
        Pipeline configuration containing directory paths.
    method : str, optional
        Extraction method used (e.g., 'optext'). Default is 'optext'.
    frame_types_to_check : list of str, optional
        Frame types to check (e.g., ['CORO', 'CENTER', 'FLUX']). 
        If None, checks all standard frame types.
    check_optional_files : bool, optional
        Whether to check for optional files like hexagon and residual cubes.
        Default is True.

    Returns
    -------
    tuple[bool, list[str]]
        (success, missing_files)
        - success: True if all expected files exist
        - missing_files: List of missing file paths

    Examples
    --------
    >>> from spherical.pipeline.cleanup import check_cube_building_success
    >>> success, missing = check_cube_building_success(observation, config)
    >>> if success:
    ...     print("Cube building completed successfully!")
    >>> else:
    ...     print(f"Missing files: {missing}")
    """
    if frame_types_to_check is None:
        frame_types_to_check = ['CORO', 'CENTER', 'FLUX']

    # Get the converted directory path
    instrument = str(observation.observation['INSTRUMENT'][0]).lower()
    name_mode_date = make_target_folder_string(observation)
    
    if config.directories.reduction_directory is None:
        raise ValueError("Reduction directory not configured")
        
    converted_dir = Path(config.directories.reduction_directory) / f"{instrument.upper()}/observation" / name_mode_date / method / "converted"

    missing_files = []

    # Check core bundled files for each frame type
    for frame_type in frame_types_to_check:
        frame_lower = frame_type.lower()
        
        # Main cube and inverse variance cube
        cube_file = converted_dir / f"{frame_lower}_cube.fits"
        ivar_file = converted_dir / f"{frame_lower}_ivar_cube.fits"
        
        if not cube_file.exists():
            missing_files.append(str(cube_file))
        if not ivar_file.exists():
            missing_files.append(str(ivar_file))
        
        # Optional files if requested
        if check_optional_files:
            hex_file = converted_dir / f"{frame_lower}_hexagons_cube.fits"
            res_file = converted_dir / f"{frame_lower}_residuals_cube.fits"
            angle_file = converted_dir / f"{frame_lower}_parallactic_angles.fits"
            
            if hex_file.exists() or res_file.exists():  # Only check if they exist
                if not hex_file.exists():
                    missing_files.append(str(hex_file))
                if not res_file.exists():
                    missing_files.append(str(res_file))
            
            if not angle_file.exists():
                missing_files.append(str(angle_file))

    # Check wavelength solution
    wavelength_file = converted_dir / "wavelengths.fits"
    if not wavelength_file.exists():
        missing_files.append(str(wavelength_file))

    success = len(missing_files) == 0
    return success, missing_files


def clean_raw_data(
    observation: Union[IFSObservation, IRDISObservation],
    config: IFSReductionConfig,
    frame_types_to_clean: list[str] | None = None,
    dry_run: bool = True
) -> tuple[list[str], float]:
    """
    Clean raw data files for a specific observation.

    Removes raw FITS files downloaded from the ESO archive after successful
    processing. This helps manage storage space by removing large raw files
    that are no longer needed.

    Parameters
    ----------
    observation : IFSObservation or IRDISObservation
        Observation object to clean raw data for.
    config : IFSReductionConfig
        Pipeline configuration containing directory paths.
    frame_types_to_clean : list of str, optional
        Frame types to clean (e.g., ['CORO', 'CENTER', 'FLUX']). 
        If None, cleans all frame types.
    dry_run : bool, optional
        If True, only report what would be deleted without actually deleting.
        Default is True for safety.

    Returns
    -------
    tuple[list[str], float]
        (deleted_files, total_size_mb)
        - deleted_files: List of deleted file paths
        - total_size_mb: Total size of deleted files in MB

    Warning
    -------
    This function permanently deletes files. Use dry_run=True first to verify
    what will be deleted. Ensure cube building was successful before cleaning.

    Examples
    --------
    >>> # First check what would be deleted
    >>> files, size = clean_raw_data(observation, config, dry_run=True)
    >>> print(f"Would delete {len(files)} files totaling {size:.1f} MB")
    >>> 
    >>> # Actually delete files
    >>> files, size = clean_raw_data(observation, config, dry_run=False)
    """
    if frame_types_to_clean is None:
        frame_types_to_clean = ['CORO', 'CENTER', 'FLUX', 'WAVECAL']

    # Get the raw data directory structure
    instrument = str(observation.observation['INSTRUMENT'][0]).lower()
    name_mode_date = make_target_folder_string(observation)
    target_name, obs_band, date = name_mode_date.split('/')
    
    if config.directories.raw_directory is None:
        raise ValueError("Raw directory not configured")
        
    raw_base = Path(config.directories.raw_directory) / f"{instrument.upper()}"
    
    deleted_files = []
    total_size = 0

    # Clean science frames
    science_frames = ['CORO', 'CENTER', 'FLUX']
    for frame_type in frame_types_to_clean:
        if frame_type in science_frames:
            frame_dir = raw_base / "science" / target_name / obs_band / date / frame_type
        elif frame_type == 'WAVECAL':
            frame_dir = raw_base / "calibration" / obs_band / "WAVECAL"
        else:
            continue  # Skip unknown frame types

        if frame_dir.exists():
            # Find all FITS files in the directory
            fits_files = list(frame_dir.glob("*.fits"))
            
            for fits_file in fits_files:
                if fits_file.exists():
                    file_size = fits_file.stat().st_size
                    total_size += file_size
                    deleted_files.append(str(fits_file))
                    
                    if not dry_run:
                        fits_file.unlink()
            
            # Remove empty directories if not dry run
            if not dry_run and frame_dir.exists():
                try:
                    frame_dir.rmdir()  # Only removes if empty
                except OSError:
                    pass  # Directory not empty, leave it

    total_size_mb = total_size / (1024 * 1024)
    return deleted_files, total_size_mb


def clean_extracted_cubes(
    observation: Union[IFSObservation, IRDISObservation],
    config: IFSReductionConfig,
    method: str = 'optext',
    frame_types_to_clean: list[str] | None = None,
    dry_run: bool = True
) -> tuple[list[str], float]:
    """
    Clean intermediate extracted cube files.

    Removes individual extracted cube files from the cube_outputdir after 
    successful bundling. These are the individual SPHER.*cube*.fits files
    created during cube extraction that are bundled into master cubes.

    Parameters
    ----------
    observation : IFSObservation or IRDISObservation
        Observation object to clean extracted cubes for.
    config : IFSReductionConfig
        Pipeline configuration containing directory paths.
    method : str, optional
        Extraction method used (e.g., 'optext'). Default is 'optext'.
    frame_types_to_clean : list of str, optional
        Frame types to clean (e.g., ['CORO', 'CENTER', 'FLUX']). 
        If None, cleans all frame types.
    dry_run : bool, optional
        If True, only report what would be deleted without actually deleting.
        Default is True for safety.

    Returns
    -------
    tuple[list[str], float]
        (deleted_files, total_size_mb)
        - deleted_files: List of deleted file paths
        - total_size_mb: Total size of deleted files in MB

    Warning
    -------
    This function permanently deletes files. Use dry_run=True first to verify
    what will be deleted. Ensure bundling was successful before cleaning.

    Examples
    --------
    >>> # Check what would be deleted
    >>> files, size = clean_extracted_cubes(observation, config, dry_run=True)
    >>> print(f"Would delete {len(files)} extracted cube files ({size:.1f} MB)")
    >>>
    >>> # Actually delete files  
    >>> files, size = clean_extracted_cubes(observation, config, dry_run=False)
    """
    if frame_types_to_clean is None:
        frame_types_to_clean = ['CORO', 'CENTER', 'FLUX']

    # Get the cube output directory
    instrument = str(observation.observation['INSTRUMENT'][0]).lower()
    name_mode_date = make_target_folder_string(observation)
    
    if config.directories.reduction_directory is None:
        raise ValueError("Reduction directory not configured")
        
    cube_outputdir = Path(config.directories.reduction_directory) / f"{instrument.upper()}/observation" / name_mode_date / method

    deleted_files = []
    total_size = 0

    # Patterns for different types of extracted cube files
    cube_patterns = [
        'SPHER.*cube_resampled_DIT*.fits',  # Resampled cubes
        'SPHER.*cube_DIT*.fits',            # Native cubes  
        'SPHER.*cube_residuals_DIT*.fits'   # Residual cubes
    ]

    for frame_type in frame_types_to_clean:
        frame_dir = cube_outputdir / frame_type
        
        if frame_dir.exists():
            for pattern in cube_patterns:
                cube_files = list(frame_dir.glob(pattern))
                
                for cube_file in cube_files:
                    if cube_file.exists():
                        file_size = cube_file.stat().st_size
                        total_size += file_size
                        deleted_files.append(str(cube_file))
                        
                        if not dry_run:
                            cube_file.unlink()
            
            # Remove empty frame directory if not dry run
            if not dry_run and frame_dir.exists():
                try:
                    frame_dir.rmdir()  # Only removes if empty
                except OSError:
                    pass  # Directory not empty, leave it

    total_size_mb = total_size / (1024 * 1024)
    return deleted_files, total_size_mb


def clean_wavelength_calibrations(
    observation: Union[IFSObservation, IRDISObservation],
    config: IFSReductionConfig,
    dry_run: bool = True
) -> tuple[list[str], float]:
    """
    Clean wavelength calibration output files.

    Removes wavelength calibration products after successful cube extraction.
    These are the calibration key files created during wavelength calibration
    that are used by the cube extraction step.

    Parameters
    ----------
    observation : IFSObservation or IRDISObservation
        Observation object to clean wavelength calibration files for.
    config : IFSReductionConfig
        Pipeline configuration containing directory paths.
    dry_run : bool, optional
        If True, only report what would be deleted without actually deleting.
        Default is True for safety.

    Returns
    -------
    tuple[list[str], float]
        (deleted_files, total_size_mb)
        - deleted_files: List of deleted file paths  
        - total_size_mb: Total size of deleted files in MB

    Warning
    -------
    This function permanently deletes files. Use dry_run=True first to verify
    what will be deleted. Ensure cube extraction was successful before cleaning.

    Examples
    --------
    >>> # Check what would be deleted
    >>> files, size = clean_wavelength_calibrations(observation, config, dry_run=True)
    >>> print(f"Would delete {len(files)} calibration files ({size:.1f} MB)")
    >>>
    >>> # Actually delete files
    >>> files, size = clean_wavelength_calibrations(observation, config, dry_run=False)
    """
    # Get the wavelength calibration directory
    instrument = str(observation.observation['INSTRUMENT'][0]).lower()
    name_mode_date = make_target_folder_string(observation)
    obs_band = str(observation.observation['FILTER'][0])
    
    if config.directories.reduction_directory is None:
        raise ValueError("Reduction directory not configured")
        
    wavecal_outputdir = Path(config.directories.reduction_directory) / f"{instrument.upper()}/calibration" / obs_band / name_mode_date

    deleted_files = []
    total_size = 0

    if wavecal_outputdir.exists():
        # Find all calibration key files (typically *.fits files in the wavecal directory)
        cal_files = list(wavecal_outputdir.glob("*.fits"))
        
        for cal_file in cal_files:
            if cal_file.exists():
                file_size = cal_file.stat().st_size
                total_size += file_size
                deleted_files.append(str(cal_file))
                
                if not dry_run:
                    cal_file.unlink()
        
        # Remove empty directories if not dry run
        if not dry_run and wavecal_outputdir.exists():
            try:
                wavecal_outputdir.rmdir()  # Only removes if empty
                # Try to remove parent directories if empty
                try:
                    wavecal_outputdir.parent.rmdir()  # obs_band directory
                    wavecal_outputdir.parent.parent.rmdir()  # calibration directory
                except OSError:
                    pass  # Directories not empty, leave them
            except OSError:
                pass  # Directory not empty, leave it

    total_size_mb = total_size / (1024 * 1024)
    return deleted_files, total_size_mb


def clean_all_intermediate_files(
    observation: Union[IFSObservation, IRDISObservation],
    config: IFSReductionConfig,
    method: str = 'optext',
    clean_raw: bool = False,
    clean_extracted: bool = True,
    clean_wavecal: bool = True,
    dry_run: bool = True
) -> dict[str, tuple[list[str], float]]:
    """
    Clean all intermediate files for an observation in one call.

    Convenient wrapper function that can clean raw data, extracted cubes,
    and wavelength calibrations in a single operation. Provides detailed
    reporting of what files would be or were deleted.

    Parameters
    ----------
    observation : IFSObservation or IRDISObservation
        Observation object to clean files for.
    config : IFSReductionConfig
        Pipeline configuration containing directory paths.
    method : str, optional
        Extraction method used (e.g., 'optext'). Default is 'optext'.
    clean_raw : bool, optional
        Whether to clean raw data files. Default is False for safety.
    clean_extracted : bool, optional
        Whether to clean extracted cube files. Default is True.
    clean_wavecal : bool, optional
        Whether to clean wavelength calibration files. Default is True.
    dry_run : bool, optional
        If True, only report what would be deleted without actually deleting.
        Default is True for safety.

    Returns
    -------
    dict[str, tuple[list[str], float]]
        Dictionary with cleanup results for each category:
        - 'raw': (deleted_files, size_mb) for raw data
        - 'extracted': (deleted_files, size_mb) for extracted cubes  
        - 'wavecal': (deleted_files, size_mb) for wavelength calibrations

    Examples
    --------
    >>> # Check what would be cleaned
    >>> results = clean_all_intermediate_files(observation, config, dry_run=True)
    >>> total_files = sum(len(files) for files, _ in results.values())
    >>> total_size = sum(size for _, size in results.values())
    >>> print(f"Would delete {total_files} files totaling {total_size:.1f} MB")
    >>>
    >>> # Actually clean intermediate files (but not raw data)
    >>> results = clean_all_intermediate_files(
    ...     observation, config, clean_raw=False, dry_run=False
    ... )
    """
    results = {}

    if clean_raw:
        results['raw'] = clean_raw_data(observation, config, dry_run=dry_run)
    else:
        results['raw'] = ([], 0.0)

    if clean_extracted:
        results['extracted'] = clean_extracted_cubes(observation, config, method=method, dry_run=dry_run)
    else:
        results['extracted'] = ([], 0.0)

    if clean_wavecal:
        results['wavecal'] = clean_wavelength_calibrations(observation, config, dry_run=dry_run)
    else:
        results['wavecal'] = ([], 0.0)

    return results
