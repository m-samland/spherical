import os
from glob import glob

import numpy as np
from astropy.io import fits
from natsort import natsorted

from spherical.pipeline.logging_utils import optional_logger


@optional_logger
def bundle_IFS_output_into_cubes(key, cube_outputdir, logger, output_type='resampled', overwrite=True):
    """
    Assemble individual IFS data cubes and associated metadata into 4D master cubes for a given frame type.

    This function collects all extracted IFS data cubes (e.g., for 'CORO', 'CENTER', or 'FLUX' frames) from a specified directory,
    stacks them along the temporal axis, and writes out combined data, inverse variance, and parallactic angle arrays as FITS files.
    It supports different output types (resampled, hexagons, residuals) and is typically called as the final step of the IFS pipeline
    for each frame type.

    Parameters
    ----------
    key : str
        Frame type to bundle (e.g., 'CORO', 'CENTER', 'FLUX').
    cube_outputdir : str
        Path to the directory containing extracted cubes for this frame type.
    logger : logging.Logger
        Logger instance to use for logging messages.
    output_type : {'resampled', 'hexagons', 'residuals'}, optional
        Which type of cube to bundle. 'resampled' (default) for wavelength-resampled cubes,
        'hexagons' for native hexagonal grid cubes, 'residuals' for residual cubes.
    overwrite : bool, optional
        If True (default), overwrite existing output FITS files.

    Returns
    -------
    None
        The function writes output FITS files to disk and returns None.

    Notes
    -----
    - The output cubes have shape (n_wave, n_time, ny, nx), with axes swapped for compatibility with downstream analysis.
    - Parallactic angles are extracted from the FITS headers if available and saved as a separate FITS file.
    - This function is called automatically by the pipeline after cube extraction, but can also be run manually.

    Examples
    --------
    >>> bundle_IFS_output_into_cubes('CORO', '/path/to/output', output_type='resampled')
    >>> bundle_IFS_output_into_cubes('CENTER', '/path/to/output', output_type='hexagons', overwrite=False)
    """

    step_name = f"bundle_output_{key.lower()}_{output_type}"
    logger.info(f"Bundling {key} cubes from {cube_outputdir} (output_type={output_type})", extra={"step": step_name, "status": "started"})

    extracted_dir = os.path.join(cube_outputdir, key)
    converted_dir = os.path.join(cube_outputdir, 'converted')
    os.makedirs(converted_dir, exist_ok=True)

    if output_type == 'resampled':
        glob_pattern = 'SPHER.*cube_resampled_DIT*.fits'
        name_suffix = ''
    elif output_type == 'hexagons':
        glob_pattern = 'SPHER.*cube_DIT*.fits'
        name_suffix = 'hexagons_'
    elif output_type == 'residuals':
        glob_pattern = 'SPHER.*cube_residuals_DIT*.fits'
        name_suffix = 'residuals_'
    else:
        logger.error(f"Invalid output_type '{output_type}' provided.", extra={"step": step_name, "status": "failed"})
        raise ValueError("Invalid output_type selected.")

    science_files = natsorted(
        glob(os.path.join(extracted_dir, glob_pattern), recursive=False))

    if len(science_files) == 0:
        logger.warning(f"No output files found in: {extracted_dir}", extra={"step": step_name, "status": "failed"})
        return None

    logger.info(f"Found {len(science_files)} cube files to process.")

    data_cube = []
    inverse_variance_cube = []
    parallactic_angles = []

    for file in science_files:
        try:
            hdus = fits.open(file)
            data_cube.append(hdus[1].data.astype('float32'))
            inverse_variance_cube.append(hdus[2].data.astype('float32'))
            try:
                angle = hdus[0].header['HIERARCH DEROT ANGLE']
                parallactic_angles.append(angle)
            except KeyError:
                logger.debug(f"No parallactic angle in {file}")
            finally:
                hdus.close()
        except Exception:
            logger.exception(f"Failed to read file {file}", extra={"step": step_name, "status": "failed"})
            continue

    if not data_cube:
        logger.error("No valid data cubes could be loaded.", extra={"step": step_name, "status": "failed"})
        return None

    data_cube = np.swapaxes(np.array(data_cube, dtype='float32'), 0, 1)
    inverse_variance_cube = np.swapaxes(np.array(inverse_variance_cube, dtype='float32'), 0, 1)

    data_path = os.path.join(converted_dir, f'{key}_{name_suffix}cube.fits'.lower())
    ivar_path = os.path.join(converted_dir, f'{key}_{name_suffix}ivar_cube.fits'.lower())

    try:
        fits.writeto(data_path, data_cube, overwrite=overwrite)
        fits.writeto(ivar_path, inverse_variance_cube, overwrite=overwrite)
        logger.info(f"Wrote bundled data cube to: {data_path}")
        logger.info(f"Wrote bundled inverse variance cube to: {ivar_path}")
    except Exception:
        logger.exception("Failed to write output cubes.", extra={"step": step_name, "status": "failed"})
        return None

    if parallactic_angles:
        angle_path = os.path.join(converted_dir, f'{key}_parallactic_angles.fits'.lower())
        try:
            fits.writeto(angle_path, np.array(parallactic_angles, dtype='float32'), overwrite=overwrite)
            logger.info(f"Wrote parallactic angles to: {angle_path}")
        except Exception:
            logger.exception("Failed to write parallactic angles.", extra={"step": step_name, "status": "failed"})

    logger.info(f"Finished bundling IFS cubes for {key}.", extra={"step": step_name, "status": "success"})

@optional_logger
def run_bundle_output(
    frame_types_to_extract,
    cube_outputdir,
    converted_dir,
    extraction_parameters,
    instrument,
    non_least_square_methods,
    overwrite_bundle,
    bundle_hexagons,
    bundle_residuals,
    logger,
):
    """Bundle extracted IFS data cubes into master cubes and write wavelength solution.

    This is the fourth step in the SPHERE/IFS data reduction pipeline. It combines
    individual extracted data cubes into master cubes for each frame type and
    writes the wavelength solution. The function can create three types of output:
    resampled cubes (default), hexagon cubes, and residual cubes.

    Required Input Files
    -------------------
    From previous step (extract_cubes):
    - cube_outputdir/CORO/
        - SPHER.*cube_resampled_DIT*.fits
        - SPHER.*cube_DIT*.fits (for hexagons)
        - SPHER.*cube_residuals_DIT*.fits (for residuals)
    - cube_outputdir/CENTER/
        - SPHER.*cube_resampled_DIT*.fits
        - SPHER.*cube_DIT*.fits (for hexagons)
        - SPHER.*cube_residuals_DIT*.fits (for residuals)
    - cube_outputdir/FLUX/
        - SPHER.*cube_resampled_DIT*.fits
        - SPHER.*cube_DIT*.fits (for hexagons)
        - SPHER.*cube_residuals_DIT*.fits (for residuals)

    Generated Output Files
    ---------------------
    In converted_dir:
    - coro_cube.fits
        Master cube of coronagraphic data (n_wave, n_time, ny, nx)
    - coro_ivar_cube.fits
        Inverse variance cube for coronagraphic data
    - center_cube.fits
        Master cube of center data
    - center_ivar_cube.fits
        Inverse variance cube for center data
    - flux_cube.fits
        Master cube of flux data
    - flux_ivar_cube.fits
        Inverse variance cube for flux data
    - coro_hexagons_cube.fits (if bundle_hexagons)
        Master cube in native hexagonal grid
    - coro_residuals_cube.fits (if bundle_residuals)
        Master cube of residuals
    - wavelengths.fits
        Wavelength solution array

    Parameters
    ----------
    frame_types_to_extract : list of str
        Frame types to bundle (e.g., ["FLUX", "CENTER", "CORO"]).
    cube_outputdir : str
        Directory containing extracted cubes from previous step.
    converted_dir : str
        Directory where bundled outputs will be written.
    extraction_parameters : dict
        Extraction parameters, must include:
        - method: str
            Extraction method used
        - linear_wavelength: bool
            Whether to use linear wavelength sampling
    instrument : object
        Instrument object with:
        - wavelength_range: tuple
            Min and max wavelengths
        - lam_midpts: array
            Wavelength midpoints for non-linear sampling
    non_least_square_methods : list of str
        List of extraction methods that are not least-squares.
    overwrite_bundle : bool
        Whether to overwrite existing bundle outputs.
    bundle_hexagons : bool
        Whether to bundle hexagon outputs.
    bundle_residuals : bool
        Whether to bundle residual outputs.
    logger : logging.Logger
        Logger instance to use for logging messages.

    Returns
    -------
    None
        This function writes bundled cubes to disk and does not return a value.

    Notes
    -----
    - Creates converted_dir if it doesn't exist
    - For non-least-square methods with linear wavelength sampling,
      generates a linear wavelength array with 39 points
    - For other cases, uses instrument's wavelength midpoints
    - All output cubes are in float32 format
    - Data axes are swapped to (n_wave, n_time, ny, nx) for compatibility

    Examples
    --------
    >>> run_bundle_output(
    ...     frame_types_to_extract=["CORO", "CENTER", "FLUX"],
    ...     cube_outputdir="/path/to/cubes",
    ...     converted_dir="/path/to/converted",
    ...     extraction_parameters={
    ...         "method": "optext",
    ...         "linear_wavelength": True
    ...     },
    ...     instrument=charis.instruments.SPHERE('YJ'),
    ...     non_least_square_methods=["optext", "apphot3", "apphot5"],
    ...     overwrite_bundle=True,
    ...     bundle_hexagons=True,
    ...     bundle_residuals=True
    ... )
    """

    logger.info("Starting bundle output step.", extra={"step": "bundle_output", "status": "started"})
    logger.debug(f"Parameters: frame_types={frame_types_to_extract}, "
                 f"method={extraction_parameters['method']}, "
                 f"linear_wavelength={extraction_parameters['linear_wavelength']}, "
                 f"overwrite_bundle={overwrite_bundle}, "
                 f"bundle_hexagons={bundle_hexagons}, bundle_residuals={bundle_residuals}")

    for key in frame_types_to_extract:
        logger.info(f"Processing frame type: {key}")
        try:
            if bundle_hexagons:
                bundle_IFS_output_into_cubes(
                    key, cube_outputdir, logger=logger, output_type='hexagons',
                    overwrite=overwrite_bundle)

            if bundle_residuals:
                bundle_IFS_output_into_cubes(
                    key, cube_outputdir, logger=logger, output_type='residuals',
                    overwrite=overwrite_bundle)

            bundle_IFS_output_into_cubes(
                key, cube_outputdir, logger=logger, output_type='resampled',
                overwrite=overwrite_bundle)

        except Exception:
            logger.exception(f"Failed to bundle cubes for frame type: {key}", extra={"step": f"bundle_output_{key.lower()}", "status": "failed"})

    logger.info("Bundling wavelength solution array...")
    try:
        if (extraction_parameters['method'] in non_least_square_methods) and \
           extraction_parameters['linear_wavelength']:
            wavelengths = np.linspace(
                instrument.wavelength_range[0].value,
                instrument.wavelength_range[1].value,
                39
            )
            logger.debug("Generated linear wavelength solution.")
        else:
            wavelengths = instrument.lam_midpts
            logger.debug("Using instrument-provided wavelength midpoints.")

        output_path = os.path.join(converted_dir, 'wavelengths.fits')
        fits.writeto(output_path, wavelengths.astype('float32'), overwrite=overwrite_bundle)
        logger.info(f"Wavelength solution written to: {output_path}")

    except Exception:
        logger.exception("Failed to write wavelength solution.", extra={"step": "bundle_output_wavelengths", "status": "failed"})

    logger.info("Finished bundle output step.", extra={"step": "bundle_output", "status": "success"})