import os
from glob import glob

import numpy as np
from astropy.io import fits
from natsort import natsorted


def bundle_IFS_output_into_cubes(key, cube_outputdir, output_type='resampled', overwrite=True):
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

    extracted_dir = os.path.join(cube_outputdir, key) + '/'
    converted_dir = os.path.join(cube_outputdir, 'converted') + '/'
    if not os.path.exists(converted_dir):
        os.makedirs(converted_dir)

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
        raise ValueError('Invalid output_type selected.')

    science_files = natsorted(
        glob(os.path.join(extracted_dir, glob_pattern), recursive=False))

    if len(science_files) == 0:
        print('No output found in: {}'.format(extracted_dir))
        return None

    data_cube = []
    inverse_variance_cube = []

    for file in science_files:
        hdus = fits.open(file)
        data_cube.append(hdus[1].data.astype('float32'))
        inverse_variance_cube.append(hdus[2].data.astype('float32'))
        hdus.close()

    data_cube = np.array(data_cube, dtype='float32')
    inverse_variance_cube = np.array(inverse_variance_cube, dtype='float32')

    data_cube = np.swapaxes(data_cube, 0, 1)
    inverse_variance_cube = np.swapaxes(inverse_variance_cube, 0, 1)

    fits.writeto(os.path.join(converted_dir, '{}_{}cube.fits'.format(
        key, name_suffix).lower()), data_cube.astype('float32'), overwrite=overwrite)
    fits.writeto(os.path.join(converted_dir, '{}_{}ivar_cube.fits'.format(
        key, name_suffix).lower()), inverse_variance_cube.astype('float32'), overwrite=overwrite)


def run_bundle_output(
    frame_types_to_extract,
    cube_outputdir,
    converted_dir,
    extraction_parameters,
    instrument,
    non_least_square_methods,
    overwrite_bundle,
    bundle_hexagons,
    bundle_residuals
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

    for key in frame_types_to_extract:
        if bundle_hexagons:
            bundle_IFS_output_into_cubes(
                key, cube_outputdir, output_type='hexagons', overwrite=overwrite_bundle)
        if bundle_residuals:
            bundle_IFS_output_into_cubes(
                key, cube_outputdir, output_type='residuals', overwrite=overwrite_bundle)
        bundle_IFS_output_into_cubes(
            key, cube_outputdir, output_type='resampled', overwrite=overwrite_bundle)

    if (extraction_parameters['method'] in non_least_square_methods) \
            and extraction_parameters['linear_wavelength']:
        wavelengths = np.linspace(
            instrument.wavelength_range[0].value,
            instrument.wavelength_range[1].value,
            39
        )
    else:
        wavelengths = instrument.lam_midpts

    fits.writeto(os.path.join(converted_dir, 'wavelengths.fits'),
                 wavelengths, overwrite=overwrite_bundle)
